"""API FastAPI da Fase 5: endpoint /query com streaming (Server-Sent
Events) e citação de fontes, endpoint /health pra checar se os
componentes carregaram.

Rodar:
    python scripts/run_api.py --reload
    # ou diretamente:
    uvicorn app.api.main:app --reload

Testar (a resposta chega em pedaços, -N desativa o buffer do curl):
    curl -N -X POST http://localhost:8000/query \\
        -H 'Content-Type: application/json' \\
        -d '{"question": "Como funciona a autenticação da API?"}'
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import Depends, FastAPI

from fastapi.responses import StreamingResponse

from app.api.dependencies import build_generator, get_generator
from app.api.schemas import QueryRequest
from app.config.settings import settings
from app.generation.pipeline import AnswerGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = build_generator()
    yield


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)

logger = logging.getLogger("rag_api")


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_events(generator: AnswerGenerator, req: QueryRequest) -> Iterator[str]:
    """Formata os eventos de AnswerGenerator.stream() como SSE.

    Uma falha no meio do streaming (rate limit, billing, timeout da
    Anthropic etc.) não pode virar um HTTP 500 normal — a resposta já
    começou (status 200 + o evento "retrieved" já foi enviado), então o
    Starlette não tem mais como trocar o status code. Sem esse
    try/except, a exceção simplesmente derruba a conexão TCP no meio
    (é o "transfer closed with outstanding read data remaining" que o
    curl reporta) e quem está chamando a API não sabe por quê. Em vez
    disso, emitimos um evento "error" explícito e fechamos o stream de
    forma limpa.
    """
    try:
        for event in generator.stream(req.question, mode=req.mode, top_k=req.top_k):
            if event["type"] == "retrieved":
                yield _sse(
                    "retrieved",
                    {
                        "mode": event["mode"],
                        "retrieved": [
                            {
                                "source": c["payload"].get("source"),
                                "section": c["payload"].get("section"),
                                "page": c["payload"].get("page"),
                            }
                            for c in event["chunks"]
                        ],
                    },
                )
            elif event["type"] == "delta":
                yield _sse("delta", {"text": event["text"]})
            elif event["type"] == "done":
                yield _sse(
                    "done",
                    {
                        "answer": event["answer"],
                        "citations": event["citations"],
                        "mode": event["mode"],
                    },
                )
    except Exception as exc:  # noqa: BLE001 — qualquer falha vira um evento "error" pro cliente
        logger.exception("Erro durante a geração da resposta para a pergunta: %r", req.question)
        yield _sse("error", {"message": str(exc)})


@app.post("/query")
def query(req: QueryRequest, generator: AnswerGenerator = Depends(get_generator)):
    return StreamingResponse(
        _sse_events(generator, req), media_type="text/event-stream"
    )


@app.get("/health")
def health(generator: AnswerGenerator = Depends(get_generator)):
    return {
        "status": "ok",
        "qdrant_collection": settings.qdrant_collection,
        "bm25_loaded": generator.retriever.sparse_index is not None,
    }
