"""Orquestra retrieval + geração: recupera chunks, monta o prompt,
transmite a resposta do LLM token a token e resolve as citações no
final.

Segue o mesmo padrão de DI por Protocol do resto do projeto — em teste,
`retriever` e `llm_client` são fakes (ver tests/fakes.py), sem precisar
de Qdrant/sentence-transformers/API key da Anthropic instalados.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from app.generation.citations import build_citations
from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt

_NO_CONTEXT_ANSWER = (
    "Não encontrei nenhum trecho relevante na base de conhecimento para "
    "responder essa pergunta."
)


class Retriever(Protocol):
    def retrieve(self, query: str, mode: str, top_k: int) -> list[dict]: ...


class StreamingLLM(Protocol):
    def stream(self, system: str, user: str) -> Iterator[str]: ...


class AnswerGenerator:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: StreamingLLM,
        default_mode: str = "hybrid_rerank_blend",
        default_top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.default_mode = default_mode
        self.default_top_k = default_top_k

    def stream(
        self, query: str, mode: str | None = None, top_k: int | None = None
    ) -> Iterator[dict]:
        """Gera eventos incrementais:
          {"type": "retrieved", "chunks": [...], "mode": ...} — uma vez, no início
          {"type": "delta", "text": "..."}                     — a cada pedaço de texto
          {"type": "done", "answer": "...", "citations": [...], "mode": ...} — no final
        """
        mode = mode or self.default_mode
        top_k = top_k or self.default_top_k

        chunks = self.retriever.retrieve(query, mode=mode, top_k=top_k)
        yield {"type": "retrieved", "chunks": chunks, "mode": mode}

        if not chunks:
            yield {"type": "delta", "text": _NO_CONTEXT_ANSWER}
            yield {
                "type": "done",
                "answer": _NO_CONTEXT_ANSWER,
                "citations": [],
                "mode": mode,
            }
            return

        user_prompt = build_user_prompt(query, chunks)

        pieces: list[str] = []
        for delta in self.llm_client.stream(SYSTEM_PROMPT, user_prompt):
            pieces.append(delta)
            yield {"type": "delta", "text": delta}

        answer = "".join(pieces)
        citations = build_citations(answer, chunks)
        yield {"type": "done", "answer": answer, "citations": citations, "mode": mode}

    def answer(
        self, query: str, mode: str | None = None, top_k: int | None = None
    ) -> dict:
        """Versão não-streaming: consome stream() até o fim e devolve só
        o resultado final. Usada pela avaliação de Answer Faithfulness
        (app/evaluation/faithfulness.py), que não precisa de token a
        token — só do texto completo + contexto usado."""
        result: dict = {}
        for event in self.stream(query, mode=mode, top_k=top_k):
            if event["type"] == "retrieved":
                result["chunks"] = event["chunks"]
            elif event["type"] == "done":
                result["answer"] = event["answer"]
                result["citations"] = event["citations"]
                result["mode"] = event["mode"]
        return result
