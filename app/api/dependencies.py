"""Monta os componentes reais do pipeline (embedder, Qdrant, BM25,
reranker, LLM) uma única vez, no startup da API — carregar os modelos
locais (bge-base, bge-reranker) é lento (alguns segundos), então isso
acontece uma vez só, não a cada request. Ver app/api/main.py (lifespan).

Import de bibliotecas pesadas (sentence-transformers, qdrant-client) é
lazy dentro de build_generator() pelo mesmo motivo do resto do projeto:
tests/test_api.py substitui essa dependência inteira por fakes via
app.dependency_overrides, então esse import nunca precisa rodar em
teste.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from app.config.settings import settings
from app.generation.llm_client import AnthropicLLMClient, LLMClient, OpenAICompatibleLLMClient
from app.generation.pipeline import AnswerGenerator
from app.retrieval.pipeline import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent.parent


def build_llm_client() -> LLMClient:
    """Escolhe o cliente de LLM com base em LLM_PROVIDER:
      - "anthropic": API oficial da Anthropic (precisa de ANTHROPIC_API_KEY
        com crédito).
      - "openai_compatible": qualquer proxy que fale o protocolo de chat
        completions da OpenAI — usado aqui pro 9Router local do usuário
        (LLM_BASE_URL=http://localhost:20128/v1), que expõe um combo de
        vários modelos atrás de um único endpoint sem custo de API externa.
    """
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens=settings.generation_max_tokens,
            temperature=settings.generation_temperature,
        )
    return AnthropicLLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        max_tokens=settings.generation_max_tokens,
        temperature=settings.generation_temperature,
    )


def build_generator() -> AnswerGenerator:
    from app.indexing.bm25_store import BM25Index
    from app.indexing.embeddings import HuggingFaceEmbedder
    from app.indexing.qdrant_store import QdrantStore
    from app.retrieval.reranker import CrossEncoderReranker

    embedder = HuggingFaceEmbedder(settings.embedding_model)
    vector_store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)

    bm25_path = ROOT / settings.bm25_index_path
    sparse_index = BM25Index.load(str(bm25_path)) if bm25_path.exists() else None

    reranker = CrossEncoderReranker(settings.reranker_model)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=sparse_index,
        reranker=reranker,
    )

    llm_client = build_llm_client()

    return AnswerGenerator(retriever, llm_client)


def get_generator(request: Request) -> AnswerGenerator:
    """Dependency do FastAPI — devolve a instância construída no startup
    (app.state.generator). Em teste, substituída inteiramente via
    app.dependency_overrides[get_generator], então o corpo desta função
    nunca roda."""
    return request.app.state.generator
