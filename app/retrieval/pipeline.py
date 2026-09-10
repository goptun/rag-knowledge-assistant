"""Orquestra os modos de retrieval comparados na Fase 4 de avaliação:
vector_only, hybrid (RRF), hybrid_rerank (RRF + cross-encoder) e
hybrid_rerank_blend (RRF + cross-encoder, mas sem deixar o cross-encoder
sobrescrever 100% o ranking — ver app/retrieval/blend.py).

Todos os componentes (embedder, vector_store, sparse_index, reranker)
são recebidos por injeção de dependência — permite testar a
orquestração com fakes, sem sentence-transformers/qdrant-client/
rank_bm25 instalados. Ver tests/fakes.py e tests/test_retrieval.py.
"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.blend import blend_scores
from app.retrieval.hybrid import reciprocal_rank_fusion

VALID_MODES = ("vector_only", "hybrid", "hybrid_rerank", "hybrid_rerank_blend")


class QueryEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DenseSearcher(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[dict]: ...


class SparseSearcher(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[dict], top_k: int | None
    ) -> list[dict]: ...


class HybridRetriever:
    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: DenseSearcher,
        sparse_index: SparseSearcher | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_pool_size: int = 20,
        rerank_blend_alpha: float = 0.7,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.rerank_blend_alpha = rerank_blend_alpha

    def retrieve(
        self, query: str, mode: str = "hybrid_rerank", top_k: int = 5
    ) -> list[dict]:
        if mode not in VALID_MODES:
            raise ValueError(f"mode inválido: '{mode}'. Use: {', '.join(VALID_MODES)}")

        query_vector = self.embedder.embed([query])[0]

        if mode == "vector_only":
            return self.vector_store.search(query_vector, top_k=top_k)

        if self.sparse_index is None:
            raise RuntimeError(f"mode='{mode}' precisa de um sparse_index (BM25).")

        dense_results = self.vector_store.search(
            query_vector, top_k=self.candidate_pool_size
        )
        sparse_results = self.sparse_index.search(
            query, top_k=self.candidate_pool_size
        )
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results], k=self.rrf_k
        )

        if mode == "hybrid":
            return fused[:top_k]

        # mode in ("hybrid_rerank", "hybrid_rerank_blend")
        if self.reranker is None:
            raise RuntimeError(f"mode='{mode}' precisa de um reranker.")
        candidates = fused[: self.candidate_pool_size]

        if mode == "hybrid_rerank":
            return self.reranker.rerank(query, candidates, top_k=top_k)

        # mode == "hybrid_rerank_blend": pega o rerank_score de TODOS os
        # candidatos (top_k=None, sem cortar) pra poder combinar com o
        # rrf_score antes de decidir o corte final.
        reranked = self.reranker.rerank(query, candidates, top_k=None)
        blended = blend_scores(reranked, alpha=self.rerank_blend_alpha)
        return blended[:top_k]
