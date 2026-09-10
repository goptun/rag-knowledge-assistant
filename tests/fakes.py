"""Fakes usados para testar o pipeline de indexação sem dependências pesadas
(sentence-transformers, qdrant-client) instaladas."""

from __future__ import annotations

import hashlib

from app.chunking.models import Chunk


class FakeEmbedder:
    """Gera vetores determinísticos (hash do texto) de dimensão fixa.

    Não tem nenhuma qualidade semântica — serve só para testar a
    orquestração (batching, chamadas ao store), não a qualidade do
    retrieval.
    """

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            digest = hashlib.md5(text.encode()).digest()
            vector = [b / 255.0 for b in digest[: self.dimension]]
            vectors.append(vector)
        return vectors


class InMemoryStore:
    """Substitui o QdrantStore em testes: guarda tudo em memória."""

    def __init__(self):
        self.vector_size: int | None = None
        self.points: list[tuple[Chunk, list[float]]] = []
        self.ensure_collection_calls = 0

    def ensure_collection(self, vector_size: int) -> None:
        self.vector_size = vector_size
        self.ensure_collection_calls += 1

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.points.extend(zip(chunks, vectors))

    def count(self) -> int:
        return len(self.points)


class FakeDenseSearcher:
    """Retorna uma lista fixa de resultados, ignorando o vetor de consulta."""

    def __init__(self, results: list[dict]):
        self._results = results
        self.calls: list[tuple[list[float], int]] = []

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        self.calls.append((vector, top_k))
        return self._results[:top_k]


class FakeSparseSearcher:
    """Retorna uma lista fixa de resultados, ignorando a query em si."""

    def __init__(self, results: list[dict]):
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[dict]:
        self.calls.append((query, top_k))
        return self._results[:top_k]


class FakeReranker:
    """Reordena candidatos por um score fixo definido por id, sem modelo real."""

    def __init__(self, score_by_id: dict | None = None):
        self.score_by_id = score_by_id or {}
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        self.calls.append((query, len(candidates)))
        scored = [
            {**c, "rerank_score": self.score_by_id.get(c["id"], 0.0)}
            for c in candidates
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k] if top_k is not None else scored


class FakeRetriever:
    """Substitui o HybridRetriever nos testes de avaliação: retorna uma
    lista pré-definida de resultados por pergunta.

    results_by_question[pergunta] pode ser:
      - uma lista de resultados (mesma resposta pra qualquer modo), ou
      - um dict {modo: lista de resultados} (resposta varia por modo —
        útil pra simular uma regressão de recall entre dois modos).
    """

    def __init__(self, results_by_question: dict):
        self.results_by_question = results_by_question
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, query: str, mode: str, top_k: int) -> list[dict]:
        self.calls.append((query, mode, top_k))
        entry = self.results_by_question.get(query, [])
        results = entry.get(mode, []) if isinstance(entry, dict) else entry
        return results[:top_k]


class FakeLLMClient:
    """Emite uma resposta pré-definida em pedaços (streaming fake), sem
    nenhuma chamada de API real. Usado pra testar app.generation.pipeline
    sem depender do SDK da Anthropic nem de uma API key."""

    def __init__(self, response: str, chunk_size: int = 5):
        self.response = response
        self.chunk_size = chunk_size
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str):
        self.calls.append((system, user))
        for i in range(0, len(self.response), self.chunk_size):
            yield self.response[i : i + self.chunk_size]

