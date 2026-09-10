"""Reranking com cross-encoder local (BAAI/bge-reranker-base).

Import de sentence_transformers.CrossEncoder é lazy, mesmo motivo dos
outros módulos que dependem de libs pesadas: não quebrar partes do
projeto que não precisam de reranking (ex.: testes de chunking/parsing).

Um cross-encoder é mais caro que um bi-encoder (roda um forward pass
por par query-documento, não dá pra pré-computar), por isso só faz
sentido aplicá-lo a um conjunto pequeno de candidatos (top-20 do RRF,
não a collection inteira) — é exatamente esse o papel dele aqui:
reordenar os poucos candidatos plausíveis antes de mandar pro LLM.
"""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[dict], top_k: int | None = None
    ) -> list[dict]:
        """candidates: lista de {"id", "payload", ...}, payload precisa
        ter a chave "text". Retorna a mesma lista com "rerank_score"
        adicionado, ordenada por esse score decrescente."""
        if not candidates:
            return []

        pairs = [(query, c["payload"]["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k] if top_k is not None else reranked
