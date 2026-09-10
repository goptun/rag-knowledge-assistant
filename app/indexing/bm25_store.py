"""Índice esparso (BM25) para o retrieval híbrido.

Import de rank_bm25 é lazy. O índice guarda, por chunk, o mesmo point_id
usado no Qdrant (app.indexing.ids.point_id) e um payload equivalente,
pra que a fusão RRF (app/retrieval/hybrid.py) consiga casar resultados
dense e sparse pelo id.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from app.chunking.models import Chunk
from app.chunking.token_utils import tokenize
from app.indexing.ids import point_id


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._ids: list[str] = []
        self._payloads: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        corpus_tokens = [tokenize(c.text.lower()) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._ids = [point_id(c.id) for c in chunks]
        self._payloads = [
            {
                "text": c.text,
                "source": c.source,
                "section": c.section,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "strategy": c.strategy,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Busca sparse. Retorna lista de dicts {id, score, payload}."""
        if self._bm25 is None:
            raise RuntimeError(
                "Índice BM25 vazio — chame build() ou load() antes de search()."
            )

        scores = self._bm25.get_scores(tokenize(query.lower()))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:  # sem termo em comum — não é um match de verdade
                break
            results.append(
                {"id": self._ids[i], "score": float(scores[i]), "payload": self._payloads[i]}
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "ids": self._ids, "payloads": self._payloads}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index._bm25 = data["bm25"]
        index._ids = data["ids"]
        index._payloads = data["payloads"]
        return index

    def __len__(self) -> int:
        return len(self._ids)
