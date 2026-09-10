"""Camada de acesso ao Qdrant: criação de collection, upsert e busca dense.

Import de qdrant_client é lazy pelo mesmo motivo do embeddings.py: não
acoplar o resto do projeto a uma dependência pesada que pode não estar
instalada (ex.: rodando só os testes de chunking).
"""

from __future__ import annotations

from app.chunking.models import Chunk
from app.indexing.ids import point_id

_DISTANCE_MAP = {"Cosine": "COSINE", "Dot": "DOT", "Euclid": "EUCLID"}


class QdrantStore:
    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(url=url)

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection in existing

    def ensure_collection(self, vector_size: int, distance: str = "Cosine") -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=getattr(Distance, _DISTANCE_MAP[distance]),
            ),
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=point_id(chunk.id),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "section": chunk.section,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    "token_count": chunk.token_count,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int = 20) -> list[dict]:
        """Busca dense. Retorna lista de dicts {id, score, payload}."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
        ).points
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count
