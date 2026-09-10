"""Orquestra parsing -> chunking -> embedding -> upsert (+ BM25 opcional).

embedder e store são recebidos por injeção de dependência (duck typing:
precisam só de .embed()/.dimension e .ensure_collection()/.upsert_chunks())
para que a lógica de orquestração seja testável sem sentence-transformers
nem qdrant-client instalados — ver tests/fakes.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.chunking import chunk_units
from app.chunking.models import Chunk
from app.parsing import parse_directory


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...


def build_index(
    source_dir: str | Path,
    embedder: Embedder,
    store: VectorStore,
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 50,
    batch_size: int = 32,
    bm25_index_path: str | Path | None = None,
) -> dict:
    units = parse_directory(source_dir)
    chunks = chunk_units(units, strategy=strategy, chunk_size=chunk_size, overlap=overlap)

    stats = {
        "documents_parsed": len({u.source for u in units}),
        "units_parsed": len(units),
        "chunks_indexed": 0,
        "strategy": strategy,
    }

    if not chunks:
        return stats

    collection_ready = False
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.embed([c.text for c in batch])

        if not collection_ready:
            store.ensure_collection(vector_size=len(vectors[0]))
            collection_ready = True

        store.upsert_chunks(batch, vectors)
        stats["chunks_indexed"] += len(batch)

    if bm25_index_path is not None:
        # import lazy: só precisa de rank_bm25 se o caller pedir o índice esparso
        from app.indexing.bm25_store import BM25Index

        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(bm25_index_path)
        stats["bm25_index_path"] = str(bm25_index_path)
        stats["bm25_entries"] = len(bm25)

    return stats
