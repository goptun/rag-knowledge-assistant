"""Dispatcher de chunking: recursive (default) ou semantic."""


from __future__ import annotations

from app.chunking.models import Chunk
from app.chunking.recursive import recursive_chunk_units
from app.chunking.semantic import semantic_chunk_units
from app.parsing.models import ParsedUnit

_STRATEGIES = {
    "recursive": recursive_chunk_units,
    "semantic": semantic_chunk_units,
}


def chunk_units(
    units: list[ParsedUnit],
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    if strategy == "recursive":
        return recursive_chunk_units(units, chunk_size=chunk_size, overlap=overlap)
    if strategy == "semantic":
        return semantic_chunk_units(units, chunk_size=chunk_size)
    raise ValueError(
        f"Estratégia desconhecida: '{strategy}'. Use: {', '.join(_STRATEGIES)}"
    )


__all__ = ["Chunk", "chunk_units", "recursive_chunk_units", "semantic_chunk_units"]
