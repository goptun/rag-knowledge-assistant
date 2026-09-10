"""Modelo de chunk pronto para embedding/indexação."""


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    token_count: int
    strategy: str  # "recursive" ou "semantic"
    section: str | None = None
    page: int | None = None

    @property
    def id(self) -> str:
        """ID determinístico, útil como point ID no Qdrant."""
        base = f"{self.source}:{self.strategy}:{self.chunk_index}"
        if self.page is not None:
            base += f":p{self.page}"
        return base
