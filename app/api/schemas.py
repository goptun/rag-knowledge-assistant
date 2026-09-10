"""Schemas Pydantic do endpoint /query."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.retrieval.pipeline import VALID_MODES


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = "hybrid_rerank_blend"
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in VALID_MODES:
            raise ValueError(f"mode inválido: '{v}'. Use: {', '.join(VALID_MODES)}")
        return v


class Citation(BaseModel):
    index: int
    source: str | None = None
    section: str | None = None
    page: int | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant_collection: str
    bm25_loaded: bool
