"""Métricas de retrieval: Precision@K, Recall@K, MRR.

Operam sobre payloads (dicts com source/section/page), não sobre
chunk_id — assim a avaliação continua válida se você mudar chunk_size
ou strategy e os ids dos chunks mudarem, desde que a granularidade
semântica (a seção ou página relevante) continue a mesma. É por isso
que o dataset de avaliação (data/eval/qa_dataset.json) marca relevância
por source+section (HTML/Markdown) ou source+page (PDF), não por id.
"""

from __future__ import annotations


def is_relevant(payload: dict, relevant: dict) -> bool:
    if payload.get("source") != relevant.get("source"):
        return False
    if "section" in relevant:
        return payload.get("section") == relevant["section"]
    if "page" in relevant:
        return payload.get("page") == relevant["page"]
    return False


def _relevant_flags(retrieved: list[dict], relevant: list[dict]) -> list[bool]:
    return [any(is_relevant(r["payload"], rel) for rel in relevant) for r in retrieved]


def precision_at_k(retrieved: list[dict], relevant: list[dict], k: int) -> float:
    """Fração dos top-k resultados que são relevantes."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(_relevant_flags(top_k, relevant)) / len(top_k)


def recall_at_k(retrieved: list[dict], relevant: list[dict], k: int) -> float:
    """Fração dos itens relevantes que aparecem nos top-k resultados."""
    if not relevant:
        return 0.0
    hits = sum(_relevant_flags(retrieved[:k], relevant))
    return min(hits, len(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[dict], relevant: list[dict]) -> float:
    """1/rank do primeiro resultado relevante; 0 se nenhum for relevante."""
    for rank, r in enumerate(retrieved, start=1):
        if any(is_relevant(r["payload"], rel) for rel in relevant):
            return 1.0 / rank
    return 0.0
