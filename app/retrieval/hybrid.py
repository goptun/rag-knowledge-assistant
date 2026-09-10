"""Reciprocal Rank Fusion (RRF): combina rankings dense e sparse.

RRF é preferido a normalizar e somar os scores brutos porque cosine
similarity (dense) e BM25 score (sparse) vivem em escalas completamente
diferentes e não normalizáveis de forma confiável — RRF ignora a
magnitude do score e usa só a posição no ranking, o que é robusto a
essa diferença de escala. Fórmula padrão (Cormack et al., 2009):

    score(d) = sum_sobre_rankings( 1 / (k + rank(d)) )

k=60 é o valor default da literatura original, funciona bem na prática
sem precisar de tuning.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """Funde N rankings (cada um: lista de {"id", "score", "payload"},
    já ordenada da mais relevante pra menos relevante) em um só.

    Retorna lista de {"id", "rrf_score", "payload"} ordenada por
    rrf_score decrescente. Um documento que aparece em mais de um
    ranking soma as contribuições — é assim que o RRF favorece
    documentos que tanto o retrieval semântico quanto o keyword
    concordam ser relevantes.
    """
    fused_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            doc_id = item["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(doc_id, item["payload"])

    fused = [
        {"id": doc_id, "rrf_score": score, "payload": payloads[doc_id]}
        for doc_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused[:top_k] if top_k is not None else fused
