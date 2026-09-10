"""Blend de rerank_score (cross-encoder) com rrf_score (fusão híbrida).

Mitigação para o caso descoberto na Fase 4 (q26 do dataset de
avaliação): o cross-encoder pode pontuar uma passagem corretamente
relevante como pouco relevante por causa do *framing* temático dela
(ex: um parâmetro de configuração mencionado dentro de uma seção cujo
título é "solução de problemas"). O cross-encoder julga o tópico geral
da passagem, não extrai fatos específicos.

Em vez de deixar o rerank_score sobrescrever 100% o ranking, blend_scores
combina o sinal de consenso do RRF (que já tinha achado a passagem
relevante, cruzando dense+sparse) com o sinal do cross-encoder — um
documento que os dois retrievers concordaram ser relevante não cai tanto
no ranking final só porque o cross-encoder discordou.
"""

from __future__ import annotations


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)  # todos iguais — normalização não discrimina
    return [(v - lo) / (hi - lo) for v in values]


def blend_scores(candidates: list[dict], alpha: float = 0.7) -> list[dict]:
    """candidates: cada um precisa ter 'rrf_score' e 'rerank_score'
    (o retorno de reranker.rerank() sobre uma lista fundida por RRF).

    alpha pesa o rerank_score: 1.0 = comportamento atual (só
    cross-encoder), 0.0 = ignora o cross-encoder e usa só o RRF. Os dois
    scores são normalizados (min-max) dentro do próprio conjunto de
    candidatos antes de combinar, porque vivem em escalas incompatíveis
    (RRF ~0.01-0.03, cross-encoder pode ser um logit de -10 a 10).

    Retorna a lista com "blended_score" adicionado, ordenada decrescente.
    """
    if not candidates:
        return []

    rerank_norm = _min_max_normalize([c["rerank_score"] for c in candidates])
    rrf_norm = _min_max_normalize([c["rrf_score"] for c in candidates])

    blended = [
        {**c, "blended_score": alpha * rk + (1 - alpha) * rf}
        for c, rk, rf in zip(candidates, rerank_norm, rrf_norm)
    ]
    blended.sort(key=lambda x: x["blended_score"], reverse=True)
    return blended
