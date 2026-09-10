"""Rate limiting simples em memória para o endpoint /query.

Em produção a geração passa pelo 9Router, roteando pra contas/free
tiers pessoais — um endpoint público sem limite é convite a abuso
(bot, scraper) que estoura a cota ou provoca ban num provider. Isso
não precisa de Redis nem de nada distribuído: a API roda numa
instância única, então uma janela deslizante em memória por IP já
resolve — sem trazer uma dependência nova pro requirements.txt, o que
forçaria reconstruir a imagem inteira na VPS (o pip install de
torch/transformers/sentence-transformers já leva ~90 minutos em ARM).
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 8

_hits: dict[str, list[float]] = defaultdict(list)


def rate_limit(request: Request) -> None:
    """Dependency do FastAPI: levanta 429 se o IP excedeu o limite na janela atual."""
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    hits = _hits[ip]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.pop(0)
    if len(hits) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail="Muitas perguntas em pouco tempo. Espera um minuto e tenta de novo.",
        )
    hits.append(now)


def reset_for_tests() -> None:
    """Limpa o estado global do rate limiter — usado só nos testes, pra isolar
    uma suíte da outra (o estado é um dict de módulo, compartilhado entre
    todos os testes do mesmo processo)."""
    _hits.clear()
