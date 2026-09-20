"""Rate limiting em memória para o endpoint /query, por visitante REAL, com teto global.

Em produção a geração passa pelo 9Router, roteando pra contas/free
tiers pessoais — um endpoint público sem limite é convite a abuso
(bot, scraper) que estoura a cota ou provoca ban num provider. Isso
não precisa de Redis nem de nada distribuído: a API roda numa
instância única, então uma janela deslizante em memória já resolve —
sem trazer uma dependência nova pro requirements.txt, o que forçaria
reconstruir a imagem inteira na VPS (o pip install de
torch/transformers/sentence-transformers já leva ~90 minutos em ARM).

Quem é o "visitante": atrás do Nginx (e da Cloudflare) o peer TCP
(`request.client.host`) é sempre o gateway do Docker, então usá-lo como
chave faria TODOS os visitantes dividirem um único bucket — 8 perguntas
por minuto no total, não por pessoa. O visitante real vem de
`CF-Connecting-IP` (Cloudflare) ou do primeiro `X-Forwarded-For` (Nginx);
`client.host` só é o último recurso (acesso direto, testes).

Risco aceito: quem acessar a origem sem passar pela Cloudflare pode forjar
esses cabeçalhos e ganhar buckets novos. O teto GLOBAL limita o dano — é
uma demo de portfólio, não um controle de acesso.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 8  # por visitante
GLOBAL_MAX_REQUESTS_PER_WINDOW = 30  # todos os visitantes juntos: protege a cota do LLM
MAX_TRACKED_VISITORS = 10_000  # teto de memória (chaves distintas em janela)

_hits: dict[str, deque[float]] = {}
_global_hits: deque[float] = deque()
_lock = threading.Lock()


def visitor_key(request: Request) -> str:
    """Identifica o visitante real (ver o docstring do módulo)."""
    headers = request.headers
    if cf_ip := headers.get("cf-connecting-ip"):
        return cf_ip.strip()[:64]
    if forwarded := headers.get("x-forwarded-for"):
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def _trim(hits: deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()


def _too_many(retry_after: float) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail="Muitas perguntas em pouco tempo. Espera um minuto e tenta de novo.",
        headers={"Retry-After": str(int(retry_after) + 1)},
    )


def rate_limit(request: Request) -> None:
    """Dependency do FastAPI: levanta 429 se o visitante (ou o total) excedeu o limite na janela atual."""
    key = visitor_key(request)
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS
    with _lock:
        _trim(_global_hits, cutoff)
        hits = _hits.setdefault(key, deque())
        _trim(hits, cutoff)
        if len(_global_hits) >= GLOBAL_MAX_REQUESTS_PER_WINDOW:
            raise _too_many(max(_global_hits[0] + WINDOW_SECONDS - now, 1.0))
        if len(hits) >= MAX_REQUESTS_PER_WINDOW:
            raise _too_many(max(hits[0] + WINDOW_SECONDS - now, 1.0))
        hits.append(now)
        _global_hits.append(now)
        if len(_hits) > MAX_TRACKED_VISITORS:  # descarta quem não tem hits recentes: memória limitada
            for stale in [k for k, v in _hits.items() if not v or v[-1] <= cutoff]:
                del _hits[stale]


def reset_for_tests() -> None:
    """Limpa o estado global do rate limiter — usado só nos testes, pra isolar
    uma suíte da outra (o estado é um dict de módulo, compartilhado entre
    todos os testes do mesmo processo)."""
    with _lock:
        _hits.clear()
        _global_hits.clear()
