"""Extrai citações [n] do texto gerado e mapeia de volta pros metadados
reais dos chunks recuperados.

Importante: não confiamos no LLM para relatar a fonte corretamente — ele
só usa o índice numérico [n], e nós resolvemos esse índice pro payload
real do chunk que ocupava aquela posição no contexto. Isso elimina uma
classe inteira de alucinação de citação (LLM inventando um source/página
que não existe no corpus).
"""

from __future__ import annotations

import re

# Aceita também 【n】 (colchetes de largura total), que alguns modelos
# emitem no lugar de [n].
_CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")


def extract_cited_indices(text: str) -> list[int]:
    """Retorna os índices citados (1-based), em ordem de primeira
    aparição, sem duplicatas."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def build_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Resolve os índices citados no texto pros metadados reais dos
    chunks. Índices fora do range (LLM citou um número inexistente) são
    ignorados silenciosamente."""
    citations = []
    for idx in extract_cited_indices(text):
        if 1 <= idx <= len(chunks):
            payload = chunks[idx - 1]["payload"]
            citations.append(
                {
                    "index": idx,
                    "source": payload.get("source"),
                    "section": payload.get("section"),
                    "page": payload.get("page"),
                }
            )
    return citations
