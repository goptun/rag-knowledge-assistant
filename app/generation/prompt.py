"""Monta o prompt de geração a partir da query e dos chunks recuperados.

O prompt instrui o LLM a responder SOMENTE com base no contexto
fornecido e a citar a fonte de cada afirmação usando a notação [n],
onde n é o índice do chunk (1-based) na lista numerada do contexto.
Isso permite mapear as citações de volta pros metadados reais do chunk
(source/section/page) sem confiar cegamente no texto gerado — ver
app/generation/citations.py.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Você é um assistente de conhecimento interno. Responda à pergunta do "
    "usuário usando exclusivamente as informações do CONTEXTO abaixo. "
    "Cada trecho do contexto é numerado como [n]. Sempre que usar uma "
    "informação de um trecho na sua resposta, cite o número dele entre "
    "colchetes logo após a afirmação, por exemplo: 'A API usa OAuth2 [1].' "
    "O marcador [n] já É a citação — não descreva em texto onde a "
    "informação está (nunca escreva frases como 'essas informações "
    "constam na documentação/seção X' ou 'de acordo com a fonte Y'); "
    "isso não agrega nada ao usuário e não é uma afirmação verificável "
    "a partir do conteúdo do trecho em si. "
    "Se o contexto não tiver informação suficiente para responder, diga "
    "isso explicitamente em vez de inventar uma resposta. Não use "
    "conhecimento fora do contexto fornecido."
)


def _format_source_label(payload: dict) -> str:
    source = payload.get("source", "desconhecido")
    if payload.get("section"):
        return f"{source} (seção: {payload['section']})"
    if payload.get("page") is not None:
        return f"{source} (página: {payload['page']})"
    return source


def build_context_block(chunks: list[dict]) -> str:
    """chunks: lista de resultados do retriever ({"payload": {...}, ...}),
    já na ordem final (pós rerank/blend) — a ordem define a numeração
    usada nas citações [n]."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk["payload"]
        label = _format_source_label(payload)
        parts.append(f"[{i}] Fonte: {label}\n{payload['text']}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return (
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {query}\n\n"
        "Responda em português, citando as fontes com [n] conforme instruído."
    )
