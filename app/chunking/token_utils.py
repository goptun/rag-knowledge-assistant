"""Contagem e (de)codificação de "tokens" para controle de tamanho de chunk.

Nota: usamos um tokenizador regex (word-ish) em vez de tiktoken/BPE do
modelo de embedding real. Motivo prático: este ambiente de build não tem
acesso a rede para instalar pacotes novos (tiktoken incluso), e de todo
modo tiktoken (BPE da OpenAI) não é o tokenizer exato do bge-base
(WordPiece) — já era uma aproximação mesmo na escolha original. Um
contador baseado em regex é determinístico, sem dependências, e serve
igualmente bem para controlar o tamanho dos chunks (chunk_size é um
parâmetro heurístico, não uma exigência exata de nenhum modelo).

Se quiser precisão exata do tokenizer do embedding model ao rodar
localmente com as dependências completas instaladas, troque a
implementação por `AutoTokenizer.from_pretrained(EMBEDDING_MODEL)`.
"""


from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def encode(text: str) -> list[str]:
    return tokenize(text)


def decode(tokens: list[str]) -> str:
    return " ".join(tokens)
