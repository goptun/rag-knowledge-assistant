"""Recursive character/token splitting com overlap.

Estratégia (equivalente ao RecursiveCharacterTextSplitter): tenta separar
o texto por parágrafo, depois por linha, depois por frase, depois por
espaço, e em último caso corta por token. Pedaços pequenos são
remontados em chunks até o limite de chunk_size, carregando um overlap
de tokens do final de um chunk para o início do próximo.
"""


from __future__ import annotations

from app.chunking.models import Chunk
from app.chunking.token_utils import count_tokens, decode, encode
from app.parsing.models import ParsedUnit

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_by_separator(text: str, sep: str) -> list[str]:
    if sep not in text:
        return [text]
    parts = text.split(sep)
    return [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        tokens = encode(text)
        return [
            decode(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]

    sep, *rest = separators
    pieces = _split_by_separator(text, sep)

    if len(pieces) == 1:
        return _split_recursive(text, chunk_size, rest)

    result: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= chunk_size:
            if piece.strip():
                result.append(piece)
        else:
            result.extend(_split_recursive(piece, chunk_size, rest))
    return result


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, overlap: int
) -> list[str]:
    chunks: list[str] = []
    current = ""
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if current and current_tokens + piece_tokens > chunk_size:
            chunks.append(current.strip())
            if overlap > 0:
                tail_tokens = encode(current)[-overlap:]
                current = decode(tail_tokens)
                current_tokens = len(tail_tokens)
            else:
                current, current_tokens = "", 0

        current += piece
        current_tokens += piece_tokens

    if current.strip():
        chunks.append(current.strip())

    return chunks


def recursive_chunk_units(
    units: list[ParsedUnit], chunk_size: int = 512, overlap: int = 50
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for unit in units:
        pieces = _split_recursive(unit.text, chunk_size, _SEPARATORS)
        merged = _merge_with_overlap(pieces, chunk_size, overlap)

        for text in merged:
            chunks.append(
                Chunk(
                    text=text,
                    source=unit.source,
                    section=unit.section,
                    page=unit.page,
                    chunk_index=index,
                    token_count=count_tokens(text),
                    strategy="recursive",
                )
            )
            index += 1

    return chunks
