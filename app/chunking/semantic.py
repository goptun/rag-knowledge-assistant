"""Chunking semântico: respeita as seções/headings do documento.

Cada ParsedUnit (já delimitada por heading no parser de HTML/Markdown,
ou por página no PDF) vira um chunk. Se uma seção for grande demais,
ela é subdividida com o mesmo splitter recursivo, mas sem overlap —
a fronteira semântica (heading) já é o critério de corte natural.
"""


from __future__ import annotations

from app.chunking.models import Chunk
from app.chunking.recursive import _SEPARATORS, _split_recursive
from app.chunking.token_utils import count_tokens
from app.parsing.models import ParsedUnit

# seções maiores que chunk_size * MAX_SECTION_MULTIPLIER são subdivididas
MAX_SECTION_MULTIPLIER = 2


def semantic_chunk_units(units: list[ParsedUnit], chunk_size: int = 512) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    max_tokens = chunk_size * MAX_SECTION_MULTIPLIER

    for unit in units:
        if count_tokens(unit.text) <= max_tokens:
            pieces = [unit.text]
        else:
            pieces = _split_recursive(unit.text, chunk_size, _SEPARATORS)

        for text in pieces:
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    source=unit.source,
                    section=unit.section,
                    page=unit.page,
                    chunk_index=index,
                    token_count=count_tokens(text),
                    strategy="semantic",
                )
            )
            index += 1

    return chunks
