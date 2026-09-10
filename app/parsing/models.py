"""Modelos de dados compartilhados pelo pipeline de parsing."""


from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedUnit:
    """Um pedaço de texto extraído de um documento, antes do chunking.

    Para PDF, cada unidade é uma página. Para HTML/Markdown, cada unidade
    é uma seção delimitada por heading. O chunker pode juntar ou dividir
    essas unidades — a metadata de origem é preservada em ambos os casos.
    """

    text: str
    source: str  # caminho ou nome do arquivo original
    section: str | None = None  # título da seção/heading, se houver
    page: int | None = None  # número da página, se aplicável (PDF)
    extra: dict = field(default_factory=dict)
