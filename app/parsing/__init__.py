"""Dispatcher de parsing: escolhe o parser certo pela extensão do arquivo."""


from __future__ import annotations

from pathlib import Path

from app.parsing.html_parser import parse_html
from app.parsing.markdown_parser import parse_markdown
from app.parsing.models import ParsedUnit
from app.parsing.pdf_parser import parse_pdf

_PARSERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}


def parse_document(path: str | Path) -> list[ParsedUnit]:
    path = Path(path)
    suffix = path.suffix.lower()

    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(
            f"Formato não suportado: '{suffix}'. "
            f"Formatos aceitos: {', '.join(_PARSERS)}"
        )

    return parser(path)


def parse_directory(directory: str | Path) -> list[ParsedUnit]:
    directory = Path(directory)
    units: list[ParsedUnit] = []

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _PARSERS:
            units.extend(parse_document(path))

    return units


__all__ = ["ParsedUnit", "parse_document", "parse_directory"]
