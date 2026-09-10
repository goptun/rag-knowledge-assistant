"""Parser de PDF: uma ParsedUnit por página."""


from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.parsing.models import ParsedUnit


def parse_pdf(path: str | Path) -> list[ParsedUnit]:
    path = Path(path)
    reader = PdfReader(str(path))
    units: list[ParsedUnit] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        units.append(
            ParsedUnit(
                text=text,
                source=path.name,
                page=page_number,
            )
        )

    return units
