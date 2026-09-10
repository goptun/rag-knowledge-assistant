"""Parser de Markdown: uma ParsedUnit por seção (delimitada por heading)."""


from __future__ import annotations

import re
from pathlib import Path

from app.parsing.models import ParsedUnit

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)


def parse_markdown(path: str | Path) -> list[ParsedUnit]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    matches = list(HEADING_RE.finditer(raw))
    units: list[ParsedUnit] = []

    if not matches:
        text = raw.strip()
        if text:
            units.append(ParsedUnit(text=text, source=path.name))
        return units

    # texto antes do primeiro heading (se houver) vira uma unidade sem seção
    preamble = raw[: matches[0].start()].strip()
    if preamble:
        units.append(ParsedUnit(text=preamble, source=path.name))

    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            units.append(
                ParsedUnit(text=body, source=path.name, section=section_title)
            )

    return units
