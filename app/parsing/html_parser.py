"""Parser de HTML: uma ParsedUnit por seção (delimitada por heading)."""


from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.parsing.models import ParsedUnit

HEADING_TAGS = ("h1", "h2", "h3", "h4")


def parse_html(path: str | Path) -> list[ParsedUnit]:
    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    # remove elementos que não carregam conteúdo relevante
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    body = soup.body or soup
    units: list[ParsedUnit] = []
    current_section: str | None = None
    current_text: list[str] = []

    def flush():
        text = "\n".join(current_text).strip()
        if text:
            units.append(
                ParsedUnit(text=text, source=path.name, section=current_section)
            )

    for element in body.descendants:
        if getattr(element, "name", None) in HEADING_TAGS:
            flush()
            current_section = element.get_text(strip=True)
            current_text = []
        elif getattr(element, "name", None) in ("p", "li", "td"):
            text = element.get_text(strip=True)
            if text:
                current_text.append(text)

    flush()

    # fallback: se não havia headings, extrai o texto todo como uma unidade
    if not units:
        text = body.get_text("\n", strip=True)
        if text:
            units.append(ParsedUnit(text=text, source=path.name))

    return units
