"""Carrega o dataset de avaliação (perguntas + ground truth de relevância)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "qa_dataset.json"


def load_dataset(path: str | Path = DEFAULT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
