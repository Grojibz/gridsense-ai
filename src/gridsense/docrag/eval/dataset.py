"""Labelled question set for evaluating DocRAG."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

DEFAULT_DATASET = Path(__file__).parent / "questions.json"


class EvalItem(BaseModel):
    """One labelled evaluation question."""

    question: str
    #: File the answer should be grounded in (``None`` for unanswerable questions).
    expected_source: str | None = None
    #: Substrings the answer should contain (case-insensitive), for answerable questions.
    expected_keywords: list[str] = []
    #: Whether the documents can answer this — unanswerable items test the refusal path.
    answerable: bool = True


def load_dataset(path: str | Path | None = None) -> list[EvalItem]:
    """Load the evaluation dataset from JSON (defaults to the bundled question set)."""
    path = Path(path) if path is not None else DEFAULT_DATASET
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalItem(**item) for item in raw]
