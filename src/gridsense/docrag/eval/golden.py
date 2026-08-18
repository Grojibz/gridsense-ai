"""The golden dataset: curated queries with retrieval ground truth and reference answers.

The set lives in ``eval/golden_dataset.json`` at the repo root (override with
``GOLDEN_DATASET_PATH``) so it can be reviewed and diffed like any other artefact.

Two kinds of ground truth are attached to each item:

- ``expected_chunks`` — which source file the answer lives in and verbatim substrings a
  correctly retrieved chunk must contain. This gives a *deterministic* retrieval score
  (:func:`retrieval_recall`) that needs no LLM, so it is cheap and stable in CI.
- ``reference_answer`` — the ground-truth answer, consumed by the RAGAS
  ``context_precision`` / ``context_recall`` metrics.

Unanswerable items (``vague``, ``out_of_scope``, ``insufficient_context``) carry no
expected chunks: they exercise the refusal path and are scored on refusal, not on RAGAS
answer quality.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

#: Repo-root dataset: ``src/gridsense/docrag/eval/golden.py`` -> up 4 -> repo root.
DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parents[4] / "eval" / "golden_dataset.json"

Category = Literal["factual", "synthesis", "vague", "out_of_scope", "insufficient_context"]


class ExpectedChunk(BaseModel):
    """Retrieval ground truth: a source file and substrings its chunk must contain."""

    source: str
    must_contain: list[str] = Field(default_factory=list)


class GoldenItem(BaseModel):
    """One curated query with its ground truth."""

    id: str
    category: Category
    answerable: bool
    query: str
    expected_chunks: list[ExpectedChunk] = Field(default_factory=list)
    reference_answer: str


class GoldenDataset(BaseModel):
    """The curated set plus provenance about the corpus it was written against."""

    corpus: str
    description: str = ""
    items: list[GoldenItem]

    @property
    def answerable(self) -> list[GoldenItem]:
        return [i for i in self.items if i.answerable]

    @property
    def unanswerable(self) -> list[GoldenItem]:
        return [i for i in self.items if not i.answerable]


class RetrievedChunk(BaseModel):
    """A chunk the retriever returned for a query."""

    source: str
    text: str
    score: float = 0.0


def load_golden_dataset(path: str | Path | None = None) -> GoldenDataset:
    """Load the golden dataset (repo-root default, or ``GOLDEN_DATASET_PATH``)."""
    if path is None:
        path = os.environ.get("GOLDEN_DATASET_PATH") or DEFAULT_GOLDEN_PATH
    return GoldenDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _chunk_found(expected: ExpectedChunk, retrieved: list[RetrievedChunk]) -> bool:
    """Whether any retrieved chunk is from ``expected.source`` and holds all its substrings."""
    for chunk in retrieved:
        if not chunk.source.endswith(expected.source):
            continue
        if all(needle.lower() in chunk.text.lower() for needle in expected.must_contain):
            return True
    return False


def retrieval_recall(item: GoldenItem, retrieved: list[RetrievedChunk]) -> float:
    """Fraction of ``item``'s expected chunks present in ``retrieved`` (1.0 if none expected).

    Deterministic and LLM-free: this is what catches a retrieval regression even when the
    generator papers over it with a plausible-sounding answer.
    """
    if not item.expected_chunks:
        return 1.0
    hits = sum(_chunk_found(e, retrieved) for e in item.expected_chunks)
    return hits / len(item.expected_chunks)
