"""The golden dataset must stay true to the corpus it claims to describe.

`must_contain` is retrieval ground truth: if someone edits `data/docs` and a substring no
longer exists, `retrieval_recall` silently becomes unachievable and the gate turns into
noise. These tests fail loudly instead — no LLM or datastore needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gridsense.docrag.eval.golden import (
    ExpectedChunk,
    GoldenItem,
    RetrievedChunk,
    load_golden_dataset,
    retrieval_recall,
)

DATASET = load_golden_dataset()
CORPUS = {p.name: p.read_text(encoding="utf-8") for p in Path(DATASET.corpus).glob("*.md")}


def test_dataset_is_large_enough_and_balanced() -> None:
    assert len(DATASET.items) >= 20
    assert len(DATASET.answerable) >= 15
    assert len(DATASET.unanswerable) >= 5


def test_every_edge_case_category_is_covered() -> None:
    categories = {item.category for item in DATASET.items}
    assert {"vague", "out_of_scope", "insufficient_context"} <= categories


def test_item_ids_are_unique() -> None:
    ids = [item.id for item in DATASET.items]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("item", DATASET.items, ids=lambda i: i.id)
def test_ground_truth_matches_the_corpus(item: GoldenItem) -> None:
    """Every `must_contain` string is a verbatim substring of the file it names."""
    for expected in item.expected_chunks:
        assert expected.source in CORPUS, f"unknown source {expected.source}"
        assert expected.must_contain, "an expected chunk with no substrings proves nothing"
        text = CORPUS[expected.source].lower()
        for needle in expected.must_contain:
            assert needle.lower() in text, f"{needle!r} is not in {expected.source}"


@pytest.mark.parametrize("item", DATASET.items, ids=lambda i: i.id)
def test_answerability_matches_the_ground_truth_shape(item: GoldenItem) -> None:
    assert item.reference_answer.strip(), "every item needs a reference answer"
    if item.answerable:
        assert item.expected_chunks, "an answerable item must name the chunks that answer it"
        assert item.category in {"factual", "synthesis"}
    else:
        assert not item.expected_chunks, "an unanswerable item must not claim ground-truth chunks"
        assert item.category in {"vague", "out_of_scope", "insufficient_context"}


def test_retrieval_recall_counts_matched_chunks() -> None:
    item = GoldenItem(
        id="x",
        category="synthesis",
        answerable=True,
        query="q",
        reference_answer="r",
        expected_chunks=[
            ExpectedChunk(source="a.md", must_contain=["alpha", "beta"]),
            ExpectedChunk(source="b.md", must_contain=["gamma"]),
        ],
    )
    retrieved = [RetrievedChunk(source="data/docs/a.md", text="ALPHA and BETA are here")]

    assert retrieval_recall(item, retrieved) == 0.5
    assert retrieval_recall(item, [*retrieved, RetrievedChunk(source="b.md", text="gamma")]) == 1.0


def test_retrieval_recall_requires_all_substrings_in_one_chunk() -> None:
    item = GoldenItem(
        id="x",
        category="factual",
        answerable=True,
        query="q",
        reference_answer="r",
        expected_chunks=[ExpectedChunk(source="a.md", must_contain=["alpha", "beta"])],
    )
    split_across_chunks = [
        RetrievedChunk(source="a.md", text="alpha only"),
        RetrievedChunk(source="a.md", text="beta only"),
    ]
    assert retrieval_recall(item, split_across_chunks) == 0.0


def test_retrieval_recall_ignores_the_right_text_from_the_wrong_source() -> None:
    item = GoldenItem(
        id="x",
        category="factual",
        answerable=True,
        query="q",
        reference_answer="r",
        expected_chunks=[ExpectedChunk(source="a.md", must_contain=["alpha"])],
    )
    assert retrieval_recall(item, [RetrievedChunk(source="other.md", text="alpha")]) == 0.0


def test_unanswerable_items_score_full_retrieval_recall() -> None:
    item = GoldenItem(
        id="x", category="out_of_scope", answerable=False, query="q", reference_answer="r"
    )
    assert retrieval_recall(item, []) == 1.0
