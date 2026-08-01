"""Post-ingest index verification.

A vector written wrong is completely silent: no error, no log, the chunk just stops being
retrievable. It happened on this corpus — one chunk scored 0.331 against a query its
correct embedding scores 0.708 on, and the symptom looked like a weak embedding model.
Self-retrieval is the cheapest tripwire for it.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from gridsense.docrag.ingest import IndexVerificationError, ingest_path, verify_index


def _chunks() -> list[Document]:
    return [
        Document(page_content="SOH end of life is 80%.", metadata={"source": "a.md"}),
        Document(page_content="Thermal runaway is exothermic.", metadata={"source": "b.md"}),
    ]


class HealthyStore:
    """Returns each chunk as its own best match, as a sound index does."""

    def similarity_search_with_relevance_scores(self, query: str, k: int = 1) -> list:
        return [(Document(page_content=query, metadata={}), 1.0)]


class CorruptedStore:
    """One chunk's vector is wrong, so it no longer matches its own text."""

    def __init__(self, broken: str) -> None:
        self._broken = broken

    def similarity_search_with_relevance_scores(self, query: str, k: int = 1) -> list:
        if query == self._broken:
            return [(Document(page_content="an unrelated chunk", metadata={}), 0.33)]
        return [(Document(page_content=query, metadata={}), 1.0)]


class EmptyStore:
    def similarity_search_with_relevance_scores(self, _q: str, k: int = 1) -> list:
        return []


def test_sound_index_reports_no_problems() -> None:
    assert verify_index(_chunks(), vectorstore=HealthyStore()) == []


def test_chunk_that_cannot_retrieve_itself_is_reported() -> None:
    problems = verify_index(_chunks(), vectorstore=CorruptedStore("SOH end of life is 80%."))
    assert len(problems) == 1
    assert "a.md" in problems[0]
    # The useful diagnosis is "a sibling won", not the score that happens to go with it.
    assert "outranks it" in problems[0]
    assert "0.33" in problems[0]


def test_self_match_that_is_merely_weak_is_reported_by_score() -> None:
    class WeakStore:
        def similarity_search_with_relevance_scores(self, query: str, k: int = 1) -> list:
            return [(Document(page_content=query, metadata={}), 0.60)]

    problems = verify_index(_chunks(), vectorstore=WeakStore())
    assert all("self-retrieval score 0.600" in p for p in problems)


def test_chunk_retrieving_nothing_is_reported() -> None:
    problems = verify_index(_chunks(), vectorstore=EmptyStore())
    assert len(problems) == 2
    assert all("retrieved nothing" in p for p in problems)


def test_a_chunk_outranked_on_its_own_text_is_reported() -> None:
    class OutrankedStore:
        def similarity_search_with_relevance_scores(self, _q: str, k: int = 1) -> list:
            return [(Document(page_content="a different chunk", metadata={}), 0.99)]

    problems = verify_index(_chunks(), vectorstore=OutrankedStore())
    assert all("outranks it" in p for p in problems)


class RecordingStore:
    def __init__(self, verifier) -> None:
        self.added: list[Document] = []
        self.reset = 0
        self._verifier = verifier

    def delete_collection(self) -> None:
        self.reset += 1

    def create_collection(self) -> None:
        pass

    def add_documents(self, docs: list[Document]) -> None:
        self.added.extend(docs)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 1) -> list:
        return self._verifier.similarity_search_with_relevance_scores(query, k)


def test_ingest_raises_when_verification_fails(tmp_path) -> None:
    (tmp_path / "doc.md").write_text("# Title\n\nSome battery content here.", encoding="utf-8")
    store = RecordingStore(CorruptedStore("# Title\n\nSome battery content here."))

    with pytest.raises(IndexVerificationError) as excinfo:
        ingest_path(tmp_path, vectorstore=store)

    assert excinfo.value.problems
    assert "doc.md" in str(excinfo.value)


def test_verification_can_be_skipped(tmp_path) -> None:
    (tmp_path / "doc.md").write_text("# Title\n\nSome battery content here.", encoding="utf-8")
    store = RecordingStore(CorruptedStore("# Title\n\nSome battery content here."))

    assert ingest_path(tmp_path, vectorstore=store, verify=False) == 1


def test_successful_ingest_verifies_and_returns_the_chunk_count(tmp_path) -> None:
    (tmp_path / "doc.md").write_text("# Title\n\nSome battery content here.", encoding="utf-8")
    store = RecordingStore(HealthyStore())

    assert ingest_path(tmp_path, vectorstore=store) == 1
    assert store.reset == 1
    assert len(store.added) == 1
