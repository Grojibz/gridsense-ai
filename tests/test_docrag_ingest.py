"""Unit tests for ingestion loading/chunking and the store-write flow (no DB)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from gridsense.docrag.ingest import chunk_documents, ingest_path, load_documents


def test_load_documents_reads_supported_files_with_source_metadata(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Title\nMarkdown body.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Plain text body.", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    docs = load_documents(tmp_path)

    sources = sorted(d.metadata["source"] for d in docs)
    assert sources == ["a.md", "b.txt"]  # .json ignored


def test_load_documents_skips_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    assert load_documents(tmp_path) == []


def test_chunk_documents_splits_long_text() -> None:
    doc = Document(page_content="word " * 1000, metadata={"source": "big.md"})
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "big.md" for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)


class RecordingVectorStore:
    """Captures store interactions so we can assert on the reset+write flow."""

    def __init__(self) -> None:
        self.deleted = False
        self.created = False
        self.added: list[Document] = []

    def delete_collection(self) -> None:
        self.deleted = True

    def create_collection(self) -> None:
        self.created = True

    def add_documents(self, docs: list[Document]) -> None:
        self.added.extend(docs)


def test_ingest_path_resets_then_writes_chunks(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("word " * 500, encoding="utf-8")
    store = RecordingVectorStore()

    n = ingest_path(tmp_path, vectorstore=store, reset=True, chunk_size=200, chunk_overlap=20)

    assert store.deleted is True
    assert store.created is True
    assert n == len(store.added) > 1


def test_ingest_path_without_reset_does_not_delete(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("hello world", encoding="utf-8")
    store = RecordingVectorStore()

    ingest_path(tmp_path, vectorstore=store, reset=False)

    assert store.deleted is False
    assert len(store.added) >= 1
