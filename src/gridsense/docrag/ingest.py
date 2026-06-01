"""Document ingestion: load -> chunk -> embed -> store in pgvector.

Supports markdown/text/PDF files. Run as::

    python -m gridsense.docrag.ingest data/docs

By default this resets the DocRAG collection first, so re-running is idempotent rather than
accumulating duplicate chunks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_postgres import PGVector

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(path: str | Path) -> list[Document]:
    """Load supported files under ``path`` (a file or directory) into Documents.

    Each document carries ``source`` metadata (path relative to the input root) so answers
    can cite where information came from.
    """
    from langchain_core.documents import Document

    root = Path(path)
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        base = root

    docs: list[Document] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = file.read_text(encoding="utf-8")
        elif suffix in PDF_SUFFIXES:
            text = _read_pdf(file)
        else:
            continue
        if not text.strip():
            continue
        source = file.relative_to(base).as_posix()
        docs.append(Document(page_content=text, metadata={"source": source}))
    return docs


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def ingest_path(
    path: str | Path,
    *,
    vectorstore: PGVector | None = None,
    reset: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """Ingest all supported docs under ``path``; return the number of chunks stored."""
    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore()

    docs = load_documents(path)
    if not docs:
        return 0
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if reset:
        # Drop and recreate the collection so re-ingesting the same corpus stays
        # idempotent (delete_collection removes the collection row entirely, so it must
        # be recreated before adding documents).
        vectorstore.delete_collection()
        vectorstore.create_collection()
    vectorstore.add_documents(chunks)
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the DocRAG vector store.")
    parser.add_argument("path", help="File or directory of .md/.txt/.pdf documents.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing collection instead of resetting it first.",
    )
    args = parser.parse_args(argv)

    n_files = len(load_documents(args.path))
    n_chunks = ingest_path(args.path, reset=not args.no_reset)
    print(f"Ingested {n_files} document(s) -> {n_chunks} chunk(s) into the DocRAG store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
