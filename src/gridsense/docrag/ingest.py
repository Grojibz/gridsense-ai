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

#: A chunk queried with its own text must score at least this against itself. A correctly
#: stored vector scores ~1.0; anything this low means the stored vector is not the
#: embedding of the text next to it.
SELF_RETRIEVAL_FLOOR = 0.95


class IndexVerificationError(RuntimeError):
    """Raised when a freshly written index fails its self-retrieval check."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            f"{len(problems)} chunk(s) failed index verification:\n  " + "\n  ".join(problems)
        )


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


def verify_index(
    chunks: list[Document],
    *,
    vectorstore: PGVector,
    min_self_score: float = SELF_RETRIEVAL_FLOOR,
) -> list[str]:
    """Check every chunk retrieves *itself* when queried with its own text.

    Returns a list of human-readable problems (empty means the index is sound).

    This exists because a bad vector is otherwise completely silent. One chunk in this
    corpus was once stored with a vector scoring 0.331 against a query its correct
    embedding scores 0.708 on — the chunk simply stopped being retrievable, no error
    anywhere, and the symptom looked exactly like "the embedding model is weak". A chunk
    that cannot retrieve itself is the cheapest possible tripwire for that.

    It is not a hypothetical: this check caught a *second* occurrence within an hour of
    being written, on an Ollama instance under concurrent load. On a sound index every
    chunk self-retrieves at exactly 1.0, so the floor has plenty of headroom and a failure
    here means the stored vector is genuinely not the embedding of its text.
    """
    problems: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        head = " ".join(chunk.page_content.split())[:60]
        hits = vectorstore.similarity_search_with_relevance_scores(chunk.page_content, k=1)
        if not hits:
            problems.append(f"{source}: chunk retrieved nothing ({head!r})")
            continue
        best, score = hits[0]
        # Identity first: "a sibling chunk won" is the useful diagnosis, and reporting the
        # score instead sends you chasing a threshold when the vector is the problem.
        if best.page_content != chunk.page_content:
            problems.append(
                f"{source}: another chunk outranks it on its own text "
                f"(score {score:.3f}) ({head!r})"
            )
        elif score < min_self_score:
            problems.append(
                f"{source}: self-retrieval score {score:.3f} < {min_self_score} ({head!r})"
            )
    return problems


def ingest_path(
    path: str | Path,
    *,
    vectorstore: PGVector | None = None,
    reset: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    verify: bool = True,
) -> int:
    """Ingest all supported docs under ``path``; return the number of chunks stored.

    With ``verify`` (the default) the freshly written index is checked chunk by chunk and
    :class:`IndexVerificationError` is raised if any chunk cannot retrieve itself.
    """
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

    if verify and (problems := verify_index(chunks, vectorstore=vectorstore)):
        raise IndexVerificationError(problems)

    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the DocRAG vector store.")
    parser.add_argument("path", help="File or directory of .md/.txt/.pdf documents.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing collection instead of resetting it first.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-ingest self-retrieval check (not recommended).",
    )
    args = parser.parse_args(argv)

    n_files = len(load_documents(args.path))
    try:
        n_chunks = ingest_path(args.path, reset=not args.no_reset, verify=not args.no_verify)
    except IndexVerificationError as exc:
        print(f"Ingest FAILED verification.\n{exc}", file=sys.stderr)
        return 1
    print(f"Ingested {n_files} document(s) -> {n_chunks} chunk(s) into the DocRAG store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
