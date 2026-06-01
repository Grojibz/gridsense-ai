"""End-to-end DocRAG test against the real stack (pgvector + Ollama).

Skipped unless RUN_INTEGRATION=1, since it needs the docker-compose Postgres and a running
Ollama with the configured models. Run with::

    RUN_INTEGRATION=1 pytest tests/test_integration_docrag.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with the live stack running",
)


def test_ingest_and_answer_end_to_end(tmp_path: Path) -> None:
    from gridsense.docrag.chain import answer_question
    from gridsense.docrag.ingest import ingest_path
    from gridsense.docrag.retriever import get_vectorstore

    doc = tmp_path / "soh.md"
    doc.write_text(
        "For grid energy-storage systems, 80% State of Health is the conventional "
        "end-of-life threshold for lithium-ion batteries.",
        encoding="utf-8",
    )

    store = get_vectorstore()
    n_chunks = ingest_path(tmp_path, vectorstore=store, reset=True)
    assert n_chunks >= 1

    result = answer_question(
        "What State of Health percentage marks end of life for grid batteries?",
        vectorstore=store,
    )

    assert "80" in result.answer
    assert result.citations, "expected at least one citation"
    assert result.citations[0].source == "soh.md"
    assert 0.0 <= result.confidence <= 1.0
