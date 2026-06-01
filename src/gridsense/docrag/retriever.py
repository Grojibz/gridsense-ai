"""pgvector-backed vector store and retriever for DocRAG.

The store is a single ``PGVector`` collection over the app's Postgres database. Ingestion
writes chunks here; the RAG chain reads from it. Embedding dimensionality is determined by
the configured provider (e.g. 768 for Ollama ``nomic-embed-text``, 1536 for Azure
``text-embedding-3-small``) — pgvector stores variable-dimension vectors, so switching
providers only requires a fresh ingest, not a schema change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridsense.config import Settings, get_settings
from gridsense.providers import get_embeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_postgres import PGVector

#: Name of the PGVector collection holding all DocRAG chunks.
COLLECTION_NAME = "docrag"


def get_vectorstore(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> PGVector:
    """Return the DocRAG :class:`PGVector` store, creating its tables if needed."""
    from langchain_postgres import PGVector

    settings = settings or get_settings()
    embeddings = embeddings or get_embeddings(settings)

    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=settings.database_url,
        use_jsonb=True,
    )
