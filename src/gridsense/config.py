"""Application configuration, driven by environment variables / a local ``.env``.

Centralising settings here keeps secrets and environment-specific values out of the
code. Import the singleton via :func:`get_settings` so the ``.env`` file is parsed once.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Which backend serves LLM + embedding calls."""

    azure = "azure"
    ollama = "ollama"


class Settings(BaseSettings):
    """Typed, env-driven settings for the whole app.

    Values are read from environment variables (case-insensitive) or a ``.env`` file.
    See ``.env.example`` for the full list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------
    app_name: str = "gridsense"
    environment: str = "local"
    log_level: str = "INFO"

    # --- LLM provider ------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.azure

    # Azure OpenAI (used when llm_provider == azure)
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Ollama (used when llm_provider == ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- DocRAG guardrails -------------------------------------------------
    # Minimum retrieval relevance (0..1) for a chunk to count as context. Below this
    # for all chunks, DocRAG refuses rather than answer from irrelevant context.
    # Tuned for Ollama nomic-embed-text; adjust per embedding provider.
    docrag_min_relevance: float = 0.5
    # Minimum self-reported answer confidence (0..1) below which DocRAG refuses.
    docrag_min_confidence: float = 0.3
    # How many chunks to retrieve per question.
    docrag_top_k: int = 4

    # --- Datastores --------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://gridsense:gridsense@localhost:5432/gridsense",
        description="Postgres (with pgvector) connection string.",
    )
    mlflow_tracking_uri: str = "http://localhost:5000"

    # --- Observability -----------------------------------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, parsing ``.env`` only once."""
    return Settings()
