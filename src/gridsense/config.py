"""Application configuration, driven by environment variables / a local ``.env``.

Centralising settings here keeps secrets and environment-specific values out of the
code. Import the singleton via :func:`get_settings` so the ``.env`` file is parsed once.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatProvider(StrEnum):
    """Which backend serves chat-completion calls."""

    anthropic = "anthropic"
    azure = "azure"
    ollama = "ollama"


class EmbeddingProvider(StrEnum):
    """Which backend serves embedding calls.

    Deliberately narrower than :class:`ChatProvider`: Anthropic ships no embeddings API, so
    the two cannot be collapsed back into one setting. Running Claude for generation while
    Ollama or Azure serves the vectors is the normal configuration, not a workaround.
    """

    azure = "azure"
    ollama = "ollama"


#: Deprecated alias kept so ``from gridsense.config import LLMProvider`` still resolves.
LLMProvider = ChatProvider


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

    # --- LLM providers -----------------------------------------------------
    # Chat and embeddings are configured separately because Anthropic has no embeddings
    # API: `chat_provider=anthropic` is only meaningful alongside an embedding provider
    # that is not Anthropic. Collapsing them back into one field would make the default
    # configuration unrepresentable.
    chat_provider: ChatProvider = ChatProvider.anthropic
    embedding_provider: EmbeddingProvider = EmbeddingProvider.ollama

    #: Deprecated. Set both providers at once, as the pre-Claude releases did. Ignored for
    #: whichever of the two fields is also set explicitly.
    llm_provider: LLMProvider | None = None

    # Anthropic (used when chat_provider == anthropic)
    anthropic_api_key: str | None = None
    #: Exact model ID, no date suffix. Opus 5 is the default; `claude-sonnet-5` and
    #: `claude-haiku-4-5` are the cheaper rungs.
    anthropic_model: str = "claude-opus-5"
    #: Thinking depth and overall token spend: low | medium | high | xhigh | max.
    anthropic_effort: str = "high"
    #: Hard ceiling on thinking *plus* answer text. Thinking is on by default on Opus 5,
    #: so a limit sized around the answer alone truncates the response mid-sentence.
    anthropic_max_tokens: int = 16000
    #: Let the API re-run a safety-declined request on a fallback model instead of just
    #: stopping. Turn off for an account without the server-side-fallback beta.
    anthropic_refusal_fallback: bool = True

    # Azure OpenAI (used when either provider == azure)
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Ollama (used when either provider == ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"
    # Layers to offload to the GPU. ``None`` lets Ollama decide (the normal setting); 0
    # forces CPU-only inference. Worth knowing that 0 is a real escape hatch: on a machine
    # whose GPU computes incorrectly, Ollama returns fluent-looking garbage rather than an
    # error — the same prompt yielding "@@@@@@" on GPU and a correct answer on CPU — and it
    # corrupts embeddings the same way, silently, on write.
    ollama_num_gpu: int | None = None
    # Context window. Ollama defaults to 2048 regardless of what the model supports, which
    # silently truncates a RAG prompt carrying several chunks.
    ollama_num_ctx: int = 8192

    # --- DocRAG guardrails -------------------------------------------------
    # Minimum retrieval relevance (0..1) for a chunk to count as context. Below this
    # for all chunks, DocRAG refuses rather than answer from irrelevant context.
    # Tuned for Ollama nomic-embed-text; adjust per embedding provider.
    docrag_min_relevance: float = 0.5
    # Minimum self-reported answer confidence (0..1) below which DocRAG refuses.
    docrag_min_confidence: float = 0.3
    # How many chunks to retrieve per question.
    docrag_top_k: int = 4

    # --- Upstream guardrails (applied to the query, before any token is spent) ---
    # Hard cap on question length. A doc-QA question longer than this is abuse or a
    # mistake, not a question; it is rejected rather than truncated.
    docrag_max_input_tokens: int = 512
    # Prompt-injection score (0..1) at or above which a query is rejected.
    docrag_injection_threshold: float = 0.5
    # Run spaCy NER as an extra PII detector on top of the regex patterns. Off by
    # default: it needs the `en_core_web_sm` model and its false positives would
    # redact legitimate technical terms out of the question.
    docrag_pii_ner_enabled: bool = False
    # Let the intent router hard-reject clearly out-of-scope queries.
    docrag_intent_router_enabled: bool = True

    # --- Downstream guardrails (applied to the model's output) -------------
    # How many times to re-ask the model when its structured output fails validation.
    docrag_output_retries: int = 1
    # Fraction of answer sentences that must be lexically supported by the retrieved
    # chunks. Below this the answer is disclosed as uncertain, not suppressed.
    docrag_min_groundedness: float = 0.6

    # --- Uncertainty disclosure --------------------------------------------
    # Confidence below this (but at or above `docrag_min_confidence`) still answers,
    # flagged as uncertain rather than passed off as solid.
    docrag_uncertain_confidence: float = 0.6
    # Best chunk relevance below this marks the answer uncertain. Calibrated against the
    # golden set on nomic-embed-text, where top-1 relevance runs median 0.70 (min 0.54) for
    # answerable queries and median 0.57 for unanswerable ones. 0.60 sits in that gap;
    # raising it toward 0.65 would flag a quarter of good queries as weak. Re-measure when
    # switching embedding provider — the scale is not comparable across models.
    docrag_uncertain_relevance: float = 0.60

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

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_llm_provider(cls, data: Any) -> Any:
        """Let a pre-Claude ``LLM_PROVIDER`` still configure both providers at once.

        Older deployments (and the checked-in ``docker-compose.yml`` / ``k8s/configmap.yaml``)
        set a single ``LLM_PROVIDER``. It fills whichever of the two new fields was not set
        explicitly, so an existing ``.env`` keeps working unchanged while an explicit
        ``CHAT_PROVIDER`` always wins.
        """
        if not isinstance(data, dict):
            return data
        legacy = data.get("llm_provider")
        if not legacy:
            return data
        if str(legacy) == ChatProvider.anthropic:
            # There is no embedding provider this could mean, so guessing one would pick a
            # vector space the caller never chose. Make them say it.
            raise ValueError(
                "LLM_PROVIDER=anthropic is ambiguous: Anthropic serves no embeddings API. "
                "Set CHAT_PROVIDER=anthropic and EMBEDDING_PROVIDER=ollama|azure instead."
            )
        data.setdefault("chat_provider", legacy)
        data.setdefault("embedding_provider", legacy)
        return data


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, parsing ``.env`` only once."""
    return Settings()
