"""Langfuse client factory for LLM tracing.

Tracing is optional: if no Langfuse keys are configured, :func:`get_langfuse` returns
``None`` and callers no-op. This keeps the app fully runnable offline and in CI without an
observability backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse


def get_langfuse(settings: Settings | None = None) -> Langfuse | None:
    """Return a configured Langfuse client, or ``None`` if keys are not set."""
    settings = settings or get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
