"""Factories for the LLM chat model and embeddings, selected by the configured providers.

Centralising provider construction here keeps the rest of the code provider-agnostic:
DocRAG and DegradeML ask for ``get_chat_model()`` / ``get_embeddings()`` and never care
whether they're talking to Anthropic, Azure OpenAI or a local Ollama.

Chat and embeddings are chosen by two *separate* settings. That is not symmetry for its own
sake: Anthropic ships no embeddings API, so the default configuration — Claude generating,
Ollama embedding — is unrepresentable with a single provider field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridsense.config import ChatProvider, EmbeddingProvider, Settings, get_settings

if TYPE_CHECKING:  # imported lazily at runtime to keep base imports cheap
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return an embeddings client for the configured embedding provider.

    ⚠️ **The relevance scale is not comparable across embedding models.**
    ``docrag_min_relevance`` and ``docrag_uncertain_relevance`` are calibrated against a
    specific model; switching provider without re-measuring them silently changes what
    counts as "relevant enough to answer from". Re-ingest *and* re-run
    ``eval/calibrate_relevance.py`` after any change here.
    """
    settings = settings or get_settings()

    if settings.embedding_provider is EmbeddingProvider.voyage:
        if not settings.voyage_api_key:
            raise ValueError(
                "embedding_provider=voyage but VOYAGE_API_KEY is unset. Get a key at "
                "voyageai.com, or switch EMBEDDING_PROVIDER to azure|ollama."
            )
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model=settings.voyage_model,
            voyage_api_key=settings.voyage_api_key,
        )

    if settings.embedding_provider is EmbeddingProvider.ollama:
        from langchain_ollama import OllamaEmbeddings

        extra = {} if settings.ollama_num_gpu is None else {"num_gpu": settings.ollama_num_gpu}
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            **extra,
        )

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )


def anthropic_chat_kwargs(settings: Settings) -> dict:
    """Build the ``ChatAnthropic`` constructor kwargs.

    Split out from :func:`get_chat_model` so the wire-level contract can be asserted in a
    unit test without importing ``langchain_anthropic`` or touching the network.

    Two things are deliberately *absent*:

    - ``temperature`` — removed on Claude Opus 5, along with ``top_p`` / ``top_k``. Sending
      it is a 400, not a warning. Determinism is steered with ``effort`` and the prompt now.
    - ``thinking`` — thinking is on by default on Opus 5 and its depth is what ``effort``
      controls. Passing ``{"type": "disabled"}`` would buy nothing here and costs two known
      failure modes (tool calls emitted as plain text, ``<thinking>`` tags leaking into the
      answer), both of which would land squarely on the structured-output path.
    """
    return {
        "model": settings.anthropic_model,
        "api_key": settings.anthropic_api_key,
        "max_tokens": settings.anthropic_max_tokens,
        # First-class parameter, not `model_kwargs`: langchain-anthropic hoists it out of
        # `model_kwargs` with a warning and would otherwise leave it silently unset.
        "output_config": {"effort": settings.anthropic_effort},
    }


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model for the configured chat provider.

    ``temperature`` applies to the Azure and Ollama backends only, where 0 is what makes a
    citing RAG assistant reproducible. Claude rejects the parameter outright, so it is
    dropped rather than quietly forwarded — see :func:`anthropic_chat_kwargs`.
    """
    settings = settings or get_settings()

    if settings.chat_provider is ChatProvider.anthropic:
        # Config is checked before the import so a missing key reports the thing the
        # operator can act on, rather than an ImportError from a dependency they would
        # only need once the key existed.
        if not settings.anthropic_api_key:
            raise ValueError(
                "chat_provider=anthropic but ANTHROPIC_API_KEY is unset. Set the key, or "
                "switch CHAT_PROVIDER to azure|ollama."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**anthropic_chat_kwargs(settings))

    if settings.chat_provider is ChatProvider.ollama:
        from langchain_ollama import ChatOllama

        extra = {} if settings.ollama_num_gpu is None else {"num_gpu": settings.ollama_num_gpu}
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
            **extra,
        )

    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=temperature,
    )
