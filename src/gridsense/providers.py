"""Factories for the LLM chat model and embeddings, selected by ``LLM_PROVIDER``.

Centralising provider construction here keeps the rest of the code provider-agnostic:
DocRAG and DegradeML ask for ``get_chat_model()`` / ``get_embeddings()`` and never care
whether they're talking to Azure OpenAI or a local Ollama.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridsense.config import LLMProvider, Settings, get_settings

if TYPE_CHECKING:  # imported lazily at runtime to keep base imports cheap
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return an embeddings client for the configured provider."""
    settings = settings or get_settings()

    if settings.llm_provider is LLMProvider.ollama:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model for the configured provider.

    Temperature defaults to 0 — for a citing RAG assistant we want determinism, not
    creativity.
    """
    settings = settings or get_settings()

    if settings.llm_provider is LLMProvider.ollama:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=temperature,
    )
