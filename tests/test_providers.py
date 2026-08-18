"""Provider selection: the Claude backend, and the two rules that make it work.

Two things about Anthropic are not interchangeable with the other backends, and both are
silent failures if they regress:

1. ``temperature`` (and ``top_p`` / ``top_k``) are **removed** on Claude Opus 5 — sending
   one is a 400, not a warning. So the Anthropic kwargs must never carry it, even though
   ``get_chat_model`` still takes a ``temperature`` argument for the other two providers.
2. Anthropic serves **no embeddings API**, so chat and embedding providers cannot be one
   setting. The legacy single ``LLM_PROVIDER`` still has to configure both, except for the
   one value that has no embedding meaning.

Every case builds ``Settings`` with ``_env_file=None``: a developer's real ``.env`` sets
``LLM_PROVIDER``, and these tests are about what the settings layer does with it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gridsense.config import ChatProvider, EmbeddingProvider, Settings
from gridsense.providers import anthropic_chat_kwargs, get_chat_model


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- defaults ---------------------------------------------------------------


def test_defaults_are_claude_for_chat_and_ollama_for_embeddings():
    settings = _settings()
    assert settings.chat_provider is ChatProvider.anthropic
    assert settings.embedding_provider is EmbeddingProvider.ollama
    assert settings.anthropic_model == "claude-opus-5"


# --- the sampling-parameter contract ----------------------------------------


@pytest.mark.parametrize("removed", ["temperature", "top_p", "top_k"])
def test_anthropic_kwargs_never_carry_a_sampling_parameter(removed):
    """These are 400s on Opus 5. The kwargs dict is the last place to catch them."""
    kwargs = anthropic_chat_kwargs(_settings(anthropic_api_key="sk-test"))
    assert removed not in kwargs


def test_anthropic_kwargs_do_not_disable_thinking():
    """Thinking is on by default on Opus 5 and `effort` is what tunes its depth.

    Disabling it buys nothing here and costs two documented failure modes — tool calls
    emitted as plain text, and `<thinking>` tags leaking into the answer — both of which
    would land on the structured-output path this model is used for.
    """
    assert "thinking" not in anthropic_chat_kwargs(_settings(anthropic_api_key="sk-test"))


def test_anthropic_kwargs_carry_model_effort_and_token_ceiling():
    kwargs = anthropic_chat_kwargs(
        _settings(anthropic_api_key="sk-test", anthropic_effort="xhigh", anthropic_max_tokens=32000)
    )
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 32000
    assert kwargs["output_config"]["effort"] == "xhigh"


def test_missing_api_key_fails_with_an_actionable_message():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_chat_model(_settings(chat_provider="anthropic", anthropic_api_key=None))


# --- the legacy LLM_PROVIDER alias ------------------------------------------


@pytest.mark.parametrize("legacy", ["ollama", "azure"])
def test_legacy_llm_provider_still_configures_both(legacy):
    """A pre-Claude .env, docker-compose.yml or configmap must keep working untouched."""
    settings = _settings(llm_provider=legacy)
    assert settings.chat_provider == legacy
    assert settings.embedding_provider == legacy


def test_explicit_chat_provider_wins_over_the_legacy_alias():
    settings = _settings(llm_provider="ollama", chat_provider="anthropic")
    assert settings.chat_provider is ChatProvider.anthropic
    assert settings.embedding_provider is EmbeddingProvider.ollama


def test_legacy_alias_rejects_anthropic_rather_than_guessing_an_embedding_provider():
    """There is no embedding provider `LLM_PROVIDER=anthropic` could mean.

    Guessing one would silently pick a vector space the caller never chose — and a wrong
    embedding model is exactly the failure that produces plausible answers from a broken
    index, with no error anywhere.
    """
    with pytest.raises(ValidationError, match="no embeddings API"):
        _settings(llm_provider="anthropic")


def test_embedding_provider_cannot_be_anthropic():
    with pytest.raises(ValidationError):
        _settings(embedding_provider="anthropic")


# --- the built client, not just the kwargs ----------------------------------


def test_built_client_drops_temperature_and_keeps_effort():
    """Assert against the constructed client, because the kwargs dict is not the contract.

    ``langchain-anthropic`` silently relocates anything it recognises out of
    ``model_kwargs`` into its own field — so a parameter passed the wrong way is accepted,
    warned about once, and then never sent. Only the built object shows what the request
    will actually carry.
    """
    model = get_chat_model(
        _settings(chat_provider="anthropic", anthropic_api_key="sk-test", anthropic_effort="max")
    )
    assert model.temperature is None, "temperature is a 400 on Opus 5; it must never be set"
    assert model.top_p is None
    assert model.top_k is None
    assert model.output_config == {"effort": "max"}
    assert not model.model_kwargs, "nothing should be left riding in model_kwargs"


# --- embeddings -------------------------------------------------------------


def test_voyage_is_an_embedding_provider_but_not_a_chat_one():
    """Voyage does embeddings and nothing else; the enums say so rather than a runtime check."""
    assert "voyage" in set(EmbeddingProvider)
    assert "voyage" not in set(ChatProvider)


def test_voyage_embeddings_build_with_the_configured_model():
    from gridsense.providers import get_embeddings

    embeddings = get_embeddings(
        _settings(embedding_provider="voyage", voyage_api_key="pa-test", voyage_model="voyage-4")
    )
    assert embeddings.model == "voyage-4"


def test_voyage_without_a_key_fails_with_an_actionable_message():
    from gridsense.providers import get_embeddings

    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        get_embeddings(_settings(embedding_provider="voyage", voyage_api_key=None))


def test_the_default_embedding_model_is_the_one_with_a_free_tier():
    """voyage-4-lite carries 200M free tokens; the older -lite variants carry none, and the
    difference is invisible until a bill arrives."""
    assert _settings().voyage_model == "voyage-4-lite"
