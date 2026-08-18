"""Downstream guardrails: everything applied to the model's output before it ships.

Three concerns:

1. **Schema validation with a bounded retry** (:func:`invoke_structured`). A small model
   will occasionally emit output that doesn't fit the schema — the offline default has
   been observed returning a partial object with `confidence` missing entirely. Left
   unhandled that is an unhandled exception on `/ask`, so the call is retried once with a
   repair instruction and then gives up cleanly for the caller to fall back on.

   It distinguishes two failures that look identical from the call site and are not the
   same thing at all. `invalid_output` means the model answered and the answer did not fit
   the schema: retrying is reasonable, and it says something about the model. A
   `provider_error` means the call never reached a model — no credit balance, a bad key, a
   rate limit, a refused connection. Collapsing the two makes an unpaid invoice read as a
   weak model, tells the user to "try again" when no amount of trying will help, and drags
   an infrastructure fault into `output_validity`, which is supposed to measure the model.
   Retries are also skipped for a provider error: the SDKs already retry what is worth
   retrying, so a second attempt only doubles the latency of a certain failure.

2. **Faithfulness post-check** (:func:`groundedness`). A deterministic, LLM-free check that
   each sentence of the answer is lexically supported by the retrieved chunks. Numbers get
   special treatment: a figure that appears nowhere in the context marks its sentence
   unsupported outright, because an invented number is the failure mode that actually
   hurts in technical documentation.

3. **Uncertainty disclosure** (:func:`assess_uncertainty`). When confidence is middling, or
   retrieval was weak, or grounding is thin, the answer still ships — but labelled, with
   the reasons attached. The point is to stop a shaky answer from *looking* like a solid
   one. The disclosure is a separate field rather than a prefix on the answer text, so it
   never contaminates the answer that RAGAS scores for faithfulness.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, Field

from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

T = TypeVar("T", bound=BaseModel)

UncertaintyLevel = Literal["low", "medium", "high"]

#: The model answered, and the answer did not fit the schema. Retrying is reasonable.
INVALID_OUTPUT = "invalid_output"
#: The call never reached a model — auth, credits, rate limit, connection. Retrying is not.
PROVIDER_ERROR = "provider_error"

#: Appended when the model's structured output failed to validate, to steer the retry.
REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed. Reply again with the required JSON object "
    "and every required field present. Do not add commentary outside the object."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z][a-z\-]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

#: Function words carry no evidence, so they must not prop up a sentence's support score.
_STOPWORDS = frozenset(
    """
    the a an and or but if then than that this these those is are was were be been being
    it its of in on at to for from by with as not no nor so such can could may might must
    will would shall should do does did have has had which who whom what when where why how
    there here also very more most much many some any each per about into over under
    """.split()  # noqa: SIM905 — a prose block is far easier to review than 70 list items
)

DISCLOSURE_TEXT = {
    "medium": "This answer is only partly supported by the documentation — verify it before acting on it.",
    "high": "Low confidence: the documentation does not clearly support this answer. Treat it as a lead, not a fact.",
}


class GroundednessReport(BaseModel):
    """How much of the answer the retrieved chunks actually back up."""

    ratio: float = Field(ge=0.0, le=1.0, description="Fraction of sentences supported.")
    unsupported: list[str] = Field(
        default_factory=list, description="Answer sentences with no support in the context."
    )


class Uncertainty(BaseModel):
    """The uncertainty verdict attached to an answer."""

    level: UncertaintyLevel
    reasons: list[str] = Field(default_factory=list)
    #: User-facing sentence to render alongside the answer. Empty when level is ``low``.
    note: str = ""


#: Top-level modules whose exceptions mean the request never reached a model.
_PROVIDER_ERROR_MODULES = frozenset({"anthropic", "openai", "ollama", "httpx", "httpcore"})


def is_provider_error(exc: BaseException) -> bool:
    """True when ``exc`` means the call failed rather than the answer being unusable.

    Two signals, both cheap and neither requiring the provider SDKs to be importable here:
    an HTTP status (every Anthropic and OpenAI SDK error carries one; a parse failure never
    does), and the top-level module the exception class comes from, which covers transport
    failures like a refused connection to a local Ollama.
    """
    if getattr(exc, "status_code", None) is not None:
        return True
    return type(exc).__module__.split(".")[0] in _PROVIDER_ERROR_MODULES


def invoke_structured(
    chat_model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    retries: int = 1,
) -> tuple[T | None, int, str | None]:
    """Invoke ``chat_model`` for structured ``schema`` output, retrying on a parse failure.

    Returns ``(parsed, attempts, failure)``. ``parsed`` is ``None`` when every attempt
    failed — the caller falls back deterministically rather than raising — and ``failure``
    names *which* failure it was: ``"invalid_output"`` when a model answered unusably, or
    ``"provider_error"`` when the call never got that far. See the module docstring for why
    the two must not be merged.
    """
    from langchain_core.messages import HumanMessage

    structured = chat_model.with_structured_output(schema)
    attempt_messages = list(messages)

    for attempt in range(1, retries + 2):
        try:
            parsed = structured.invoke(attempt_messages)
        except Exception as exc:
            if is_provider_error(exc):
                # Nothing to repair: there is no output. Retrying a bad key or an empty
                # credit balance just spends latency on a certain failure, and the SDK has
                # already retried anything transient.
                return None, attempt, PROVIDER_ERROR
            # Otherwise the model did answer and the answer was unusable — a malformed
            # completion, a validation error, or a refusal to emit the schema at all.
            attempt_messages = [*messages, HumanMessage(content=REPAIR_INSTRUCTION)]
            continue
        if isinstance(parsed, schema):
            return parsed, attempt, None
        attempt_messages = [*messages, HumanMessage(content=REPAIR_INSTRUCTION)]

    return None, retries + 1, INVALID_OUTPUT


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def groundedness(
    answer: str, contexts: list[str], *, support_threshold: float = 0.6
) -> GroundednessReport:
    """Fraction of the answer's sentences lexically supported by ``contexts``.

    A sentence counts as supported when ``support_threshold`` of its content words appear
    in the context *and* every number it states appears there too. This is a cheap
    hallucination tripwire, not a semantic judge — it catches invented figures and
    off-corpus claims, and it will not catch a fluent paraphrase that is subtly wrong.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]
    if not sentences or not contexts:
        return GroundednessReport(ratio=0.0, unsupported=sentences)

    joined = " ".join(contexts)
    context_words = _content_words(joined)
    context_numbers = set(_NUMBER_RE.findall(joined))

    unsupported: list[str] = []
    for sentence in sentences:
        words = _content_words(sentence)
        numbers = set(_NUMBER_RE.findall(sentence))
        overlap = len(words & context_words) / len(words) if words else 1.0
        if overlap < support_threshold or not numbers <= context_numbers:
            unsupported.append(sentence)

    supported = len(sentences) - len(unsupported)
    return GroundednessReport(ratio=supported / len(sentences), unsupported=unsupported)


def assess_uncertainty(
    *,
    confidence: float,
    relevance_scores: list[float],
    grounding: float,
    settings: Settings | None = None,
) -> Uncertainty:
    """Decide how much to trust an answer, and say why.

    One weak signal makes the answer ``medium``; two or more make it ``high``. Both get a
    disclosure note — the caller shows it instead of presenting a shaky answer as solid.
    """
    settings = settings or get_settings()

    reasons: list[str] = []
    if confidence < settings.docrag_uncertain_confidence:
        reasons.append("low_confidence")
    if not relevance_scores or max(relevance_scores) < settings.docrag_uncertain_relevance:
        reasons.append("weak_retrieval")
    if grounding < settings.docrag_min_groundedness:
        reasons.append("weak_grounding")

    level: UncertaintyLevel = "low" if not reasons else "medium" if len(reasons) == 1 else "high"
    return Uncertainty(level=level, reasons=reasons, note=DISCLOSURE_TEXT.get(level, ""))
