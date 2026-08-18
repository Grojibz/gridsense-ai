"""Guardrails around the DocRAG chain.

:mod:`upstream` runs on the query before any token is spent; :mod:`downstream` runs on
the model's output before it reaches the caller. Both are pure functions over plain data
so they can be tested — and reasoned about — without a model or a datastore.
"""

from gridsense.docrag.guardrails.downstream import (
    INVALID_OUTPUT,
    PROVIDER_ERROR,
    GroundednessReport,
    Uncertainty,
    UncertaintyLevel,
    assess_uncertainty,
    groundedness,
    invoke_structured,
    is_provider_error,
)
from gridsense.docrag.guardrails.upstream import (
    InputVerdict,
    check_input,
    detect_injection,
    estimate_tokens,
    route_intent,
    scrub_pii,
)

__all__ = [
    "GroundednessReport",
    "InputVerdict",
    "Uncertainty",
    "UncertaintyLevel",
    "assess_uncertainty",
    "check_input",
    "detect_injection",
    "estimate_tokens",
    "groundedness",
    "INVALID_OUTPUT",
    "PROVIDER_ERROR",
    "invoke_structured",
    "is_provider_error",
    "route_intent",
    "scrub_pii",
]
