"""Downstream guardrails: schema retry/fallback, groundedness post-check, uncertainty."""

from __future__ import annotations

from pydantic import BaseModel

from gridsense.config import Settings
from gridsense.docrag.guardrails.downstream import (
    REPAIR_INSTRUCTION,
    assess_uncertainty,
    groundedness,
    invoke_structured,
)

SETTINGS = Settings(
    docrag_min_groundedness=0.6, docrag_uncertain_confidence=0.6, docrag_uncertain_relevance=0.65
)

CONTEXT = [
    "State of Health expresses a battery's current maximum capacity as a percentage of its "
    "original rated capacity. 80% SOH is the conventional end-of-life threshold.",
    "Degradation roughly doubles for every 10 degree increase in temperature.",
]


class Answer(BaseModel):
    text: str


class ScriptedModel:
    """Replays a script of outcomes; an Exception instance is raised, anything else returned."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[object]] = []

    def with_structured_output(self, _schema: object) -> ScriptedModel:
        return self

    def invoke(self, messages: list[object], config: object = None) -> object:
        self.calls.append(list(messages))
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --- schema validation + retry --------------------------------------------


def test_valid_output_returns_on_the_first_attempt() -> None:
    model = ScriptedModel(Answer(text="ok"))
    parsed, attempts = invoke_structured(model, Answer, ["m"])

    assert parsed == Answer(text="ok")
    assert attempts == 1
    assert len(model.calls) == 1


def test_parse_failure_is_retried_once_with_a_repair_instruction() -> None:
    model = ScriptedModel(ValueError("Failed to parse Answer"), Answer(text="repaired"))
    parsed, attempts = invoke_structured(model, Answer, ["m"], retries=1)

    assert parsed == Answer(text="repaired")
    assert attempts == 2
    assert len(model.calls) == 2
    # The retry carries the original messages plus the repair nudge.
    assert model.calls[1][-1].content == REPAIR_INSTRUCTION


def test_exhausted_retries_return_none_rather_than_raise() -> None:
    """The real regression: an unparseable completion used to blow up /ask with a 500."""
    model = ScriptedModel(ValueError("bad"), ValueError("still bad"))
    parsed, attempts = invoke_structured(model, Answer, ["m"], retries=1)

    assert parsed is None
    assert attempts == 2


def test_wrong_type_is_treated_as_a_parse_failure() -> None:
    model = ScriptedModel({"text": "a dict is not the schema"}, Answer(text="ok"))
    parsed, _ = invoke_structured(model, Answer, ["m"], retries=1)
    assert parsed == Answer(text="ok")


def test_retries_can_be_disabled() -> None:
    model = ScriptedModel(ValueError("bad"), Answer(text="never reached"))
    parsed, attempts = invoke_structured(model, Answer, ["m"], retries=0)

    assert parsed is None
    assert attempts == 1
    assert len(model.calls) == 1


# --- groundedness post-check ----------------------------------------------


def test_grounded_answer_scores_high() -> None:
    report = groundedness("80% SOH is the conventional end-of-life threshold.", CONTEXT)
    assert report.ratio == 1.0
    assert report.unsupported == []


def test_offtopic_sentence_is_flagged_unsupported() -> None:
    answer = "80% SOH is the conventional end-of-life threshold. Paris hosts the Olympic games."
    report = groundedness(answer, CONTEXT)
    assert report.ratio == 0.5
    assert "Paris" in report.unsupported[0]


def test_invented_number_makes_its_sentence_unsupported() -> None:
    """The failure that actually hurts in technical docs: a plausible, fabricated figure."""
    report = groundedness("The conventional end-of-life threshold is 72% SOH.", CONTEXT)
    assert report.ratio == 0.0
    assert report.unsupported == ["The conventional end-of-life threshold is 72% SOH."]


def test_number_present_in_context_is_accepted() -> None:
    assert groundedness("The threshold is 80% SOH.", CONTEXT).ratio == 1.0


def test_empty_context_grounds_nothing() -> None:
    report = groundedness("Anything at all.", [])
    assert report.ratio == 0.0
    assert report.unsupported == ["Anything at all."]


def test_empty_answer_is_not_grounded() -> None:
    assert groundedness("", CONTEXT).ratio == 0.0


# --- uncertainty disclosure ------------------------------------------------


def _assess(confidence: float, relevance: list[float], grounding: float):
    return assess_uncertainty(
        confidence=confidence, relevance_scores=relevance, grounding=grounding, settings=SETTINGS
    )


def test_strong_signals_give_low_uncertainty_and_no_note() -> None:
    verdict = _assess(0.9, [0.85], 1.0)
    assert verdict.level == "low"
    assert verdict.reasons == []
    assert verdict.note == ""


def test_one_weak_signal_is_medium_and_discloses() -> None:
    verdict = _assess(0.4, [0.85], 1.0)
    assert verdict.level == "medium"
    assert verdict.reasons == ["low_confidence"]
    assert verdict.note


def test_two_weak_signals_escalate_to_high() -> None:
    verdict = _assess(0.4, [0.3], 1.0)
    assert verdict.level == "high"
    assert set(verdict.reasons) == {"low_confidence", "weak_retrieval"}


def test_thin_grounding_alone_is_disclosed() -> None:
    verdict = _assess(0.9, [0.85], 0.2)
    assert verdict.level == "medium"
    assert verdict.reasons == ["weak_grounding"]


def test_no_retrieval_scores_counts_as_weak_retrieval() -> None:
    assert "weak_retrieval" in _assess(0.9, [], 1.0).reasons
