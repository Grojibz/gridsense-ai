"""Upstream guardrails: everything applied to the query *before* a token is spent.

Four checks, cheapest first, in the order :func:`check_input` runs them:

1. **Length cap** — a doc-QA question longer than the cap is abuse or a mistake, not a
   question. Rejected, never silently truncated.
2. **Prompt-injection detection** — weighted regex patterns over the known families
   (instruction override, role reassignment, system-prompt exfiltration, delimiter
   smuggling, guardrail bypass). Not an ML classifier, and deliberately not described as
   one: the score is the sum of matched pattern weights.
3. **Intent routing** — hard-rejects queries that clearly want something this service does
   not do, so out-of-scope traffic never reaches the embedder or the LLM. Tuned for
   *precision*: anything ambiguous falls through to retrieval, which refuses on its own.
4. **PII scrubbing** — the query is redacted before it is logged *and* before it is sent
   to the model, so neither the trace store nor the provider ever sees the raw value.

Every check is a pure function over text, so the whole upstream path is testable without a
model, a datastore, or a network.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from gridsense.config import Settings, get_settings

Intent = Literal["in_domain", "unknown", "out_of_scope"]
BlockReason = Literal["ok", "empty", "too_long", "prompt_injection", "out_of_scope"]

# --- Prompt injection ------------------------------------------------------
# (name, pattern, weight). Weights are calibrated so one unambiguous instruction-override
# or exfiltration attempt trips the default 0.5 threshold on its own, while a single weak
# signal needs corroboration.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.?!]{0,40}\b"
            r"(previous|prior|above|preceding|earlier|all)\b[^.?!]{0,20}\b"
            r"(instruction|prompt|rule|direction|context)",
            re.I,
        ),
        0.6,
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|output|display|tell me)\b[^.?!]{0,30}\b"
            r"(your |the )?(system prompt|initial instruction|original instruction|"
            r"system message)",
            re.I,
        ),
        0.6,
    ),
    (
        "repeat_everything_above",
        re.compile(r"\brepeat\b[^.?!]{0,20}\b(everything|all|text)\b[^.?!]{0,20}\babove\b", re.I),
        0.6,
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(you are now|from now on,? you|act as if you|pretend (to be|you are)|"
            r"roleplay as|new (system )?instructions?:)",
            re.I,
        ),
        0.5,
    ),
    (
        "guardrail_bypass",
        re.compile(
            r"\b(without any (restrictions?|filters?|rules?|limits?)|bypass\b[^.?!]{0,20}"
            r"(guardrail|filter|restriction|safety)|no longer (bound|restricted))",
            re.I,
        ),
        0.5,
    ),
    (
        "jailbreak_persona",
        re.compile(r"\b(DAN mode|developer mode|jailbreak|unfiltered mode)\b", re.I),
        0.5,
    ),
    (
        "delimiter_smuggling",
        re.compile(
            r"(</?(system|instructions?|im_start|im_end)>|\[/?INST\]|###\s*(system|instruction))",
            re.I,
        ),
        0.5,
    ),
    (
        "encoded_payload",
        # A long unbroken base64-ish run has no business in a question about batteries.
        re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b"),
        0.4,
    ),
]

# --- Intent routing --------------------------------------------------------
#: Vocabulary of the indexed corpus. A query touching any of these is plausibly in scope.
DOMAIN_TERMS: frozenset[str] = frozenset(
    """
    battery batteries cell cells bess pack module rack
    soh soc dod state-of-health state-of-charge capacity nameplate
    degradation degrade ageing aging calendar lifetime end-of-life eol stressor plating sei
    lithium li-ion nmc lfp electrode electrolyte anode cathode
    charge charging discharge discharging cycle cycles cycling
    c-rate crate current voltage contactor bms balancing protection estimation
    thermal temperature cooling coolant runaway exothermic venting deflagration fusing
    enclosure efficiency round-trip storage grid energy
    """.split()  # noqa: SIM905 — grouped by theme; a flat 65-item list is unreviewable
)

#: Intents this service does not serve. Matching one of these with no domain term in the
#: query is treated as conclusive; matching with a domain term present is not.
OUT_OF_SCOPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "code_generation",
        re.compile(
            r"\b(write|generate|create|give me|show me)\b[^.?!]{0,30}\b"
            r"(script|program|function|code|snippet|class|query|regex)\b",
            re.I,
        ),
    ),
    ("code_literal", re.compile(r"(```|\bdef \w+\(|\bimport \w+|SELECT .+ FROM )", re.I)),
    (
        "translation",
        re.compile(r"\btranslate\b[^.?!]{0,30}\b(to|into)\b\s+\w+", re.I),
    ),
    (
        "chitchat",
        re.compile(
            r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening)|how are you)\b", re.I
        ),
    ),
    (
        "live_data",
        re.compile(
            r"\b(today'?s|current|latest|live)\b[^.?!]{0,20}\b"
            r"(price|quote|weather|news|stock|rate today)",
            re.I,
        ),
    ),
]

# --- PII -------------------------------------------------------------------
#: (label, pattern). Ordered: the greedier numeric patterns run last so an email or an
#: IBAN is consumed before the phone matcher can nibble at its digits.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CREDIT_CARD", re.compile(r"(?<![\d.])(?:\d[ -]?){13,19}(?![\d.])")),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d[\d\s.\-()]{7,18}\d(?![\w.])")),
]
#: A phone number has this many digits; anything outside the band is a measurement, a part
#: number or a year range, and must not be redacted out of a technical question.
PHONE_DIGIT_RANGE = (9, 15)

#: spaCy entity labels treated as PII when the NER layer is enabled.
NER_PII_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC"})

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


class InjectionVerdict(BaseModel):
    """Outcome of the prompt-injection scan."""

    score: float = Field(ge=0.0, le=1.0)
    patterns: list[str] = Field(default_factory=list)


class PIIResult(BaseModel):
    """A scrubbed string plus the *labels* that were redacted — never the values."""

    text: str
    labels: list[str] = Field(default_factory=list)


class InputVerdict(BaseModel):
    """What the upstream guardrails decided about a query."""

    allowed: bool
    reason: BlockReason
    #: User-facing explanation. Empty when the query was allowed.
    message: str = ""
    #: The query as it should be logged and sent to the model (PII redacted).
    query: str
    intent: Intent = "unknown"
    injection_score: float = 0.0
    injection_patterns: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    pii_labels: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _tiktoken_encoding() -> object | None:
    """The cl100k tokenizer, or ``None`` when tiktoken isn't installed."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Token count for ``text``, exact via tiktoken when available.

    The fallback (max of chars/4 and word count) is an *estimate*, and errs high so the
    cap can't be walked past by feeding the estimator its blind spot.
    """
    encoding = _tiktoken_encoding()
    if encoding is not None:
        return len(encoding.encode(text))  # type: ignore[attr-defined]
    return max(len(text) // 4, len(text.split()))


def detect_injection(text: str) -> InjectionVerdict:
    """Score ``text`` for prompt-injection patterns (0.0 = clean, 1.0 = flagrant)."""
    matched = [
        (name, weight) for name, pattern, weight in INJECTION_PATTERNS if pattern.search(text)
    ]
    return InjectionVerdict(
        score=min(1.0, sum(weight for _, weight in matched)),
        patterns=[name for name, _ in matched],
    )


def _domain_terms_in(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w in DOMAIN_TERMS}


def route_intent(text: str) -> Intent:
    """Classify ``text`` as in-domain, out-of-scope, or too ambiguous to judge.

    Only ``out_of_scope`` causes a rejection, so this is tuned for precision: a query is
    out of scope only when it asks for something the service doesn't do *and* mentions
    nothing from the corpus vocabulary.
    """
    has_domain_term = bool(_domain_terms_in(text))
    if any(pattern.search(text) for _, pattern in OUT_OF_SCOPE_PATTERNS) and not has_domain_term:
        return "out_of_scope"
    return "in_domain" if has_domain_term else "unknown"


@lru_cache(maxsize=1)
def _spacy_model() -> object | None:
    """The spaCy pipeline, or ``None`` if spaCy or its model isn't installed."""
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _redact_phone(text: str, found: list[str]) -> str:
    """Redact phone-shaped runs, skipping digit runs that are really measurements."""
    low, high = PHONE_DIGIT_RANGE

    def replace(match: re.Match[str]) -> str:
        digits = sum(c.isdigit() for c in match.group())
        if not low <= digits <= high:
            return match.group()
        found.append("PHONE")
        return "[PHONE]"

    return PII_PATTERNS[-1][1].sub(replace, text)


def scrub_pii(text: str, *, use_ner: bool = False) -> PIIResult:
    """Replace personal data in ``text`` with ``[LABEL]`` placeholders.

    ``use_ner`` adds a spaCy NER pass for names/orgs/places on top of the regex patterns.
    It is off by default because its false positives would redact legitimate technical
    terms out of the question; it no-ops when spaCy isn't installed.
    """
    found: list[str] = []
    scrubbed = text
    for label, pattern in PII_PATTERNS:
        if label == "PHONE":
            continue  # handled separately: needs a digit-count check
        scrubbed, hits = pattern.subn(lambda _m, lbl=label: f"[{lbl}]", scrubbed)
        found += [label] * hits
    scrubbed = _redact_phone(scrubbed, found)

    if use_ner and (nlp := _spacy_model()) is not None:
        doc = nlp(scrubbed)  # type: ignore[operator]
        for entity in reversed(doc.ents):  # reversed: edit from the end, keep offsets valid
            if entity.label_ in NER_PII_LABELS:
                scrubbed = (
                    scrubbed[: entity.start_char]
                    + f"[{entity.label_}]"
                    + scrubbed[entity.end_char :]
                )
                found.append(entity.label_)

    return PIIResult(text=scrubbed, labels=sorted(set(found)))


def check_input(query: str, settings: Settings | None = None) -> InputVerdict:
    """Run every upstream guardrail over ``query`` and decide whether to proceed."""
    settings = settings or get_settings()

    if not query.strip():
        return InputVerdict(allowed=False, reason="empty", message="Ask a question.", query=query)

    tokens = estimate_tokens(query)
    if tokens > settings.docrag_max_input_tokens:
        return InputVerdict(
            allowed=False,
            reason="too_long",
            message=(
                f"That question is {tokens} tokens; the limit is "
                f"{settings.docrag_max_input_tokens}. Please shorten it."
            ),
            query=query[:200],
            estimated_tokens=tokens,
        )

    injection = detect_injection(query)
    if injection.score >= settings.docrag_injection_threshold:
        return InputVerdict(
            allowed=False,
            reason="prompt_injection",
            message=(
                "That request looks like an attempt to change how this assistant works. "
                "Ask a question about the battery documentation instead."
            ),
            # Redacted before it is logged: the payload itself must not land in the trace.
            query="[REDACTED]",
            injection_score=injection.score,
            injection_patterns=injection.patterns,
            estimated_tokens=tokens,
        )

    intent = route_intent(query)
    if intent == "out_of_scope" and settings.docrag_intent_router_enabled:
        return InputVerdict(
            allowed=False,
            reason="out_of_scope",
            message=(
                "This assistant only answers from the indexed battery energy-storage "
                "documentation, and that question is outside it."
            ),
            query=query,
            intent=intent,
            estimated_tokens=tokens,
        )

    pii = scrub_pii(query, use_ner=settings.docrag_pii_ner_enabled)
    return InputVerdict(
        allowed=True,
        reason="ok",
        query=pii.text,
        intent=intent,
        injection_score=injection.score,
        injection_patterns=injection.patterns,
        estimated_tokens=tokens,
        pii_labels=pii.labels,
    )
