"""The tool layer both the MCP server and the Claude agent loop expose.

Every tool here is a plain function over plain data, with its collaborators injectable.
That is what lets one definition serve two very different surfaces — an MCP server spoken
to over stdio, and an in-process Anthropic tool runner — without either one owning the
logic or the two drifting apart.

Nothing here re-implements Module A or Module B. Each function is a thin adapter over an
existing entry point (:func:`~gridsense.docrag.chain.answer_question`,
:func:`~gridsense.degrade.serve.predict_soh`, ...), so the guardrails, the relevance gate
and the prediction logging apply identically whether a call arrives from ``/ask``, from an
MCP client, or from a model deciding to reach for a tool.

**The docstrings are load-bearing.** They are what a model reads to decide whether to call
a tool, so each one says *when* to call it, not only what it does — that is the difference
between a tool that gets used and one that sits idle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_postgres import PGVector
    from sqlalchemy.engine import Engine

#: Default location of the eval run written by ``eval/run_ragas.py``.
DEFAULT_RESULTS_PATH = "eval/results.json"


def search_docs(
    query: str,
    k: int = 4,
    *,
    vectorstore: PGVector | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Retrieve raw passages from the BESS technical documentation, without answering.

    Call this when you need the source text itself: to quote a standard verbatim, to check
    whether the corpus covers a topic at all, or to gather evidence before reasoning across
    several documents. Prefer ask_docs when a written answer with citations is what you
    actually want.

    Returns the passages that cleared the relevance floor, each with its source file and
    relevance score, plus the scores of every candidate considered — a query whose scores
    are uniformly low has fallen off the corpus, which is worth knowing before building an
    answer on top of it.
    """
    from gridsense.docrag.chain import retrieve_context

    settings = settings or get_settings()
    if vectorstore is None:
        from gridsense.docrag.retriever import get_vectorstore

        vectorstore = get_vectorstore(settings)

    kept, all_scores = retrieve_context(query, vectorstore=vectorstore, k=k, settings=settings)
    return {
        "passages": [
            {
                "source": str(doc.metadata.get("source", "unknown")),
                "text": doc.page_content,
                "relevance": score,
            }
            for doc, score in kept
        ],
        "candidate_scores": all_scores,
        "relevance_floor": settings.docrag_min_relevance,
    }


def ask_docs(
    question: str,
    *,
    vectorstore: PGVector | None = None,
    chat_model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Answer a question from the BESS technical documentation, with citations.

    Call this for any question about battery standards, state-of-health thresholds,
    operating limits, test procedures, or the contents of the ingested reports. This is the
    only tool that reads the documentation *and* writes an answer.

    The full guardrail path applies: the question is length-capped, scanned for prompt
    injection, routed for intent and PII-scrubbed before it reaches a model, and the answer
    is schema-validated and checked sentence by sentence against the retrieved text.

    Check `refused` before using `answer` — a refusal is a structural decision carrying a
    `refusal_reason`, not a phrasing. When `uncertainty_level` is medium or high the answer
    is still usable, but must be presented with its `uncertainty_note` rather than as
    settled fact.
    """
    from gridsense.docrag.chain import answer_question

    result = answer_question(
        question,
        vectorstore=vectorstore,
        chat_model=chat_model,
        settings=settings,
    )
    return result.model_dump()


def predict_soh(
    cycle_count: float,
    avg_temperature_c: float,
    avg_dod: float,
    avg_c_rate: float,
    calendar_age_days: float,
    *,
    model: Any | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Predict a battery pack's State of Health (%) from its operating conditions.

    Call this whenever the question involves how a pack has aged, or will age, under given
    duty — temperature, depth of discharge, C-rate, cycle count, calendar age. It returns a
    number, not a verdict: deciding whether that number is acceptable needs the end-of-life
    threshold from the documentation, so pair it with search_docs or ask_docs.

    `avg_dod` is a fraction between 0 and 1, not a percentage. The registered model version
    used is returned alongside the prediction, and the call is logged for drift monitoring.
    """
    from gridsense.degrade.serve import predict_soh as _predict_soh

    return _predict_soh(
        {
            "cycle_count": cycle_count,
            "avg_temperature_c": avg_temperature_c,
            "avg_dod": avg_dod,
            "avg_c_rate": avg_c_rate,
            "calendar_age_days": calendar_age_days,
        },
        model=model,
        settings=settings,
        engine=engine,
        log=log,
    )


def drift_report(
    *,
    current_limit: int = 500,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compare recent /predict inputs against the model's training distribution.

    Call this to answer whether the SOH model is still being asked about the kind of packs
    it was trained on — before trusting a prediction on unusual duty, or when investigating
    predictions that look wrong.

    This is *data* drift only. No ground-truth SOH ever comes back from production, so
    concept drift is not measurable here and must not be inferred from this result. `alert`
    is true when the drifted share of input columns exceeds the configured threshold.
    """
    from gridsense.degrade.monitor import run_drift_report

    return run_drift_report(
        engine=engine, settings=settings, current_limit=current_limit
    ).model_dump()


def eval_gate(results_path: str | Path = DEFAULT_RESULTS_PATH) -> dict[str, Any]:
    """Check the latest eval run against the merge-gate thresholds.

    Call this to answer whether the current DocRAG pipeline is shippable, or to explain why
    the eval CI job is failing. Returns every gated metric with its threshold, and a
    rendered explanation for each violation.

    A metric missing from the results file counts as 0.0 rather than being skipped, so a
    scorer that silently stopped producing a number cannot pass the gate by omission.
    """
    from gridsense.docrag.eval.thresholds import THRESHOLDS, check_thresholds

    path = Path(results_path)
    if not path.is_file():
        return {
            "passed": False,
            "error": f"No eval results at '{path}'. Run `make eval-ragas` first.",
            "violations": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores", payload)
    violations = check_thresholds(
        scores,
        coverage=payload.get("coverage"),
        n_answerable=payload.get("n_answerable", 0),
    )
    return {
        "passed": not violations,
        "scores": {metric: scores.get(metric) for metric in THRESHOLDS},
        "thresholds": dict(THRESHOLDS),
        "violations": [{**v.model_dump(), "message": v.render().strip()} for v in violations],
    }


def as_tool_result(payload: dict[str, Any]) -> str:
    """Serialise a tool's return value for the Anthropic tool runner.

    The functions above return dicts because that is the right shape for Python callers and
    for the MCP server, whose SDK serialises a dict return to JSON text itself. The Anthropic
    tool runner does **not**: it puts whatever the function returned straight into
    ``tool_result.content``, which the Messages API defines as a string or a list of content
    blocks. A bare object there is not a shape the API accepts.

    That is also what ``beta_tool``'s own type says — its ``FunctionT`` is bound to a callable
    returning ``str`` or an iterable of block params — and it is the reason the two surfaces
    wrap the shared functions rather than registering them directly. No test caught it,
    because every agent test replaces the client with a fake and so never serialises a
    request; ``tests/test_agent_loop.py`` now drives the real SDK over a mock transport for
    exactly this.

    ``default=str`` so a stray non-serialisable value degrades to its repr rather than
    raising inside the loop, where the failure would surface as an opaque tool error.
    """
    return json.dumps(payload, ensure_ascii=False, default=str)
