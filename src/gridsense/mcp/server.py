"""MCP server exposing GridSense as tools any MCP client can call.

Run it over stdio::

    python -m gridsense.mcp.server        # or: make mcp

Each tool is a thin wrapper over :mod:`gridsense.agent.tools`, which is in turn a thin
wrapper over the real entry points. The indirection buys one thing worth having: the
guardrails are not re-implemented at the protocol boundary. A question arriving from an MCP
client goes through the same injection scan, PII scrub, relevance gate and output
validation as one arriving at ``/ask`` — there is no second, weaker path into the model.

The wrappers exist rather than registering the shared functions directly because those
carry keyword-only injection parameters (``vectorstore``, ``engine``, ``settings``) that
exist for tests. MCP derives each tool's schema from the signature, so exposing them
verbatim would put test seams into the public contract.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from gridsense.agent import tools

INSTRUCTIONS = """\
GridSense exposes a battery energy-storage (BESS) documentation assistant and a
state-of-health prediction model.

Answering "will this pack still be healthy in N years?" needs both: predict_soh gives the
projected SOH, and the end-of-life threshold to compare it against lives in the
documentation. Neither tool decides that on its own.

Answers from ask_docs carry a structural `refused` flag and an `uncertainty_level`. Respect
them: report a refusal as a refusal, and pass on the uncertainty note rather than
presenting a hedged answer as a settled one.\
"""

server = MCPServer(
    name="gridsense",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


@server.tool()
def search_docs(query: str, k: int = 4) -> dict[str, Any]:
    """Retrieve raw passages from the BESS technical documentation, without answering.

    Call this when you need the source text itself: to quote a standard verbatim, to check
    whether the corpus covers a topic at all, or to gather evidence before reasoning across
    several documents. Prefer ask_docs when a written answer with citations is what you
    want.

    Returns the passages that cleared the relevance floor with their source file and score,
    plus the scores of every candidate considered — uniformly low scores mean the query has
    fallen off the corpus, which is worth knowing before building an answer on top of it.
    """
    return tools.search_docs(query, k)


@server.tool()
def ask_docs(question: str) -> dict[str, Any]:
    """Answer a question from the BESS technical documentation, with citations.

    Call this for any question about battery standards, state-of-health thresholds,
    operating limits, test procedures, or the contents of the ingested reports.

    The full guardrail path applies — the question is length-capped, scanned for prompt
    injection, routed for intent and PII-scrubbed before reaching a model, and the answer is
    schema-validated and checked sentence by sentence against the retrieved text.

    Check `refused` before using `answer`: a refusal is a structural decision carrying a
    `refusal_reason`, not a phrasing. When `uncertainty_level` is medium or high, present
    the answer together with its `uncertainty_note`, not as settled fact.
    """
    return tools.ask_docs(question)


@server.tool()
def predict_soh(
    cycle_count: float,
    avg_temperature_c: float,
    avg_dod: float,
    avg_c_rate: float,
    calendar_age_days: float,
) -> dict[str, Any]:
    """Predict a battery pack's State of Health (%) from its operating conditions.

    Call this whenever the question involves how a pack has aged, or will age, under given
    duty — temperature, depth of discharge, C-rate, cycle count, calendar age. It returns a
    number, not a verdict: deciding whether that number is acceptable needs the end-of-life
    threshold from the documentation, so pair it with search_docs or ask_docs.

    `avg_dod` is a fraction between 0 and 1, not a percentage. The registered model version
    is returned alongside the prediction, and the call is logged for drift monitoring.
    """
    return tools.predict_soh(
        cycle_count=cycle_count,
        avg_temperature_c=avg_temperature_c,
        avg_dod=avg_dod,
        avg_c_rate=avg_c_rate,
        calendar_age_days=calendar_age_days,
    )


@server.tool()
def drift_report(current_limit: int = 500) -> dict[str, Any]:
    """Compare recent /predict inputs against the model's training distribution.

    Call this to answer whether the SOH model is still being asked about the kind of packs
    it was trained on — before trusting a prediction on unusual duty, or when investigating
    predictions that look wrong.

    This is *data* drift only. No ground-truth SOH ever comes back from production, so
    concept drift is not measurable here and must not be inferred from this result. `alert`
    is true when the drifted share of input columns exceeds the configured threshold.
    """
    return tools.drift_report(current_limit=current_limit)


@server.tool()
def eval_gate(results_path: str = tools.DEFAULT_RESULTS_PATH) -> dict[str, Any]:
    """Check the latest eval run against the merge-gate thresholds.

    Call this to answer whether the current DocRAG pipeline is shippable, or to explain why
    the eval CI job is failing. Returns every gated metric with its threshold, and a
    rendered explanation for each violation.

    A metric missing from the results file counts as 0.0 rather than being skipped, so a
    scorer that silently stopped producing a number cannot pass the gate by omission.
    """
    return tools.eval_gate(results_path)


def main() -> None:
    """Serve over stdio — the transport MCP clients launch a local server with."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
