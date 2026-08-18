"""A cheap research subagent the coordinator can hand reading-heavy work to.

The coordinator runs on Opus because the work that matters — deciding which tools a
question needs, comparing a predicted number against a documented threshold, and saying
plainly when the two disagree — is judgement. Trawling the corpus for the passages that
bear on a sub-question is not: it is mostly input tokens and extraction, which is what the
cheapest capable model is for.

So `delegate_research` runs its own tool loop on Haiku with `search_docs` as its only tool,
and returns a short written finding. Two things follow from that shape:

- the coordinator's context stays small — it reads one finding, not twenty passages;
- the bulk of the reading is billed at roughly a fifth of Opus rates.

**On the word "subagent".** The Messages API has no subagent primitive: this is a tool whose
implementation happens to be another tool loop. Managed Agents does have one
(``multiagent: {"type": "coordinator", "agents": [...]}``), with real per-thread isolation
and its own event stream — but it needs an Anthropic-hosted sandbox, which this repo
deliberately does not require. Naming the difference is more useful than pretending the two
are the same thing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anthropic import beta_tool

from gridsense.agent import tools
from gridsense.config import Settings, get_settings

if TYPE_CHECKING:
    from anthropic import Anthropic
    from langchain_postgres import PGVector

#: The worker's model. Cheap on purpose — this thread reads, it does not decide.
RESEARCH_MODEL = "claude-haiku-4-5"

#: Bounded so a worker that keeps searching cannot quietly become the expensive half.
MAX_RESEARCH_ITERATIONS = 6

RESEARCH_SYSTEM = """\
You research one question against a battery energy-storage (BESS) documentation corpus.

Search as many times as the question needs, then report your findings and stop. Quote the
figures and thresholds you find verbatim, and name the source file for every claim.

If the corpus does not answer the question, say so plainly. A stated gap is a useful
finding; a plausible-sounding guess is not, and the caller has no way to tell them apart.\
"""


def build_research_tool(
    *,
    settings: Settings | None = None,
    client: Anthropic | None = None,
    vectorstore: PGVector | None = None,
) -> Any:
    """Build the `delegate_research` tool, closing over its dependencies.

    Returning a closure rather than a module-level tool keeps the injection seams
    (``client``, ``vectorstore``) out of the schema the model sees, and lets a test drive
    the whole delegation path with fakes.
    """
    settings = settings or get_settings()

    @beta_tool
    def search_docs(query: str, k: int = 4) -> dict[str, Any]:
        """Retrieve passages from the BESS technical documentation.

        Args:
            query: What to look for. Search several phrasings rather than one.
            k: How many candidate passages to consider.
        """
        return tools.search_docs(query, k, vectorstore=vectorstore, settings=settings)

    @beta_tool
    def delegate_research(question: str) -> str:
        """Hand one self-contained research question to a fast documentation researcher.

        Call this when a question needs several searches across the corpus, or when you
        need background you would otherwise read at length yourself. The researcher sees
        none of this conversation, so state the question in full — it cannot ask you what
        you meant.

        Use it for breadth. For a single lookup, search the documentation yourself: a
        delegation costs a round trip and a re-briefing, which one search does not justify.

        Args:
            question: One complete, self-contained question, with any context it needs.
        """
        inner = client
        if inner is None:
            from anthropic import Anthropic

            inner = Anthropic(api_key=settings.anthropic_api_key)

        runner = inner.beta.messages.tool_runner(
            model=RESEARCH_MODEL,
            max_tokens=4096,
            system=RESEARCH_SYSTEM,
            max_iterations=MAX_RESEARCH_ITERATIONS,
            tools=[search_docs],
            messages=[{"role": "user", "content": question}],
        )

        final = None
        for message in runner:
            final = message

        if final is None:
            return "The researcher returned nothing. Search the documentation directly."
        if final.stop_reason == "refusal":
            return "The researcher declined this question. Do not retry it verbatim."

        text = "\n".join(b.text for b in final.content if b.type == "text").strip()
        return text or "The researcher finished without a written finding."

    return delegate_research
