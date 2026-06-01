"""LLM-as-judge scorers for DocRAG answers (groundedness and answer relevance).

These use the configured chat model to score an answer on a 0..1 scale. They are
intentionally small and prompt-driven; with a strong model (Azure OpenAI) the scores are
reliable, with a small local model they are indicative. Citation correctness is scored
deterministically in :mod:`gridsense.docrag.eval.run`, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class JudgeScore(BaseModel):
    """A judge's verdict."""

    score: float = Field(ge=0.0, le=1.0, description="0 = fails the criterion, 1 = fully meets it.")
    reason: str = Field(default="", description="Brief justification.")


def _judge(chat_model: BaseChatModel, system: str, user: str) -> float:
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = chat_model.with_structured_output(JudgeScore)
    verdict: JudgeScore = structured.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return max(0.0, min(1.0, verdict.score))


def judge_groundedness(answer: str, context: str, *, chat_model: BaseChatModel) -> float:
    """Score how fully ``answer`` is supported by ``context`` (0..1)."""
    system = (
        "You are a strict evaluator. Given CONTEXT and an ANSWER, score how fully the answer "
        "is supported by the context. 1.0 = every claim is supported; 0.0 = unsupported or "
        "contradicted. Judge support only, not whether the answer is helpful."
    )
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    return _judge(chat_model, system, user)


def judge_relevance(question: str, answer: str, *, chat_model: BaseChatModel) -> float:
    """Score how well ``answer`` addresses ``question`` (0..1)."""
    system = (
        "You are a strict evaluator. Given a QUESTION and an ANSWER, score how directly the "
        "answer addresses the question. 1.0 = directly and completely answers it; 0.0 = "
        "off-topic or non-responsive."
    )
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return _judge(chat_model, system, user)
