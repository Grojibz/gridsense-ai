"""`/ask` route — DocRAG question answering over the ingested documents."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gridsense.docrag.chain import RagAnswer, answer_question

router = APIRouter(tags=["docrag"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question about the docs.")
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve.")


@router.post("/ask", response_model=RagAnswer)
def ask(request: AskRequest) -> RagAnswer:
    """Answer a question from the ingested documents with citations and confidence."""
    return answer_question(request.question, k=request.k)
