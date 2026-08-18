"""FastAPI application entrypoint.

Run locally with::

    uvicorn gridsense.api.main:app --reload

Exposes ``/health``, the DocRAG ``/ask`` endpoint (M1), the DegradeML ``/predict``
endpoint (M4), and the Claude tool-use ``/agent`` endpoint (M8) that spans both.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from gridsense import __version__
from gridsense.api.routes_agent import router as agent_router
from gridsense.api.routes_ask import router as ask_router
from gridsense.api.routes_chat import router as chat_router
from gridsense.api.routes_predict import router as predict_router
from gridsense.config import get_settings


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


def create_app() -> FastAPI:
    """Application factory — keeps the app importable and testable."""
    settings = get_settings()
    app = FastAPI(
        title="GridSense AI",
        version=__version__,
        summary="RAG over technical docs (DocRAG) + battery-degradation MLOps (DegradeML).",
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe used by docker-compose, k8s, and CI smoke tests."""
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=__version__,
            environment=settings.environment,
        )

    app.include_router(chat_router)
    app.include_router(ask_router)
    app.include_router(agent_router)
    app.include_router(predict_router)
    return app


app = create_app()
