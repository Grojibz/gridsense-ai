"""FastAPI application entrypoint.

Run locally with::

    uvicorn gridsense.api.main:app --reload

M0 exposes only ``/health`` and a service banner. The ``/ask`` (DocRAG) and
``/predict`` (DegradeML) endpoints are added in later milestones (M1, M4).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from gridsense import __version__
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

    return app


app = create_app()
