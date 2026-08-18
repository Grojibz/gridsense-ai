"""FastAPI application entrypoint.

Run locally with::

    uvicorn gridsense.api.main:app --reload

Exposes ``/health``, the DocRAG ``/ask`` endpoint (M1), the DegradeML ``/predict``
endpoint (M4), and the Claude tool-use ``/agent`` endpoint (M8) that spans both.

The factory does three things before returning the app, in this order for a reason:
logging first so that anything the rest of startup has to say is actually recorded; the
auth check second so a deployment without authentication fails here rather than at the
first request; routers last.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from gridsense import __version__
from gridsense.api.middleware import install_middleware
from gridsense.api.ratelimit import rate_limit
from gridsense.api.routes_agent import router as agent_router
from gridsense.api.routes_ask import router as ask_router
from gridsense.api.routes_chat import router as chat_router
from gridsense.api.routes_health import router as health_router
from gridsense.api.routes_predict import router as predict_router
from gridsense.api.security import require_api_key, verify_auth_configuration
from gridsense.config import get_settings
from gridsense.logconfig import configure_logging

logger = logging.getLogger("gridsense.api")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


def create_app() -> FastAPI:
    """Application factory — keeps the app importable and testable."""
    settings = get_settings()

    configure_logging(settings.log_level, json_output=settings.environment.lower() != "local")
    verify_auth_configuration(settings)

    app = FastAPI(
        title="GridSense AI",
        version=__version__,
        summary="RAG over technical docs (DocRAG) + battery-degradation MLOps (DegradeML).",
    )
    install_middleware(app)
    app.include_router(health_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe: is the process up and serving?

        Deliberately touches nothing external. A liveness probe that fails when Postgres is
        briefly unreachable gets the container *killed* for a fault it cannot fix by
        restarting — which turns a dependency blip into an outage. Readiness is a separate
        question with a separate endpoint: ``/ready``.
        """
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=__version__,
            environment=settings.environment,
        )

    # Everything below is authenticated and rate-limited. `/health` and `/ready` are not:
    # a probe cannot present a key, and a readiness check that 401s takes the pod out of
    # service for a reason unrelated to whether it can serve.
    guarded = [
        Depends(require_api_key),
        Depends(rate_limit("default", lambda s: s.rate_limit_per_minute)),
    ]
    app.include_router(chat_router, dependencies=guarded)
    app.include_router(ask_router, dependencies=guarded)
    app.include_router(predict_router, dependencies=guarded)
    # The agent gets its own, tighter bucket on top of the shared one: one request there is
    # a multi-turn loop plus a sub-agent, not a single call.
    app.include_router(
        agent_router,
        dependencies=[
            Depends(require_api_key),
            Depends(rate_limit("agent", lambda s: s.agent_rate_limit_per_minute)),
        ],
    )

    logger.info(
        "application started",
        extra={
            "environment": settings.environment,
            "chat_provider": settings.chat_provider.value,
            "embedding_provider": settings.embedding_provider.value,
            "auth": "disabled" if settings.api_auth_disabled else "api-key",
        },
    )
    return app


app = create_app()
