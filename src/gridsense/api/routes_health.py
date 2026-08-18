"""`/ready` — readiness, as distinct from liveness.

`/health` answers "is this process alive?"; this answers "can it actually serve?". Both
probes in ``k8s/deployment.yaml`` used to point at `/health`, which meant a pod whose
Postgres was unreachable reported itself ready and got traffic routed to it — every request
failing, the deployment reporting healthy.

The split matters in both directions. Readiness failing takes the pod out of the load
balancer until it recovers, which is right. Liveness failing *kills the container*, which
would be exactly wrong for a database blip a restart cannot fix.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from gridsense.config import get_settings

logger = logging.getLogger("gridsense.api.health")

router = APIRouter(tags=["meta"])


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, str] = Field(description="One entry per dependency: 'ok' or the error.")


def _check_database() -> str:
    """Round-trip the database. Cheap, and it fails the way a real query would."""
    settings = get_settings()
    # A short, dedicated pool: a readiness probe that blocks on the request pool's last
    # connection would report not-ready precisely when the service is busiest.
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    finally:
        engine.dispose()


@router.get("/ready", response_model=ReadyResponse)
def ready(response: Response) -> ReadyResponse:
    """Report whether the service's dependencies are reachable.

    The provider is deliberately *not* probed. A readiness check that calls the model costs
    money on a timer and rate-limits itself out of existence, and a provider outage is
    something to surface per request — as ``provider_error`` — not by removing every pod
    from service at once.
    """
    checks: dict[str, str] = {}
    try:
        checks["database"] = _check_database()
    except Exception as exc:  # noqa: BLE001 - any failure to reach it means not ready
        checks["database"] = f"{type(exc).__name__}: {exc}"
        logger.warning("readiness check failed", extra={"check": "database"})

    ok = all(value == "ok" for value in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(ready=ok, checks=checks)
