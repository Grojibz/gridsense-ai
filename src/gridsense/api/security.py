"""API-key authentication, and the startup check that stops it being forgotten.

The endpoints spend money on someone else's behalf: `/ask` runs a model call per request
and `/agent` runs a bounded loop of them plus a sub-agent. Leaving them open is not a
missing nicety, it is an unmetered bill payable by whoever holds the provider key.

**Fail closed, and fail at startup.** A configuration flag that defaults to "no auth"
reproduces exactly the gap it was meant to close, because the default is what ships. So
outside a local run the app refuses to *start* without either a key or an explicit
`API_AUTH_DISABLED=true`. Refusing at startup rather than at the first request means the
mistake surfaces in a deploy, in front of whoever is deploying — not weeks later in a bill.

`API_AUTH_DISABLED` exists because a genuinely correct deployment can put auth in front of
this service (an API gateway, a service mesh, an ingress with mTLS). What it must not be is
the silent default, so it is opt-out, explicit, and logged loudly at every startup.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, status

from gridsense.config import Settings, get_settings

logger = logging.getLogger("gridsense.api.security")

#: Environments where an unauthenticated app is a reasonable default: a developer's machine
#: and the test suite. Deliberately a narrow allowlist rather than a "production" denylist —
#: an environment named `staging`, `dev-eu` or `k8s` is reachable by someone, and a denylist
#: silently exempts every name nobody thought to add.
UNAUTHENTICATED_ENVIRONMENTS = frozenset({"local", "test"})

API_KEY_HEADER = "X-API-Key"


def parse_api_keys(raw: str | None) -> list[str]:
    """Split the configured key list. Comma-separated so a key can be rotated.

    Rotation needs two valid keys at once: publish the new one, let clients move, retire the
    old one. A single-key setting forces a flag-day cutover, which in practice means the key
    never gets rotated at all.
    """
    if not raw:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def auth_enabled(settings: Settings) -> bool:
    """Whether requests must present a key."""
    return not settings.api_auth_disabled and bool(parse_api_keys(settings.api_keys))


def verify_auth_configuration(settings: Settings) -> None:
    """Refuse to start a deployed app that has no authentication. Called from the factory."""
    if settings.api_auth_disabled:
        logger.warning(
            "API authentication is DISABLED by API_AUTH_DISABLED. Every endpoint is open, "
            "including /agent, which runs a multi-turn model loop per request. This is only "
            "safe behind a gateway that authenticates on this service's behalf.",
            extra={"environment": settings.environment},
        )
        return

    if parse_api_keys(settings.api_keys):
        return

    if settings.environment.lower() in UNAUTHENTICATED_ENVIRONMENTS:
        logger.warning(
            "No API_KEYS set — endpoints are open. Fine for %s; set API_KEYS before "
            "deploying anywhere reachable.",
            settings.environment,
        )
        return

    raise RuntimeError(
        f"ENVIRONMENT={settings.environment!r} but no API_KEYS is set. The API would start "
        "with /ask and /agent open to anyone, and both spend provider credits per request. "
        "Set API_KEYS (comma-separated allows rotation), or set API_AUTH_DISABLED=true if "
        "authentication is genuinely handled in front of this service."
    )


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the ``X-API-Key`` header.

    Re-reads settings per request rather than closing over them so that ``get_settings``
    stays the single source of truth (it is ``lru_cache``d, so this is a dict lookup).
    """
    settings = get_settings()
    if not auth_enabled(settings):
        return

    valid = parse_api_keys(settings.api_keys)
    # `compare_digest` on every candidate, and no early exit: comparing with `in` or
    # returning on first match leaks key material through response timing.
    presented = x_api_key or ""
    matched = False
    for candidate in valid:
        if secrets.compare_digest(presented, candidate):
            matched = True

    if not matched:
        # Deliberately identical for a missing and a wrong key: distinguishing them tells a
        # prober which half of the problem to work on.
        logger.warning("rejected unauthenticated request", extra={"key_present": bool(x_api_key)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"A valid {API_KEY_HEADER} header is required.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
