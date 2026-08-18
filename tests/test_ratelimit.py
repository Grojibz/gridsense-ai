"""The token bucket, and the rate-limit dependency built on it.

The bucket is worth testing directly because its interesting behaviour is temporal, and
temporal behaviour asserted through HTTP is asserted through a sleep. `TokenBucket.take`
takes the clock as an argument precisely so the refill can be tested by moving the clock
rather than by waiting for it.

The property that motivates a bucket over a fixed window is the burst at a window boundary:
a fixed counter lets a caller spend a full allowance at the end of one window and another
at the start of the next — twice the intended rate. The refill test below is what pins that.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from gridsense.api import ratelimit
from gridsense.api.ratelimit import RateLimiter, TokenBucket, rate_limit
from gridsense.config import Settings

# --- the bucket --------------------------------------------------------------


def test_a_full_bucket_permits_a_burst_then_stops():
    bucket = TokenBucket(capacity=3, now=0.0)
    assert [bucket.take(0.0)[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = bucket.take(0.0)
    assert allowed is False
    assert retry_after == pytest.approx(20.0)  # 3/min -> one token every 20s


def test_tokens_refill_continuously_rather_than_at_a_window_edge():
    """A fixed window would allow 2x the rate across a boundary; this must not."""
    bucket = TokenBucket(capacity=60, now=0.0)
    for _ in range(60):
        bucket.take(0.0)
    assert bucket.take(0.0)[0] is False

    assert bucket.take(0.5)[0] is False  # half a second is half a token
    assert bucket.take(1.0)[0] is True  # one second is one token


def test_refill_never_exceeds_capacity():
    """An idle caller banks one minute of allowance, not an hour of it."""
    bucket = TokenBucket(capacity=5, now=0.0)
    bucket.take(0.0)
    bucket.take(3600.0)
    assert bucket.tokens == pytest.approx(4.0)


# --- the registry ------------------------------------------------------------


def test_scopes_are_independent_so_a_cheap_route_cannot_exhaust_an_expensive_one():
    limiter = RateLimiter()
    assert limiter.check("agent", "caller", per_minute=1)[0] is True
    assert limiter.check("agent", "caller", per_minute=1)[0] is False
    assert limiter.check("default", "caller", per_minute=1)[0] is True


def test_callers_are_independent():
    limiter = RateLimiter()
    assert limiter.check("default", "a", per_minute=1)[0] is True
    assert limiter.check("default", "a", per_minute=1)[0] is False
    assert limiter.check("default", "b", per_minute=1)[0] is True


def test_the_bucket_map_is_bounded():
    """Unbounded, tracking callers is itself the memory-exhaustion vector."""
    limiter = RateLimiter(max_callers=10)
    for i in range(50):
        limiter.check("default", f"caller-{i}", per_minute=5)
    assert len(limiter._buckets) == 10


# --- the dependency ----------------------------------------------------------


def _client(monkeypatch: pytest.MonkeyPatch, **settings_overrides) -> TestClient:
    settings = Settings(_env_file=None, **settings_overrides)
    monkeypatch.setattr(ratelimit, "get_settings", lambda: settings)
    ratelimit.limiter.reset()

    app = FastAPI()

    @app.get(
        "/limited",
        dependencies=[Depends(rate_limit("test", lambda s: s.rate_limit_per_minute))],
    )
    def limited() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_exceeding_the_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch, rate_limit_per_minute=2)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200

    resp = client.get("/limited")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_zero_disables_the_limit(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch, rate_limit_per_minute=0)
    assert all(client.get("/limited").status_code == 200 for _ in range(50))


def test_the_agent_is_budgeted_more_tightly_than_the_cheap_routes():
    """One /agent request is a multi-turn loop plus a sub-agent, not one call."""
    settings = Settings(_env_file=None)
    assert settings.agent_rate_limit_per_minute < settings.rate_limit_per_minute


# --- caller identity ---------------------------------------------------------


def test_a_forged_forwarded_header_cannot_mint_a_fresh_identity(monkeypatch):
    """Trusting X-Forwarded-For would make the limiter bypassable by any client."""
    client = _client(monkeypatch, rate_limit_per_minute=1)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited", headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 429


def test_only_a_key_prefix_is_retained_as_an_identity(monkeypatch):
    """The bucket key ends up in warning logs; a whole credential must not."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"x-api-key", b"supersecretkeyvalue")],
        "client": ("1.2.3.4", 1234),
    }
    identity = ratelimit.caller_identity(Request(scope))
    assert identity == "key:supersec"
    assert "keyvalue" not in identity
