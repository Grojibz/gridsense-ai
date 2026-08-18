"""A per-caller token-bucket rate limiter.

`/agent` runs up to `MAX_ITERATIONS` model turns plus a sub-agent per request. Without a
limit, one client in a loop is an unbounded bill, and the endpoint's cost is set by whoever
calls it rather than by whoever pays for it. So the expensive routes carry a tighter budget
than the cheap ones — a single number applied to `/health` and `/agent` alike would have to
be loose enough for the first, which makes it useless for the second.

**A token bucket, not a fixed window.** A fixed window lets a caller spend the whole
allowance in the last second of one window and again in the first second of the next —
double the intended rate, at the worst possible moment. A bucket refilling continuously
smooths that out while still permitting a short burst, which is what a human clicking twice
actually needs.

**⚠️ The state is per process.** Two uvicorn workers, or the two replicas in
`k8s/deployment.yaml`, each keep their own buckets, so the effective limit is N times the
configured one. That is a real limitation, stated rather than hidden: this is a guardrail
against a runaway client and an accidental loop, not a defence against a distributed
attacker. A limit that must hold across replicas needs shared state (Redis) or an ingress
that enforces it — and either belongs in front of the app, not inside it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, status

from gridsense.config import Settings, get_settings

logger = logging.getLogger("gridsense.api.ratelimit")

#: Cap on tracked callers. Unbounded, the bucket map is itself a memory-exhaustion vector:
#: an attacker rotating source addresses would grow it without limit. Least-recently-used
#: entries are dropped, which at worst forgives an idle caller its history.
MAX_TRACKED_CALLERS = 10_000


class TokenBucket:
    """Continuous-refill bucket. ``capacity`` tokens, refilled to full over one minute."""

    __slots__ = ("capacity", "tokens", "updated")

    def __init__(self, capacity: float, now: float) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.updated = now

    def take(self, now: float) -> tuple[bool, float]:
        """Try to spend one token. Returns ``(allowed, seconds_until_next_token)``."""
        elapsed = max(0.0, now - self.updated)
        self.updated = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.capacity / 60.0)

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0

        deficit = 1.0 - self.tokens
        return False, deficit * 60.0 / self.capacity


class RateLimiter:
    """Thread-safe bucket registry. One instance per app."""

    def __init__(self, max_callers: int = MAX_TRACKED_CALLERS) -> None:
        self._buckets: OrderedDict[tuple[str, str], TokenBucket] = OrderedDict()
        self._lock = threading.Lock()
        self._max_callers = max_callers

    def check(self, scope: str, caller: str, per_minute: int) -> tuple[bool, float]:
        """Spend one token for ``caller`` in ``scope``."""
        now = time.monotonic()
        key = (scope, caller)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(float(per_minute), now)
                self._buckets[key] = bucket
            elif bucket.capacity != per_minute:
                # The setting changed under a live bucket; resize rather than keep applying
                # a limit nobody configured any more.
                bucket.capacity = float(per_minute)
                bucket.tokens = min(bucket.tokens, bucket.capacity)
            self._buckets.move_to_end(key)

            while len(self._buckets) > self._max_callers:
                self._buckets.popitem(last=False)

            return bucket.take(now)

    def reset(self) -> None:
        """Drop all buckets. For tests."""
        with self._lock:
            self._buckets.clear()


#: Process-wide registry. Module-level for the same reason the settings singleton is:
#: a limiter rebuilt per request would grant everyone a full bucket every time.
limiter = RateLimiter()


def caller_identity(request: Request) -> str:
    """Identify the caller: the API key when there is one, otherwise the peer address.

    The key is preferred because it is the thing that was actually granted a budget — one
    key behind a NAT is one caller, and one key from three machines is still one caller.
    Only a prefix is retained, so no log line or bucket key can carry a whole credential.

    The address fallback is weak on purpose-honesty grounds: behind a proxy, every request
    appears to come from the proxy. `X-Forwarded-For` is *not* consulted, because a client
    can set it freely and trusting it would let anyone mint a fresh identity per request —
    a limiter that is trivially bypassed is worse than a visibly coarse one. If this runs
    behind a proxy you control, run uvicorn with `--proxy-headers --forwarded-allow-ips`
    so the address is rewritten by the server before it ever reaches here.
    """
    key = request.headers.get("x-api-key")
    if key:
        return f"key:{key[:8]}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


def rate_limit(
    scope: str, limit_of: Callable[[Settings], int]
) -> Callable[[Request], Coroutine[Any, Any, None]]:
    """Build a dependency limiting ``scope`` to the configured rate. ``0`` disables it."""

    async def dependency(request: Request) -> None:
        settings = get_settings()
        per_minute = limit_of(settings)
        if per_minute <= 0:
            return

        caller = caller_identity(request)
        allowed, retry_after = limiter.check(scope, caller, per_minute)
        if allowed:
            return

        logger.warning(
            "rate limit exceeded",
            extra={"scope": scope, "caller": caller, "limit_per_minute": per_minute},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {scope}: {per_minute} requests/minute.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )

    return dependency
