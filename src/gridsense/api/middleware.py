"""Request-scoped context, the access log, and the last line of defence for exceptions.

Two responsibilities that belong together because both need the same thing: an id that
identifies one request across every line it produces.

The exception handler is the part that changes behaviour visibly. Before it, any unhandled
exception left FastAPI to return a 500 whose body carried the traceback — file paths,
local variables, sometimes a connection string with credentials in it. Now the traceback
goes to the log, and the caller gets a request id to quote. That is the whole trade:
the operator keeps the detail, the caller keeps the correlation, and the internet gets
neither.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from gridsense.logconfig import request_id_var

logger = logging.getLogger("gridsense.api")

#: Echoed on every response, and accepted on the way in so an id assigned by an upstream
#: proxy or gateway survives instead of being replaced.
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the outcome, and let nothing escape as a raw traceback."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ) -> JSONResponse:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        def endpoint_of(req: Request) -> str:
            # The route *pattern*, not the raw path, so `/items/{id}` groups instead of
            # minting a distinct log dimension per id. Only readable after routing has run,
            # which is why this is a closure called below rather than a value read above.
            # Falls back to the path for a 404, which never matched a route.
            return getattr(req.scope.get("route"), "path", None) or req.url.path

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.exception(
                "unhandled exception",
                extra={
                    "method": request.method,
                    "endpoint": endpoint_of(request),
                    "status_code": 500,
                    "duration_ms": elapsed_ms,
                },
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error.",
                    # The one piece of internal state worth handing out: it is the only way
                    # a caller can report a fault the operator can then actually find.
                    "request_id": request_id,
                },
            )
        else:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "endpoint": endpoint_of(request),
                    "status_code": response.status_code,
                    "duration_ms": elapsed_ms,
                },
            )
        finally:
            request_id_var.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def install_middleware(app: FastAPI) -> None:
    """Attach the request context middleware to ``app``."""
    app.add_middleware(RequestContextMiddleware)
