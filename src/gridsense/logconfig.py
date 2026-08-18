"""Structured JSON logging, and the request-scoped context that makes it useful.

Before this module the service logged nothing at all: ``LOG_LEVEL`` was configured and read
by no one, so a guardrail refusal, a provider outage or a 500 left no trace anywhere except
the response the caller received. Langfuse traces the *model* calls; it says nothing about
the service around them.

Two decisions worth naming.

**JSON, not text.** The lines are meant to be queried by a log backend, and a message that
has to be regex-parsed to find out which request it belonged to is not queryable. Extra
fields ride in the record's ``extra`` dict and land as top-level JSON keys.

**A ContextVar, not a parameter.** The request id has to reach a log call five frames deep
inside the RAG chain, which has no idea it is serving HTTP. Threading it through every
signature would couple the whole codebase to the API layer; a ContextVar is
concurrency-correct under both threads and asyncio and costs the call sites nothing.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

#: The id of the HTTP request being served, when there is one. Empty outside a request —
#: CLI entry points (ingest, eval, the MCP server) log through the same handler.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

#: Attributes every ``LogRecord`` carries. Anything outside this set was put there by an
#: ``extra=`` at the call site, which is exactly what we want to serialise.
_STANDARD_FIELDS = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)

#: Never log these, whatever a call site passes. A secret that reaches a log line has
#: leaked to every system that ingests logs, and redacting it afterwards does not un-leak
#: it. Substring match, so `anthropic_api_key` and `x-api-key` are both caught.
_REDACTED_SUBSTRINGS = ("api_key", "apikey", "secret", "password", "token", "authorization")

REDACTED = "***"


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    # `*_tokens` are usage counts, not credentials, and they are among the most useful
    # numbers in the log. Exclude them before the substring check swallows them.
    if lowered.endswith("_tokens") or lowered == "tokens":
        return False
    return any(marker in lowered for marker in _REDACTED_SUBSTRINGS)


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in _STANDARD_FIELDS or key.startswith("_"):
                continue
            payload[key] = REDACTED if _is_sensitive(key) else value

        if record.exc_info:
            # The traceback belongs in the log and nowhere else — never in a response body.
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Install the root handler. Idempotent — safe to call from several entry points.

    ``json_output=False`` gives plain text for a human reading ``docker compose logs``.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(handler)

    # Uvicorn installs its own access log with a different format and no request id. Ours
    # carries strictly more (duration, status, route, id), so silence the duplicate.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False

    # httpx logs every request it makes at INFO, and every provider SDK here is built on it.
    # At INFO that is one line per model call, per embedding call, per retry — enough noise
    # to bury the lines that carry a decision. Their warnings and errors still come through.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
