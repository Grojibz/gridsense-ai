"""Structured logging: the redaction rule, and the request id that makes lines correlatable.

The redaction test is the one with teeth. Everything else here is formatting; that one is
the difference between a log backend holding an API key and not. It is asserted on the
*formatter* rather than on any particular call site, because the guarantee has to hold for
call sites nobody has written yet.
"""

from __future__ import annotations

import json
import logging

import pytest

from gridsense.logconfig import JsonFormatter, request_id_var


def _record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, msg, None, None)
    record.__dict__.update(extra)
    return record


def _format(**extra) -> dict:
    return json.loads(JsonFormatter().format(_record(**extra)))


def test_a_line_is_one_json_object_with_the_standard_fields():
    payload = _format()
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello"
    assert "ts" in payload


def test_extras_become_top_level_keys_so_they_are_queryable():
    assert _format(endpoint="/ask", duration_ms=12.5)["endpoint"] == "/ask"


@pytest.mark.parametrize(
    "field",
    ["api_key", "anthropic_api_key", "AUTHORIZATION", "db_password", "langfuse_secret_key"],
)
def test_credentials_are_redacted_whatever_the_call_site_passes(field):
    """A secret that reaches a log has leaked to everything that ingests logs."""
    payload = _format(**{field: "sk-ant-real-value"})
    assert payload[field] == "***"
    assert "sk-ant-real-value" not in json.dumps(payload)


def test_usage_counts_are_not_mistaken_for_credentials():
    """`*_tokens` are the most useful numbers in the log; the word must not swallow them."""
    payload = _format(input_tokens=120, output_tokens=44, billable_tokens=164)
    assert payload["input_tokens"] == 120
    assert payload["billable_tokens"] == 164


def test_the_request_id_rides_along_without_being_passed():
    """It has to reach log calls deep in the chain, which know nothing about HTTP."""
    token = request_id_var.set("abc123")
    try:
        assert _format()["request_id"] == "abc123"
    finally:
        request_id_var.reset(token)


def test_no_request_id_outside_a_request():
    """Ingest, eval and the MCP server log through the same handler."""
    assert "request_id" not in _format()


def test_a_traceback_goes_to_the_log_and_carries_the_type():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_transport_loggers_are_quieted_so_they_cannot_bury_decisions():
    """Every provider SDK here rides on httpx, which logs each call at INFO."""
    from gridsense.logconfig import configure_logging

    configure_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
