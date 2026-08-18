"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """A TestClient bound to a fresh app instance.

    Both imports sit inside the fixture on purpose. At module level they pull the whole
    application graph — including ``routes_predict`` -> ``degrade.serve`` -> pandas — into
    *every* collection, so a CI job that only scores the golden dataset or the trajectory
    metrics would need the full ML stack installed just to collect a single test. Deferring
    them means the ML dependencies are required by the tests that actually exercise the app,
    and by nothing else.
    """
    from fastapi.testclient import TestClient

    from gridsense.api.main import create_app

    return TestClient(create_app())
