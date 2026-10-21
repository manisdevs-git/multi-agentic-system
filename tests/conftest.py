"""Unit tests stay on the cue path. Live DeepInfra calls are opt-in and use fake clients."""

import pytest


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    for name in (
        "DEEPINFRA_API_KEY",
        "DEEPINFRA_MODEL",
        "DEEPINFRA_REVIEW_MODEL",
        "MCP_SERVER_URL",
        "MCP_AUTH_TOKEN",
        "MAIL_TO",
    ):
        monkeypatch.delenv(name, raising=False)
