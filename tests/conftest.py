"""Unit tests stay on the cue path. Groq and Gemini calls are opt-in and use fake clients."""

import pytest


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)
