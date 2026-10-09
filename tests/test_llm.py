"""Groq classifier: JSON in, redacted text only, no call when the key is unset."""

import json
from types import SimpleNamespace

import pytest

from support_reply.llm import (
    DEFAULT_GROQ_MODEL,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_REASONING_EFFORT,
    GroqClassifier,
    _apply_env_lines,
    classify_or_none,
    default_classifier,
    instructions,
    match_allowed,
    parse_json_object,
)


class _Completions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, content: str) -> None:
        self.completions = _Completions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_default_classifier_is_none_without_a_key():
    assert default_classifier() is None


def test_default_classifier_uses_groq_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    classifier = default_classifier()

    assert isinstance(classifier, GroqClassifier)
    assert classifier.model == DEFAULT_GROQ_MODEL


def test_groq_model_name_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    assert default_classifier().model == "openai/gpt-oss-20b"


def test_groq_sends_the_redacted_complaint_and_asks_for_json():
    client = _Client('{"theme_id": "fees_brokerage"}')
    classifier = GroqClassifier(api_key="test-key", model=DEFAULT_GROQ_MODEL, client=client)
    complaint = "Why an exit load on HDFC Large Cap? [PAN]"

    payload = classifier.classify(
        task="pattern",
        redacted_text=complaint,
        allowed={"theme_id": ("fees_brokerage", "other")},
        labels={"fees_brokerage": "Fees & brokerage"},
    )

    assert payload == {"theme_id": "fees_brokerage"}
    sent = client.completions.kwargs
    assert sent["model"] == DEFAULT_GROQ_MODEL
    assert sent["temperature"] == 0
    assert sent["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS
    assert sent["reasoning_effort"] == GROQ_REASONING_EFFORT
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][1] == {"role": "user", "content": complaint}
    system = sent["messages"][0]["content"]
    assert "json" in system.lower()
    assert "fees_brokerage" in system
    assert "Fees & brokerage" in system
    assert complaint not in system
    assert "test-key" not in system
    assert "test-key" not in sent["messages"][1]["content"]


def test_groq_omits_reasoning_effort_for_a_model_that_does_not_support_it():
    client = _Client('{"theme_id": "other"}')
    classifier = GroqClassifier(
        api_key="test-key", model="llama-3.3-70b-versatile", client=client
    )

    classifier.classify(
        task="pattern",
        redacted_text="the app is slow",
        allowed={"theme_id": ("reliability", "other")},
    )

    sent = client.completions.kwargs
    assert sent["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS
    assert "reasoning_effort" not in sent


def test_env_lines_fill_missing_keys_and_skip_blanks():
    env = {"GROQ_MODEL": "already-set"}

    _apply_env_lines(
        [
            "# comment",
            "GROQ_API_KEY=",
            "GEMINI_API_KEY=from-file",
            'export GEMINI_MODEL="gemini-2.5-pro"',
            "GROQ_MODEL=openai/gpt-oss-20b",
        ],
        env,
    )

    assert "GROQ_API_KEY" not in env
    assert env["GEMINI_API_KEY"] == "from-file"
    assert env["GEMINI_MODEL"] == "gemini-2.5-pro"
    assert env["GROQ_MODEL"] == "already-set"


def test_groq_rejects_a_message_that_is_not_json():
    client = _Client("not json")
    classifier = GroqClassifier(api_key="test-key", model=DEFAULT_GROQ_MODEL, client=client)

    with pytest.raises(ValueError, match="JSON"):
        classifier.classify(
            task="compliance",
            redacted_text="Should I redeem?",
            allowed={"flags": ("advice", "performance_claim")},
        )


def test_parse_json_object_strips_a_fence_and_rejects_a_list():
    assert parse_json_object('```json\n{"flags": []}\n```') == {"flags": []}
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("[1, 2]")


def test_classify_or_none_is_none_when_the_call_fails():
    class Boom:
        def classify(self, **kwargs):
            raise TimeoutError("slow")

    assert (
        classify_or_none(
            task="facts",
            redacted_text="exit load",
            allowed={"scheme_id": ("large_cap",)},
            classifier=Boom(),
        )
        is None
    )


def test_instructions_name_the_allowed_ids_and_not_the_complaint():
    complaint = "CASE-9f3a exit load"
    text = instructions(
        "facts",
        {"scheme_id": ("large_cap",), "section_title": ("Exit load",)},
        {"large_cap": "HDFC Large Cap Fund Direct Growth"},
    )

    assert "large_cap" in text
    assert "Exit load" in text
    assert "HDFC Large Cap Fund Direct Growth" in text
    assert complaint not in text
    with pytest.raises(ValueError, match="unknown classifier task"):
        instructions("draft", {})


def test_match_allowed_accepts_case_and_rejects_unknown_ids():
    allowed = ("large_cap", "Exit load")

    assert match_allowed("Large_Cap", ("large_cap",)) == "large_cap"
    assert match_allowed(" null ", allowed) is None
    assert match_allowed("nippon", ("large_cap",)) is None
    assert match_allowed(None, allowed) is None


def test_groq_response_body_is_parsed_from_json_text():
    body = json.dumps({"flags": ["advice"]})
    client = _Client(body)
    classifier = GroqClassifier(api_key="test-key", model=DEFAULT_GROQ_MODEL, client=client)

    assert classifier.classify(
        task="compliance",
        redacted_text="Should I redeem?",
        allowed={"flags": ("advice", "performance_claim")},
    ) == {"flags": ["advice"]}
