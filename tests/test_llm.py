"""DeepInfra classifier: JSON in, redacted text only, no call when the key is unset."""

import json
from types import SimpleNamespace

import pytest

from support_reply.llm import (
    DEFAULT_DEEPINFRA_MODEL,
    DeepInfraClassifier,
    _apply_env_lines,
    classify_or_none,
    default_classifier,
    default_review_classifier,
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
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _Client:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_Completions(content))


def test_default_classifier_is_none_without_a_key():
    assert default_classifier() is None
    assert default_review_classifier() is None


def test_a_deepinfra_key_configures_agents_and_the_reviewer(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.delenv("DEEPINFRA_MODEL", raising=False)
    monkeypatch.delenv("DEEPINFRA_REVIEW_MODEL", raising=False)

    agents = default_classifier()
    review = default_review_classifier()

    assert isinstance(agents, DeepInfraClassifier)
    assert isinstance(review, DeepInfraClassifier)
    assert agents.model == DEFAULT_DEEPINFRA_MODEL == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert review.model == DEFAULT_DEEPINFRA_MODEL


def test_deepinfra_model_name_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setenv("DEEPINFRA_MODEL", "zai-org/GLM-5.3-Flash")
    monkeypatch.delenv("DEEPINFRA_REVIEW_MODEL", raising=False)

    assert default_classifier().model == "zai-org/GLM-5.3-Flash"
    assert default_review_classifier().model == "zai-org/GLM-5.3-Flash"

    monkeypatch.setenv("DEEPINFRA_REVIEW_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")
    assert default_classifier().model == "zai-org/GLM-5.3-Flash"
    assert default_review_classifier().model == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_deepinfra_sends_the_redacted_complaint_and_asks_for_json():
    client = _Client('{"theme_id": "fees_brokerage"}')
    classifier = DeepInfraClassifier(
        api_key="test-key", model=DEFAULT_DEEPINFRA_MODEL, client=client
    )
    complaint = "Why an exit load on HDFC Large Cap? [PAN]"

    payload = classifier.classify(
        task="pattern",
        redacted_text=complaint,
        allowed={"theme_id": ("fees_brokerage", "other")},
        labels={"fees_brokerage": "Fees & brokerage"},
    )

    assert payload == {"theme_id": "fees_brokerage"}
    sent = client.chat.completions.kwargs
    assert sent["model"] == DEFAULT_DEEPINFRA_MODEL
    assert sent["temperature"] == 0
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][1] == {"role": "user", "content": complaint}
    system = sent["messages"][0]["content"]
    assert sent["messages"][0]["role"] == "system"
    assert "json" in system.lower()
    assert "fees_brokerage" in system
    assert "Fees & brokerage" in system
    assert complaint not in system
    assert "test-key" not in system


def test_env_lines_fill_missing_keys_and_skip_blanks():
    env = {"DEEPINFRA_MODEL": "already-set"}

    _apply_env_lines(
        [
            "# comment",
            "DEEPINFRA_API_KEY=",
            "DEEPINFRA_API_KEY=from-file",
            'export DEEPINFRA_REVIEW_MODEL="zai-org/GLM-5.3-Flash"',
            "DEEPINFRA_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731",
        ],
        env,
    )

    assert env["DEEPINFRA_API_KEY"] == "from-file"
    assert env["DEEPINFRA_REVIEW_MODEL"] == "zai-org/GLM-5.3-Flash"
    assert env["DEEPINFRA_MODEL"] == "already-set"


def test_deepinfra_rejects_a_message_that_is_not_json():
    client = _Client("not json")
    classifier = DeepInfraClassifier(
        api_key="test-key", model=DEFAULT_DEEPINFRA_MODEL, client=client
    )

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


def test_deepinfra_response_body_is_parsed_from_json_text():
    body = json.dumps({"flags": ["advice"]})
    client = _Client(body)
    classifier = DeepInfraClassifier(
        api_key="test-key", model=DEFAULT_DEEPINFRA_MODEL, client=client
    )

    assert classifier.classify(
        task="compliance",
        redacted_text="Should I redeem?",
        allowed={"flags": ("advice", "performance_claim")},
    ) == {"flags": ["advice"]}
