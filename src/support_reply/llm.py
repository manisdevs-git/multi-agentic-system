"""Classifiers for the Facts, Pattern, Compliance, and reviewer nodes.

Groq is used for the three agent nodes when `GROQ_API_KEY` is set. `GROQ_MODEL`
overrides the model name. Each Groq call asks for `reasoning_effort=low` on the
gpt-oss models and caps the completion at 128 tokens, so the three calls started
together stay inside the free-tier 8,000 tokens per minute. Gemini is used for
the reviewer's advice and performance scan when `GEMINI_API_KEY` is set;
`GEMINI_MODEL` overrides it. Keys are read from the environment, then from a
repo `.env` (see `.env.example`) for any name that is still unset.
With no key, or when a call fails, `classify_or_none` returns None and the
node uses its cue table or code checks.

The model sees redacted text and the allowed ids. It returns JSON. It does not
write a fact, a count, a citation, or a reply. The reviewer model only names a
check and copies the offending span out of the draft.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

TASK_FACTS = "facts"
TASK_PATTERN = "pattern"
TASK_COMPLIANCE = "compliance"
TASK_REVIEW = "review"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_TIMEOUT_SECONDS = 20.0
# Free tier for gpt-oss is 8_000 tokens per minute, prompt plus completion,
# and reasoning tokens count. Facts, Pattern, and Compliance each make one
# call, and the graph starts the three together. The JSON answer is a few
# dozen tokens, so the completion cap stays at the short-answer floor and
# reasoning effort stays low.
GROQ_MAX_COMPLETION_TOKENS = 128
GROQ_REASONING_EFFORT = "low"
_LOW_REASONING_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    }
)

_COMPLAINT_PREAMBLE = (
    "You classify one redacted support complaint. "
    "Reply with a JSON object and nothing else. "
    "Do not write a fact, a count, a citation, or a reply.\n"
)
_REVIEW_PREAMBLE = (
    "You review one outgoing support reply from a mutual-fund platform before a human "
    "sees it. Reply with a JSON object and nothing else. "
    "Do not rewrite the reply, add a fact, or add a citation.\n"
)
_PREAMBLES: dict[str, str] = {
    TASK_FACTS: _COMPLAINT_PREAMBLE,
    TASK_PATTERN: _COMPLAINT_PREAMBLE,
    TASK_COMPLIANCE: _COMPLAINT_PREAMBLE,
    TASK_REVIEW: _REVIEW_PREAMBLE,
}

_TASK_INSTRUCTIONS: dict[str, str] = {
    TASK_FACTS: (
        'Return a JSON object with keys "scheme_id" and "section_title". '
        "Each value is one allowed id, or null. Return the id, not the display name. "
        "Use null for scheme_id when the complaint names none of the schemes, more than one, "
        "or a different AMC. "
        "Use null for section_title when none of the sections answer the question, "
        "including a request for returns, CAGR, performance, or NAV history."
    ),
    TASK_PATTERN: (
        'Return a JSON object with key "theme_id" set to one allowed id. '
        "Return the id, not the display name."
    ),
    TASK_COMPLIANCE: (
        'Return a JSON object with key "flags", an array of zero or more of '
        '"advice" and "performance_claim". '
        "advice: the user asks whether to redeem, buy, sell, hold, stay invested, "
        "whether a scheme is good or worth it, or what they should do with a holding. "
        "performance_claim: the user asks for or states a return, CAGR, or performance figure. "
        "A complaint about an exit load, a fee, a tax, an order, or the app is neither. "
        'Do not include "pii".'
    ),
    TASK_REVIEW: (
        'Return a JSON object with key "failures": an array of objects, each with '
        '"check" set to one allowed id and "span" set to the exact text copied from the '
        "reply that trips that check. Return an empty array when the reply is clean. "
        "advice: the reply tells or nudges the reader to buy, sell, redeem, hold, switch, "
        "exit, stay invested, or otherwise recommends what to do with a holding, "
        'for example "you may want to" or "consider staying invested". '
        "performance: the reply states or implies a return, CAGR, yield, or how a scheme "
        "has performed or will perform. "
        "Quoting a scheme rule such as an exit load, expense ratio, stamp duty, tax rate, "
        "NAV, AUM, benchmark, or minimum investment copied from the scheme page is a fact, "
        "not advice and not performance. A reply that says the team cannot give advice "
        "is not advice."
    ),
}

_NULL_WORDS = frozenset({"", "null", "none"})
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class Classifier(Protocol):
    """One classification call. Implementations must not retain raw PII."""

    def classify(
        self,
        *,
        task: str,
        redacted_text: str,
        allowed: Mapping[str, Sequence[str]],
        labels: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """Return the JSON object for this task."""


class GroqClassifier:
    """Groq chat call that must answer with one JSON object."""

    def __init__(self, *, api_key: str, model: str, client: Any = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GROQ_API_KEY is empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("GROQ_MODEL is empty")
        self.model = model.strip()
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key.strip(), timeout=_TIMEOUT_SECONDS)
        self._client = client

    def classify(
        self,
        *,
        task: str,
        redacted_text: str,
        allowed: Mapping[str, Sequence[str]],
        labels: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(redacted_text, str):
            raise TypeError("classifier expects redacted text")
        system = instructions(task, allowed, labels)
        logger.info("groq classify task=%s model=%s", task, self.model)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": redacted_text},
            ],
            **_groq_limits(self.model),
        )
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("model response has no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model response is empty")
        return parse_json_object(content)


class GeminiClassifier:
    """Gemini call that must answer with one JSON object. Used by the reviewer."""

    def __init__(self, *, api_key: str, model: str, client: Any = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("GEMINI_MODEL is empty")
        self.model = model.strip()
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=api_key.strip(),
                http_options={"timeout": int(_TIMEOUT_SECONDS * 1000)},
            )
        self._client = client

    def classify(
        self,
        *,
        task: str,
        redacted_text: str,
        allowed: Mapping[str, Sequence[str]],
        labels: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(redacted_text, str):
            raise TypeError("classifier expects redacted text")
        system = instructions(task, allowed, labels)
        logger.info("gemini classify task=%s model=%s", task, self.model)
        response = self._client.models.generate_content(
            model=self.model,
            contents=redacted_text,
            config={
                "system_instruction": system,
                "temperature": 0,
                "seed": 0,
                "response_mime_type": "application/json",
            },
        )
        content = getattr(response, "text", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model response is empty")
        return parse_json_object(content)


def _groq_limits(model: str) -> dict[str, Any]:
    """Completion budget for one Groq call under the 8_000 token minute cap."""
    limits: dict[str, Any] = {
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
        "timeout": _TIMEOUT_SECONDS,
    }
    if model in _LOW_REASONING_MODELS:
        limits["reasoning_effort"] = GROQ_REASONING_EFFORT
    return limits


def _apply_env_lines(lines: Iterable[str], env: MutableMapping[str, str]) -> None:
    """Fill blank keys from `.env` lines. Existing values and blanks stay put."""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and value and key not in env:
            env[key] = value


def _load_project_env() -> None:
    """Load the repo `.env` once. A value already in the environment wins."""
    for directory in Path(__file__).resolve().parents:
        if not (directory / "pyproject.toml").is_file():
            continue
        env_file = directory / ".env"
        if env_file.is_file():
            _apply_env_lines(env_file.read_text(encoding="utf-8").splitlines(), os.environ)
        return


def default_classifier() -> GroqClassifier | None:
    """Return a Groq classifier for the agent nodes when `GROQ_API_KEY` is set."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    return GroqClassifier(api_key=key, model=model)


def default_review_classifier() -> GeminiClassifier | None:
    """Return a Gemini classifier for the reviewer when `GEMINI_API_KEY` is set."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = (
        os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    )
    return GeminiClassifier(api_key=key, model=model)


def default_for(task: str) -> Classifier | None:
    """The configured model for a task: Gemini for the review, Groq for the agents."""
    if task == TASK_REVIEW:
        return default_review_classifier()
    return default_classifier()


def classify_or_none(
    *,
    task: str,
    redacted_text: str,
    allowed: Mapping[str, Sequence[str]],
    classifier: Classifier | None = None,
    labels: Mapping[str, str] | None = None,
) -> Mapping[str, Any] | None:
    """Return the model JSON, or None when no model is configured or the call fails."""
    active = classifier if classifier is not None else default_for(task)
    if active is None:
        return None
    try:
        payload = active.classify(
            task=task,
            redacted_text=redacted_text,
            allowed=allowed,
            labels=labels,
        )
    except Exception as exc:
        logger.warning("%s classifier failed (%s); using cues", task, type(exc).__name__)
        return None
    if not isinstance(payload, Mapping):
        logger.warning(
            "%s classifier returned %s; using cues", task, type(payload).__name__
        )
        return None
    return payload


_load_project_env()


def instructions(
    task: str,
    allowed: Mapping[str, Sequence[str]],
    labels: Mapping[str, str] | None = None,
) -> str:
    """System prompt: the task, then the allowed ids. The classified text is not included."""
    if task not in _TASK_INSTRUCTIONS:
        raise ValueError(f"unknown classifier task {task!r}")
    names = labels or {}
    lines = [_TASK_INSTRUCTIONS[task]]
    for key, values in allowed.items():
        lines.append(f"{key}:")
        for value in values:
            label = names.get(value)
            if label:
                lines.append(f"- {value}: {label}")
            else:
                lines.append(f"- {value}")
    return _PREAMBLES[task] + "\n".join(lines)


def match_allowed(value: object, allowed: Sequence[str]) -> str | None:
    """Return the canonical allowed id, or None for null and unknown ids."""
    if value is None or not isinstance(value, str):
        return None
    text = value.strip()
    if text.casefold() in _NULL_WORDS:
        return None
    for item in allowed:
        if text == item or text.casefold() == item.casefold():
            return item
    return None


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a model message that is a JSON object, with or without a fence."""
    text = _FENCE.sub("", content.strip()).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not a JSON object")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload
