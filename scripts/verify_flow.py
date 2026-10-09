#!/usr/bin/env python3
"""Run one complaint through the full support-reply graph with live model calls.

Facts, Pattern, and Compliance each make one Groq call. The reviewer makes one
Gemini call. Code still copies the fact, counts the reviews, writes the draft,
and decides the review. Nothing is approved or emailed.

Usage (from the repo root):

    uv run python scripts/verify_flow.py
    uv run python scripts/verify_flow.py "why was I charged an exit load on HDFC gold mutual fund?"
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_reply.contracts import Session  # noqa: E402
from support_reply.gate import approvable  # noqa: E402
from support_reply.graph import handle_complaint  # noqa: E402
from support_reply.llm import (  # noqa: E402
    Classifier,
    default_classifier,
    default_review_classifier,
)
from support_reply.trace import Trace  # noqa: E402

DEFAULT_COMPLAINT = (
    "why was I charged an exit load on HDFC gold mutual fund , "
    "what was the amount and why I was charge ?"
)


class RecordingClassifier:
    """Delegates to a real model and keeps each call's task, timing, and JSON."""

    def __init__(self, inner: Classifier, provider: str) -> None:
        self._inner = inner
        self.provider = provider
        self.model = getattr(inner, "model", "")
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def classify(
        self,
        *,
        task: str,
        redacted_text: str,
        allowed: dict[str, Any] | Any,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = self._inner.classify(
                task=task,
                redacted_text=redacted_text,
                allowed=allowed,
                labels=labels,
            )
        except Exception as exc:
            self._record(task, started, error=type(exc).__name__)
            raise
        self._record(task, started, payload=dict(payload))
        return payload

    def _record(
        self,
        task: str,
        started: float,
        *,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        entry = {
            "provider": self.provider,
            "model": self.model,
            "task": task,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "payload": payload,
            "error": error,
        }
        with self._lock:
            self.calls.append(entry)


class FallbackWatch(logging.Handler):
    """Catches the cue-table fallback the graph uses when a model call fails."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "using cues" in message or "classifier failed" in message:
            self.messages.append(message)


def main(argv: list[str]) -> int:
    complaint = argv[1] if len(argv) > 1 else DEFAULT_COMPLAINT
    if len(argv) > 2:
        print("usage: python scripts/verify_flow.py [complaint]", file=sys.stderr)
        return 2

    groq = default_classifier()
    gemini = default_review_classifier()
    missing = [
        name
        for name, client in (("GROQ_API_KEY", groq), ("GEMINI_API_KEY", gemini))
        if client is None
    ]
    if missing or groq is None or gemini is None:
        print(
            "Missing "
            + ", ".join(missing)
            + ". Put them in .env so this run calls the models instead of the cue tables.",
            file=sys.stderr,
        )
        return 1

    agents = RecordingClassifier(groq, "groq")
    reviewer = RecordingClassifier(gemini, "gemini")
    watch = FallbackWatch()
    logging.getLogger("support_reply.llm").addHandler(watch)

    trace = Trace()
    session = handle_complaint(
        Session(),
        complaint,
        classifier=agents,
        review_classifier=reviewer,
        trace=trace,
    )

    _print_report(complaint, agents, reviewer, trace, session, watch.messages)
    if watch.messages or any(call["error"] for call in agents.calls + reviewer.calls):
        return 1
    if trace.violations():
        return 1
    expected = {"facts", "pattern", "compliance"}
    seen = {call["task"] for call in agents.calls if call["error"] is None}
    if seen != expected or not any(call["task"] == "review" for call in reviewer.calls):
        print("Expected one Groq call each for facts, pattern, and compliance, and one Gemini review call.")
        return 1
    return 0


def _print_report(
    complaint: str,
    agents: RecordingClassifier,
    reviewer: RecordingClassifier,
    trace: Trace,
    session: Session,
    fallbacks: list[str],
) -> None:
    print("Complaint")
    print(f"  {complaint}")
    print()
    print("Models")
    print(f"  agents:   {agents.provider} {agents.model}")
    print(f"  reviewer: {reviewer.provider} {reviewer.model}")
    print()
    print("LLM calls")
    for index, call in enumerate(agents.calls + reviewer.calls, start=1):
        body = call["error"] or json.dumps(call["payload"], ensure_ascii=False)
        print(f"  {index}. {call['provider']} {call['task']}  {call['elapsed_ms']:.0f}ms  {body}")
    if fallbacks:
        print()
        print("Cue-table fallback")
        for message in fallbacks:
            print(f"  {message}")
    print()
    print("Trace")
    print(trace.render())
    problems = trace.violations()
    print()
    print("Parallel rule: " + ("pass" if not problems else "fail"))
    for problem in problems:
        print(f"  {problem}")
    print()
    print("Session")
    print(json.dumps(session.to_dict(), indent=2, ensure_ascii=False))
    print()
    print("Approval gate")
    print(f"  approvable: {approvable(session)}")
    print("  approved: no — this script does not click approve, so nothing is appended or emailed")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
