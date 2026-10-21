#!/usr/bin/env python3
"""Run one complaint through the full support-reply graph with live model calls.

Facts, Pattern, Compliance, and the reviewer each make one DeepInfra call. Code
still copies the fact, counts the reviews, writes the draft, and decides the
review. Nothing is approved or emailed.

Usage (from the repo root):

    uv run python scripts/verify_flow.py
    uv run python scripts/verify_flow.py "why was I charged an exit load on HDFC gold mutual fund?"
    uv run python scripts/verify_flow.py --approve
    uv run python scripts/verify_flow.py --approve "why was I charged an exit load on HDFC gold mutual fund?"
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
from support_reply.gate import approve, approvable  # noqa: E402
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
    args = argv[1:]
    human_approve = "--approve" in args
    args = [arg for arg in args if arg != "--approve"]
    if len(args) > 1:
        print(
            "usage: python scripts/verify_flow.py [--approve] [complaint]",
            file=sys.stderr,
        )
        return 2
    complaint = args[0] if args else DEFAULT_COMPLAINT

    agents_model = default_classifier()
    review_model = default_review_classifier()
    if agents_model is None or review_model is None:
        print(
            "Missing DEEPINFRA_API_KEY. Put it in .env so this run calls DeepInfra "
            "instead of the cue tables.",
            file=sys.stderr,
        )
        return 1

    agents = RecordingClassifier(agents_model, "deepinfra")
    reviewer = RecordingClassifier(review_model, "deepinfra")
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

    _print_report(
        complaint,
        agents,
        reviewer,
        trace,
        session,
        watch.messages,
        will_approve=human_approve,
    )
    if watch.messages or any(call["error"] for call in agents.calls + reviewer.calls):
        return 1
    if trace.violations():
        return 1
    expected = {"facts", "pattern", "compliance"}
    seen = {call["task"] for call in agents.calls if call["error"] is None}
    if seen != expected or not any(call["task"] == "review" for call in reviewer.calls):
        print("Expected one DeepInfra call each for facts, pattern, and compliance, and one DeepInfra review call.")
        return 1
    if human_approve:
        approval = approve(session, human_approved=True)
        print()
        print("Human approve")
        print(f"  session_id: {approval.session_id}")
        print(f"  record:     {approval.record_path}")
        print(f"  email:      {approval.email_path}")
        print("  sent:       no — approve writes a local draft file only")
        print()
        print("Email draft file")
        print(f"  {approval.email_path.read_text(encoding='utf-8')}")
    return 0


def _print_report(
    complaint: str,
    agents: RecordingClassifier,
    reviewer: RecordingClassifier,
    trace: Trace,
    session: Session,
    fallbacks: list[str],
    *,
    will_approve: bool = False,
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
    if not will_approve:
        print("  approved: no — pass --approve to write the record and email draft file")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
