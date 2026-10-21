"""Offline eval runner for `evals/cases.json`. Cue path only."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from support_reply.contracts import Session
from support_reply.draft import sentences
from support_reply.graph import handle_complaint

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases.json"


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    category: str
    complaint: str
    expect: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Failure:
    case_id: str
    check: str
    expected: str
    got: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    ok: bool
    failures: tuple[Failure, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalReport:
    ok: bool
    results: tuple[CaseResult, ...]
    failures: tuple[Failure, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.ok)

    @property
    def total(self) -> int:
        return len(self.results)


def force_offline() -> None:
    """Drop the DeepInfra key so `.env` cannot turn evals into a live run."""
    os.environ.pop("DEEPINFRA_API_KEY", None)


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or CASES_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cases" not in payload:
        raise ValueError("evals catalog must be an object with a cases array")
    return payload


def load_cases(path: Path | None = None) -> list[EvalCase]:
    payload = load_catalog(path)
    return [_case(item) for item in payload["cases"]]


def run_evals(
    *,
    path: Path | None = None,
    file: TextIO = sys.stdout,
) -> EvalReport:
    force_offline()
    payload = load_catalog(path)
    results: list[CaseResult] = []
    for item in payload["cases"]:
        case = _case(item)
        session = handle_complaint(Session(), case.complaint)
        results.append(score_session(case, session))
    for thread in payload.get("threads", ()):
        results.append(_run_thread(thread))
    failures = tuple(fail for result in results for fail in result.failures)
    report = EvalReport(ok=not failures, results=tuple(results), failures=failures)
    _print_report(report, file)
    return report


def score_session(case: EvalCase, session: Session) -> CaseResult:
    found: list[Failure] = []
    expect = case.expect
    draft = session.draft
    facts = session.facts
    pattern = session.pattern
    review = session.review
    reply = "" if draft is None else draft.reply
    dumped = json.dumps(session.to_dict())

    def miss(check: str, expected: object, got: object) -> None:
        found.append(
            Failure(
                case_id=case.id,
                check=check,
                expected=repr(expected),
                got=repr(got),
            )
        )

    if "facts_status" in expect:
        got = None if facts is None else facts.status
        if got != expect["facts_status"]:
            miss("facts_status", expect["facts_status"], got)
    if "scheme_id" in expect:
        got = None if facts is None else facts.scheme_id
        if got != expect["scheme_id"]:
            miss("scheme_id", expect["scheme_id"], got)
    if "section_title" in expect:
        got = None if facts is None else facts.section_title
        if got != expect["section_title"]:
            miss("section_title", expect["section_title"], got)
    if "theme_id" in expect:
        got = None if pattern is None else pattern.theme_id
        if got != expect["theme_id"]:
            miss("theme_id", expect["theme_id"], got)
    if "disposition" in expect:
        got = None if draft is None else draft.disposition
        if got != expect["disposition"]:
            miss("disposition", expect["disposition"], got)
    if "dispositions" in expect:
        allowed = list(expect["dispositions"])
        got = None if draft is None else draft.disposition
        if got not in allowed:
            miss("dispositions", allowed, got)
    if "review_status" in expect:
        got = None if review is None else review.status
        if got != expect["review_status"]:
            miss("review_status", expect["review_status"], got)
    if "advice_refusal" in expect:
        if session.advice_refusal is not expect["advice_refusal"]:
            miss("advice_refusal", expect["advice_refusal"], session.advice_refusal)
    if expect.get("ticket_open"):
        why = "" if draft is None else draft.routing_note.why
        if "stays open" not in why:
            miss("ticket_open", "stays open", why)
    if "citation" in expect:
        got = None if draft is None else draft.citation
        if got != expect["citation"]:
            miss("citation", expect["citation"], got)
        elif expect["citation"] and expect["citation"] not in reply:
            miss("citation_in_reply", expect["citation"], reply)
    if "last_checked" in expect:
        got = None if draft is None else draft.last_checked
        if got != expect["last_checked"]:
            miss("last_checked", expect["last_checked"], got)
    if "pii_types" in expect:
        got = list(session.pii_types)
        if got != list(expect["pii_types"]):
            miss("pii_types", expect["pii_types"], got)
    if "max_sentences" in expect:
        count = len(sentences(reply)) if reply else 0
        if count > int(expect["max_sentences"]):
            miss("max_sentences", f"<= {expect['max_sentences']}", count)
    for needle in _strings(expect.get("reply_contains")):
        if needle not in reply:
            miss("reply_contains", needle, reply)
    for needle in _strings(expect.get("reply_absent")):
        if needle in reply:
            miss("reply_absent", f"absent {needle!r}", reply)
    for pattern_text in _strings(expect.get("reply_forbidden_patterns")):
        if re.search(pattern_text, reply):
            miss("reply_forbidden_patterns", pattern_text, reply)
    for raw in _strings(expect.get("raw_absent")):
        if raw in reply or raw in dumped:
            miss("raw_absent", f"absent {raw!r}", "present")

    return CaseResult(id=case.id, ok=not found, failures=tuple(found))


def _run_thread(payload: Mapping[str, Any]) -> CaseResult:
    thread_id = str(payload["id"])
    session = Session()
    found: list[Failure] = []
    for index, turn in enumerate(payload["turns"]):
        case = EvalCase(
            id=f"{thread_id}[{index}]",
            category="thread",
            complaint=turn["complaint"],
            expect=turn.get("expect") or {},
        )
        session = handle_complaint(session, case.complaint)
        scored = score_session(case, session)
        found.extend(scored.failures)
    return CaseResult(id=thread_id, ok=not found, failures=tuple(found))


def _case(item: Mapping[str, Any]) -> EvalCase:
    return EvalCase(
        id=str(item["id"]),
        category=str(item.get("category", "")),
        complaint=str(item["complaint"]),
        expect=item.get("expect") or {},
    )


def _strings(value: object) -> Sequence[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def _print_report(report: EvalReport, file: TextIO) -> None:
    print(f"evals: {report.passed}/{report.total} pass", file=file)
    for failure in report.failures:
        print(
            f"FAIL {failure.case_id} {failure.check}: expected {failure.expected} got {failure.got}",
            file=file,
        )
    print("evals: pass" if report.ok else "evals: fail", file=file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline support-reply evals.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args(argv)
    report = run_evals(path=args.cases)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
