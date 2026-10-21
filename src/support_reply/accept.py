"""Phase 10 acceptance run. Offline. Exits non-zero on the first miss.

Usage from the repo root:

    python -m support_reply.accept
    python -m support_reply.accept --out out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

from support_reply.contracts import Session
from support_reply.corpus import data_dir
from support_reply.gate import (
    DEFAULT_OUT_DIR,
    ApprovalRefused,
    approvable,
    approve,
)
from support_reply.graph import handle_complaint
import support_reply.graph as graph_module
from support_reply.session import MAX_REFINEMENTS, RefinementRefused, refine
from support_reply.trace import AGENT_NODES, Trace

SAMPLE_SIZE = 20
_ADVICE_SPAN = "You may want to stay invested."


class AcceptanceFailure(RuntimeError):
    """One acceptance check missed. The script exits 1."""


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    ok: bool
    review_ids: tuple[str, ...]
    branches: dict[str, dict[str, Any]]


def run_acceptance(
    *,
    out_dir: Path | None = None,
    file: TextIO = sys.stdout,
) -> AcceptanceReport:
    """Run every Phase 10 check. Raises `AcceptanceFailure` on the first miss."""
    dest = Path(DEFAULT_OUT_DIR if out_dir is None else out_dir)
    fx = load_fixtures()
    review_ids = _twenty_complaints(file)
    branches = _four_branches(fx, dest, file)
    _advice_thread(fx, file)
    _caps(fx, dest, file)
    print("acceptance: pass", file=file)
    return AcceptanceReport(ok=True, review_ids=review_ids, branches=branches)


def sample_reviews(path: Path | None = None) -> list[dict[str, Any]]:
    """The first `SAMPLE_SIZE` reviews with non-empty text, in file order."""
    reviews_path = path or (data_dir() / "normalized_reviews.json")
    payload = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise AcceptanceFailure("normalized_reviews.json must be a list")
    picked: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ident = item.get("id")
        if not isinstance(ident, str) or not ident.strip():
            raise AcceptanceFailure("a sampled review is missing id")
        picked.append(item)
        if len(picked) == SAMPLE_SIZE:
            return picked
    raise AcceptanceFailure(
        f"needed {SAMPLE_SIZE} non-empty reviews, found {len(picked)}"
    )


def load_fixtures() -> ModuleType:
    """Load `tests/fixtures.py` from the checkout. The script does not copy the strings."""
    repo = _repo_root()
    path = repo / "tests" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("support_reply_accept_fixtures", path)
    if spec is None or spec.loader is None:
        raise AcceptanceFailure(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "tests" / "fixtures.py").is_file() and (
            parent / "data" / "normalized_reviews.json"
        ).is_file():
            return parent
    raise AcceptanceFailure("tests/fixtures.py and data/normalized_reviews.json not found")


def _require(ok: object, message: str) -> None:
    if not ok:
        raise AcceptanceFailure(message)


def _twenty_complaints(file: TextIO) -> tuple[str, ...]:
    reviews = sample_reviews()
    ids = tuple(item["id"] for item in reviews)
    print(f"20 reviews: {', '.join(ids)}", file=file)
    for item in reviews:
        session = handle_complaint(Session(), item["text"])
        _require_structured(session, label=item["id"])
        _require_cited_or_uncovered(session, label=item["id"])
    return ids


def _require_structured(session: Session, *, label: str) -> None:
    for name in ("facts", "pattern", "compliance"):
        _require(getattr(session, name) is not None, f"{label}: {name} returned no JSON result")
    _require(session.draft is not None, f"{label}: no draft")
    _require(session.review is not None, f"{label}: no reviewer result")


def _require_cited_or_uncovered(session: Session, *, label: str) -> None:
    facts = session.facts
    draft = session.draft
    assert facts is not None and draft is not None
    if facts.status == "covered" and draft.disposition != "refuse":
        _require(facts.citation, f"{label}: covered fact has no source_url")
        _require(
            facts.citation in draft.reply,
            f"{label}: reply states a fact without {facts.citation}",
        )
        _require(
            draft.citation == facts.citation,
            f"{label}: draft citation is not the scheme source_url",
        )
        return
    if facts.status == "not_covered":
        _require(draft.citation is None, f"{label}: not_covered reply has a citation")
        _require(
            "https://" not in draft.reply and "http://" not in draft.reply,
            f"{label}: not_covered reply contains a URL",
        )


def _four_branches(fx: ModuleType, out_dir: Path, file: TextIO) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    reports["clean_hit"] = _clean_hit(fx, file)
    reports["corpus_miss"] = _corpus_miss(fx, file)
    reports["recurring"] = _recurring(fx, file)
    reports["pii"] = _pii(fx, out_dir, file)
    return reports


def _clean_hit(fx: ModuleType, file: TextIO) -> dict[str, Any]:
    trace = Trace()
    session = handle_complaint(Session(), fx.CLEAN_HIT, trace=trace)
    _require_structured(session, label="clean_hit")
    assert session.facts is not None
    _require(session.facts.status == "covered", "clean_hit: Facts was not covered")
    _require(
        session.facts.citation == fx.LARGE_CAP_URL,
        "clean_hit: citation is not the large-cap source_url",
    )
    _require(not trace.violations(), "clean_hit: " + "; ".join(trace.violations()))
    _print_branch("clean_hit", session, trace, file)
    return _branch_payload(session, trace)


def _corpus_miss(fx: ModuleType, file: TextIO) -> dict[str, Any]:
    trace = Trace()
    session = handle_complaint(Session(), fx.CORPUS_MISS, trace=trace)
    _require_structured(session, label="corpus_miss")
    assert session.facts is not None and session.draft is not None
    _require(session.facts.status == "not_covered", "corpus_miss: Facts was covered")
    _require(session.facts.searched, "corpus_miss: searched is empty")
    _require(session.draft.citation is None, "corpus_miss: reply has a citation")
    _require(
        "https://" not in session.draft.reply and "http://" not in session.draft.reply,
        "corpus_miss: reply has a URL",
    )
    _require("1%" not in session.draft.reply, "corpus_miss: invented exit-load figure")
    _require(
        "NAV" not in session.draft.reply,
        "corpus_miss: invented NAV",
    )
    _print_branch("corpus_miss", session, trace, file)
    return _branch_payload(session, trace)


def _recurring(fx: ModuleType, file: TextIO) -> dict[str, Any]:
    trace = Trace()
    session = handle_complaint(Session(), fx.RECURRING, trace=trace)
    _require_structured(session, label="recurring")
    assert session.pattern is not None and session.draft is not None
    _require(
        session.pattern.theme_id == "fees_brokerage",
        f"recurring: theme_id is {session.pattern.theme_id!r}",
    )
    _require(
        session.pattern.count == fx.FEES_BROKERAGE_COUNT,
        f"recurring: count is {session.pattern.count}, not {fx.FEES_BROKERAGE_COUNT}",
    )
    _require(
        session.pattern.signal in ("recurring", "rising"),
        f"recurring: signal is {session.pattern.signal!r}",
    )
    _require(
        "stays open" in session.draft.routing_note.why,
        "recurring: routing note does not keep the ticket open",
    )
    _print_branch("recurring", session, trace, file)
    return _branch_payload(session, trace)


def _pii(fx: ModuleType, out_dir: Path, file: TextIO) -> dict[str, Any]:
    trace = Trace()
    session = handle_complaint(Session(), fx.PII, trace=trace)
    _require_structured(session, label="pii")
    _require(
        session.pii_types == ("PAN", "PHONE"),
        f"pii: ingress types are {session.pii_types!r}",
    )
    _require(session.review is not None and session.review.status == "pass", "pii: reviewer did not pass")
    dumped = json.dumps(session.to_dict())
    _require(fx.RAW_PAN not in dumped and fx.RAW_PHONE not in dumped, "pii: raw value in session")
    _require("[PAN]" in session.redacted_text and "[PHONE]" in session.redacted_text, "pii: tokens missing")
    ran = [node for node in AGENT_NODES if trace.starts(node)]
    _require(ran == list(AGENT_NODES), f"pii: agents that ran are {ran}")
    fired = [entry.to_dict() for entry in session.agents_fired]
    _print_branch("pii", session, trace, file)
    result = approve(session, human_approved=True, out_dir=out_dir, session_id="accept-pii")
    record_text = result.record_path.read_text(encoding="utf-8")
    email_text = result.email_path.read_text(encoding="utf-8")
    _require(fx.RAW_PAN not in record_text and fx.RAW_PHONE not in record_text, "pii: raw value in record")
    _require(fx.RAW_PAN not in email_text and fx.RAW_PHONE not in email_text, "pii: raw value in email")
    _require("[PAN]" in record_text, "pii: record dropped the PAN token")
    return _branch_payload(session, trace) | {"agents_fired": fired, "approved": True}


def _advice_thread(fx: ModuleType, file: TextIO) -> None:
    print("== advice_thread ==", file=file)
    session = handle_complaint(Session(), fx.ADVICE_REDEEM)
    _require_refuse(session, "redeem")
    session = handle_complaint(session, fx.ADVICE_GOOD)
    _require_refuse(session, "good")
    session = handle_complaint(session, fx.ADVICE_RETURNS)
    _require_refuse(session, "returns")
    refine(session, fx.ADVICE_REPHRASE)
    _require_refuse(session, "rephrase")
    session = handle_complaint(session, fx.ADVICE_AGAIN)
    _require_refuse(session, "again")
    _require(session.advice_refusal is True, "advice_thread: advice_refusal reset")
    print("advice_thread: every turn refuse", file=file)


def _require_refuse(session: Session, label: str) -> None:
    _require(session.draft is not None, f"advice_thread {label}: no draft")
    _require(
        session.draft.disposition == "refuse",
        f"advice_thread {label}: disposition is {session.draft.disposition!r}",
    )
    _require(session.advice_refusal is True, f"advice_thread {label}: advice_refusal is false")


def _caps(fx: ModuleType, out_dir: Path, file: TextIO) -> None:
    print("== caps ==", file=file)
    session = _blocked_session(fx.CLEAN_HIT)
    _require(session.review is not None and session.review.status == "blocked", "caps: not blocked")
    _require(approvable(session) is False, "caps: blocked session is approvable")
    try:
        approve(session, human_approved=True, out_dir=out_dir, session_id="accept-blocked")
    except ApprovalRefused:
        pass
    else:
        raise AcceptanceFailure("caps: blocked session was approved")
    _require(
        not (out_dir / "email_drafts" / "accept-blocked.txt").exists(),
        "caps: blocked session wrote an email draft",
    )

    session = handle_complaint(Session(), fx.CLEAN_HIT)
    for _ in range(MAX_REFINEMENTS):
        refine(session, "Shorter")
    _require(session.refinement_count == MAX_REFINEMENTS, "caps: refinement_count is not 5")
    kept = session.draft.reply if session.draft is not None else None
    try:
        refine(session, "Shorter")
    except RefinementRefused:
        pass
    else:
        raise AcceptanceFailure("caps: sixth pushback was accepted")
    _require(session.refinement_count == MAX_REFINEMENTS, "caps: sixth pushback changed the count")
    _require(session.draft is not None and session.draft.reply == kept, "caps: sixth pushback changed the draft")
    print("caps: blocked cannot approve; sixth refinement refused", file=file)


def _blocked_session(text: str) -> Session:
    """Force two bad drafts so the reviewer ends `blocked`. Restore compose after."""
    real = graph_module.compose

    def corrupt(*args: Any, **kwargs: Any):
        drafted = real(*args, **kwargs)
        marker = f"Last checked {drafted.last_checked}."
        if drafted.last_checked and marker in drafted.reply:
            tail = drafted.reply[drafted.reply.index(marker) :]
            return replace(drafted, reply=f"{_ADVICE_SPAN} {tail}")
        return replace(drafted, reply=f"{_ADVICE_SPAN} {drafted.reply}")

    graph_module.compose = corrupt
    try:
        return handle_complaint(Session(), text)
    finally:
        graph_module.compose = real


def _print_branch(name: str, session: Session, trace: Trace, file: TextIO) -> None:
    print(f"== {name} ==", file=file)
    print(trace.render(), file=file)
    fired = [entry.to_dict() for entry in session.agents_fired]
    print(f"agents_fired: {json.dumps(fired, ensure_ascii=False)}", file=file)
    print(
        f"facts={session.facts.status if session.facts else None} "
        f"theme={session.pattern.theme_id if session.pattern else None} "
        f"disposition={session.draft.disposition if session.draft else None} "
        f"review={session.review.status if session.review else None}",
        file=file,
    )


def _branch_payload(session: Session, trace: Trace) -> dict[str, Any]:
    return {
        "facts": None if session.facts is None else session.facts.to_dict(),
        "pattern": None if session.pattern is None else session.pattern.to_dict(),
        "compliance": None if session.compliance is None else session.compliance.to_dict(),
        "disposition": None if session.draft is None else session.draft.disposition,
        "review": None if session.review is None else session.review.to_dict(),
        "agents_fired": [entry.to_dict() for entry in session.agents_fired],
        "trace": trace.to_list(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 10 acceptance checks.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for approved records and email drafts (default: repo out/)",
    )
    args = parser.parse_args(argv)
    try:
        run_acceptance(out_dir=args.out)
    except AcceptanceFailure as exc:
        print(f"acceptance: fail: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
