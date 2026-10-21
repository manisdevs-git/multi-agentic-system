"""Approval gate. A human click is the only trigger; nothing is sent.

`approvable` / `require_approvable` refuse `fail` and `blocked`. `approve` calls
that guard first, then appends one redacted record and writes one email draft
file. There is no SMTP client and no send function. The payload is scanned for
raw PAN, Aadhaar, account numbers, phone numbers, emails, and OTPs before any
file is created; a hit raises and writes nothing.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from support_reply.contracts import Session
from support_reply.ingress import scan

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "out"
RECORDS_NAME = "records.jsonl"
EMAIL_DIR_NAME = "email_drafts"


class ApprovalRefused(RuntimeError):
    """Raised when something tries to approve a session the gate does not accept."""


class RawPIIInRecord(RuntimeError):
    """Raised when the record payload still contains a raw PII value. The message
    names the token type only; it does not include the value.
    """


@dataclass(frozen=True, slots=True)
class Approval:
    """Paths written by one successful `approve` call."""

    session_id: str
    record_path: Path
    email_path: Path
    record: dict[str, Any]


def approvable(session: Session) -> bool:
    """True only when a draft exists and its last review is `pass`."""
    if not isinstance(session, Session):
        raise TypeError("gate expects a Session")
    return (
        session.draft is not None
        and session.review is not None
        and session.review.status == "pass"
    )


def require_approvable(session: Session, *, human_approved: bool) -> None:
    """Raise `ApprovalRefused` unless a human approved a reviewer-passed draft."""
    if type(human_approved) is not bool:
        raise TypeError("human_approved must be a bool")
    if not human_approved:
        raise ApprovalRefused("approve requires the human-approve flag")
    if session.draft is None or session.review is None:
        raise ApprovalRefused("nothing to approve: the session has no reviewed draft")
    if session.review.status != "pass":
        checks = [failure.check for failure in session.review.failures]
        raise ApprovalRefused(
            f"review status is {session.review.status!r} (round {session.review.round}, "
            f"checks {checks}); approve is not available"
        )


def approve(
    session: Session,
    *,
    human_approved: bool,
    out_dir: Path | None = None,
    session_id: str | None = None,
) -> Approval:
    """Append one redacted record and write an email draft. Do not send it.

    `out_dir` defaults to the repo `out/` directory. `session_id` names the
    email file; a new id is generated when the caller does not pass one.
    """
    require_approvable(session, human_approved=human_approved)
    if session.draft is None or session.pattern is None:
        raise ApprovalRefused("nothing to approve: the session has no reviewed draft")

    sid = session_id if session_id is not None else uuid.uuid4().hex
    if not isinstance(sid, str) or not sid.strip() or "/" in sid or "\\" in sid:
        raise ValueError("session_id must be a file-safe non-empty str")

    record = {
        "session_id": sid,
        "redacted_text": session.redacted_text,
        "disposition": session.draft.disposition,
        "reply": session.draft.reply,
        "citation": session.draft.citation,
        "last_checked": session.draft.last_checked,
        "routing_note": session.draft.routing_note.to_dict(),
        "signal": session.pattern.signal,
        "agents_fired": [entry.to_dict() for entry in session.agents_fired],
    }
    _reject_raw_pii(record)

    root = Path(DEFAULT_OUT_DIR if out_dir is None else out_dir)
    records_path = root / RECORDS_NAME
    email_path = root / EMAIL_DIR_NAME / f"{sid}.txt"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    email_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    email_path.write_text(session.draft.reply, encoding="utf-8")
    return Approval(
        session_id=sid,
        record_path=records_path,
        email_path=email_path,
        record=record,
    )


def _reject_raw_pii(payload: Mapping[str, Any]) -> None:
    """Scan the JSON that would be stored. Token types only in any error."""
    dumped = json.dumps(payload, ensure_ascii=False, default=str)
    found = scan(dumped)
    if found.pii_types:
        types = ", ".join(found.pii_types)
        raise RawPIIInRecord(f"record payload contains {types}")
