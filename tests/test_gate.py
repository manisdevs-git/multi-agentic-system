"""Phase 9 exit: human approve appends a redacted record and writes an email draft.

Nothing is sent. `fail` and `blocked` write neither file. The payload scan raises
if a raw PAN, Aadhaar, phone, email, or OTP is still in what would be stored.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from support_reply.contracts import ReviewFailure, ReviewerResult, Session
from support_reply.gate import ApprovalRefused, RawPIIInRecord, approve, approvable
from support_reply.graph import handle_complaint

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"


def _passed(text=EXIT_LOAD):
    session = handle_complaint(Session(), text)
    assert session.review.status == "pass"
    return session


def _paths(out_dir: Path, session_id: str):
    return out_dir / "records.jsonl", out_dir / "email_drafts" / f"{session_id}.txt"


def test_a_passed_large_cap_draft_appends_one_record_and_one_email_file(tmp_path):
    session = _passed()
    session_id = "large-cap"

    result = approve(
        session, human_approved=True, out_dir=tmp_path, session_id=session_id
    )

    records_path, email_path = _paths(tmp_path, session_id)
    assert result.record_path == records_path
    assert result.email_path == email_path
    lines = records_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["redacted_text"] == EXIT_LOAD
    assert record["disposition"] == session.draft.disposition
    assert record["reply"] == session.draft.reply
    assert record["citation"] == LARGE_CAP_URL
    assert record["last_checked"] == "25-Jun-2026"
    assert record["routing_note"] == session.draft.routing_note.to_dict()
    assert record["signal"] == session.pattern.signal
    assert record["agents_fired"] == [entry.to_dict() for entry in session.agents_fired]
    body = email_path.read_text(encoding="utf-8")
    assert body == session.draft.reply
    assert LARGE_CAP_URL in body
    assert not hasattr(approve, "send")


def test_a_pan_complaint_record_keeps_the_token_and_drops_the_pan(tmp_path):
    session = _passed(f"{EXIT_LOAD} My PAN is {RAW_PAN} and phone {RAW_PHONE}.")

    approve(session, human_approved=True, out_dir=tmp_path, session_id="pii")

    dumped = (tmp_path / "records.jsonl").read_text(encoding="utf-8")
    email = (tmp_path / "email_drafts" / "pii.txt").read_text(encoding="utf-8")
    assert "[PAN]" in dumped
    assert RAW_PAN not in dumped and RAW_PAN not in email
    assert RAW_PHONE not in dumped and RAW_PHONE not in email
    record = json.loads(dumped)
    assert record["redacted_text"] == session.redacted_text


def test_a_blocked_session_creates_neither_file(tmp_path):
    session = _passed()
    session.review = ReviewerResult(
        status="blocked",
        round=2,
        failures=(ReviewFailure(check="advice", detail="you may want to"),),
    )
    assert approvable(session) is False

    with pytest.raises(ApprovalRefused, match="blocked"):
        approve(session, human_approved=True, out_dir=tmp_path, session_id="blocked")

    assert not (tmp_path / "records.jsonl").exists()
    assert not (tmp_path / "email_drafts").exists()


def test_without_the_human_flag_nothing_is_written(tmp_path):
    session = _passed()

    with pytest.raises(ApprovalRefused, match="human-approve"):
        approve(session, human_approved=False, out_dir=tmp_path, session_id="no")

    assert list(tmp_path.iterdir()) == []


def test_a_raw_value_in_the_payload_raises_and_writes_nothing(tmp_path):
    session = _passed()
    session.draft = replace(session.draft, reply=f"{session.draft.reply} PAN {RAW_PAN}.")

    with pytest.raises(RawPIIInRecord, match="PAN") as caught:
        approve(session, human_approved=True, out_dir=tmp_path, session_id="leak")

    assert RAW_PAN not in str(caught.value)
    assert not (tmp_path / "records.jsonl").exists()
    assert not (tmp_path / "email_drafts").exists()
