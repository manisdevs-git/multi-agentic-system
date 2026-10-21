"""After approve: a mail preview (configurable From, To) then MCP send. Approve does not send."""

import pytest

from support_reply.contracts import Session
from support_reply.gate import approve
from support_reply.graph import handle_complaint
from support_reply.mail import (
    MailError,
    compose_preview,
    send_preview,
)

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"


def _approved(tmp_path):
    session = handle_complaint(Session(), EXIT_LOAD)
    approval = approve(
        session, human_approved=True, out_dir=tmp_path, session_id="mail-1"
    )
    return session, approval


def test_preview_uses_the_customer_reply_and_to_address(tmp_path):
    session, approval = _approved(tmp_path)

    preview = compose_preview(
        session, approval=approval, to_address="ops@example.com"
    )

    assert preview.from_address == ""
    assert preview.to_address == "ops@example.com"
    assert preview.body == session.draft.reply
    assert "exit load" in preview.body.lower() or "1%" in preview.body
    assert preview.subject


def test_send_requires_a_to_address(tmp_path):
    session, approval = _approved(tmp_path)
    preview = compose_preview(session, approval=approval, to_address="  ")
    with pytest.raises(MailError, match="To"):
        send_preview(preview, client=_FakeMcp())


def test_send_calls_mcp_send_email_and_reflects_the_record(tmp_path):
    session, approval = _approved(tmp_path)
    preview = compose_preview(
        session, approval=approval, to_address="ops@example.com"
    )
    fake = _FakeMcp()

    result = send_preview(preview, client=fake, record=approval.record)

    assert [name for name, _ in fake.calls] == ["reflect_content", "send_email"]
    doc_args = fake.calls[0][1]
    assert "Fees" in doc_args["content"] or "routing" in doc_args["content"].lower() or "escalate" in doc_args["content"]
    mail_args = fake.calls[1][1]
    assert mail_args["to"] == "ops@example.com"
    assert mail_args["body_text"] == session.draft.reply
    assert "from" not in mail_args
    assert result.message_id == "m-1"
    assert result.document_id == "d-1"


def test_approve_still_does_not_send(tmp_path):
    session, approval = _approved(tmp_path)
    assert approval.email_path.read_text(encoding="utf-8") == session.draft.reply


class _FakeMcp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def initialize(self) -> dict:
        return {}

    def close(self) -> None:
        return None

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "reflect_content":
            return {"success": True, "document_id": "d-1", "document_url": "https://docs.google.com/d-1"}
        if name == "send_email":
            return {"success": True, "message_id": "m-1"}
        raise AssertionError(name)
