"""Mail preview after approve, then MCP send. Approve itself does not send."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from support_reply.contracts import Session
from support_reply.gate import Approval, require_approvable
from support_reply.mcp import McpClient, McpError, ToolClient, mcp_configured


class MailError(RuntimeError):
    """Preview or send was refused."""


@dataclass(frozen=True, slots=True)
class MailPreview:
    from_address: str
    to_address: str
    subject: str
    body: str
    cc: str = ""


@dataclass(frozen=True, slots=True)
class SendResult:
    message_id: str | None
    document_id: str | None
    document_url: str | None


def mail_from_address() -> str:
    """Unused as a Gmail header. Send always uses the MCP workspace account."""
    return ""


def default_mail_to() -> str:
    return os.environ.get("MAIL_TO", os.environ.get("MCP_GMAIL_TO", "")).strip()


def compose_preview(
    session: Session,
    *,
    approval: Approval,
    to_address: str,
    from_address: str | None = None,
    cc: str | None = None,
) -> MailPreview:
    """Build the mail the human sees after Approve. Body is the customer reply only."""
    require_approvable(session, human_approved=True)
    if session.draft is None:
        raise MailError("nothing to preview")
    if approval.record.get("reply") != session.draft.reply:
        raise MailError("preview requires the approved reply")
    to = (to_address or "").strip()
    disposition = session.draft.disposition
    return MailPreview(
        from_address=(from_address if from_address is not None else mail_from_address()).strip(),
        to_address=to,
        subject=f"Support reply · {disposition}",
        body=session.draft.reply,
        cc=(cc if cc is not None else os.environ.get("MCP_GMAIL_CC", "")).strip(),
    )


def _record_doc_body(record: dict[str, Any]) -> str:
    note = record.get("routing_note") or {}
    lines = [
        record.get("reply") or "",
        "",
        "Internal (not in the customer email)",
        f"Disposition: {record.get('disposition')}",
        f"Team: {note.get('team')}",
        f"Priority: {note.get('priority')}",
        f"Why: {note.get('why')}",
        f"Signal: {record.get('signal')}",
        f"Citation: {record.get('citation')}",
        f"Last checked: {record.get('last_checked')}",
    ]
    return "\n".join(str(part) for part in lines)


def send_preview(
    preview: MailPreview,
    *,
    client: ToolClient | None = None,
    record: dict[str, Any] | None = None,
) -> SendResult:
    """Write the ticket to a Google Doc, then send the customer reply via Gmail.

    From in the preview is display-only. Gmail sends from the workspace account.
    """
    to = preview.to_address.strip()
    if not to:
        raise MailError("To address is required to send")
    if "@" not in to:
        raise MailError("To address is not a valid email")
    body = preview.body.strip()
    if not body:
        raise MailError("Mail body is empty")

    own = client is None
    mcp = client or _live_client()
    try:
        if own:
            mcp.initialize()
        document_id = None
        document_url = None
        if record is not None:
            doc = mcp.call_tool(
                os.environ.get("MCP_DOCS_TOOL", "reflect_content").strip()
                or "reflect_content",
                {
                    "title": preview.subject,
                    "content": _record_doc_body(record),
                    "subtitle": "MCP workspace Gmail",
                },
            )
            if doc.get("success") is False:
                raise MailError(str(doc.get("error") or "Google Doc was not created"))
            document_id = str(doc.get("document_id") or doc.get("id") or "") or None
            document_url = str(doc.get("document_url") or doc.get("url") or "") or None
        mail = mcp.call_tool(
            os.environ.get("MCP_GMAIL_TOOL", "send_email").strip() or "send_email",
            {
                "to": to,
                "subject": preview.subject,
                "body_text": body,
                "cc": preview.cc,
            },
        )
        if mail.get("success") is False:
            raise MailError(str(mail.get("error") or "Email was not sent"))
        message_id = str(mail.get("message_id") or mail.get("id") or "") or None
        return SendResult(
            message_id=message_id,
            document_id=document_id,
            document_url=document_url,
        )
    except McpError as exc:
        raise MailError(str(exc)) from exc
    finally:
        if own:
            mcp.close()


def _mcp_server_url() -> str:
    """Prefer `.env` so Send tracks manismcpserver without restarting Streamlit."""
    for directory in Path(__file__).resolve().parents:
        if not (directory / "pyproject.toml").is_file():
            continue
        env_file = directory / ".env"
        if env_file.is_file():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line.startswith("MCP_SERVER_URL="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
        break
    return os.environ.get("MCP_SERVER_URL", "").strip() or "http://127.0.0.1:8080/mcp"


def _live_client() -> McpClient:
    url = _mcp_server_url()
    if not mcp_configured(url):
        raise MailError("MCP_SERVER_URL is not configured")
    return McpClient(
        server_url=url,
        auth_token=os.environ.get("MCP_AUTH_TOKEN") or None,
    )
