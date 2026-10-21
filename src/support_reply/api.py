"""HTTP backend for the support desk. The Vercel UI calls this; it owns the graph."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from support_reply.contracts import Session
from support_reply.gate import Approval, ApprovalRefused, approvable, approve
from support_reply.graph import handle_complaint
from support_reply.mail import MailError, MailPreview, compose_preview, default_mail_to, send_preview
from support_reply.trace import Trace

app = FastAPI(title="support-reply", version="0.0.1")


def _origins() -> list[str]:
    raw = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
    return [part.strip() for part in raw.split(",") if part.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Stored:
    session: Session
    trace: str
    approval: Approval | None = None
    sent: dict[str, Any] | None = None


_STORE: dict[str, Stored] = {}


def reset_store() -> None:
    _STORE.clear()


class ComplaintBody(BaseModel):
    complaint: str = Field(min_length=1)


class MailBody(BaseModel):
    to: str = ""


def _out_dir() -> Path:
    value = os.environ.get("OUT_DIR", "").strip()
    return Path(value) if value else Path(__file__).resolve().parents[2] / "out"


def _public(sid: str, stored: Stored, *, preview: MailPreview | None = None) -> dict[str, Any]:
    session = stored.session
    draft = session.draft
    return {
        "session_id": sid,
        "redacted_text": session.redacted_text,
        "pii_types": list(session.pii_types),
        "facts": None if session.facts is None else session.facts.to_dict(),
        "pattern": None if session.pattern is None else session.pattern.to_dict(),
        "compliance": None if session.compliance is None else session.compliance.to_dict(),
        "draft": None if draft is None else draft.to_dict(),
        "review": None if session.review is None else session.review.to_dict(),
        "trace": stored.trace,
        "approvable": approvable(session),
        "approved": stored.approval is not None,
        "default_to": default_mail_to() or "investor.support@groww.in",
        "preview": None
        if preview is None
        else {
            "to_address": preview.to_address,
            "subject": preview.subject,
            "body": preview.body,
        },
        "sent": stored.sent,
    }


def _get(sid: str) -> Stored:
    stored = _STORE.get(sid)
    if stored is None:
        raise HTTPException(status_code=404, detail="session not found")
    return stored


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/sessions")
def create_session(body: ComplaintBody) -> dict[str, Any]:
    text = body.complaint.strip()
    if not text:
        raise HTTPException(status_code=400, detail="paste a complaint first")
    trace = Trace()
    session = handle_complaint(Session(), text, trace=trace)
    del text
    # Hyphenated chunks so a 12-digit Aadhaar scan cannot match the id.
    raw = uuid.uuid4().hex
    sid = "sr-" + "-".join(raw[i : i + 4] for i in range(0, len(raw), 4))
    stored = Stored(session=session, trace=trace.render())
    _STORE[sid] = stored
    return _public(sid, stored)


@app.get("/v1/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    return _public(sid, _get(sid))


@app.post("/v1/sessions/{sid}/approve")
def approve_session(sid: str) -> dict[str, Any]:
    stored = _get(sid)
    try:
        stored.approval = approve(
            stored.session,
            human_approved=True,
            out_dir=_out_dir(),
            session_id=sid,
        )
    except ApprovalRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _public(sid, stored)


@app.post("/v1/sessions/{sid}/mail/preview")
def preview_mail(sid: str, body: MailBody) -> dict[str, Any]:
    stored = _get(sid)
    if stored.approval is None:
        raise HTTPException(status_code=409, detail="approve first")
    try:
        preview = compose_preview(
            stored.session,
            approval=stored.approval,
            to_address=body.to or default_mail_to(),
        )
    except (MailError, ApprovalRefused) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _public(sid, stored, preview=preview)


@app.post("/v1/sessions/{sid}/mail/send")
def send_mail(sid: str, body: MailBody) -> dict[str, Any]:
    stored = _get(sid)
    if stored.approval is None:
        raise HTTPException(status_code=409, detail="approve first")
    try:
        preview = compose_preview(
            stored.session,
            approval=stored.approval,
            to_address=body.to or default_mail_to(),
        )
        result = send_preview(
            preview,
            client=getattr(app.state, "mail_client", None),
            record=stored.approval.record,
        )
    except (MailError, ApprovalRefused) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    stored.sent = {
        "message_id": result.message_id,
        "document_id": result.document_id,
        "document_url": result.document_url,
    }
    return _public(sid, stored, preview=preview)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "support_reply.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
