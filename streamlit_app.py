"""Support-reply desk styled to match .stitch/designs intake + result screens."""

from __future__ import annotations

import html

import streamlit as st

from support_reply.contracts import Session
from support_reply.gate import ApprovalRefused, approve, approvable
from support_reply.graph import handle_complaint
from support_reply.mail import (
    MailError,
    compose_preview,
    default_mail_to,
    send_preview,
)
from support_reply.trace import Trace

st.set_page_config(page_title="Support reply", layout="wide")

st.html(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  html, body, [data-testid="stAppViewContainer"], .stApp { background: #faf8ff !important; font-family: Inter, sans-serif; }
  [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
  .block-container { padding: 0.75rem 2rem 2.5rem !important; max-width: 1120px !important; }
  [data-testid="stMainBlockContainer"] { background: #faf8ff; }
  .stButton > button[kind="primary"] {
    background: #00d09c !important; color: #06231a !important; border: 0 !important;
    font-weight: 600 !important; height: 2.5rem !important;
  }
  .stButton > button[kind="primary"]:hover { background: #00b887 !important; }
  .stButton > button[kind="secondary"] {
    background: #fff !important; color: #131b2e !important;
    border: 1px solid #cbd5e1 !important; font-weight: 600 !important; height: 2.5rem !important;
  }
  textarea, .stTextArea textarea {
    font-size: 1rem !important; line-height: 1.625rem !important; color: #131b2e !important;
  }
</style>
"""
)

if "session" not in st.session_state:
    st.session_state.session = None
    st.session_state.trace_text = ""
    st.session_state.approval = None
    st.session_state.send_result = None


def _run(complaint: str) -> None:
    trace = Trace()
    st.session_state.session = handle_complaint(Session(), complaint, trace=trace)
    st.session_state.trace_text = trace.render()
    st.session_state.approval = None
    st.session_state.send_result = None


def _header(*, clear: bool) -> None:
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        st.html(
            """
<div style="display:flex;align-items:center;gap:12px;padding:4px 0 12px;border-bottom:1px solid #bacac1;margin-bottom:8px">
  <div style="width:28px;height:28px;border-radius:8px;background:#006c4f;color:#fff;display:flex;align-items:center;justify-content:center;font-size:14px">↗</div>
  <div style="display:flex;align-items:baseline;gap:8px">
    <span style="font-size:1.125rem;font-weight:600;color:#131b2e">Support reply</span>
    <span style="font-size:0.6875rem;letter-spacing:0.04em;color:#545f73;border:1px solid #bacac1;background:#eaedff;padding:2px 6px;border-radius:4px">MF OPS</span>
  </div>
</div>
"""
        )
    with right:
        if clear and st.button("Clear", type="secondary"):
            st.session_state.session = None
            st.session_state.approval = None
            st.session_state.send_result = None
            st.rerun()


@st.dialog("Mail preview")
def _mail_dialog() -> None:
    session = st.session_state.session
    approval = st.session_state.approval
    if session is None or approval is None:
        st.stop()
    to_addr = st.text_input("To", value=default_mail_to() or "investor.support@groww.in")
    preview = compose_preview(session, approval=approval, to_address=to_addr)
    st.text_input("Subject", value=preview.subject, disabled=True)
    st.text_area("Body", value=preview.body, height=140, disabled=True)
    st.caption("Saves nothing extra until Send. Sends from the MCP Gmail workspace account.")
    cancel, send = st.columns(2)
    with cancel:
        if st.button("Cancel", type="secondary", width="stretch"):
            st.rerun()
    with send:
        if st.button("Send", type="primary", width="stretch"):
            try:
                st.session_state.send_result = send_preview(
                    compose_preview(session, approval=approval, to_address=to_addr),
                    record=approval.record,
                )
                st.rerun()
            except MailError as exc:
                st.error(str(exc))


_header(clear=st.session_state.session is not None)

if st.session_state.session is None:
    st.markdown("### Paste a complaint, then run.")
    st.caption("Direct complaint ingestion and contextual query resolver for mutual fund operations.")
    complaint = st.text_area(
        "Customer message",
        value="Why was I charged an exit load on HDFC Large Cap Fund?",
        height=180,
        help="PAN and phone numbers are masked before agents run.",
    )
    hint, run = st.columns([3, 1], vertical_alignment="bottom")
    with hint:
        words = len(complaint.split()) if complaint.strip() else 0
        st.caption(
            f"PAN and phone numbers are masked before agents run.  ·  {words} words  ·  {len(complaint)} characters"
        )
    with run:
        if st.button("Run", type="primary", width="stretch"):
            if not complaint.strip():
                st.warning("Paste a complaint first.")
            else:
                with st.spinner("Facts, Pattern, and Compliance starting together…"):
                    _run(complaint.strip())
                st.rerun()
    st.stop()

session = st.session_state.session
review = session.review
draft = session.draft
note = session.routing_note
pattern = session.pattern
status = review.status if review else "—"
passed = review is not None and review.status == "pass"
blocked = review is not None and review.status == "blocked"
open_ticket = bool(draft and draft.disposition in {"escalate", "disagree"})

st.html(
    f"""
<div style="display:flex;align-items:center;gap:10px;margin:8px 0 18px">
  <span style="font-size:0.75rem;font-weight:600;color:#047857;background:#ecfdf5;border:1px solid #a7f3d0;padding:2px 8px;border-radius:4px">Reviewer {html.escape(status)}</span>
</div>
"""
)

left, right = st.columns([7, 5], gap="large")
with left:
    st.markdown("#### Reply")
    st.caption("Draft message to customer")
    with st.container(border=True):
        st.write(draft.reply if draft else "")
        if draft and draft.citation:
            st.markdown(f"Primary source: [{draft.citation}]({draft.citation})")
        st.caption(
            "SEBI scheme disclosure validation: **Compliant**"
            if passed
            else f"Review: {status}"
        )
with right:
    st.markdown("#### Internal")
    st.caption("Operational routing metadata")
    with st.container(border=True):
        if note:
            a, b = st.columns(2)
            a.caption("Team")
            b.markdown(f"**{note.team}**")
            a, b = st.columns(2)
            a.caption("Priority")
            b.markdown(f"`{note.priority}`")
            a, b = st.columns(2)
            a.caption("Theme")
            theme = (
                f"{pattern.theme_name} · {pattern.count} reviews" if pattern else "—"
            )
            b.write(theme)
            a, b = st.columns(2)
            a.caption("Status line")
            b.markdown("**Ticket stays open**" if open_ticket else "May close after reply")
        st.caption("Facts · Pattern · Compliance  ·  3 specialists executed")
        st.caption("Internal is not emailed.")

bar_l, bar_r = st.columns([3, 2], vertical_alignment="center")
with bar_l:
    if blocked:
        st.error("Blocked. Approve is not available.")
    elif st.session_state.approval is not None:
        st.success("Approved. Draft saved. Does not send until Preview mail → Send.")
    else:
        st.caption("Ready for supervisor sign-off")
        st.caption("Saves a draft. Does not send.")
with bar_r:
    pcol, acol = st.columns(2)
    with pcol:
        preview_ok = st.session_state.approval is not None
        if st.button("Preview mail", type="secondary", width="stretch", disabled=not preview_ok):
            _mail_dialog()
    with acol:
        if blocked:
            st.button("Approve", type="primary", width="stretch", disabled=True)
        elif st.session_state.approval is None and approvable(session):
            if st.button("Approve", type="primary", width="stretch"):
                try:
                    st.session_state.approval = approve(session, human_approved=True)
                    st.rerun()
                except ApprovalRefused as exc:
                    st.error(str(exc))
        else:
            st.button("Approved", type="primary", width="stretch", disabled=True)

if st.session_state.send_result is not None:
    sent = st.session_state.send_result
    st.success(f"Sent. message_id={sent.message_id or '—'}  document_id={sent.document_id or '—'}")

with st.expander("Trace"):
    st.code(st.session_state.trace_text or "—", language="text")
