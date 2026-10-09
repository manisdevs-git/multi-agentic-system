"""Approval gate guard. The record append and the email draft are Phase 9.

The human click is the only trigger, and the reviewer status must be `pass`.
`fail` and `blocked` sessions are not approvable: the UI hides Approve and Send,
and a programmatic call lands here and raises. Phase 9's `approve` calls
`require_approvable` first and writes nothing when it raises.
"""

from __future__ import annotations

from support_reply.contracts import Session


class ApprovalRefused(RuntimeError):
    """Raised when something tries to approve a session the gate does not accept."""


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
