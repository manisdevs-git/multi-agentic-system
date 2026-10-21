"""Offline eval suite: docs/edgecases.md implemented as evals/cases.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_reply.contracts import Session
from support_reply.evals import (
    CASES_PATH,
    EvalCase,
    run_evals,
    score_session,
)
from support_reply.graph import handle_complaint


def test_cases_file_is_the_eval_catalog():
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    ids = [item["id"] for item in payload["cases"]]
    assert "gold_fof_alleged_2pct" in ids
    assert "gold_fof_user_gave_phrasing" in ids
    assert "other_amc_with_hdfc" in ids
    assert "advice_then_gold_exit_load" in [item["id"] for item in payload["threads"]]
    assert (Path("docs") / "edgecases.md").is_file()
    assert (Path("docs") / "evals.md").is_file()


def test_offline_evals_all_pass(monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    report = run_evals()
    assert report.ok, report.failures


def test_scorer_fails_when_reply_confirms_alleged_two_percent():
    session = handle_complaint(
        Session(),
        "Why 2% charged for exit load on HDFC Gold ETF Fund of Fund Direct Plan Growth?",
    )
    case = EvalCase(
        id="planted_miss",
        category="alleged_figure",
        complaint=session.redacted_text,
        expect={"reply_contains": ["charged 2% as the exit load"]},
    )
    result = score_session(case, session)
    assert result.ok is False
    assert result.failures[0].check == "reply_contains"


def test_gold_two_percent_is_not_advice():
    session = handle_complaint(
        Session(),
        "I gave HDFC Gold ETF Fund of Fund Direct Plan Growth and asked why 2% charged for exit load",
    )
    assert session.draft is not None
    assert session.draft.disposition != "refuse"
    assert "1%, if redeemed within 15 days" in session.draft.reply
    assert session.advice_refusal is False
