"""Phase 0 contract exit: legal citations, and no raw complaint string on Session."""

import logging

import pytest

from support_reply.contracts import (
    FULL_WINDOW,
    RECENT_WINDOW,
    AgentsFired,
    ComplianceResult,
    Draft,
    FactsResult,
    PatternResult,
    ReviewFailure,
    ReviewerResult,
    RoutingNote,
    Session,
)
from support_reply.corpus import load as load_corpus
from support_reply.themes import load as load_themes

RAW_PAN = "ABCDE1234F"
REPLY_MARKER = "REPLY-BODY-9f3a"
WHY_MARKER = "WHY-BODY-9f3a"
SPAN_MARKER = "SPAN-9f3a"


def _covered_facts() -> FactsResult:
    scheme = load_corpus().scheme("large_cap")
    block = scheme.section("Exit load")
    return FactsResult(
        status="covered",
        scheme_id="large_cap",
        section_title="Exit load",
        fact=block.text,
        citation=scheme.source_url,
        last_checked=scheme.nav_date,
        searched=("large_cap", "mid_cap", "small_cap", "gold_fof", "silver_fof"),
    )


def _pattern() -> PatternResult:
    theme = load_themes().theme("fees_brokerage")
    return PatternResult(
        theme_id=theme.id,
        theme_name=theme.name,
        signal="recurring",
        count=theme.count,
        recent_count=12,
        window=FULL_WINDOW,
        recent_window=RECENT_WINDOW,
    )


def _draft() -> Draft:
    scheme = load_corpus().scheme("large_cap")
    return Draft(
        reply=f"{REPLY_MARKER} Last checked {scheme.nav_date}. Source: {scheme.source_url}",
        last_checked=scheme.nav_date,
        citation=scheme.source_url,
        disposition="escalate",
        disagreement=False,
        routing_note=RoutingNote(
            team="Fees and mutual fund ops",
            priority="P2",
            why=WHY_MARKER,
        ),
    )


def test_facts_result_rejects_citation_outside_corpus():
    scheme = load_corpus().scheme("large_cap")
    block = scheme.section("Exit load")

    with pytest.raises(ValueError, match="source_url"):
        FactsResult(
            status="covered",
            scheme_id="large_cap",
            section_title="Exit load",
            fact=block.text,
            citation="https://groww.in/mutual-funds/not-a-real-fund",
            last_checked=scheme.nav_date,
            searched=("large_cap",),
        )


def test_facts_result_rejects_another_schemes_url():
    large = load_corpus().scheme("large_cap")
    mid = load_corpus().scheme("mid_cap")
    block = large.section("Exit load")

    with pytest.raises(ValueError, match="source_url"):
        FactsResult(
            status="covered",
            scheme_id="large_cap",
            section_title="Exit load",
            fact=block.text,
            citation=mid.source_url,
            last_checked=large.nav_date,
            searched=("large_cap",),
        )


def test_covered_facts_result_uses_the_scheme_text_block():
    result = _covered_facts()
    payload = result.to_dict()

    assert payload["status"] == "covered"
    assert payload["citation"] == load_corpus().scheme("large_cap").source_url
    assert payload["fact"] == result.fact
    assert result.fact not in repr(result)
    assert "large_cap" in repr(result)


def test_not_covered_fact_has_null_citation():
    result = FactsResult(
        status="not_covered",
        scheme_id=None,
        section_title=None,
        fact=None,
        citation=None,
        last_checked=None,
        searched=("large_cap", "Exit load"),
    )

    assert result.to_dict()["citation"] is None
    assert result.to_dict()["fact"] is None


def test_session_rejects_a_raw_string_on_any_other_name():
    session = Session(redacted_text="Why was I charged an exit load on HDFC Large Cap Fund?")
    raw = f"raw complaint {RAW_PAN}"

    for name in ("complaint", "raw_text", "raw_complaint", "message", "user_text"):
        with pytest.raises(AttributeError, match="redacted_text"):
            setattr(session, name, raw)

    for name in Session.__dataclass_fields__:
        if name == "redacted_text":
            continue
        with pytest.raises(TypeError, match="redacted_text"):
            setattr(session, name, raw)

    with pytest.raises(ValueError, match="redacted_text"):
        Session.from_dict({"redacted_text": "ok", "complaint": raw})

    assert RAW_PAN not in repr(session)
    assert "complaint" not in session.to_dict()
    session.redacted_text = "redacted only"
    assert session.redacted_text == "redacted only"


def test_session_field_bounds():
    session = Session()
    session.advice_refusal = True
    with pytest.raises(ValueError, match="sticky"):
        session.advice_refusal = False

    with pytest.raises(ValueError, match="refinement_count"):
        Session(refinement_count=6)
    with pytest.raises(ValueError, match="refinement_count"):
        session.refinement_count = -1


def test_repr_and_logs_keep_redacted_text_and_ids(caplog):
    redacted = "Why was I charged an exit load on HDFC Large Cap Fund?"
    facts = _covered_facts()
    review = ReviewerResult(
        status="fail",
        round=1,
        failures=(ReviewFailure(check="pii", detail=f"{SPAN_MARKER} {RAW_PAN}"),),
    )
    session = Session(
        redacted_text=redacted,
        pii_types=("PAN",),
        facts=facts,
        pattern=_pattern(),
        compliance=ComplianceResult(
            flags=(),
            pii_types=(),
            advice_refusal=False,
            redacted_text=redacted,
        ),
        draft=_draft(),
        routing_note=_draft().routing_note,
        review=review,
        agents_fired=(
            AgentsFired(
                round=1,
                kind="reviewer_rework",
                ran=("orchestrator",),
                skipped=("facts", "pattern", "compliance"),
            ),
        ),
    )

    logged = " ".join(
        (
            repr(session),
            repr(facts),
            repr(session.pattern),
            repr(session.compliance),
            repr(session.draft),
            repr(session.review),
            repr(review.failures[0]),
        )
    )
    assert redacted in logged
    assert "large_cap" in logged
    assert "fees_brokerage" in logged
    assert REPLY_MARKER not in logged
    assert WHY_MARKER not in logged
    assert SPAN_MARKER not in logged
    assert RAW_PAN not in logged
    assert facts.fact not in logged

    caplog.set_level(logging.INFO)
    logging.getLogger("support_reply.contracts").info("%s", session)
    assert redacted in caplog.text
    assert REPLY_MARKER not in caplog.text
    assert WHY_MARKER not in caplog.text
    assert RAW_PAN not in caplog.text
    assert facts.fact not in caplog.text
