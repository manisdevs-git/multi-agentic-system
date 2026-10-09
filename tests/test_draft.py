"""Phase 5 exit: disposition precedence, the three-sentence reply, and the routing note."""

import ast
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from support_reply.compliance import answer as compliance_answer
from support_reply.contracts import FULL_WINDOW, RECENT_WINDOW, Draft, PatternResult, RoutingNote
from support_reply.draft import (
    REFUSAL_REPLY,
    SENTENCE_LIMIT,
    compose,
    disposition_for,
    fact_sentence,
    priority_for,
    sentences,
    team_for,
)
from support_reply.facts import answer as facts_answer
from support_reply.ingress import redact
from support_reply.pattern import answer as pattern_answer
from support_reply.themes import ThemeIndex, load as load_themes

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
EXIT_LOAD_FACT = "Exit load of 1% if redeemed within 1 year"
LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
NIPPON_ORDER = "My order failed on Nippon Small Cap"
SHOULD_I_REDEEM = "Should I redeem HDFC Large Cap?"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"


def _draft(text: str, **kwargs) -> Draft:
    return compose(
        facts_answer(text), pattern_answer(text), compliance_answer(text), text, **kwargs
    )


def _rising_fees() -> PatternResult:
    """fees_brokerage with 40 reviews in the recent slice, as in the Pattern tests."""
    real = load_themes().theme("fees_brokerage")
    review_ids = tuple(f"fixture-{i}" for i in range(real.count))
    dates = {
        review_id: date(2026, 6, 28) if i < 40 else date(2026, 4, 12)
        for i, review_id in enumerate(review_ids)
    }
    index = ThemeIndex(
        review_count=real.count,
        all_themes=(replace(real, review_ids=review_ids),),
        top_theme_ids=(),
        review_dates=dates,
    )
    return pattern_answer(EXIT_LOAD, index=index)


def test_architecture_walkthrough_is_three_sentences_cited_and_kept_open():
    draft = _draft(EXIT_LOAD)
    pattern = pattern_answer(EXIT_LOAD)

    assert len(sentences(draft.reply)) == SENTENCE_LIMIT
    assert EXIT_LOAD_FACT in draft.reply
    assert "Last checked 25-Jun-2026." in draft.reply
    assert draft.reply.endswith(f"Source: {LARGE_CAP_URL}")
    assert draft.reply.count("https://") == 1
    assert draft.citation == LARGE_CAP_URL
    assert draft.last_checked == "25-Jun-2026"

    assert pattern.signal in {"recurring", "rising"}
    expected = "escalate" if pattern.signal == "recurring" else "disagree"
    assert draft.disposition == expected
    assert draft.disagreement is (expected == "disagree")

    note = draft.routing_note
    assert note.team == "Fees and mutual fund ops"
    assert note.priority == ("P2" if pattern.signal == "recurring" else "P1")
    assert "stays open" in note.why
    for token in ("fees_brokerage", pattern.signal, "136", "2026-04-12", "2026-07-04"):
        assert token in note.why
    assert "large_cap" in note.why
    assert Draft.from_dict(draft.to_dict()) == draft


def test_walkthrough_matches_the_real_recent_count():
    draft = _draft(EXIT_LOAD)

    assert draft.disposition == "escalate"
    assert draft.routing_note.priority == "P2"
    assert draft.routing_note.why == (
        "Facts covered for large_cap Exit load. Pattern fees_brokerage is recurring "
        "across 136 reviews from 2026-04-12 to 2026-07-04, so this ticket stays open."
    )


def test_rising_pattern_on_a_covered_fact_is_disagree_with_the_recent_count():
    pattern = _rising_fees()
    draft = compose(facts_answer(EXIT_LOAD), pattern, compliance_answer(EXIT_LOAD), EXIT_LOAD)

    assert pattern.signal == "rising"
    assert draft.disposition == "disagree"
    assert draft.disagreement is True
    assert draft.citation == LARGE_CAP_URL
    assert EXIT_LOAD_FACT in draft.reply
    assert len(sentences(draft.reply)) == SENTENCE_LIMIT
    assert draft.routing_note.priority == "P1"
    assert "stays open" in draft.routing_note.why
    assert "40 of them from 2026-06-28 to 2026-07-04" in draft.routing_note.why
    assert "rising" in draft.routing_note.why


def test_covered_one_off_clean_is_reply_routed_to_mutual_fund_support():
    text = "What is the benchmark of HDFC Mid Cap Fund?"
    draft = _draft(text)

    assert pattern_answer(text).theme_id == "other"
    assert draft.disposition == "reply"
    assert draft.disagreement is False
    assert draft.reply.startswith("Fund benchmark: NIFTY Midcap 150 Total Return Index.")
    assert draft.citation == "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
    assert len(sentences(draft.reply)) == SENTENCE_LIMIT
    assert draft.routing_note.team == "Mutual fund support (mid_cap)"
    assert draft.routing_note.priority == "P3"
    assert "stays open" not in draft.routing_note.why
    assert "other" in draft.routing_note.why and "one-off" in draft.routing_note.why


def test_corpus_miss_is_handoff_with_no_url_and_the_searched_names():
    facts = facts_answer(NIPPON_ORDER)
    draft = _draft(NIPPON_ORDER)

    assert facts.status == "not_covered"
    assert draft.disposition == "handoff"
    assert draft.citation is None
    assert draft.last_checked is None
    assert "http" not in draft.reply
    assert "groww.in" not in draft.reply
    assert len(sentences(draft.reply)) == SENTENCE_LIMIT
    for scheme_id in ("large_cap", "mid_cap", "small_cap", "gold_fof", "silver_fof"):
        assert scheme_id in draft.reply
    assert "Exit load" in draft.reply and "Key fund facts" in draft.reply
    assert "do not cover this question" in draft.reply
    # No invented fact: nothing from any text block, no NAV, no exit-load rule.
    assert "1%" not in draft.reply
    assert "NAV" not in draft.reply
    assert draft.routing_note.team == "Trading platform"
    assert "not_covered" in draft.routing_note.why
    assert "trading_charts" in draft.routing_note.why


def test_should_i_redeem_is_refuse_and_does_not_quote_the_exit_load_rule():
    # Facts could cover exit load. Compliance says advice. The refusal wins.
    facts = facts_answer(EXIT_LOAD)
    compliance = compliance_answer(SHOULD_I_REDEEM)
    draft = compose(facts, pattern_answer(EXIT_LOAD), compliance, SHOULD_I_REDEEM)

    assert facts.status == "covered"
    assert compliance.advice_refusal is True
    assert draft.disposition == "refuse"
    assert draft.disagreement is False
    assert draft.reply == REFUSAL_REPLY
    assert EXIT_LOAD_FACT not in draft.reply
    assert "exit load" not in draft.reply.lower()
    assert draft.citation is None
    assert draft.last_checked is None
    assert "http" not in draft.reply
    assert len(sentences(draft.reply)) <= SENTENCE_LIMIT
    assert "advice" in draft.routing_note.why and "refuses" in draft.routing_note.why


def test_refusal_text_carries_no_figure_and_no_action_word():
    assert not any(character.isdigit() for character in REFUSAL_REPLY)
    lowered = REFUSAL_REPLY.lower()
    for word in ("buy", "sell", "redeem", "hold", "cagr", "return", "%"):
        assert word not in lowered


@pytest.mark.parametrize(
    "text",
    [
        "What CAGR should I expect from HDFC Large Cap Fund?",
        "What returns should I expect?",
        "Is HDFC Large Cap a good fund?",
        "Should I redeem HDFC Large Cap before the exit load period ends?",
    ],
)
def test_advice_and_performance_asks_refuse(text):
    draft = _draft(text)

    assert draft.disposition == "refuse"
    assert draft.citation is None
    assert "http" not in draft.reply
    assert "%" not in draft.reply


def test_sticky_session_refusal_overrides_a_clean_covered_turn():
    clean = _draft(EXIT_LOAD)
    sticky = _draft(EXIT_LOAD, advice_refusal=True)

    assert clean.disposition == "escalate"
    assert sticky.disposition == "refuse"
    assert sticky.reply == REFUSAL_REPLY
    assert EXIT_LOAD_FACT not in sticky.reply
    assert sticky.citation is None
    assert "sticky" in sticky.routing_note.why


def test_pii_flag_alone_does_not_change_the_disposition_and_nothing_raw_leaks():
    ingress = redact(f"{EXIT_LOAD} My PAN is {RAW_PAN} and phone {RAW_PHONE}.")
    facts = facts_answer(ingress.redacted_text)
    pattern = pattern_answer(ingress.redacted_text)
    compliance = compliance_answer(ingress.redacted_text, ingress.pii_types)

    draft = compose(facts, pattern, compliance, ingress.redacted_text)

    assert compliance.flags == ("pii",)
    assert draft.disposition == "escalate"
    assert draft.citation == LARGE_CAP_URL
    serialized = str(draft.to_dict()) + repr(draft)
    assert RAW_PAN not in serialized
    assert RAW_PHONE not in serialized
    assert "[PAN]" not in draft.reply  # the reply quotes the corpus, not the complaint
    assert "PAN" in draft.routing_note.why and "PHONE" in draft.routing_note.why


def test_disposition_precedence_table():
    covered = facts_answer(EXIT_LOAD)
    missed = facts_answer(NIPPON_ORDER)
    recurring = pattern_answer(EXIT_LOAD)
    rising = _rising_fees()
    one_off = pattern_answer("The mutual fund page looks fine")
    clean = compliance_answer(EXIT_LOAD)
    advice = compliance_answer(SHOULD_I_REDEEM)

    assert disposition_for(covered, one_off, advice) == "refuse"
    assert disposition_for(missed, rising, advice) == "refuse"
    assert disposition_for(covered, rising, clean, advice_refusal=True) == "refuse"
    assert disposition_for(missed, rising, clean) == "handoff"
    assert disposition_for(missed, one_off, clean) == "handoff"
    assert disposition_for(covered, rising, clean) == "disagree"
    assert disposition_for(covered, recurring, clean) == "escalate"
    assert disposition_for(covered, one_off, clean) == "reply"


@pytest.mark.parametrize(
    "text,team",
    [
        ("Why was I charged an exit load on Nippon Small Cap?", "Fees and mutual fund ops"),
        ("My UPI payment to Nippon Small Cap is stuck", "Payments"),
        (NIPPON_ORDER, "Trading platform"),
        ("The app crashed on login", "App stability"),
        ("The Nippon fund page looks fine", "General support"),
    ],
)
def test_team_follows_the_theme_when_facts_are_not_covered(text, team):
    draft = _draft(text)

    assert draft.disposition == "handoff"
    assert draft.routing_note.team == team


def test_covered_fact_with_theme_other_routes_to_mutual_fund_support_by_scheme_id():
    text = "What is the exit load on HDFC Gold ETF Fund of Fund?"
    facts = facts_answer(text)
    other = pattern_answer("The mutual fund page looks fine")

    assert facts.scheme_id == "gold_fof"
    assert team_for(facts, other) == "Mutual fund support (gold_fof)"
    assert team_for(facts, pattern_answer(text)) == "Fees and mutual fund ops"


def test_priority_follows_the_signal_with_a_p2_floor_for_open_tickets():
    recurring = pattern_answer(EXIT_LOAD)
    rising = _rising_fees()
    one_off = pattern_answer("The mutual fund page looks fine")

    assert priority_for("reply", one_off) == "P3"
    assert priority_for("handoff", one_off) == "P3"
    assert priority_for("refuse", recurring) == "P2"
    assert priority_for("escalate", recurring) == "P2"
    assert priority_for("disagree", rising) == "P1"
    assert priority_for("handoff", rising) == "P1"
    assert priority_for("escalate", one_off) == "P2"
    assert priority_for("disagree", one_off) == "P2"


@pytest.mark.parametrize(
    "text,quoted",
    [
        ("What is the expense ratio of HDFC Small Cap?", "Expense ratio: 0.77%."),
        ("What is the NAV of HDFC Gold ETF Fund of Fund?", "NAV: 43.279."),
        ("How is HDFC Small Cap taxed?", "If you redeem within one year, returns are taxed at 20%."),
        ("What is the stamp duty on HDFC Mid Cap?", "Stamp duty on investment: 0.005% (from July 1st, 2020)."),
    ],
)
def test_multi_sentence_text_block_quotes_the_matching_sentence_within_the_budget(text, quoted):
    facts = facts_answer(text)
    draft = _draft(text)

    assert facts.status == "covered"
    assert quoted in facts.fact
    assert draft.reply.startswith(quoted)
    assert len(sentences(draft.reply)) == SENTENCE_LIMIT
    # Every word of the quoted sentence is in the text block; nothing is added.
    assert draft.reply.split(" Last checked ")[0] in facts.fact


def test_fact_sentence_keeps_the_first_sentence_when_nothing_overlaps_and_adds_a_period():
    assert fact_sentence(EXIT_LOAD_FACT, "zzz") == EXIT_LOAD_FACT + "."
    assert fact_sentence("Already ends.", "zzz") == "Already ends."
    assert fact_sentence("One. Two. Three.", "nothing here") == "One."
    assert fact_sentence("One. Two. Three.", "give me two") == "Two."


def test_sentences_do_not_split_inside_urls_decimals_or_abbreviations():
    reply = (
        "Mr. Baijal manages it. Rs 1.25 lakh is the limit. Last checked 25-Jun-2026. "
        f"Source: {LARGE_CAP_URL}"
    )

    assert sentences(reply) == [
        "Mr. Baijal manages it.",
        "Rs 1.25 lakh is the limit.",
        "Last checked 25-Jun-2026.",
        f"Source: {LARGE_CAP_URL}",
    ]
    assert sentences("Education Mr. Setalvad is a B. Sc and MBA. Next one.") == [
        "Education Mr. Setalvad is a B. Sc and MBA.",
        "Next one.",
    ]
    assert sentences("NAV: 1217.44. NAV date: 25-Jun-2026.") == ["NAV: 1217.44.", "NAV date: 25-Jun-2026."]
    assert sentences("") == []
    with pytest.raises(TypeError):
        sentences(None)  # type: ignore[arg-type]


def test_routing_note_is_a_contract_and_is_separate_from_the_reply():
    draft = _draft(EXIT_LOAD)

    assert isinstance(draft.routing_note, RoutingNote)
    assert draft.routing_note.why not in draft.reply
    assert "Fees and mutual fund ops" not in draft.reply
    assert RoutingNote.from_dict(draft.routing_note.to_dict()) == draft.routing_note


def test_draft_module_does_not_open_the_corpus_the_themes_or_the_agents():
    import support_reply.draft as draft_module

    tree = ast.parse(Path(draft_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)

    forbidden = ("corpus", "themes", "facts", "pattern", "compliance", "llm", "groq", "langgraph")
    assert not any(
        name in forbidden or name.endswith(tuple(f".{item}" for item in forbidden))
        for name in imported
    )


def test_draft_rejects_bad_input():
    facts = facts_answer(EXIT_LOAD)
    pattern = pattern_answer(EXIT_LOAD)
    compliance = compliance_answer(EXIT_LOAD)

    with pytest.raises(TypeError):
        compose(None, pattern, compliance, EXIT_LOAD)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compose(facts, None, compliance, EXIT_LOAD)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compose(facts, pattern, None, EXIT_LOAD)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compose(facts, pattern, compliance, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compose(facts, pattern, compliance, EXIT_LOAD, advice_refusal="yes")  # type: ignore[arg-type]
