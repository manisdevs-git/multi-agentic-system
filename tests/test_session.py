"""Phase 8 exit: five refinements, selective re-runs, sticky refusal, and the sixth pushback.

Every agent is reached through the compiled graph. A refinement starts only the agents
the pushback names and copies the others from the session.
"""

import json
import logging
from dataclasses import replace

import pytest

import support_reply.graph as graph_module
from support_reply.contracts import AgentsFired, Session
from support_reply.draft import REFUSAL_REPLY
from support_reply.graph import REWRITE_NODE, handle_complaint, run
from support_reply.session import (
    MAX_REFINEMENTS,
    Pushback,
    RefinementRefused,
    classify_pushback,
    refine,
    schemes_named,
)
from support_reply.trace import AGENT_NODES, Trace

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
EXIT_LOAD_FACT = "Exit load of 1% if redeemed within 1 year"
GOLD_EXIT_LOAD_FACT = "Exit load of 1%, if redeemed within 15 days."
LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
GOLD_URL = "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth"
MID_CAP_URL = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
MID_CAP_BENCHMARK = "What is the benchmark for HDFC Mid Cap?"
SHOULD_I_REDEEM = "Should I redeem HDFC Large Cap?"
GOLD_PUSHBACK = "They're asking about HDFC Gold ETF Fund of Fund"
FEE_PUSHBACK = "Check if this is the fee issue from last week"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"
ADVICE = "You may want to stay invested."

ORCHESTRATOR_ONLY = AgentsFired(
    round=1, kind="refinement", ran=("orchestrator",), skipped=AGENT_NODES
)


def _reviewed(text=EXIT_LOAD):
    session = handle_complaint(Session(), text)
    assert session.review.status == "pass"
    return session


def _refine(session, pushback, **kwargs):
    trace = Trace()
    refine(session, pushback, trace=trace, **kwargs)
    return trace


def _started(trace):
    return tuple(node for node in AGENT_NODES if trace.starts(node))


# --- The routing table --------------------------------------------------------------------


def test_shorter_logs_orchestrator_only_and_keeps_every_result():
    session = _reviewed()
    before = (session.facts, session.pattern, session.compliance, session.draft)

    trace = _refine(session, "Shorter")

    assert session.refinement_count == 1
    assert session.agents_fired == (ORCHESTRATOR_ONLY,)
    assert (session.facts, session.pattern, session.compliance, session.draft) == before
    assert session.review.status == "pass"
    assert session.redacted_text == EXIT_LOAD
    assert _started(trace) == ()
    assert trace.sequence() == (
        ("start", "ingress"),
        ("return", "ingress"),
        ("start", "draft"),
        ("return", "draft"),
        ("start", "reviewer"),
        ("return", "reviewer"),
    )
    assert trace.violations(agents=()) == []


def test_a_different_scheme_reruns_facts_cites_the_gold_url_and_skips_pattern():
    session = _reviewed()
    pattern_before, compliance_before = session.pattern, session.compliance

    trace = _refine(session, GOLD_PUSHBACK)

    assert session.facts.scheme_id == "gold_fof"
    assert session.facts.section_title == "Exit load"
    assert session.facts.citation == GOLD_URL
    assert session.draft.citation == GOLD_URL
    assert session.draft.reply.startswith(GOLD_EXIT_LOAD_FACT)
    assert session.draft.reply.endswith(f"Source: {GOLD_URL}")
    assert LARGE_CAP_URL not in session.draft.reply
    assert session.review.status == "pass"
    # Pattern and Compliance are copied, not re-run. The complaint text is unchanged.
    assert session.pattern == pattern_before
    assert session.compliance == compliance_before
    assert session.redacted_text == EXIT_LOAD
    assert _started(trace) == ("facts",)
    assert trace.violations(agents=("facts",)) == []
    assert session.agents_fired == (
        AgentsFired(
            round=1,
            kind="refinement",
            ran=("facts", "orchestrator"),
            skipped=("pattern", "compliance"),
        ),
    )
    assert session.agents_fired[0].to_dict() == {
        "round": 1,
        "kind": "refinement",
        "ran": ["facts", "orchestrator"],
        "skipped": ["pattern", "compliance"],
    }


def test_the_fee_issue_from_last_week_reruns_pattern_only():
    session = _reviewed(MID_CAP_BENCHMARK)
    assert session.pattern.theme_id == "other"
    assert session.draft.disposition == "reply"
    facts_before, compliance_before = session.facts, session.compliance

    trace = _refine(session, FEE_PUSHBACK)

    assert session.pattern.theme_id == "fees_brokerage"
    assert session.pattern.count == 136
    assert session.pattern.signal in {"recurring", "rising"}
    assert session.draft.disposition in {"escalate", "disagree"}
    assert "stays open" in session.routing_note.why
    assert session.routing_note.team == "Fees and mutual fund ops"
    # The fact, the citation, and the compliance result are the stored ones.
    assert session.facts == facts_before
    assert session.compliance == compliance_before
    assert session.draft.citation == MID_CAP_URL
    assert _started(trace) == ("pattern",)
    assert trace.violations(agents=("pattern",)) == []
    assert session.agents_fired == (
        AgentsFired(
            round=1,
            kind="refinement",
            ran=("pattern", "orchestrator"),
            skipped=("facts", "compliance"),
        ),
    )


def test_a_scheme_and_a_theme_rerun_facts_and_pattern_but_not_compliance():
    session = _reviewed()
    compliance_before = session.compliance

    trace = _refine(session, "They're asking about the withdrawal from the gold fund")

    assert session.facts.scheme_id == "gold_fof"
    assert session.pattern.theme_id == "money_movement"
    assert session.compliance == compliance_before
    assert set(_started(trace)) == {"facts", "pattern"}
    assert trace.violations(agents=("facts", "pattern")) == []
    assert session.agents_fired[-1] == AgentsFired(
        round=1,
        kind="refinement",
        ran=("facts", "pattern", "orchestrator"),
        skipped=("compliance",),
    )


def test_a_changed_complaint_reruns_compliance_as_well_and_replaces_the_text():
    session = _reviewed()
    new_text = f"My UPI deposit to HDFC Mid Cap failed. My phone is {RAW_PHONE}."

    trace = _refine(session, "The user wrote something else", complaint=new_text)

    assert _started(trace) == AGENT_NODES
    assert trace.violations() == []
    assert session.redacted_text == "My UPI deposit to HDFC Mid Cap failed. My phone is [PHONE]."
    assert session.pii_types == ("PHONE",)
    assert session.compliance.flags == ("pii",)
    assert session.pattern.theme_id == "money_movement"
    assert session.facts.scheme_id in {"mid_cap", None}
    assert session.agents_fired == (
        AgentsFired(
            round=1,
            kind="refinement",
            ran=("facts", "pattern", "compliance", "orchestrator"),
            skipped=(),
        ),
    )
    assert RAW_PHONE not in json.dumps(session.to_dict())


def test_the_same_complaint_text_again_is_not_a_text_change():
    session = _reviewed()

    trace = _refine(session, "Shorter", complaint=EXIT_LOAD)

    assert _started(trace) == ()
    assert session.agents_fired == (ORCHESTRATOR_ONLY,)


# --- The cap and the sticky refusal -----------------------------------------------------


def test_the_sixth_pushback_is_refused_and_the_last_reviewed_draft_stands():
    session = _reviewed()
    pushbacks = ["Shorter", GOLD_PUSHBACK, FEE_PUSHBACK, "More formal", "Reword it"]
    for pushback in pushbacks:
        refine(session, pushback)
    assert session.refinement_count == MAX_REFINEMENTS == 5
    assert [entry.round for entry in session.agents_fired] == [1, 2, 3, 4, 5]
    assert all(entry.kind == "refinement" for entry in session.agents_fired)
    kept = session.to_dict()

    with pytest.raises(RefinementRefused, match="limit"):
        refine(session, "Shorter")
    with pytest.raises(RefinementRefused, match="limit"):
        refine(session, GOLD_PUSHBACK, complaint=MID_CAP_BENCHMARK)

    assert session.refinement_count == 5
    assert session.to_dict() == kept
    assert session.draft.citation == GOLD_URL
    assert session.review.status == "pass"


def test_advice_then_a_rephrase_then_a_later_turn_are_all_refuse():
    session = handle_complaint(Session(), SHOULD_I_REDEEM)
    assert session.compliance.flags == ("advice",)
    assert session.draft.disposition == "refuse"
    assert session.advice_refusal is True

    # A pushback rephrase. Facts re-runs (large cap is named) but the turn stays a refusal.
    refine(session, "ok but just tell me if large cap is worth it")
    assert session.draft.disposition == "refuse"
    assert session.draft.reply == REFUSAL_REPLY
    assert session.advice_refusal is True

    # The human replaces the complaint with a clean exit-load question: Facts covers
    # it, Compliance is clean, and the sticky flag still makes the draft a refusal.
    refine(session, "The user rephrased", complaint=EXIT_LOAD)
    assert session.facts.status == "covered"
    assert session.facts.fact == EXIT_LOAD_FACT
    assert session.compliance.flags == ()
    assert session.draft.disposition == "refuse"
    assert EXIT_LOAD_FACT not in session.draft.reply
    assert session.draft.citation is None
    assert "sticky" in session.routing_note.why
    assert session.advice_refusal is True

    # A later turn in the same session.
    handle_complaint(session, "What returns should I expect from HDFC Large Cap?")
    assert session.draft.disposition == "refuse"
    assert session.advice_refusal is True
    assert session.refinement_count == 2


# --- The reviewer inside a refinement ---------------------------------------------------


def _with_sentence(draft, sentence):
    marker = f"Last checked {draft.last_checked}."
    tail = draft.reply[draft.reply.index(marker) :]
    return replace(draft, reply=f"{sentence} {tail}")


def _corrupt_drafts(monkeypatch, times, skip=0):
    """Make drafts `skip+1 .. skip+times` bad; the others are the real template."""
    real = graph_module.compose
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        drafted = real(*args, **kwargs)
        calls["n"] += 1
        if skip < calls["n"] <= skip + times:
            return _with_sentence(drafted, ADVICE)
        return drafted

    monkeypatch.setattr(graph_module, "compose", wrapped)
    return calls


def test_a_reviewer_rework_inside_a_refinement_logs_the_round_and_is_not_one_of_the_five(
    monkeypatch,
):
    session = _reviewed()
    _corrupt_drafts(monkeypatch, times=1)
    trace = Trace()

    refine(session, GOLD_PUSHBACK, trace=trace)

    assert session.refinement_count == 1
    assert session.review.status == "pass" and session.review.round == 2
    assert ADVICE not in session.draft.reply
    assert session.draft.citation == GOLD_URL
    assert session.agents_fired == (
        AgentsFired(
            round=1,
            kind="refinement",
            ran=("facts", "orchestrator"),
            skipped=("pattern", "compliance"),
        ),
        AgentsFired(
            round=1,
            kind="reviewer_rework",
            ran=("orchestrator",),
            skipped=AGENT_NODES,
        ),
    )
    assert len(trace.starts("facts")) == 1
    assert trace.starts("pattern") == () and trace.starts("compliance") == ()
    assert len(trace.starts("reviewer")) == 2 and len(trace.starts(REWRITE_NODE)) == 1
    assert trace.first("start", REWRITE_NODE).seq > trace.first("return", "facts").seq

    # Four more pushbacks are still available: the rework did not use one.
    for pushback in ["Shorter"] * 4:
        refine(session, pushback)
    assert session.refinement_count == 5
    with pytest.raises(RefinementRefused):
        refine(session, "Shorter")


def test_a_blocked_draft_can_be_refined_and_the_new_draft_is_reviewed_afresh(monkeypatch):
    _corrupt_drafts(monkeypatch, times=2)
    session = handle_complaint(Session(), EXIT_LOAD)
    assert session.review.status == "blocked"

    refine(session, "Shorter")

    assert session.review.status == "pass"
    assert ADVICE not in session.draft.reply
    assert [entry.kind for entry in session.agents_fired] == ["reviewer_rework", "refinement"]
    assert [entry.round for entry in session.agents_fired] == [0, 1]


# --- PII and logging --------------------------------------------------------------------


def test_a_pushback_with_pii_is_redacted_before_any_agent_sees_it(caplog):
    caplog.set_level(logging.INFO)
    session = _reviewed(f"{EXIT_LOAD} My PAN is {RAW_PAN}.")
    assert session.pii_types == ("PAN",)

    trace = _refine(session, f"Check the fee pattern, the user's phone is {RAW_PHONE}")

    assert _started(trace) == ("pattern",)
    assert session.pattern.theme_id == "fees_brokerage"
    assert session.pii_types == ("PAN",)
    assert session.review.status == "pass"
    everywhere = "\n".join([repr(session), json.dumps(session.to_dict()), caplog.text, trace.render()])
    assert "[PAN]" in everywhere and "[PHONE]" in caplog.text
    assert RAW_PAN not in everywhere and RAW_PHONE not in everywhere


def test_a_session_with_refinements_round_trips_through_its_dict():
    session = _reviewed()
    refine(session, GOLD_PUSHBACK)
    refine(session, "Shorter")

    restored = Session.from_dict(json.loads(json.dumps(session.to_dict())))

    assert restored == session
    assert restored.refinement_count == 2
    assert [entry.kind for entry in restored.agents_fired] == ["refinement", "refinement"]


# --- Classification ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pushback", "kind", "rerun"),
    [
        ("Shorter", "wording", ()),
        ("Make it more formal", "wording", ()),
        ("Too long, trim the second sentence", "wording", ()),
        (GOLD_PUSHBACK, "scheme", ("facts",)),
        ("It's the mid cap, not the large cap", "scheme", ("facts",)),
        ("They're asking about a different scheme", "scheme", ("facts",)),
        (FEE_PUSHBACK, "theme", ("pattern",)),
        ("Is this a wider pattern?", "theme", ("pattern",)),
        ("How many other users reported this recently?", "theme", ("pattern",)),
        ("This is about withdrawals", "theme", ("pattern",)),
        ("They're asking about the withdrawal from the gold fund", "scheme_and_theme", ("facts", "pattern")),
    ],
)
def test_classify_pushback_follows_the_table(pushback, kind, rerun):
    session = _reviewed()

    plan = classify_pushback(pushback, session)

    assert plan.kind == kind
    assert plan.rerun == rerun
    assert plan.ran == (*rerun, "orchestrator")
    assert plan.skipped == tuple(node for node in AGENT_NODES if node not in rerun)
    assert set(plan.ran) | set(plan.skipped) == {*AGENT_NODES, "orchestrator"}


def test_classify_pushback_builds_the_text_each_agent_classifies():
    session = _reviewed()

    gold = classify_pushback(GOLD_PUSHBACK, session)
    assert gold.scheme_id == "gold_fof"
    assert gold.facts_text == "Why was I charged an exit load on HDFC Gold ETF Fund of Fund?"
    assert gold.pattern_text is None
    assert gold.agent_text == {"facts": gold.facts_text}

    other = classify_pushback("They're asking about a different scheme", session)
    assert other.scheme_id is None
    assert other.facts_text == "Why was I charged an exit load on?"

    fee = classify_pushback(FEE_PUSHBACK, session)
    assert fee.pattern_text == FEE_PUSHBACK
    assert fee.facts_text is None

    window = classify_pushback("Is this a wider pattern?", session)
    assert window.pattern_text == EXIT_LOAD

    same = classify_pushback("Yes, HDFC Large Cap", session)
    assert same.kind == "scheme" and same.scheme_id == "large_cap"
    assert same.facts_text == EXIT_LOAD

    changed = classify_pushback(GOLD_PUSHBACK, session, complaint_changed=True)
    assert changed.kind == "complaint"
    assert changed.rerun == AGENT_NODES
    assert changed.facts_text is None and changed.pattern_text is None


def test_a_different_scheme_with_no_name_hands_off():
    session = _reviewed()

    refine(session, "They're asking about a different scheme")

    assert session.facts.status == "not_covered"
    assert session.draft.disposition == "handoff"
    assert "http" not in session.draft.reply
    assert session.agents_fired[-1].ran == ("facts", "orchestrator")


def test_a_complaint_without_a_scheme_gets_one_appended():
    session = _reviewed("What is the exit load?")
    assert session.facts.status == "not_covered"

    refine(session, "It's HDFC Small Cap Fund")

    assert session.facts.scheme_id == "small_cap"
    assert session.facts.section_title == "Exit load"
    assert session.draft.citation == "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


def test_schemes_named_and_the_pushback_contract():
    assert schemes_named("gold fund or the HDFC Silver ETF FoF, not large-cap") == (
        "gold_fof",
        "silver_fof",
        "large_cap",
    )
    assert schemes_named("golden hour, midcap") == ("mid_cap",)
    assert schemes_named("Shorter") == ()

    plan = classify_pushback(f"Check the fee issue, PAN {RAW_PAN}", _reviewed())
    assert plan.redacted_pushback == "Check the fee issue, PAN [PAN]"
    assert plan.pii_types == ("PAN",)
    assert RAW_PAN not in repr(plan) and RAW_PAN not in json.dumps(plan.to_dict())

    with pytest.raises(ValueError):
        Pushback(kind="refuse", rerun=(), redacted_pushback="x", pii_types=())
    with pytest.raises(ValueError):
        Pushback(kind="scheme", rerun=("reviewer",), redacted_pushback="x", pii_types=())
    with pytest.raises(ValueError):
        Pushback(kind="wording", rerun=(), redacted_pushback="x", pii_types=(), facts_text="y")


# --- Guards -----------------------------------------------------------------------------


def test_refine_rejects_bad_input_and_an_unreviewed_session():
    session = _reviewed()
    with pytest.raises(TypeError):
        refine("session", "Shorter")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        refine(session, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        refine(session, "   ")
    with pytest.raises(TypeError):
        refine(session, "Shorter", complaint=42)  # type: ignore[arg-type]
    with pytest.raises(RefinementRefused, match="no reviewed draft"):
        refine(Session(), "Shorter")
    with pytest.raises(TypeError):
        classify_pushback("Shorter", "session")  # type: ignore[arg-type]
    assert session.refinement_count == 0 and session.agents_fired == ()


def test_run_rejects_a_rerun_that_mixes_fresh_and_stored_results():
    session = _reviewed()
    stored = {"pattern": session.pattern, "compliance": session.compliance}

    with pytest.raises(ValueError, match="not both"):
        run(EXIT_LOAD, redacted_text=EXIT_LOAD)
    with pytest.raises(TypeError):
        run(None)
    with pytest.raises(ValueError, match="stored PatternResult"):
        run(None, redacted_text=EXIT_LOAD, rerun=("facts",), reuse={"compliance": session.compliance})
    with pytest.raises(ValueError, match="do not also pass"):
        run(None, redacted_text=EXIT_LOAD, rerun=("facts",), reuse={**stored, "facts": session.facts})
    with pytest.raises(ValueError, match="unknown agent"):
        run(None, redacted_text=EXIT_LOAD, rerun=("reviewer",), reuse=stored)
    with pytest.raises(ValueError, match="needs pattern in rerun"):
        run(None, redacted_text=EXIT_LOAD, rerun=("facts",), reuse=stored, agent_text={"pattern": "x"})
    with pytest.raises(ValueError, match="not 'compliance'"):
        run(None, redacted_text=EXIT_LOAD, rerun=AGENT_NODES, agent_text={"compliance": "x"})
    with pytest.raises(ValueError, match="only when rerun"):
        run(EXIT_LOAD, reuse=stored)

    # A valid subset run through the graph alone: the stored results come back as given.
    state = run(None, redacted_text=EXIT_LOAD, rerun=("facts",), reuse=stored, agent_text={"facts": "What is the benchmark for HDFC Gold ETF Fund of Fund?"})
    assert state["facts"].scheme_id == "gold_fof" and state["facts"].section_title == "Benchmark"
    assert state["pattern"] == session.pattern and state["compliance"] == session.compliance
    assert state["redacted_text"] == EXIT_LOAD
    assert "raw_text" not in state


def test_ingress_rescans_text_handed_in_as_redacted():
    session = _reviewed()
    leaked = f"{EXIT_LOAD} PAN {RAW_PAN}."

    state = run(
        None,
        redacted_text=leaked,
        pii_types=("PHONE",),
        rerun=(),
        reuse={"facts": session.facts, "pattern": session.pattern, "compliance": session.compliance},
    )

    assert state["redacted_text"] == f"{EXIT_LOAD} PAN [PAN]."
    assert state["pii_types"] == ("PHONE", "PAN")
    assert RAW_PAN not in json.dumps({k: str(v) for k, v in state.items()})
