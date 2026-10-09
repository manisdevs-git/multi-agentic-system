"""Phase 6 exit: one StateGraph runs ingress, three agents together, draft, reviewer.

Facts, Pattern, and Compliance are reached only through the compiled graph here.
No test calls `facts.answer`, `pattern.answer`, or `compliance.answer`.
"""

import ast
import logging
import threading
from pathlib import Path

import pytest

import support_reply.compliance as compliance_module
import support_reply.draft as draft_module
import support_reply.facts as facts_module
import support_reply.pattern as pattern_module
from support_reply.contracts import (
    ComplianceResult,
    Draft,
    FactsResult,
    PatternResult,
    ReviewerResult,
    Session,
)
from support_reply.draft import REFUSAL_REPLY
from support_reply.graph import (
    NODES,
    REWRITE_NODE,
    build_graph,
    compile_graph,
    handle_complaint,
    run,
)
from support_reply.trace import AGENT_NODES, Trace

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
EXIT_LOAD_FACT = "Exit load of 1% if redeemed within 1 year"
LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
NIPPON_ORDER = "My order failed on Nippon Small Cap"
SHOULD_I_REDEEM = "Should I redeem HDFC Large Cap?"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"
PII_COMPLAINT = f"{EXIT_LOAD} My PAN is {RAW_PAN} and phone {RAW_PHONE}."


class FakeClassifier:
    """Records every call, answers by task, and is safe to call from three threads."""

    def __init__(self, answers=None, raises=False):
        self.answers = answers or {}
        self.raises = raises
        self.calls = []
        self._lock = threading.Lock()

    def classify(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("down")
        return self.answers.get(kwargs["task"], {})


def _handle(text, session=None, **kwargs):
    trace = Trace()
    session = handle_complaint(session or Session(), text, trace=trace, **kwargs)
    return session, trace


# --- One complaint, three results, a draft, a reviewer --------------------------------


def test_one_complaint_produces_all_three_results_and_a_draft():
    session, trace = _handle(EXIT_LOAD)

    assert session.redacted_text == EXIT_LOAD
    assert session.pii_types == ()
    assert isinstance(session.facts, FactsResult)
    assert isinstance(session.pattern, PatternResult)
    assert isinstance(session.compliance, ComplianceResult)
    assert isinstance(session.draft, Draft)
    assert isinstance(session.review, ReviewerResult)

    assert session.facts.status == "covered"
    assert session.facts.scheme_id == "large_cap"
    assert session.facts.citation == LARGE_CAP_URL
    assert session.pattern.theme_id == "fees_brokerage"
    assert session.pattern.count == 136
    assert session.compliance.flags == ()
    assert session.compliance.advice_refusal is False

    assert session.draft.disposition == "escalate"
    assert EXIT_LOAD_FACT in session.draft.reply
    assert session.draft.reply.endswith(f"Source: {LARGE_CAP_URL}")
    assert session.routing_note == session.draft.routing_note
    assert session.routing_note.team == "Fees and mutual fund ops"
    assert session.review.status == "pass"
    assert session.advice_refusal is False
    assert session.refinement_count == 0
    assert session.agents_fired == ()

    # Every result round-trips through its JSON shape.
    for item in (session.facts, session.pattern, session.compliance, session.draft):
        assert type(item).from_dict(item.to_dict()) == item
    # A pass never fires the rewrite node: one start and one return for every other node.
    assert len(trace) == 2 * (len(NODES) - 1)
    assert trace.starts(REWRITE_NODE) == ()


def test_trace_shows_three_starts_then_three_returns_then_the_reviewer():
    _, trace = _handle(EXIT_LOAD)
    sequence = trace.sequence()

    assert sequence[:2] == (("start", "ingress"), ("return", "ingress"))
    fan_out = sequence[2:8]
    assert {node for _, node in fan_out[:3]} == set(AGENT_NODES)
    assert all(kind == "start" for kind, _ in fan_out[:3])
    assert {node for _, node in fan_out[3:]} == set(AGENT_NODES)
    assert all(kind == "return" for kind, _ in fan_out[3:])
    assert sequence[8:] == (
        ("start", "draft"),
        ("return", "draft"),
        ("start", "reviewer"),
        ("return", "reviewer"),
    )

    last_agent_start = max(trace.first("start", node).seq for node in AGENT_NODES)
    first_agent_return = min(trace.first("return", node).seq for node in AGENT_NODES)
    last_agent_return = max(trace.first("return", node).seq for node in AGENT_NODES)
    assert last_agent_start < first_agent_return
    assert last_agent_return < trace.first("start", "reviewer").seq
    assert trace.violations() == []
    for node in AGENT_NODES:
        assert len(trace.starts(node)) == 1
        assert len(trace.returns(node)) == 1


def test_trace_violations_catch_a_sequential_run_and_an_early_reviewer():
    sequential = Trace()
    for node in AGENT_NODES:
        sequential.start(node)
        sequential.returned(node)
    sequential.start("reviewer")
    sequential.returned("reviewer")
    problems = sequential.violations()
    assert "facts returned before all agents started" in problems
    assert "pattern returned before all agents started" in problems
    assert "compliance returned before all agents started" not in problems

    early = Trace()
    for node in AGENT_NODES:
        early.start(node)
    early.returned("facts")
    early.start("reviewer")
    early.returned("pattern")
    early.returned("compliance")
    problems = early.violations()
    assert "reviewer started before pattern returned" in problems
    assert "reviewer started before compliance returned" in problems

    missing = Trace()
    missing.start("facts")
    assert "facts never returned" in missing.violations()
    assert "pattern never started" in missing.violations()
    assert "reviewer never started" in missing.violations()


def test_trace_keeps_names_and_order_only_and_renders_one_line_per_event():
    _, trace = _handle(PII_COMPLAINT)
    rendered = trace.render()

    assert len(rendered.splitlines()) == len(trace)
    assert "start  facts" in rendered
    assert RAW_PAN not in rendered and RAW_PHONE not in rendered
    assert RAW_PAN not in repr(trace) and RAW_PAN not in str(trace.to_list())
    assert [event["seq"] for event in trace.to_list()] == list(range(len(trace)))
    with pytest.raises(ValueError):
        trace.record("facts", "finish")
    with pytest.raises(ValueError):
        trace.record("", "start")


# --- Graph structure ------------------------------------------------------------------


def test_graph_edges_fan_out_from_ingress_and_join_into_draft():
    edges = {(edge.source, edge.target) for edge in compile_graph().get_graph().edges}

    assert ("__start__", "ingress") in edges
    for node in AGENT_NODES:
        assert ("ingress", node) in edges
        assert (node, "draft") in edges
    assert ("draft", "reviewer") in edges
    assert ("reviewer", "__end__") in edges
    # The one rework edge goes reviewer -> rewrite -> reviewer and nowhere else.
    assert ("reviewer", "rewrite") in edges
    assert ("rewrite", "reviewer") in edges
    assert {target for source, target in edges if source == "rewrite"} == {"reviewer"}
    # No agent feeds another agent, and nothing feeds back into an agent.
    for node in AGENT_NODES:
        assert not any(source == node and target in AGENT_NODES for source, target in edges)
        assert not any(target == node and source != "ingress" for source, target in edges)
    assert set(NODES) == {"ingress", *AGENT_NODES, "draft", "reviewer", "rewrite"}
    assert set(build_graph().nodes) == set(NODES)


def test_agents_are_reached_through_the_compiled_graph_alone():
    state = compile_graph().invoke({"raw_text": EXIT_LOAD})

    assert "raw_text" not in state
    assert state["redacted_text"] == EXIT_LOAD
    assert state["facts"].scheme_id == "large_cap"
    assert state["pattern"].theme_id == "fees_brokerage"
    assert state["compliance"].flags == ()
    assert state["draft"].disposition == "escalate"
    assert state["review"].status == "pass"


def test_each_agent_node_receives_only_its_own_input(monkeypatch):
    seen = {}

    def spy(name, real):
        compiled = real()

        class Spy:
            def invoke(self, payload, *args, **kwargs):
                seen[name] = dict(payload)
                return compiled.invoke(payload, *args, **kwargs)

        return lambda **kwargs: Spy()

    monkeypatch.setattr(facts_module, "compile_graph", spy("facts", facts_module.compile_graph))
    monkeypatch.setattr(
        pattern_module, "compile_graph", spy("pattern", pattern_module.compile_graph)
    )
    monkeypatch.setattr(
        compliance_module, "compile_graph", spy("compliance", compliance_module.compile_graph)
    )

    session, _ = _handle(PII_COMPLAINT)

    redacted = session.redacted_text
    assert "[PAN]" in redacted and "[PHONE]" in redacted
    assert seen["facts"] == {"redacted_text": redacted}
    assert seen["pattern"] == {"redacted_text": redacted}
    assert seen["compliance"] == {"redacted_text": redacted, "pii_types": ("PAN", "PHONE")}
    for payload in seen.values():
        assert RAW_PAN not in str(payload) and RAW_PHONE not in str(payload)


def test_a_fake_classifier_is_called_once_per_agent_node_with_redacted_text_only():
    fake = FakeClassifier(
        answers={
            "facts": {"scheme_id": "large_cap", "section_title": "Exit load"},
            "pattern": {"theme_id": "fees_brokerage"},
            "compliance": {"flags": []},
        }
    )
    session, trace = _handle(PII_COMPLAINT, classifier=fake)
    cue_session, _ = _handle(PII_COMPLAINT)

    assert sorted(call["task"] for call in fake.calls) == ["compliance", "facts", "pattern"]
    for call in fake.calls:
        assert call["redacted_text"] == session.redacted_text
        assert RAW_PAN not in call["redacted_text"]
        assert RAW_PHONE not in call["redacted_text"]
    assert session.facts == cue_session.facts
    assert session.pattern == cue_session.pattern
    assert session.compliance == cue_session.compliance
    assert session.compliance.flags == ("pii",)
    assert session.draft == cue_session.draft
    assert trace.violations() == []


def test_a_failing_classifier_falls_back_to_the_cue_path():
    fake = FakeClassifier(raises=True)
    session, _ = _handle(EXIT_LOAD, classifier=fake)

    assert len(fake.calls) == 3
    assert session.facts.scheme_id == "large_cap"
    assert session.pattern.theme_id == "fees_brokerage"
    assert session.draft.disposition == "escalate"


def test_agent_modules_do_not_import_the_graph_the_trace_or_the_reviewer():
    forbidden = ("graph", "trace", "reviewer", "orchestrator", "session", "draft")
    for module in (facts_module, pattern_module, compliance_module, draft_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(alias.name for alias in node.names)
        names = [name for name in forbidden if name != module.__name__.rsplit(".", 1)[-1]]
        for name in imported:
            assert name not in names, f"{module.__name__} imports {name}"
            assert not name.startswith("support_reply.") or not name.endswith(
                tuple(f".{item}" for item in names)
            ), f"{module.__name__} imports {name}"


# --- Session and PII ------------------------------------------------------------------


def test_pii_complaint_stores_tokens_only_and_nothing_raw_is_logged(caplog):
    caplog.set_level(logging.INFO)
    session, trace = _handle(PII_COMPLAINT)

    assert session.redacted_text == f"{EXIT_LOAD} My PAN is [PAN] and phone [PHONE]."
    assert session.pii_types == ("PAN", "PHONE")
    assert session.compliance.flags == ("pii",)
    assert session.compliance.pii_types == ("PAN", "PHONE")
    assert session.draft.disposition == "escalate"
    assert session.draft.citation == LARGE_CAP_URL
    assert trace.violations() == []

    everywhere = "\n".join([repr(session), str(session.to_dict()), caplog.text, trace.render()])
    assert "[PAN]" in everywhere
    assert RAW_PAN not in everywhere
    assert RAW_PHONE not in everywhere


def test_the_graph_output_has_no_raw_text_channel():
    state = run(PII_COMPLAINT)

    assert "raw_text" not in state
    assert RAW_PAN not in str(state) and RAW_PHONE not in str(state)
    assert set(state) >= {
        "redacted_text",
        "pii_types",
        "advice_refusal",
        "facts",
        "pattern",
        "compliance",
        "draft",
        "review",
    }


def test_corpus_miss_is_a_handoff_with_searched_filled_and_no_url():
    session, trace = _handle(NIPPON_ORDER)

    assert session.facts.status == "not_covered"
    assert session.facts.citation is None
    assert len(session.facts.searched) >= 5
    assert session.pattern.theme_id == "trading_charts"
    assert session.draft.disposition == "handoff"
    assert "http" not in session.draft.reply
    assert session.review.status == "pass"
    assert trace.violations() == []


def test_advice_turn_sets_the_sticky_refusal_and_a_later_clean_turn_stays_refused():
    session, _ = _handle(SHOULD_I_REDEEM)

    assert session.compliance.flags == ("advice",)
    assert session.draft.disposition == "refuse"
    assert session.draft.reply == REFUSAL_REPLY
    assert session.advice_refusal is True

    session, _ = _handle(EXIT_LOAD, session=session)

    assert session.compliance.flags == ()
    assert session.facts.status == "covered"
    assert session.draft.disposition == "refuse"
    assert EXIT_LOAD_FACT not in session.draft.reply
    assert session.advice_refusal is True
    assert "sticky" in session.routing_note.why


def test_a_clean_turn_never_resets_a_sticky_refusal():
    session = Session()
    session.advice_refusal = True

    session, _ = _handle(EXIT_LOAD, session=session)

    assert session.advice_refusal is True
    assert session.draft.disposition == "refuse"


def test_a_missing_agent_result_is_an_orchestrator_error(monkeypatch):
    class Silent:
        def invoke(self, payload, *args, **kwargs):
            return {}

    monkeypatch.setattr(pattern_module, "compile_graph", lambda **kwargs: Silent())

    with pytest.raises(RuntimeError, match="pattern node returned no result"):
        handle_complaint(Session(), EXIT_LOAD)


def test_handle_complaint_rejects_bad_input():
    with pytest.raises(TypeError):
        handle_complaint(None, EXIT_LOAD)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        handle_complaint(Session(), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run(EXIT_LOAD, advice_refusal="yes")  # type: ignore[arg-type]
