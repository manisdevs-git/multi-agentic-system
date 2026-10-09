"""Phase 7 exit: the five reviewer checks, one template rework, blocked, and the gate.

The reviewer is reached directly for the check-level cases and through the compiled
graph for the rework. Gemini is a fake here; the offline path is the code checks.
"""

import json
import logging
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

import support_reply.graph as graph_module
from support_reply.compliance import answer as compliance_answer
from support_reply.contracts import (
    AgentsFired,
    Draft,
    FactsResult,
    ReviewerResult,
    Session,
)
from support_reply.corpus import SECTION_TITLES, load as load_corpus
from support_reply.draft import REFUSAL_REPLY, compose, sentences
from support_reply.facts import answer as facts_answer
from support_reply.gate import ApprovalRefused, approvable, require_approvable
from support_reply.graph import REWRITE_NODE, handle_complaint
from support_reply.llm import (
    DEFAULT_GEMINI_MODEL,
    TASK_REVIEW,
    GeminiClassifier,
    default_classifier,
    default_for,
    default_review_classifier,
    instructions,
)
from support_reply.pattern import answer as pattern_answer
from support_reply.reviewer import CHECK_ORDER, check, review, spans
from support_reply.trace import AGENT_NODES, Trace

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
EXIT_LOAD_FACT = "Exit load of 1% if redeemed within 1 year"
LARGE_CAP_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
MID_CAP_URL = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
NIPPON_ORDER = "My order failed on Nippon Small Cap"
STAMP_DUTY = "How is HDFC Large Cap taxed on redemption and what is the stamp duty?"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"
ADVICE = "You may want to stay invested."


def _results(text):
    return facts_answer(text), pattern_answer(text), compliance_answer(text)


def _good(text=EXIT_LOAD):
    facts, pattern, compliance = _results(text)
    return compose(facts, pattern, compliance, text), facts, pattern, compliance


def _with_sentence(draft: Draft, sentence: str) -> Draft:
    """Swap the quoted fact for `sentence`; the date and the link stay, so three sentences."""
    marker = f"Last checked {draft.last_checked}."
    tail = draft.reply[draft.reply.index(marker) :]
    return replace(draft, reply=f"{sentence} {tail}")


def _rework(draft, failures, facts, pattern, compliance, text=EXIT_LOAD):
    return compose(facts, pattern, compliance, text, exclude=spans(failures))


class FakeReviewModel:
    """Answers the review task with a fixed payload and records every call."""

    def __init__(self, payload=None, raises=False):
        self.payload = {} if payload is None else payload
        self.raises = raises
        self.calls = []
        self._lock = threading.Lock()

    def classify(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("down")
        return self.payload


# --- Every template passes --------------------------------------------------------------


def test_every_corpus_section_template_passes_all_checks():
    for scheme in load_corpus().schemes:
        for title in SECTION_TITLES:
            block = scheme.section(title)
            facts = FactsResult(
                status="covered",
                scheme_id=scheme.scheme_id,
                section_title=title,
                fact=block.text,
                citation=scheme.source_url,
                last_checked=scheme.nav_date,
                searched=(),
            )
            question = f"What is the {title.lower()} of {scheme.scheme_name}?"
            pattern, compliance = pattern_answer(question), compliance_answer(question)
            draft = compose(facts, pattern, compliance, question)
            record = {"facts": facts.to_dict(), "draft": draft.to_dict()}
            assert check(draft, facts=facts, record=record) == (), (scheme.scheme_id, title)


def test_handoff_and_refusal_templates_pass():
    for text in (NIPPON_ORDER, "Should I redeem HDFC Large Cap?", "What CAGR should I expect?"):
        draft, facts, _, _ = _good(text)
        assert draft.disposition in {"handoff", "refuse"}
        assert review(draft, facts=facts).status == "pass"
    assert review(_good("Should I redeem HDFC Large Cap?")[0], facts=facts_answer(EXIT_LOAD)).status == "pass"
    assert _good("Should I redeem HDFC Large Cap?")[0].reply == REFUSAL_REPLY


# --- Round 1 fails, template rewrite passes ---------------------------------------------


def test_an_uncited_fact_fails_citation_and_passes_after_the_rewrite():
    good, facts, pattern, compliance = _good()
    draft = replace(good, reply=good.reply.replace(f" Source: {LARGE_CAP_URL}", ""))

    first = review(draft, facts=facts)

    assert first.status == "fail" and first.round == 1
    assert [failure.check for failure in first.failures] == ["citation"]
    assert "no source_url" in first.failures[0].detail

    second = review(_rework(draft, first.failures, facts, pattern, compliance), facts=facts, round_number=2)
    assert second.status == "pass" and second.round == 2 and second.failures == ()


def test_a_second_url_fails_citation_with_that_link_as_the_span():
    good, facts, pattern, compliance = _good()
    draft = _with_sentence(good, f"See also {MID_CAP_URL}.")

    first = review(draft, facts=facts)

    assert first.status == "fail"
    checks = {failure.check for failure in first.failures}
    assert "citation" in checks
    assert any(failure.detail == MID_CAP_URL for failure in first.failures)
    assert review(_rework(draft, first.failures, facts, pattern, compliance), facts=facts, round_number=2).status == "pass"


def test_a_link_that_is_not_the_facts_source_url_fails_citation():
    good, facts, _, _ = _good()
    draft = replace(good, reply=good.reply.replace(LARGE_CAP_URL, MID_CAP_URL))

    first = review(draft, facts=facts)

    assert first.status == "fail"
    assert ("citation", MID_CAP_URL) in [(f.check, f.detail) for f in first.failures]


def test_advice_wording_fails_advice_with_the_span_and_passes_after_the_rewrite():
    good, facts, pattern, compliance = _good()
    draft = _with_sentence(good, ADVICE)

    first = review(draft, facts=facts)

    assert first.status == "fail"
    assert first.failures[0].check == "advice"
    assert first.failures[0].detail in ADVICE
    assert all(failure.check == "advice" for failure in first.failures)

    reworked = _rework(draft, first.failures, facts, pattern, compliance)
    assert ADVICE not in reworked.reply
    assert reworked == good
    assert review(reworked, facts=facts, round_number=2).status == "pass"


@pytest.mark.parametrize(
    "sentence",
    [
        "You should redeem now.",
        "Consider staying invested for the long term.",
        "We recommend holding your units.",
        "It is a good time to invest more.",
        "Do not sell at this point.",
        "Hold your units until next year.",
    ],
)
def test_recommendation_phrasings_trip_the_advice_check(sentence):
    good, facts, _, _ = _good()
    draft = _with_sentence(good, sentence)

    assert "advice" in {failure.check for failure in check(draft, facts=facts)}


def test_a_raw_phone_fails_pii_and_the_detail_never_shows_the_number():
    good, facts, pattern, compliance = _good()
    draft = _with_sentence(good, f"Call us on {RAW_PHONE}.")

    first = review(draft, facts=facts)

    assert first.status == "fail"
    assert [failure.check for failure in first.failures] == ["pii"]
    assert "PHONE" in first.failures[0].detail
    assert "[PHONE]" in first.failures[0].detail
    assert RAW_PHONE not in first.failures[0].detail
    assert RAW_PHONE not in repr(first) and RAW_PHONE not in json.dumps(first.to_dict())

    assert review(_rework(draft, first.failures, facts, pattern, compliance), facts=facts, round_number=2).status == "pass"


def test_a_pan_token_in_the_draft_passes_the_pii_check():
    good, facts, _, _ = _good()
    draft = _with_sentence(good, "Your [PAN] and [PHONE] stay on file as tokens.")

    assert review(draft, facts=facts).status == "pass"


def test_a_raw_pan_in_the_routing_note_or_the_session_record_fails_pii():
    good, facts, _, _ = _good()
    noted = replace(good, routing_note=replace(good.routing_note, why=f"{good.routing_note.why} PAN {RAW_PAN}."))

    failures = check(noted, facts=facts)
    assert [failure.check for failure in failures] == ["pii"]
    assert "routing note" in failures[0].detail and RAW_PAN not in failures[0].detail

    record = {"redacted_text": f"{EXIT_LOAD} My PAN is {RAW_PAN}.", "draft": good.to_dict()}
    failures = check(good, facts=facts, record=record)
    assert [failure.check for failure in failures] == ["pii"]
    assert failures[0].detail == "PAN in session record"

    clean = {"redacted_text": f"{EXIT_LOAD} My PAN is [PAN].", "draft": good.to_dict()}
    assert check(good, facts=facts, record=clean) == ()


def test_a_four_sentence_draft_fails_shape_and_passes_after_the_rewrite():
    good, facts, pattern, compliance = _good()
    draft = replace(good, reply=good.reply.replace("Last checked", "We checked this for you. Last checked"))
    assert len(sentences(draft.reply)) == 4

    first = review(draft, facts=facts)

    assert first.status == "fail"
    assert [failure.check for failure in first.failures] == ["shape"]
    assert first.failures[0].detail == f"Source: {LARGE_CAP_URL}"

    reworked = _rework(draft, first.failures, facts, pattern, compliance)
    assert len(sentences(reworked.reply)) == 3
    assert review(reworked, facts=facts, round_number=2).status == "pass"


def test_a_covered_reply_without_a_last_checked_date_fails_shape():
    good, facts, _, _ = _good()
    draft = replace(good, reply=good.reply.replace(f"Last checked {good.last_checked}.", "Checked recently."))

    failures = check(draft, facts=facts)

    assert [failure.check for failure in failures] == ["shape"]
    assert "last-checked" in failures[0].detail


# --- Performance and the stamp-duty exception -------------------------------------------


def test_a_return_figure_fails_performance_with_the_span():
    good, facts, pattern, compliance = _good()
    draft = _with_sentence(good, "The fund gave 15% CAGR last year.")

    first = review(draft, facts=facts)

    assert first.status == "fail"
    assert {failure.check for failure in first.failures} == {"performance"}
    details = [failure.detail for failure in first.failures]
    assert "CAGR" in details and "15%" in details
    assert review(_rework(draft, first.failures, facts, pattern, compliance), facts=facts, round_number=2).status == "pass"


def test_the_tax_figure_is_allowed_only_when_facts_returned_the_stamp_duty_section():
    tax_draft, tax_facts, _, _ = _good(STAMP_DUTY)
    assert tax_facts.section_title == "Stamp duty and tax"
    assert "%" in tax_draft.reply
    assert review(tax_draft, facts=tax_facts).status == "pass"

    exit_draft, exit_facts, _, _ = _good()
    borrowed = _with_sentence(exit_draft, "If you redeem within one year, returns are taxed at 20%.")
    failures = check(borrowed, facts=exit_facts)
    assert [failure.check for failure in failures] == ["performance"]
    assert failures[0].detail == "20%"


def test_a_not_covered_draft_with_a_fact_or_a_url_fails_citation():
    handoff, facts, _, _ = _good(NIPPON_ORDER)
    assert handoff.disposition == "handoff"

    quoted = replace(handoff, reply=f"{EXIT_LOAD_FACT}. A specialist will reply.")
    failures = check(quoted, facts=facts)
    assert "citation" in {failure.check for failure in failures}
    assert any(EXIT_LOAD_FACT in failure.detail for failure in failures)

    linked = replace(handoff, reply=f"A specialist will reply. Source: {LARGE_CAP_URL}")
    failures = check(linked, facts=facts)
    assert ("citation", LARGE_CAP_URL) in [(f.check, f.detail) for f in failures]

    figure = replace(handoff, reply="The expense ratio is 1.04%. A specialist will reply.")
    assert "citation" in {failure.check for failure in check(figure, facts=facts)}


def test_failures_come_back_in_check_order_without_duplicates():
    good, facts, _, _ = _good()
    draft = replace(
        good,
        reply=f"{EXIT_LOAD_FACT}. {ADVICE} Call {RAW_PHONE}. The fund gave 15% CAGR. "
        f"See {MID_CAP_URL}. Source: {LARGE_CAP_URL}",
    )

    failures = check(draft, facts=facts)

    checks = [failure.check for failure in failures]
    assert checks == sorted(checks, key=CHECK_ORDER.index)
    assert set(checks) == {"citation", "advice", "performance", "pii", "shape"}
    assert len({(f.check, f.detail) for f in failures}) == len(failures)


def test_review_validates_its_arguments():
    good, facts, _, _ = _good()
    with pytest.raises(ValueError):
        review(good, facts=facts, round_number=3)
    with pytest.raises(TypeError):
        review("draft", facts=facts)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        review(good, facts=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        check(good, facts=facts, record="record")  # type: ignore[arg-type]


# --- Gemini second opinion --------------------------------------------------------------


def test_default_review_classifier_is_gemini_only_with_a_key(monkeypatch):
    assert default_review_classifier() is None
    assert default_for(TASK_REVIEW) is None
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    classifier = default_review_classifier()
    assert isinstance(classifier, GeminiClassifier)
    assert classifier.model == DEFAULT_GEMINI_MODEL
    assert isinstance(default_for(TASK_REVIEW), GeminiClassifier)
    # The agents keep Groq; a Gemini key alone configures nothing for them.
    assert default_classifier() is None and default_for("facts") is None
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert default_review_classifier().model == "gemini-2.5-pro"


def test_gemini_sends_the_draft_as_content_and_the_rules_as_system_instruction():
    class _Models:
        def __init__(self):
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(text='{"failures": [{"check": "advice", "span": "stay invested"}]}')

    client = SimpleNamespace(models=_Models())
    classifier = GeminiClassifier(api_key="test-key", model=DEFAULT_GEMINI_MODEL, client=client)
    reply = f"{EXIT_LOAD_FACT}. {ADVICE} Source: {LARGE_CAP_URL}"

    payload = classifier.classify(task=TASK_REVIEW, redacted_text=reply, allowed={"check": ("advice", "performance")})

    assert payload == {"failures": [{"check": "advice", "span": "stay invested"}]}
    sent = client.models.kwargs
    assert sent["model"] == DEFAULT_GEMINI_MODEL
    assert sent["contents"] == reply
    assert sent["config"]["response_mime_type"] == "application/json"
    assert sent["config"]["temperature"] == 0
    system = sent["config"]["system_instruction"]
    assert system == instructions(TASK_REVIEW, {"check": ("advice", "performance")})
    assert "advice" in system and "performance" in system and "span" in system
    assert reply not in system and "test-key" not in system

    with pytest.raises(ValueError):
        GeminiClassifier(api_key=" ", model=DEFAULT_GEMINI_MODEL, client=client)
    client.models.generate_content = lambda **kwargs: SimpleNamespace(text="")
    with pytest.raises(ValueError):
        classifier.classify(task=TASK_REVIEW, redacted_text=reply, allowed={"check": ("advice",)})


def test_a_grounded_model_finding_fails_the_review_and_the_rewrite_drops_it():
    good, facts, pattern, compliance = _good()
    phrase = "Many customers keep their units for the long run"
    draft = _with_sentence(good, f"{phrase}.")
    assert check(draft, facts=facts) == ()  # the code cues alone let this through
    model = FakeReviewModel({"failures": [{"check": "advice", "span": phrase.lower()}]})

    first = review(draft, facts=facts, classifier=model)

    assert first.status == "fail"
    assert [(f.check, f.detail) for f in first.failures] == [("advice", phrase)]
    assert len(model.calls) == 1
    assert model.calls[0]["task"] == TASK_REVIEW
    assert model.calls[0]["redacted_text"] == draft.reply
    assert model.calls[0]["allowed"] == {"check": ("advice", "performance")}

    reworked = _rework(draft, first.failures, facts, pattern, compliance)
    assert phrase not in reworked.reply
    assert review(reworked, facts=facts, round_number=2, classifier=FakeReviewModel()).status == "pass"


def test_model_findings_that_are_not_in_the_reply_or_are_corpus_figures_are_ignored():
    good, facts, _, _ = _good()
    model = FakeReviewModel(
        {
            "failures": [
                {"check": "advice", "span": "you should redeem"},  # not in the reply
                {"check": "performance", "span": "1%"},  # copied from the cited text block
                {"check": "pii", "span": EXIT_LOAD_FACT},  # not a model check
                {"check": "advice", "span": ""},
                "garbage",
            ]
        }
    )

    assert review(good, facts=facts, classifier=model).status == "pass"

    tax_draft, tax_facts, _, _ = _good(STAMP_DUTY)
    model = FakeReviewModel({"failures": [{"check": "performance", "span": "20%"}]})
    assert review(tax_draft, facts=tax_facts, classifier=model).status == "pass"


def test_a_failing_or_silent_model_leaves_the_code_checks_as_the_review():
    good, facts, _, _ = _good()
    bad = _with_sentence(good, ADVICE)

    assert review(good, facts=facts, classifier=FakeReviewModel(raises=True)).status == "pass"
    assert review(bad, facts=facts, classifier=FakeReviewModel(raises=True)).status == "fail"
    assert review(bad, facts=facts, classifier=FakeReviewModel({})).status == "fail"
    assert review(good, facts=facts, classifier=FakeReviewModel({"failures": "none"})).status == "pass"


# --- Through the graph: one rework, no agent re-entry, blocked, and the gate -------------


def _corrupting_compose(monkeypatch, corrupt, times=1):
    """Make the graph's first `times` drafts bad; later drafts are the real template."""
    real = graph_module.compose
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        drafted = real(*args, **kwargs)
        calls["n"] += 1
        return corrupt(drafted) if calls["n"] <= times else drafted

    monkeypatch.setattr(graph_module, "compose", wrapped)
    return calls


def test_the_graph_reworks_once_from_stored_results_and_logs_the_rework(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    calls = _corrupting_compose(monkeypatch, lambda d: _with_sentence(d, ADVICE))
    trace = Trace()
    session = Session()

    session = handle_complaint(session, EXIT_LOAD, trace=trace)

    assert calls["n"] == 2
    assert session.review == ReviewerResult(status="pass", round=2, failures=())
    assert ADVICE not in session.draft.reply
    assert session.draft.citation == LARGE_CAP_URL
    assert session.refinement_count == 0
    assert session.agents_fired == (
        AgentsFired(
            round=0,
            kind="reviewer_rework",
            ran=("orchestrator",),
            skipped=("facts", "pattern", "compliance"),
        ),
    )
    assert "reviewer_rework" in repr(session)

    # The rework trace: one start per agent, reviewer twice, rewrite once, no agent re-entry.
    for node in AGENT_NODES:
        assert len(trace.starts(node)) == 1 and len(trace.returns(node)) == 1
    assert len(trace.starts("reviewer")) == 2
    assert len(trace.starts(REWRITE_NODE)) == 1
    assert trace.violations() == []
    sequence = trace.sequence()
    assert sequence[-6:] == (
        ("start", "reviewer"),
        ("return", "reviewer"),
        ("start", REWRITE_NODE),
        ("return", REWRITE_NODE),
        ("start", "reviewer"),
        ("return", "reviewer"),
    )
    rewrite_start = trace.first("start", REWRITE_NODE).seq
    assert all(event.seq < rewrite_start for node in AGENT_NODES for event in trace.starts(node))
    assert "rewrite Draft(" in caplog.text


def test_a_draft_that_still_fails_after_the_rewrite_is_blocked_and_cannot_be_approved(monkeypatch):
    _corrupting_compose(monkeypatch, lambda d: _with_sentence(d, ADVICE), times=2)
    trace = Trace()

    session = handle_complaint(Session(), EXIT_LOAD, trace=trace)

    assert session.review.status == "blocked"
    assert session.review.round == 2
    assert [failure.check for failure in session.review.failures] == ["advice"] * len(session.review.failures)
    assert ADVICE in session.draft.reply  # the draft is kept, with the flag
    assert len(session.agents_fired) == 1 and session.agents_fired[0].kind == "reviewer_rework"
    for node in AGENT_NODES:
        assert len(trace.starts(node)) == 1
    assert len(trace.starts("reviewer")) == 2 and len(trace.starts(REWRITE_NODE)) == 1

    assert approvable(session) is False
    with pytest.raises(ApprovalRefused, match="blocked"):
        require_approvable(session, human_approved=True)


def test_a_fact_block_that_cannot_be_quoted_without_the_span_ends_blocked():
    good, facts, pattern, compliance = _good()
    # The whole exit-load sentence is the only sentence in its block.
    model = FakeReviewModel({"failures": [{"check": "advice", "span": EXIT_LOAD_FACT}]})

    first = review(good, facts=facts, classifier=model)
    assert first.status == "fail"
    reworked = compose(facts, pattern, compliance, EXIT_LOAD, exclude=spans(first.failures))
    assert reworked == good  # nothing else to quote, so the second round decides
    second = review(reworked, facts=facts, round_number=2, classifier=model)
    assert second.status == "blocked"


def test_the_gate_accepts_only_a_human_approved_pass():
    session = handle_complaint(Session(), EXIT_LOAD)
    assert session.review.status == "pass"
    assert approvable(session) is True
    require_approvable(session, human_approved=True)

    with pytest.raises(ApprovalRefused, match="human-approve"):
        require_approvable(session, human_approved=False)
    with pytest.raises(TypeError):
        require_approvable(session, human_approved="yes")  # type: ignore[arg-type]
    with pytest.raises(ApprovalRefused, match="no reviewed draft"):
        require_approvable(Session(), human_approved=True)
    with pytest.raises(TypeError):
        approvable("session")  # type: ignore[arg-type]


def test_the_graph_passes_the_review_classifier_to_the_reviewer_only():
    model = FakeReviewModel({"failures": []})

    session = handle_complaint(Session(), EXIT_LOAD, review_classifier=model)

    assert session.review.status == "pass"
    assert [call["task"] for call in model.calls] == [TASK_REVIEW]
    assert model.calls[0]["redacted_text"] == session.draft.reply


def test_a_pii_complaint_through_the_graph_passes_the_pii_check_on_tokens():
    session = handle_complaint(Session(), f"{EXIT_LOAD} My PAN is {RAW_PAN} and phone {RAW_PHONE}.")

    assert session.pii_types == ("PAN", "PHONE")
    assert session.review.status == "pass"
    assert session.agents_fired == ()
    assert RAW_PAN not in json.dumps(session.to_dict()) and RAW_PHONE not in json.dumps(session.to_dict())
