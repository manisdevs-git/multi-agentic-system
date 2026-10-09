"""Phase 2 Facts Agent exit: one scheme, one cited section, or not_covered."""

import ast
import logging
from pathlib import Path

import pytest

from support_reply.corpus import SECTION_TITLES, load as load_corpus
from support_reply.facts import answer, compile_graph

LARGE_CAP_EXIT = "Why was I charged an exit load on HDFC Large Cap Fund?"
NIPPON_ORDER = "My order failed on Nippon Small Cap"


def test_large_cap_exit_load_is_covered_from_the_corpus():
    scheme = load_corpus().scheme("large_cap")
    result = answer(LARGE_CAP_EXIT)

    assert result.status == "covered"
    assert result.scheme_id == "large_cap"
    assert result.section_title == "Exit load"
    assert result.fact == "Exit load of 1% if redeemed within 1 year"
    assert result.fact == scheme.section("Exit load").text
    assert result.citation == scheme.source_url
    assert result.citation == "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    assert result.last_checked == "25-Jun-2026"
    assert result.searched == tuple(item.scheme_id for item in load_corpus().schemes)


@pytest.mark.parametrize(
    "scheme_id,question",
    [
        ("gold_fof", "Why was I charged an exit load on HDFC Gold ETF Fund of Fund?"),
        ("silver_fof", "What is the exit load on HDFC Silver ETF FoF?"),
    ],
)
def test_gold_and_silver_exit_load_cite_the_15_day_rule(scheme_id, question):
    scheme = load_corpus().scheme(scheme_id)
    result = answer(question)

    assert result.status == "covered"
    assert result.scheme_id == scheme_id
    assert result.section_title == "Exit load"
    assert result.fact == "Exit load of 1%, if redeemed within 15 days."
    assert result.fact == scheme.section("Exit load").text
    assert result.citation == scheme.source_url
    assert result.last_checked == scheme.nav_date
    assert result.scheme_id not in {"large_cap", "mid_cap", "small_cap"}


def test_order_on_another_amc_is_not_covered():
    result = answer(NIPPON_ORDER)

    assert result.status == "not_covered"
    assert result.fact is None
    assert result.citation is None
    assert result.scheme_id is None
    assert result.section_title is None
    assert result.last_checked is None
    for scheme in load_corpus().schemes:
        assert scheme.scheme_id in result.searched
    for title in SECTION_TITLES:
        assert title in result.searched
    assert "https://" not in " ".join(result.searched)


def test_exit_load_on_another_amc_does_not_cite_hdfc():
    result = answer("What is the exit load on Nippon Small Cap?")

    assert result.status == "not_covered"
    assert result.citation is None
    assert result.fact is None


def test_performance_question_is_not_covered_and_adds_no_number():
    result = answer("What CAGR should I expect from HDFC Large Cap Fund?")

    assert result.status == "not_covered"
    assert result.fact is None
    assert result.citation is None


def test_nav_history_is_not_the_current_nav_fact():
    result = answer("Show me the NAV history for HDFC Large Cap Fund")

    assert result.status == "not_covered"
    assert result.fact is None


def test_two_schemes_are_not_covered():
    result = answer("Compare the exit load on HDFC Large Cap and HDFC Mid Cap")

    assert result.status == "not_covered"
    assert result.citation is None


@pytest.mark.parametrize(
    "scheme_id,question,section",
    [
        ("mid_cap", "What is the benchmark of HDFC Mid Cap Fund?", "Benchmark"),
        (
            "large_cap",
            "What is the investment objective of HDFC Large Cap Fund?",
            "Investment objective",
        ),
        ("small_cap", "How is HDFC Small Cap taxed?", "Stamp duty and tax"),
        ("mid_cap", "Who manages the HDFC Mid Cap Fund?", "Fund management"),
        ("large_cap", "What is the minimum SIP for HDFC Large Cap?", "Minimum investments"),
        ("small_cap", "What is the expense ratio of HDFC Small Cap?", "Key fund facts"),
        ("gold_fof", "What is the NAV of HDFC Gold ETF Fund of Fund?", "Key fund facts"),
    ],
)
def test_section_cues_copy_that_text_block(scheme_id, question, section):
    scheme = load_corpus().scheme(scheme_id)
    result = answer(question)

    assert result.status == "covered"
    assert result.scheme_id == scheme_id
    assert result.section_title == section
    assert result.fact == scheme.section(section).text
    assert result.citation == scheme.source_url


def test_facts_log_is_the_result_repr(caplog):
    caplog.set_level(logging.INFO, logger="support_reply.facts")
    result = answer(LARGE_CAP_EXIT)

    assert caplog.text
    assert "large_cap" in caplog.text
    assert result.citation in caplog.text
    assert result.fact not in caplog.text
    assert result.fact not in repr(result)


def test_facts_module_does_not_import_pattern_or_compliance():
    import support_reply.facts as facts

    tree = ast.parse(Path(facts.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)

    assert not any("pattern" in name or "compliance" in name for name in imported)


def test_facts_rejects_non_text():
    with pytest.raises(TypeError):
        answer(None)  # type: ignore[arg-type]


class _Spy:
    def __init__(self, payload, *, raises: bool = False) -> None:
        self.payload = payload
        self.raises = raises
        self.calls: list[dict] = []

    def classify(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("down")
        return self.payload


def test_facts_graph_classifies_then_copies():
    edges = {(edge.source, edge.target) for edge in compile_graph().get_graph().edges}

    assert edges == {
        ("__start__", "classify"),
        ("classify", "copy"),
        ("copy", "__end__"),
    }


def test_fake_classifier_matches_the_cue_path_for_large_cap_exit_load():
    fact = "Exit load of 1% if redeemed within 1 year"
    spy = _Spy({"scheme_id": "large_cap", "section_title": "Exit load"})

    assert answer(LARGE_CAP_EXIT, classifier=spy) == answer(LARGE_CAP_EXIT)
    assert spy.calls[0]["task"] == "facts"
    assert spy.calls[0]["redacted_text"] == LARGE_CAP_EXIT
    assert "large_cap" in spy.calls[0]["allowed"]["scheme_id"]
    assert "Exit load" in spy.calls[0]["allowed"]["section_title"]
    assert fact not in str(spy.calls)


def test_unknown_scheme_id_is_not_covered():
    spy = _Spy({"scheme_id": "nippon_small_cap", "section_title": "Exit load"})

    result = answer(LARGE_CAP_EXIT, classifier=spy)

    assert result.status == "not_covered"
    assert result.citation is None
    assert result.fact is None


def test_classifier_failure_uses_the_cue_table():
    spy = _Spy({}, raises=True)

    assert answer(LARGE_CAP_EXIT, classifier=spy) == answer(LARGE_CAP_EXIT)


def test_model_cannot_cite_hdfc_for_another_amc():
    spy = _Spy({"scheme_id": "small_cap", "section_title": "Exit load"})

    result = answer("What is the exit load on Nippon Small Cap?", classifier=spy)

    assert result.status == "not_covered"
    assert result.citation is None


def test_model_nav_history_does_not_copy_the_nav_paragraph():
    spy = _Spy({"scheme_id": "large_cap", "section_title": "Key fund facts"})

    result = answer("Show me the NAV history for HDFC Large Cap Fund", classifier=spy)

    assert result.status == "not_covered"
    assert result.fact is None
