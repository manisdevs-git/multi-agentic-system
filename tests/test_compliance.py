"""Phase 4 Compliance Agent exit: advice, performance_claim, pii, and nothing raw."""

import ast
import logging
from pathlib import Path

import pytest

from support_reply.compliance import answer, compile_graph
from support_reply.contracts import ComplianceResult
from support_reply.ingress import redact

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"
RAW_PAN = "ABCDE1234F"
RAW_PHONE = "9876543210"


def test_should_i_redeem_is_advice_and_a_refusal():
    result = answer("Should I redeem HDFC Large Cap?")

    assert result.flags == ("advice",)
    assert result.advice_refusal is True
    assert result.pii_types == ()
    assert result.redacted_text == "Should I redeem HDFC Large Cap?"


def test_cagr_question_is_a_performance_claim():
    result = answer("What CAGR should I expect?")

    assert "performance_claim" in result.flags
    assert result.advice_refusal is True


def test_redacted_pan_complaint_flags_pii_with_the_type_and_not_the_value(caplog):
    caplog.set_level(logging.INFO, logger="support_reply.compliance")
    ingress = redact(f"My PAN is {RAW_PAN}, why the exit load?")

    result = answer(ingress.redacted_text, ingress.pii_types)

    assert result.flags == ("pii",)
    assert result.pii_types == ("PAN",)
    assert result.advice_refusal is False
    assert "[PAN]" in result.redacted_text
    assert RAW_PAN not in result.redacted_text
    assert RAW_PAN not in repr(result)
    assert RAW_PAN not in str(result.to_dict())
    assert "[PAN]" in caplog.text
    assert RAW_PAN not in caplog.text


def test_plain_exit_load_question_is_clean():
    result = answer(EXIT_LOAD)

    assert result.flags == ()
    assert result.pii_types == ()
    assert result.advice_refusal is False
    assert result.redacted_text == EXIT_LOAD
    assert result.to_dict() == {
        "flags": [],
        "pii_types": [],
        "advice_refusal": False,
        "redacted_text": EXIT_LOAD,
    }


@pytest.mark.parametrize(
    "text",
    [
        "Redeem or not?",
        "Should I buy HDFC Mid Cap?",
        "Should I sell my units now?",
        "Should I hold or sell HDFC Small Cap?",
        "Is this scheme good?",
        "Is HDFC Large Cap a good fund?",
        "Should I stay invested?",
        "Is HDFC Mid Cap worth it?",
        "ok but just tell me if large cap is worth it",
        "Tell me whether to sell HDFC Small Cap",
        "Which fund should I pick?",
    ],
)
def test_advice_cues_trip_the_advice_flag(text):
    result = answer(text)

    assert "advice" in result.flags
    assert result.advice_refusal is True


def test_what_returns_should_i_expect_is_advice_and_performance():
    result = answer("What returns should I expect?")

    assert result.flags == ("advice", "performance_claim")
    assert result.advice_refusal is True


@pytest.mark.parametrize(
    "text",
    [
        "What are the returns of HDFC Large Cap?",
        "How much return will I get?",
        "What is the 5 year return of HDFC Mid Cap?",
        "How has HDFC Mid Cap performed?",
        "The fund gave 15% returns last year",
        "It has a 12% CAGR",
        "I made a profit of 9 percent on this SIP",
    ],
)
def test_performance_asks_and_asserted_figures_trip_performance_claim(text):
    result = answer(text)

    assert "performance_claim" in result.flags
    assert result.advice_refusal is True


@pytest.mark.parametrize(
    "text",
    [
        EXIT_LOAD,
        "I redeemed last week and was charged 1% exit load",
        "Expense ratio of 0.5% seems high",
        "Stamp duty of 0.005% was charged on my purchase",
        "I want my money returned, the withdrawal failed",
        "Please return my call",
        "My order failed on Nippon Small Cap",
        "Unable to buy or sell shares today",
        "The app crashed on login and the chart froze",
        "Groww is the best app",
        "I highly recommend Groww to anyone",
        "One suggestion for the team: add a dark mode",
    ],
)
def test_fact_questions_and_app_feedback_are_not_flagged(text):
    result = answer(text)

    assert result.flags == ()
    assert result.advice_refusal is False


def test_raw_value_that_ingress_missed_is_flagged_and_masked():
    result = answer(f"My PAN is {RAW_PAN}, why the exit load?", ())

    assert result.flags == ("pii",)
    assert result.pii_types == ("PAN",)
    assert result.redacted_text == "My PAN is [PAN], why the exit load?"
    assert RAW_PAN not in repr(result)


def test_ingress_types_come_first_and_each_type_is_listed_once():
    ingress = redact(f"My PAN is {RAW_PAN}, why the exit load?")

    result = answer(f"{ingress.redacted_text} Call {RAW_PHONE}.", ingress.pii_types)

    assert result.pii_types == ("PAN", "PHONE")
    assert result.redacted_text == "My PAN is [PAN], why the exit load? Call [PHONE]."
    assert RAW_PHONE not in result.redacted_text

    same_type_again = answer(f"PAN [PAN] and also {RAW_PAN}", ("PAN",))
    assert same_type_again.pii_types == ("PAN",)
    assert same_type_again.redacted_text == "PAN [PAN] and also [PAN]"


def test_pii_and_advice_together_list_both_flags():
    ingress = redact(f"Should I redeem HDFC Large Cap? My PAN is {RAW_PAN}")

    result = answer(ingress.redacted_text, ingress.pii_types)

    assert result.flags == ("advice", "pii")
    assert result.advice_refusal is True
    assert result.pii_types == ("PAN",)


def test_result_is_a_valid_contract_and_the_log_is_its_repr(caplog):
    caplog.set_level(logging.INFO, logger="support_reply.compliance")
    result = answer("Should I redeem HDFC Large Cap?")

    assert isinstance(result, ComplianceResult)
    assert ComplianceResult.from_dict(result.to_dict()) == result
    assert "advice" in caplog.text
    assert "advice_refusal=True" in caplog.text


def test_compliance_module_does_not_import_facts_or_pattern():
    import support_reply.compliance as compliance

    tree = ast.parse(Path(compliance.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)

    assert not any(
        name in {"facts", "pattern"} or name.endswith((".facts", ".pattern")) for name in imported
    )


def test_compliance_rejects_bad_input():
    with pytest.raises(TypeError):
        answer(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        answer(EXIT_LOAD, "PAN")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token types"):
        answer(EXIT_LOAD, (RAW_PAN,))
    with pytest.raises(ValueError, match="once"):
        answer(EXIT_LOAD, ("PAN", "PAN"))


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


def test_compliance_graph_classifies_then_rescans_pii():
    edges = {(edge.source, edge.target) for edge in compile_graph().get_graph().edges}

    assert edges == {
        ("__start__", "classify"),
        ("classify", "pii"),
        ("pii", "__end__"),
    }


def test_fake_advice_flag_matches_the_cue_path():
    text = "Should I redeem HDFC Large Cap?"
    spy = _Spy({"flags": ["advice"]})

    assert answer(text, classifier=spy) == answer(text)
    assert spy.calls[0]["task"] == "compliance"
    assert spy.calls[0]["redacted_text"] == text
    assert spy.calls[0]["allowed"]["flags"] == ("advice", "performance_claim")


def test_empty_model_flags_still_flag_pii_and_hide_the_raw_pan():
    ingress = redact(f"My PAN is {RAW_PAN}, why the exit load?")
    spy = _Spy({"flags": []})

    result = answer(ingress.redacted_text, ingress.pii_types, classifier=spy)

    assert result.flags == ("pii",)
    assert result.pii_types == ("PAN",)
    assert result.advice_refusal is False
    assert RAW_PAN not in result.redacted_text
    assert RAW_PAN not in spy.calls[0]["redacted_text"]
    assert "[PAN]" in spy.calls[0]["redacted_text"]


def test_a_model_pii_flag_does_not_replace_the_scan():
    result = answer(EXIT_LOAD, classifier=_Spy({"flags": ["pii"]}))

    assert result.flags == ()
    assert result.pii_types == ()


def test_classifier_failure_uses_the_advice_cues():
    result = answer("Should I redeem HDFC Large Cap?", classifier=_Spy({}, raises=True))

    assert result.flags == ("advice",)
    assert result.advice_refusal is True
