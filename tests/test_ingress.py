"""Phase 1 ingress exit: typed tokens, pii_types once per class, nothing raw in the logs."""

import logging

import pytest

from support_reply.contracts import PII_TYPES, ComplianceResult, Session
from support_reply.ingress import IngressResult, redact

CLEAN = "Why was I charged an exit load on HDFC Large Cap Fund?"

RAW_PAN = "ABCDE1234F"
RAW_AADHAAR_SPACED = "1234 5678 9012"
RAW_AADHAAR_PLAIN = "123456789012"
RAW_EMAIL = "ravi.kumar+groww@example.co.in"
RAW_PHONE = "9876543210"
RAW_PHONE_PREFIXED = "+91 98765 43210"
RAW_OTP = "482913"
RAW_ACCOUNT = "123456789012345"

# (input, token, pii type, raw value that must disappear)
CLASS_FIXTURES = [
    (f"My PAN is {RAW_PAN}, why the exit load?", "[PAN]", "PAN", RAW_PAN),
    (f"Aadhaar {RAW_AADHAAR_SPACED} linked, KYC still pending", "[AADHAAR]", "AADHAAR", RAW_AADHAAR_SPACED),
    (f"Aadhaar {RAW_AADHAAR_PLAIN} linked", "[AADHAAR]", "AADHAAR", RAW_AADHAAR_PLAIN),
    (f"Mail me at {RAW_EMAIL} please", "[EMAIL]", "EMAIL", RAW_EMAIL),
    (f"Call me on {RAW_PHONE} about the charge", "[PHONE]", "PHONE", RAW_PHONE),
    (f"Call me on {RAW_PHONE_PREFIXED} about the charge", "[PHONE]", "PHONE", RAW_PHONE_PREFIXED),
    (f"The OTP {RAW_OTP} never arrived", "[OTP]", "OTP", RAW_OTP),
    (f"{RAW_OTP} is my one-time password and it failed", "[OTP]", "OTP", RAW_OTP),
    (f"Debit from account {RAW_ACCOUNT} failed", "[ACCOUNT]", "ACCOUNT", RAW_ACCOUNT),
    (f"a/c no. {RAW_ACCOUNT} shows the wrong balance", "[ACCOUNT]", "ACCOUNT", RAW_ACCOUNT),
]


@pytest.mark.parametrize("text,token,pii_type,raw", CLASS_FIXTURES)
def test_each_class_redacts_to_its_token_and_lists_the_type_once(text, token, pii_type, raw):
    result = redact(text)

    assert token in result.redacted_text
    assert raw not in result.redacted_text
    assert "".join(raw.split()) not in "".join(result.redacted_text.split())
    assert result.pii_types == (pii_type,)
    assert result.to_dict() == {"redacted_text": result.redacted_text, "pii_types": [pii_type]}


def test_no_pii_returns_the_same_text_and_empty_types():
    result = redact(CLEAN)

    assert result.redacted_text == CLEAN
    assert result.pii_types == ()
    assert result.to_dict()["pii_types"] == []


def test_pan_fixture_log_has_the_token_and_not_the_pan(caplog):
    caplog.set_level(logging.INFO, logger="support_reply.ingress")

    result = redact(f"My PAN is {RAW_PAN}, why the exit load?")

    assert caplog.text
    assert "[PAN]" in caplog.text
    assert RAW_PAN not in caplog.text
    assert RAW_PAN not in repr(result)
    assert RAW_PAN not in str(result.to_dict())


def test_repeated_values_of_one_class_list_the_type_once():
    result = redact(f"Call {RAW_PHONE} or {RAW_PHONE_PREFIXED}, PAN {RAW_PAN} or {RAW_PAN}")

    assert result.redacted_text == "Call [PHONE] or [PHONE], PAN [PAN] or [PAN]"
    assert result.pii_types == ("PHONE", "PAN")


def test_pan_and_phone_are_not_relabeled_as_account():
    result = redact(f"{RAW_PAN} and {RAW_PHONE} and account {RAW_ACCOUNT}")

    assert result.redacted_text == "[PAN] and [PHONE] and account [ACCOUNT]"
    assert result.pii_types == ("PAN", "PHONE", "ACCOUNT")

    phone_in_account_context = redact(f"My phone {RAW_PHONE} is linked to my account")
    assert phone_in_account_context.redacted_text == "My phone [PHONE] is linked to my account"
    assert phone_in_account_context.pii_types == ("PHONE",)


def test_twelve_digits_without_a_label_are_aadhaar():
    result = redact(f"Random {RAW_AADHAAR_PLAIN} with no label")

    assert result.redacted_text == "Random [AADHAAR] with no label"
    assert result.pii_types == ("AADHAAR",)


def test_digit_runs_without_pii_context_are_left_alone():
    text = f"Order id {RAW_OTP} failed on 2026-07-04 for Rs 10000, ref 20260704001"

    result = redact(text)

    assert result.redacted_text == text
    assert result.pii_types == ()


def test_result_types_are_valid_for_session_and_compliance():
    result = redact(f"PAN {RAW_PAN}, phone {RAW_PHONE}, OTP {RAW_OTP}")

    assert set(result.pii_types) <= PII_TYPES
    session = Session(redacted_text=result.redacted_text, pii_types=result.pii_types)
    assert session.pii_types == ("PAN", "PHONE", "OTP")
    compliance = ComplianceResult(
        flags=("pii",),
        pii_types=result.pii_types,
        advice_refusal=False,
        redacted_text=result.redacted_text,
    )
    assert RAW_PAN not in repr(session)
    assert RAW_PAN not in repr(compliance)


def test_ingress_result_rejects_values_as_types():
    with pytest.raises(ValueError, match="token types"):
        IngressResult(redacted_text="x", pii_types=(RAW_PAN,))
    with pytest.raises(ValueError, match="once"):
        IngressResult(redacted_text="x", pii_types=("PAN", "PAN"))
    with pytest.raises(TypeError):
        redact(None)  # type: ignore[arg-type]
