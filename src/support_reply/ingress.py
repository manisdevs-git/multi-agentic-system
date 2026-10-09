"""Deterministic PII redaction. The orchestrator calls `redact` once, before fan-out.

Each match is replaced with a typed token and the matched characters are dropped.
Nothing here keeps or logs a raw value. The log line and `repr` carry the redacted
text and the token types only.

Order matters. PAN, email, and phone run before the bare digit-run pass, so a PAN
or a phone number is never also labeled an account number.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from support_reply.contracts import PII_TYPES

logger = logging.getLogger(__name__)

PAN_TOKEN = "[PAN]"
AADHAAR_TOKEN = "[AADHAAR]"
EMAIL_TOKEN = "[EMAIL]"
PHONE_TOKEN = "[PHONE]"
OTP_TOKEN = "[OTP]"
ACCOUNT_TOKEN = "[ACCOUNT]"

TOKENS: dict[str, str] = {
    "PAN": PAN_TOKEN,
    "AADHAAR": AADHAAR_TOKEN,
    "EMAIL": EMAIL_TOKEN,
    "PHONE": PHONE_TOKEN,
    "OTP": OTP_TOKEN,
    "ACCOUNT": ACCOUNT_TOKEN,
}
assert set(TOKENS) == PII_TYPES

# PAN: five letters, four digits, one letter. Word-bounded so it does not match inside a
# longer alphanumeric run.
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Standard email address.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# Indian mobile: optional +91 / 0 prefix, ten digits starting 6-9, with an optional single
# separator after the fifth digit. Digit lookarounds keep it out of longer digit runs such
# as a 12-digit Aadhaar or a 16-digit account number.
_PHONE = re.compile(
    r"(?<![\d])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\d])"
)

# A run of digits, allowing a single space or hyphen between digit groups. Candidate for
# OTP, Aadhaar, or account classification by digit count and nearby words.
_DIGIT_RUN = re.compile(r"(?<![\d])\d(?:\d|[ -](?=\d))*(?![\d])")

_OTP_LABEL = re.compile(r"\b(?:otp|one[\s-]?time[\s-]?pass(?:word|code)|verification code)\b", re.I)
_AADHAAR_LABEL = re.compile(r"\b(?:aadhaar|aadhar|adhaar|adhar|uid|uidai)\b", re.I)
_ACCOUNT_LABEL = re.compile(r"\b(?:account|acct|acc)\b|\ba/c\b", re.I)

_CONTEXT_BEFORE = 40
_CONTEXT_AFTER = 25

_OTP_MIN, _OTP_MAX = 4, 8
_AADHAAR_DIGITS = 12
_ACCOUNT_MIN_DIGITS = 9


@dataclass(frozen=True, slots=True, repr=False)
class IngressResult:
    """What the orchestrator fans out: redacted text and the token types found."""

    redacted_text: str
    pii_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.redacted_text, str):
            raise TypeError("redacted_text must be a str")
        types = tuple(self.pii_types)
        unknown = [item for item in types if item not in PII_TYPES]
        if unknown:
            raise ValueError(f"pii_types must be token types, not values: {unknown}")
        if len(types) != len(set(types)):
            raise ValueError("pii_types lists each type once")
        object.__setattr__(self, "pii_types", types)

    def to_dict(self) -> dict[str, Any]:
        return {"redacted_text": self.redacted_text, "pii_types": list(self.pii_types)}

    def __repr__(self) -> str:
        return (
            f"IngressResult(pii_types={list(self.pii_types)!r}, "
            f"redacted_text={self.redacted_text!r})"
        )


def redact(text: str) -> IngressResult:
    """Mask PAN, Aadhaar, email, phone, OTP, and account numbers with typed tokens.

    The raw string is read once and not stored. `pii_types` is ordered by the first
    position at which each type appeared, and names each type once. This is the
    ingress entry point: it logs the redacted result.
    """
    result = scan(text)
    logger.info("ingress %r", result)
    return result


def scan(text: str) -> IngressResult:
    """Same masking as `redact`, without the log line.

    Compliance and the reviewer call this on text that should already be redacted,
    to find any raw value ingress missed. A non-empty `pii_types` means one was visible.
    """
    if not isinstance(text, str):
        raise TypeError("ingress expects a str")

    first_seen: dict[str, int] = {}

    def _mark(pii_type: str, position: int) -> str:
        if pii_type not in first_seen or position < first_seen[pii_type]:
            first_seen[pii_type] = position
        return TOKENS[pii_type]

    # Pass 1: typed patterns that must win over bare digit runs.
    redacted = _PAN.sub(lambda m: _mark("PAN", m.start()), text)
    redacted = _EMAIL.sub(lambda m: _mark("EMAIL", m.start()), redacted)
    redacted = _PHONE.sub(lambda m: _mark("PHONE", m.start()), redacted)

    # Pass 2: remaining digit runs, classified by length and nearby words.
    redacted = _DIGIT_RUN.sub(_classify_digit_run(redacted, _mark), redacted)

    pii_types = tuple(sorted(first_seen, key=first_seen.__getitem__))
    return IngressResult(redacted_text=redacted, pii_types=pii_types)


def _classify_digit_run(
    source: str, mark: Callable[[str, int], str]
) -> Callable[[re.Match[str]], str]:
    def _replace(match: re.Match[str]) -> str:
        run = match.group(0)
        digits = sum(char.isdigit() for char in run)
        before = source[max(0, match.start() - _CONTEXT_BEFORE) : match.start()]
        after = source[match.end() : match.end() + _CONTEXT_AFTER]
        nearby = before + " " + after

        if _OTP_MIN <= digits <= _OTP_MAX and _OTP_LABEL.search(nearby):
            return mark("OTP", match.start())
        if digits == _AADHAAR_DIGITS and _AADHAAR_LABEL.search(nearby):
            return mark("AADHAAR", match.start())
        if digits >= _ACCOUNT_MIN_DIGITS and _ACCOUNT_LABEL.search(nearby):
            return mark("ACCOUNT", match.start())
        if digits == _AADHAAR_DIGITS:
            return mark("AADHAAR", match.start())
        return run

    return _replace
