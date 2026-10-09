"""Draft and disposition. Reads the three agent results and the redacted complaint.

Plain code, no model call. This module does not open the corpus or the theme file:
the fact comes from `FactsResult.fact`, the counts from `PatternResult`. A missing
fact stays missing. The user-facing reply is at most three sentences. The routing
note is internal and does not count toward the three.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from support_reply.contracts import (
    COVERED_DISPOSITIONS,
    ComplianceResult,
    Disposition,
    Draft,
    FactsResult,
    PatternResult,
    Priority,
    RoutingNote,
)

SENTENCE_LIMIT = 3

TEAM_FOR_THEME: dict[str, str] = {
    "fees_brokerage": "Fees and mutual fund ops",
    "money_movement": "Payments",
    "trading_charts": "Trading platform",
    "reliability": "App stability",
    "other": "General support",
}
MUTUAL_FUND_SUPPORT = "Mutual fund support"

PRIORITY_FOR_SIGNAL: dict[str, Priority] = {
    "rising": "P1",
    "recurring": "P2",
    "one-off": "P3",
}
_PRIORITY_RANK: dict[str, int] = {"P1": 1, "P2": 2, "P3": 3}
_OPEN_TICKET: frozenset[str] = frozenset({"escalate", "disagree"})

# The refusal names no action word the reviewer scans for and carries no figure.
REFUSAL_REPLY = (
    "We are not able to give investment advice, so we cannot say whether to invest in, "
    "stay in, or exit a scheme, or what it may earn. "
    "A SEBI-registered investment adviser can help with that decision. "
    "We can still help with scheme charges, SIP setup, and account questions."
)

# A scheme id is a lowercase identifier. Section titles have spaces and capitals.
# This splits `FactsResult.searched` without opening the corpus.
_SCHEME_ID = re.compile(r"[a-z0-9_]+")

_URL = re.compile(r"https?://\S+")
# A sentence ends at . ! or ? followed by whitespace and an opener. "Rs 1.25 lakh",
# "groww.in/..." and "Min. for SIP" do not split; "NAV: 1217.44. NAV date" does.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹(\[\"'])")
# "Mr. Baijal", "Rs. 100", and an initial such as "B. Sc" or "N.V. Also" do not end a sentence.
_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Rs|Ltd|Pvt|No|vs)|(?<![A-Za-z])[A-Z])\.$")
_WORD = re.compile(r"[a-z0-9%]+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "was", "were", "what", "why", "how", "who", "when", "which",
        "this", "that", "with", "from", "have", "has", "had", "are", "you", "your", "about",
        "please", "tell", "show", "does", "did", "can", "could", "would", "should", "will",
        "much", "many", "there", "here", "any", "all", "its", "our", "but", "not", "hdfc",
        "fund", "scheme", "direct", "growth", "plan", "cap", "large", "mid", "small", "gold",
        "silver", "etf", "fof",
    }
)


def compose(
    facts: FactsResult,
    pattern: PatternResult,
    compliance: ComplianceResult,
    redacted_text: str,
    *,
    advice_refusal: bool = False,
    exclude: Sequence[str] = (),
) -> Draft:
    """Pick the disposition, write the reply, and attach the routing note.

    `advice_refusal` is the session's sticky flag. When true, the turn is a refusal
    even if this message's compliance flags are empty.

    `exclude` is the reviewer's offending spans on a rework. The quoted fact skips
    any text-block sentence that contains one of them. Nothing else is generated:
    the templates are the same, so a span the templates never produce simply
    disappears from the second draft.
    """
    _check_inputs(facts, pattern, compliance, redacted_text, advice_refusal)
    disposition = disposition_for(facts, pattern, compliance, advice_refusal=advice_refusal)
    reply = reply_for(disposition, facts, redacted_text, exclude=exclude)
    note = routing_note_for(
        disposition, facts, pattern, compliance, advice_refusal=advice_refusal
    )
    covered = disposition in COVERED_DISPOSITIONS
    return Draft(
        reply=reply,
        last_checked=facts.last_checked if covered else None,
        citation=facts.citation if covered else None,
        disposition=disposition,
        disagreement=disposition == "disagree",
        routing_note=note,
    )


def disposition_for(
    facts: FactsResult,
    pattern: PatternResult,
    compliance: ComplianceResult,
    *,
    advice_refusal: bool = False,
) -> Disposition:
    """Precedence: refuse, then handoff, then disagree, escalate, reply.

    A `pii` flag alone does not change the disposition. The complaint is already
    redacted, and the reviewer checks the draft for raw values.
    """
    if advice_refusal or _asks_for_advice(compliance):
        return "refuse"
    if facts.status != "covered":
        return "handoff"
    if pattern.signal == "rising":
        return "disagree"
    if pattern.signal == "recurring":
        return "escalate"
    return "reply"


def reply_for(
    disposition: Disposition,
    facts: FactsResult,
    redacted_text: str,
    *,
    exclude: Sequence[str] = (),
) -> str:
    """At most three sentences. The fact is quoted from the text block or not at all."""
    if disposition == "refuse":
        return REFUSAL_REPLY
    if disposition == "handoff":
        return _handoff_reply(facts.searched)
    if facts.status != "covered" or facts.fact is None:
        raise ValueError(f"{disposition} needs a covered fact")
    fact = fact_sentence(facts.fact, redacted_text, exclude=exclude)
    return " ".join([fact, f"Last checked {facts.last_checked}.", f"Source: {facts.citation}"])


def routing_note_for(
    disposition: Disposition,
    facts: FactsResult,
    pattern: PatternResult,
    compliance: ComplianceResult,
    *,
    advice_refusal: bool = False,
) -> RoutingNote:
    """Team from the theme, priority from the signal, why from ids and counts only."""
    return RoutingNote(
        team=team_for(facts, pattern),
        priority=priority_for(disposition, pattern),
        why=_why(disposition, facts, pattern, compliance, advice_refusal),
    )


def team_for(facts: FactsResult, pattern: PatternResult) -> str:
    if pattern.theme_id == "other" and facts.status == "covered":
        return f"{MUTUAL_FUND_SUPPORT} ({facts.scheme_id})"
    return TEAM_FOR_THEME.get(pattern.theme_id, TEAM_FOR_THEME["other"])


def priority_for(disposition: Disposition, pattern: PatternResult) -> Priority:
    """Signal sets the priority. `escalate` and `disagree` never drop below P2."""
    priority = PRIORITY_FOR_SIGNAL[pattern.signal]
    if disposition in _OPEN_TICKET and _PRIORITY_RANK[priority] > _PRIORITY_RANK["P2"]:
        return "P2"
    return priority


def sentences(text: str) -> list[str]:
    """Split on sentence boundaries. A URL never ends a sentence mid-link."""
    if not isinstance(text, str):
        raise TypeError("sentences expects a str")
    # Mask links with same-length filler so their dots are not boundaries, then
    # slice the original text at the boundaries found in the masked copy.
    masked = _URL.sub(lambda match: "x" * len(match.group()), text)
    parts: list[str] = []
    start = 0
    for boundary in _BOUNDARY.finditer(masked):
        if _ABBREVIATION.search(masked, 0, boundary.start()):
            continue
        parts.append(text[start : boundary.start()])
        start = boundary.end()
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def fact_sentence(fact: str, redacted_text: str, *, exclude: Sequence[str] = ()) -> str:
    """One sentence of the text block, chosen by word overlap with the complaint.

    A text block with several sentences keeps the reply within the budget by quoting
    the sentence that shares the most words with the complaint; ties and no overlap
    keep the earliest. Nothing outside the block is added except a final period.

    Sentences containing an `exclude` span are skipped. When every sentence contains
    one, the block cannot be quoted without it, so the choice falls back to the full
    list and the reviewer's second round decides.
    """
    if isinstance(exclude, str) or not isinstance(exclude, (tuple, list)):
        raise TypeError("exclude must be a tuple of spans")
    candidates = sentences(fact) or [fact.strip()]
    banned = [span.strip().casefold() for span in exclude if span.strip()]
    allowed = [
        candidate
        for candidate in candidates
        if not any(span in candidate.casefold() for span in banned)
    ]
    if allowed:
        candidates = allowed
    asked = {
        word for word in _WORD.findall(redacted_text.lower()) if len(word) >= 3
    } - _STOPWORDS
    best = candidates[0]
    best_score = -1
    for candidate in candidates:
        score = len(asked & set(_WORD.findall(candidate.lower())))
        if score > best_score:
            best, best_score = candidate, score
    if not best.endswith((".", "!", "?")):
        best += "."
    return best


def _handoff_reply(searched: Sequence[str]) -> str:
    schemes = [item for item in searched if _SCHEME_ID.fullmatch(item)]
    sections = [item for item in searched if not _SCHEME_ID.fullmatch(item)]
    searched_sentence = "We searched " + ", ".join(schemes)
    if sections:
        searched_sentence += " for " + ", ".join(sections)
    searched_sentence += " and found no section that answers it."
    return " ".join(
        [
            "The scheme pages we hold do not cover this question.",
            searched_sentence,
            "A support specialist will review this ticket and reply directly.",
        ]
    )


def _why(
    disposition: Disposition,
    facts: FactsResult,
    pattern: PatternResult,
    compliance: ComplianceResult,
    advice_refusal: bool,
) -> str:
    parts: list[str] = []
    if facts.status == "covered":
        parts.append(f"Facts covered for {facts.scheme_id} {facts.section_title}.")
    else:
        parts.append("Facts not_covered.")
    if disposition == "refuse":
        flagged = [flag for flag in compliance.flags if flag in ("advice", "performance_claim")]
        if flagged:
            parts.append(f"Compliance flagged {', '.join(flagged)}, so the reply refuses.")
        elif advice_refusal:
            parts.append("Session advice_refusal is sticky, so the reply refuses.")
    if compliance.pii_types:
        parts.append(f"Ingress redacted {', '.join(compliance.pii_types)}.")
    signal = (
        f"Pattern {pattern.theme_id} is {pattern.signal} across {pattern.count} reviews "
        f"from {pattern.window.start} to {pattern.window.end}"
    )
    if disposition == "disagree":
        signal += (
            f", {pattern.recent_count} of them from {pattern.recent_window.start} "
            f"to {pattern.recent_window.end}"
        )
    signal += ", so this ticket stays open." if disposition in _OPEN_TICKET else "."
    parts.append(signal)
    return " ".join(parts)


def _asks_for_advice(compliance: ComplianceResult) -> bool:
    return "advice" in compliance.flags or "performance_claim" in compliance.flags


def _check_inputs(
    facts: object,
    pattern: object,
    compliance: object,
    redacted_text: object,
    advice_refusal: object,
) -> None:
    if not isinstance(facts, FactsResult):
        raise TypeError("draft expects a FactsResult")
    if not isinstance(pattern, PatternResult):
        raise TypeError("draft expects a PatternResult")
    if not isinstance(compliance, ComplianceResult):
        raise TypeError("draft expects a ComplianceResult")
    if not isinstance(redacted_text, str):
        raise TypeError("draft expects the redacted complaint as a str")
    if type(advice_refusal) is not bool:
        raise TypeError("advice_refusal must be a bool")
