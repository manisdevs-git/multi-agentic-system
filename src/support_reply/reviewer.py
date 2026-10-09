"""Reviewer. The outgoing scan. It runs after the merge and never starts an agent.

Five checks, in this order: `citation`, `advice`, `performance`, `pii`, `shape`.
Each is plain code over the draft, the routing note, and the session record, with
the `FactsResult` the draft was built from as the only reference. Gemini adds a
second opinion on `advice` and `performance` when `GEMINI_API_KEY` is set; a model
finding counts only when the span it quotes is really in the reply, and never for
a figure that Facts copied from the cited text block. Citation, PII, and shape stay
code-only: a model must not be the last line that catches a raw value or a second
link. With no key, or when the call fails, the code checks are the review.

Round 1 with failures is `fail`. Round 2 with failures is `blocked`. No failures is
`pass`. The orchestrator owns the rewrite between the rounds.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from support_reply.contracts import (
    COVERED_DISPOSITIONS,
    Draft,
    FactsResult,
    ReviewFailure,
    ReviewerResult,
)
from support_reply.draft import SENTENCE_LIMIT, sentences
from support_reply.ingress import scan
from support_reply.llm import TASK_REVIEW, Classifier, classify_or_none

logger = logging.getLogger(__name__)

CHECK_ORDER: tuple[str, ...] = ("citation", "advice", "performance", "pii", "shape")
MODEL_CHECKS: tuple[str, ...] = ("advice", "performance")

_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_URL_TRAIL = ".,;:!?"

# --- advice: a recommendation about a holding, not the mere word "redeem" -------------
_ACTION = (
    r"(?:redeem(?:ing)?|buy(?:ing)?|sell(?:ing)?|hold(?:ing)?|exit(?:ing)?|switch(?:ing)?|"
    r"stay(?:ing)?\s+invested|invest(?:ing)?|continu(?:e|ing)|keep(?:ing)?|stop(?:ping)?|"
    r"pause|top(?:ping)?\s+up|book(?:ing)?\s+profits?|averag(?:e|ing))"
)
_ADVICE_CUES: tuple[re.Pattern[str], ...] = (
    # "you may want to stay invested", "you should redeem", "you could consider switching"
    re.compile(
        r"\b(?:you|one)\s+(?:may|might|should|could|can|ought\s+to|would\s+do\s+well\s+to)\s+"
        r"(?:want\s+to\s+|wish\s+to\s+|like\s+to\s+)?(?:also\s+)?(?:consider\s+)?"
        rf"(?:\w+\s+){{0,2}}?{_ACTION}\b",
        re.I,
    ),
    # "consider staying invested", "consider redeeming"
    re.compile(rf"\bconsider\s+(?:\w+\s+){{0,2}}?{_ACTION}\b", re.I),
    # "we recommend", "we would suggest", "recommend that you hold", "is advisable"
    re.compile(
        r"\bwe\s+(?:would\s+|strongly\s+)?(?:recommend|suggest|advise|encourage)\b"
        rf"|\b(?:recommend|suggest|advise)\s+(?:that\s+)?(?:you\s+)?(?:to\s+)?{_ACTION}\b"
        r"|\b(?:is|are|would\s+be)\s+(?:recommended|advisable|advised)\b",
        re.I,
    ),
    # "it is a good time to invest", "it's best to hold", "best option"
    re.compile(
        r"\b(?:it\s+is|it's|this\s+is|now\s+is)\s+(?:a\s+|the\s+)?"
        r"(?:good|wise|better|best|advisable|safe|smart|prudent|right|bad|wrong)\s+"
        r"(?:idea|time|move|choice|option|decision)?\s*(?:to\s+)?\b"
        r"|\bbest\s+(?:to|option|choice|bet)\b"
        r"|\b(?:good|bad|right|wrong)\s+time\s+to\b",
        re.I,
    ),
    # "worth holding", "worth it", "stay invested"
    re.compile(
        r"\bworth\s+(?:it|holding|staying|investing|keeping|buying|continuing)\b"
        r"|\bstay(?:ing)?\s+invested\b",
        re.I,
    ),
    # "do not redeem now", "don't sell"
    re.compile(rf"\b(?:don't|do\s+not|never)\s+(?:\w+\s+)?{_ACTION}\b", re.I),
)
# An imperative opening a sentence: "Redeem now.", "Hold your units." "Exit load of 1%"
# and "Hold on" are not.
_IMPERATIVE = re.compile(
    r"^(?:please\s+|kindly\s+|do\s+)?"
    r"(?:redeem|buy|sell|hold(?!\s+on\b)|exit(?!\s+loads?\b)|switch|stay\s+invested|"
    r"keep\s+(?:your|the|holding|investing)|continue\s+(?:your|the|holding|investing|with)|"
    r"invest(?:\s+(?:in|more|now))?|book\s+profits?)\b",
    re.I,
)

# --- performance: a return, a CAGR, a judgement, or any figure not in the text block ---
_NUMBER = r"\d+(?:,\d{2,3})*(?:\.\d+)?"
_PERCENT = rf"{_NUMBER}\s*(?:%|percent|per\s*cent)"
_MONEY = rf"(?:₹|Rs\.?|INR)\s*{_NUMBER}(?:\s*(?:cr|crore|crores|lakh|lakhs|k))?\b"
_PERFORMANCE_CUES: tuple[re.Pattern[str], ...] = (
    # Named figures and judgements
    re.compile(
        r"\b(?:cagr|xirr|irr|absolute\s+returns?|rolling\s+returns?|trailing\s+returns?|"
        r"annuali[sz]ed\s+returns?|historical\s+returns?|past\s+returns?|track\s+record|"
        r"out\s*perform\w*|under\s*perform\w*|perform(?:ed|ance|ing|s)?)\b",
        re.I,
    ),
    # "15% returns", "12% p.a.", "returns of 18%", "gave 10%", "doubled"
    re.compile(
        rf"\b{_PERCENT}\s*(?:returns?|p\.?\s*a\.?\b|per\s+annum|annual(?:ly)?|cagr|gains?|"
        rf"growth|in\s+(?:a|one|{_NUMBER})\s+(?:year|yr|month)s?)"
        rf"|\b(?:returns?|gains?|growth|profits?|losses?|yield|gave|given|gained|grew|grown|"
        rf"lost|fell|dropped|rose|jumped|delivered|generated|earned|made|up|down)\s+"
        rf"(?:of|by|was|were|is|are|at|around|about|over|to)?\s*{_PERCENT}"
        r"|\b(?:doubled|tripled|halved)\b",
        re.I,
    ),
)
# Any figure at all. Allowed in a covered reply only when it is in the cited text block.
_FIGURE = re.compile(
    rf"{_PERCENT}|{_MONEY}|\bNAV\b[:\s]{{0,3}}{_NUMBER}|\b{_NUMBER}\s*(?:x|times)\b",
    re.I,
)

# --- citation: a factual claim in a draft that cites nothing ---------------------------
_CLAIM_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexit\s+load\s+(?:of|is|was)\b", re.I),
    re.compile(rf"\bexpense\s+ratio\b[^.?!]{{0,12}}?{_NUMBER}", re.I),
    re.compile(rf"\bNAV\b[^.?!]{{0,12}}?{_NUMBER}", re.I),
    re.compile(r"\btaxed\s+(?:at|as)\b", re.I),
    re.compile(r"\bbenchmark\s*(?::|is|was)\s*\w", re.I),
    re.compile(rf"\b(?:AUM|fund\s+size)\b[^.?!]{{0,12}}?{_NUMBER}", re.I),
    re.compile(rf"\bstamp\s+duty\b[^.?!]{{0,20}}?{_NUMBER}", re.I),
    re.compile(rf"\bminimum\s+(?:\w+\s+)?investment\b[^.?!]{{0,12}}?{_NUMBER}", re.I),
    re.compile(rf"\bmin\.\s+for\b[^.?!]{{0,20}}?{_NUMBER}", re.I),
)
_MIN_CORPUS_SENTENCE = 15

_LAST_CHECKED = re.compile(r"\bLast checked \d{2}-[A-Za-z]{3}-\d{4}\.")


def review(
    draft: Draft,
    *,
    facts: FactsResult,
    round_number: int = 1,
    record: Mapping[str, Any] | None = None,
    classifier: Classifier | None = None,
) -> ReviewerResult:
    """Return the review for one draft. Round 1 or 2 only.

    `record` is the session record that would be stored with this draft. It is
    scanned for raw PII along with the reply and the routing note. `classifier`
    overrides the Gemini default for the advice and performance scan.
    """
    if round_number not in (1, 2):
        raise ValueError("review round is 1 or 2")
    failures = check(draft, facts=facts, record=record, classifier=classifier)
    if not failures:
        status = "pass"
    elif round_number == 1:
        status = "fail"
    else:
        status = "blocked"
    result = ReviewerResult(status=status, round=round_number, failures=failures)
    logger.info("reviewer %r", result)
    return result


def check(
    draft: Draft,
    *,
    facts: FactsResult,
    record: Mapping[str, Any] | None = None,
    classifier: Classifier | None = None,
) -> tuple[ReviewFailure, ...]:
    """Run every check and return the failures in check order. Empty is a pass."""
    if not isinstance(draft, Draft):
        raise TypeError("reviewer expects a Draft")
    if not isinstance(facts, FactsResult):
        raise TypeError("reviewer expects the FactsResult the draft was built from")
    if record is not None and not isinstance(record, Mapping):
        raise TypeError("record must be a mapping or None")

    found: list[tuple[str, str]] = []
    found.extend(("citation", detail) for detail in check_citation(draft, facts))
    found.extend(("advice", detail) for detail in check_advice(draft.reply))
    found.extend(("performance", detail) for detail in check_performance(draft, facts))
    found.extend(("pii", detail) for detail in check_pii(draft, record))
    found.extend(("shape", detail) for detail in check_shape(draft))
    found.extend(_model_findings(draft, facts, classifier))

    ordered: list[ReviewFailure] = []
    seen: set[tuple[str, str]] = set()
    for check_name in CHECK_ORDER:
        for name, detail in found:
            if name == check_name and (name, detail) not in seen:
                seen.add((name, detail))
                ordered.append(ReviewFailure(check=name, detail=detail))  # type: ignore[arg-type]
    return tuple(ordered)


# --- the five checks ------------------------------------------------------------------


def check_citation(draft: Draft, facts: FactsResult) -> list[str]:
    """Covered: exactly one link, and it is the `source_url` Facts returned.

    Handoff and refuse: no link and no factual claim.
    """
    links = urls(draft.reply)
    details: list[str] = []
    if draft.disposition in COVERED_DISPOSITIONS:
        if facts.status != "covered" or facts.citation is None:
            details.append("covered reply but Facts returned not_covered")
            details.extend(links)
            return details
        if draft.citation != facts.citation:
            details.append(f"draft citation is not the Facts source_url: {draft.citation}")
        if not links:
            details.append("no source_url link in reply")
        for link in links:
            if link != facts.citation:
                details.append(link)
        if len(links) > 1 and all(link == facts.citation for link in links):
            details.append(f"source_url repeated {len(links)} times: {links[1]}")
        return details

    details.extend(links)
    for claim in factual_claims(draft.reply):
        if claim not in details:
            details.append(claim)
    return details


def check_advice(reply: str) -> list[str]:
    """Spans that recommend what to do with a holding."""
    details: list[str] = []
    for cue in _ADVICE_CUES:
        for match in cue.finditer(reply):
            _add(details, match.group().strip())
    for sentence in sentences(reply):
        match = _IMPERATIVE.match(sentence)
        if match:
            _add(details, sentence)
    return details


def check_performance(draft: Draft, facts: FactsResult) -> list[str]:
    """Return, CAGR, or performance spans, and any figure outside the cited block."""
    allowed_block = facts.fact if draft.disposition in COVERED_DISPOSITIONS else None
    details: list[str] = []
    for cue in _PERFORMANCE_CUES:
        for match in cue.finditer(draft.reply):
            span = match.group().strip()
            if not _in_block(span, allowed_block):
                _add(details, span)
    for match in _FIGURE.finditer(draft.reply):
        span = match.group().strip()
        if not _in_block(span, allowed_block):
            _add(details, span)
    return details


def check_pii(draft: Draft, record: Mapping[str, Any] | None) -> list[str]:
    """A raw value in the reply, the routing note, or the session record.

    The detail names the token type and shows the masked text, never the value.
    """
    details: list[str] = []
    for place, text in (
        ("reply", draft.reply),
        ("routing note", f"{draft.routing_note.team} {draft.routing_note.why}"),
        ("session record", "" if record is None else json.dumps(record, default=str)),
    ):
        if not text:
            continue
        result = scan(text)
        if not result.pii_types:
            continue
        types = ", ".join(result.pii_types)
        if place == "session record":
            _add(details, f"{types} in session record")
            continue
        for sentence in sentences(result.redacted_text):
            if any(f"[{item}]" in sentence for item in result.pii_types):
                _add(details, f"{types} in {place}: {sentence}")
    return details


def check_shape(draft: Draft) -> list[str]:
    """More than three sentences, or a covered reply without the last-checked date."""
    details: list[str] = []
    parts = sentences(draft.reply)
    if len(parts) > SENTENCE_LIMIT:
        details.append(" ".join(parts[SENTENCE_LIMIT:]))
    if draft.disposition in COVERED_DISPOSITIONS:
        dated = _LAST_CHECKED.search(draft.reply)
        if dated is None or (
            draft.last_checked is not None and draft.last_checked not in dated.group()
        ):
            details.append("covered reply has no last-checked date")
    return details


# --- helpers --------------------------------------------------------------------------


def urls(text: str) -> list[str]:
    """Links in the text, in order, without trailing punctuation."""
    return [match.group().rstrip(_URL_TRAIL) for match in _URL.finditer(text)]


def factual_claims(reply: str) -> list[str]:
    """Spans that state a scheme fact: a claim cue, a figure, or a corpus sentence."""
    details: list[str] = []
    for cue in _CLAIM_CUES:
        for match in cue.finditer(reply):
            _add(details, _sentence_around(reply, match.start()))
    for match in _FIGURE.finditer(reply):
        _add(details, match.group().strip())
    lowered = " ".join(reply.split()).casefold()
    for sentence in _corpus_sentences():
        if sentence.casefold() in lowered:
            _add(details, sentence)
    return details


def _model_findings(
    draft: Draft, facts: FactsResult, classifier: Classifier | None
) -> list[tuple[str, str]]:
    """Gemini's advice and performance findings, kept only when grounded in the reply."""
    payload = classify_or_none(
        task=TASK_REVIEW,
        redacted_text=draft.reply,
        allowed={"check": MODEL_CHECKS},
        classifier=classifier,
    )
    if payload is None:
        return []
    raw = payload.get("failures")
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        logger.warning("review model failures were %s; ignored", type(raw).__name__)
        return []
    allowed_block = facts.fact if draft.disposition in COVERED_DISPOSITIONS else None
    lowered = draft.reply.casefold()
    findings: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        check_name = item.get("check")
        span = item.get("span")
        if check_name not in MODEL_CHECKS or not isinstance(span, str) or not span.strip():
            continue
        span = span.strip()
        start = lowered.find(span.casefold())
        if start < 0:
            logger.warning("review model quoted a span that is not in the reply; ignored")
            continue
        span = draft.reply[start : start + len(span)]
        if check_name == "performance" and _in_block(span, allowed_block):
            continue
        findings.append((check_name, span))
    return findings


def _in_block(span: str, block: str | None) -> bool:
    if block is None:
        return False
    return " ".join(span.split()).casefold() in " ".join(block.split()).casefold()


def _add(details: list[str], detail: str) -> None:
    if detail and detail not in details:
        details.append(detail)


def _sentence_around(text: str, position: int) -> str:
    offset = 0
    for sentence in sentences(text):
        start = text.find(sentence, offset)
        if start < 0:
            continue
        end = start + len(sentence)
        if start <= position < end:
            return sentence
        offset = end
    return text.strip()


_CORPUS_SENTENCES: tuple[str, ...] | None = None


def _corpus_sentences() -> Iterable[str]:
    """Every sentence of every text block, so an uncited quote is caught verbatim."""
    global _CORPUS_SENTENCES
    if _CORPUS_SENTENCES is None:
        from support_reply.corpus import load

        found: list[str] = []
        for scheme in load().schemes:
            for block in scheme.text_blocks:
                for sentence in sentences(block.text):
                    if len(sentence) >= _MIN_CORPUS_SENTENCE and sentence not in found:
                        found.append(sentence)
        _CORPUS_SENTENCES = tuple(found)
    return _CORPUS_SENTENCES


def spans(failures: Sequence[ReviewFailure]) -> tuple[str, ...]:
    """The offending details, for the orchestrator's rewrite to exclude."""
    return tuple(failure.detail for failure in failures)
