"""Compliance Agent. Reads the redacted complaint and the ingress `pii_types` only.

LangGraph subgraph: the model flags advice and performance, then code rescans PII.
DeepInfra classifies when `DEEPINFRA_API_KEY` is set. With no key, or when the call fails,
the cue table supplies those two flags. A `pii` flag from the model is ignored.
It does not read Facts or Pattern output and it does not see the draft; the
reviewer applies the same policy to the draft after the orchestrator writes it.

`advice_refusal` here is for this message. The orchestrator makes it sticky on the
session, so a later rephrase in the same thread stays a refusal.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from support_reply.contracts import ComplianceResult
from support_reply.ingress import scan
from support_reply.llm import TASK_COMPLIANCE, Classifier, classify_or_none

logger = logging.getLogger(__name__)

# What the user might be asked to do with a holding. Used inside modal framings so
# "I redeemed last week and was charged" is a fact question, not an advice request.
_ACTION = (
    r"(?:redeem|buy|sell|hold|invest|exit|switch|stay\s+invested|stay|continue|keep|"
    r"stop|pause|withdraw|top\s+up)"
)
_MODAL = r"(?:should|shall|do|can|could|would|must|may|ought\s+to)"
_SUBJECT = r"(?:i|we|one|someone)"
_SCHEME_WORD = r"(?:scheme|fund|sip|etf|fof|investment)"
# A scheme named in the sentence. Keeps "is the best app" out of "is this scheme good".
_SCHEME_REF = (
    rf"(?:hdfc\b[^.?!]{{0,30}}?|{_SCHEME_WORD}|large\s*cap|mid\s*cap|small\s*cap|gold|silver)"
)

_ADVICE_CUES: tuple[re.Pattern[str], ...] = (
    # "should I redeem", "do I sell", "can we hold", "should I still stay invested"
    re.compile(
        rf"\b{_MODAL}\s+{_SUBJECT}\s+(?:\w+\s+){{0,3}}?{_ACTION}\b",
        re.I,
    ),
    # "redeem or not", and "hold or sell?" / "to redeem or exit?" asked as a question.
    # "Unable to buy or sell" is a trading complaint, not an advice request.
    re.compile(
        rf"\b{_ACTION}\s+or\s+not\b|\b(?:to\s+)?{_ACTION}\s+or\s+{_ACTION}\s*\?",
        re.I,
    ),
    # "whether to redeem", "whether I should hold", "if I should sell"
    re.compile(
        rf"\b(?:whether|if)\s+(?:to|{_SUBJECT}\s+{_MODAL})\s+(?:\w+\s+){{0,2}}?{_ACTION}\b",
        re.I,
    ),
    # "is this scheme good", "is HDFC Large Cap a good fund", "is the gold FoF safe"
    re.compile(
        rf"\bis\s+(?:this\s+|that\s+|it\s+|the\s+)?{_SCHEME_REF}\b[^.?!]{{0,30}}?"
        r"\b(?:good|bad|safe|risky|worth|better|right|wise)\b",
        re.I,
    ),
    # "a good fund", "good idea to", "good time to invest"
    re.compile(rf"\b(?:a\s+)?good\s+(?:{_SCHEME_WORD}|idea|time|choice|bet)\b", re.I),
    # "worth it", "worth investing", "worth holding on to"
    re.compile(r"\bworth\s+(?:it|investing|buying|holding|keeping|continuing|staying)\b", re.I),
    # "should I stay invested", "stay invested or not"
    re.compile(r"\bstay(?:ing)?\s+invested\b", re.I),
    # "what returns should I expect" is both an advice ask and a performance ask
    re.compile(
        r"\bwhat\s+(?:kind\s+of\s+|sort\s+of\s+)?returns?\s+(?:should|can|could|will|would|do|may|might)\s+"
        rf"{_SUBJECT}\s+(?:expect|get|hope\s+for|see|make|earn)\b",
        re.I,
    ),
    # Asking for a recommendation: "recommend a fund", "advise whether to hold",
    # "please advise what should I do", "which fund should I", "best fund".
    # "Recommend Groww to anyone" and "one suggestion for the team" do not trip.
    re.compile(
        r"\b(?:recommend|advi[cs]e|suggest)\w*\b[^.?!]{0,40}?"
        rf"\b(?:{_ACTION}|which\s+{_SCHEME_WORD}|what\s+(?:should|to)\s+{_SUBJECT}\s+do)\b"
        rf"|\b(?:which|what)\s+{_SCHEME_WORD}\s+(?:should|to|is\s+best|is\s+better)\b"
        r"|\bbest\s+(?:scheme|fund|etf|fof)s?\b",
        re.I,
    ),
)

_NUMBER = r"\d+(?:\.\d+)?"
_PERCENT = rf"{_NUMBER}\s*(?:%|percent|per\s*cent|pc)"
_RETURN_WORD = (
    r"(?:returns?|cagr|xirr|irr|gains?|growth|profits?|losses?|yield|"
    r"appreciation|upside|downside)"
)
_SCHEME_SUBJECT = (
    r"(?:fund|scheme|sip|etf|fof|large\s*cap|mid\s*cap|small\s*cap|gold|silver|"
    r"it|this|that|they|portfolio|investment)"
)

_PERFORMANCE_CUES: tuple[re.Pattern[str], ...] = (
    # Named figures
    re.compile(r"\b(?:cagr|xirr|irr|absolute\s+returns?|rolling\s+returns?|trailing\s+returns?)\b", re.I),
    # A return asked for: "what are the returns", "how much return", "expected returns",
    # "returns of the fund", "return on my investment". Bare "return(s)" alone is not
    # enough: "return my call" and "my returns page is broken" are not performance asks.
    re.compile(
        r"\b(?:what|how\s+much|how\s+many|show|tell|expect(?:ed|ing)?|get|earn|give|deliver)\b"
        r"[^.?!]{0,30}?\breturns?\b"
        rf"|\breturns?\s+(?:on|of|from|for|in)\s+(?:the\s+|this\s+|that\s+|my\s+)?"
        rf"(?:{_SCHEME_REF}|capital|money)\b",
        re.I,
    ),
    # Qualified "return": "annual return", "5 year return", "rate of return", "past returns"
    re.compile(
        rf"\b(?:rate\s+of|annual(?:i[sz]ed)?|yearly|expected|past|historical|average|"
        rf"total|{_NUMBER}[\s-]?(?:year|yr|y|month|m))\s+returns?\b",
        re.I,
    ),
    # "how has the fund performed", "scheme performance", "past performance", "track record"
    re.compile(
        rf"\b{_SCHEME_SUBJECT}\s+(?:has\s+|have\s+|is\s+|was\s+|been\s+)?perform(?:ed|ing|s|ance)\b"
        rf"|\b(?:past|historical|fund|scheme|recent|{_NUMBER}[\s-]?(?:year|yr))\s+performance\b"
        r"|\bhow\s+(?:is|has|did|does|will|would)\b[^.?!]{0,40}?\bperform(?:ed|ing|s)?\b"
        r"|\btrack\s+record\b|\bout\s*perform(?:ed|ing|s)?\b|\bunder\s*perform(?:ed|ing|s)?\b",
        re.I,
    ),
    # An asserted figure: "15% returns", "gave 12% cagr", "returns of 18%", "lost 10%",
    # "up 20% in a year", "doubled my money"
    re.compile(
        rf"\b{_PERCENT}\s*(?:{_RETURN_WORD}|p\.?\s*a\.?\b|per\s+annum|annual(?:ly)?|"
        rf"in\s+(?:a|one|{_NUMBER})\s+(?:year|yr|month))"
        rf"|\b(?:{_RETURN_WORD}|gave|given|gained|grew|grown|up|down|lost|fell|dropped|"
        rf"rose|jumped|delivered|generated|earned|made)\s+(?:of|by|was|were|is|are|at|around|about|me|over)?\s*{_PERCENT}",
        re.I,
    ),
    re.compile(r"\b(?:doubled|tripled|halved)\s+(?:my\s+)?(?:money|investment|capital|corpus)\b", re.I),
)


class ComplianceState(TypedDict, total=False):
    redacted_text: str
    pii_types: tuple[str, ...]
    model_flags: tuple[str, ...]
    result: ComplianceResult


def answer(
    redacted_text: str,
    pii_types: Sequence[str] = (),
    *,
    classifier: Classifier | None = None,
) -> ComplianceResult:
    """Scan one redacted complaint. `pii_types` is what ingress reported for it.

    The returned complaint string is redacted text only. If a raw value survived
    ingress, it is masked here and its type is added to `pii_types`.
    """
    if not isinstance(redacted_text, str):
        raise TypeError("compliance expects a str")
    if isinstance(pii_types, str) or not isinstance(pii_types, (tuple, list)):
        raise TypeError("pii_types must be a tuple of token types")

    state = compile_graph(classifier=classifier).invoke(
        {"redacted_text": redacted_text, "pii_types": tuple(pii_types)}
    )
    result = state["result"]
    if not isinstance(result, ComplianceResult):
        raise TypeError("compliance graph did not return a ComplianceResult")
    return result


def compile_graph(*, classifier: Classifier | None = None):
    """Classify, then rescan PII. Phase 6 mounts this graph as the Compliance node."""

    def classify(state: ComplianceState) -> dict[str, tuple[str, ...]]:
        return {"model_flags": _advice_and_performance(state["redacted_text"], classifier)}

    def pii(state: ComplianceState) -> dict[str, ComplianceResult]:
        rescan = scan(state["redacted_text"])
        reported = state.get("pii_types", ())
        merged_types = tuple(reported) + tuple(
            item for item in rescan.pii_types if item not in reported
        )
        flags = list(state.get("model_flags", ()))
        if merged_types:
            flags.append("pii")
        result = ComplianceResult(
            flags=tuple(flags),
            pii_types=merged_types,
            advice_refusal="advice" in flags or "performance_claim" in flags,
            redacted_text=rescan.redacted_text,
        )
        logger.info("compliance %r", result)
        return {"result": result}

    graph = StateGraph(ComplianceState)
    graph.add_node("classify", classify)
    graph.add_node("pii", pii)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "pii")
    graph.add_edge("pii", END)
    return graph.compile()


def _advice_and_performance(text: str, classifier: Classifier | None) -> tuple[str, ...]:
    payload = classify_or_none(
        task=TASK_COMPLIANCE,
        redacted_text=text,
        allowed={"flags": ("advice", "performance_claim")},
        classifier=classifier,
    )
    if payload is None or "flags" not in payload:
        return _cue_flags(text)
    raw = payload.get("flags")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        logger.warning("compliance classifier flags were %s; using cues", type(raw).__name__)
        return _cue_flags(text)
    return tuple(flag for flag in ("advice", "performance_claim") if flag in raw)


def _cue_flags(text: str) -> tuple[str, ...]:
    flags: list[str] = []
    if _any(_ADVICE_CUES, text):
        flags.append("advice")
    if _any(_PERFORMANCE_CUES, text):
        flags.append("performance_claim")
    return tuple(flags)


def _any(cues: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(cue.search(text) for cue in cues)
