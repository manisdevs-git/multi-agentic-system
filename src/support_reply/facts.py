"""Facts Agent. Reads the redacted complaint and `data/parsed/` only.

LangGraph subgraph: classify, then copy one text block.
Groq classifies when `GROQ_API_KEY` is set. With no key, or when the call
fails, the cue tables pick the scheme and the section.
The fact is copied from that text block. No match is `not_covered`.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from support_reply.contracts import FactsResult
from support_reply.corpus import SECTION_TITLES, Corpus, load
from support_reply.llm import (
    TASK_FACTS,
    Classifier,
    classify_or_none,
    match_allowed,
)

logger = logging.getLogger(__name__)

# Gold and silver are listed first. Their cues do not contain an equity-cap phrase,
# and a commodity question must not fall through to Large, Mid, or Small Cap.
_SCHEME_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "gold_fof",
        re.compile(r"\bgold etf fund of fund\b|\bgold etf fof\b|\bgold etf\b|\bhdfc gold\b", re.I),
    ),
    (
        "silver_fof",
        re.compile(r"\bsilver etf fof\b|\bsilver etf\b|\bsilver fof\b|\bhdfc silver\b", re.I),
    ),
    ("large_cap", re.compile(r"\blarge[\s-]?cap\b", re.I)),
    ("mid_cap", re.compile(r"\bmid[\s-]?cap\b", re.I)),
    ("small_cap", re.compile(r"\bsmall[\s-]?cap\b", re.I)),
)

# First matching cue wins. SIP is minimum investment, before key fund facts,
# so a minimum-SIP question is not answered with the NAV paragraph.
_SECTION_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Exit load", re.compile(r"\bexit[\s-]?load\b", re.I)),
    ("Benchmark", re.compile(r"\bbenchmark\b", re.I)),
    (
        "Investment objective",
        re.compile(r"\binvestment objective\b|\bobjective of\b|\bwhat(?:'s| is) the objective\b", re.I),
    ),
    ("Stamp duty and tax", re.compile(r"\bstamp\s*duty\b|\btax(?:es|ed|ation)?\b", re.I)),
    ("Fund management", re.compile(r"\bfund manag(?:er|ement)\b|\bwho manages\b", re.I)),
    (
        "Minimum investments",
        re.compile(r"\bminimum investment\b|\bmin(?:imum)?\.?\s+sip\b|\bsip\b|\blumpsum\b", re.I),
    ),
    (
        "Key fund facts",
        re.compile(r"\bkey fund facts\b|\bexpense ratio\b|\briskometer\b|\baum\b|\bfund size\b|\bnav\b", re.I),
    ),
)

# A category word plus a different AMC is not one of the five HDFC schemes.
_OTHER_AMC = re.compile(
    r"\b(?:nippon|sbi|icici|axis|kotak|mirae|uti|quant|motilal|tata|birla|"
    r"franklin|dsp|bandhan|edelweiss|canara|invesco|ppfas|parag)\b",
    re.I,
)

# "nav history" must not satisfy the NAV cue. The corpus has one NAV, not a history.
_NAV_HISTORY = re.compile(r"\bnav history\b", re.I)


class FactsState(TypedDict, total=False):
    redacted_text: str
    scheme_id: str | None
    section_title: str | None
    result: FactsResult


def answer(redacted_text: str, *, classifier: Classifier | None = None) -> FactsResult:
    """Return a covered fact from one scheme section, or `not_covered`."""
    if not isinstance(redacted_text, str):
        raise TypeError("facts expects a str")

    state = compile_graph(classifier=classifier).invoke({"redacted_text": redacted_text})
    result = state["result"]
    if not isinstance(result, FactsResult):
        raise TypeError("facts graph did not return a FactsResult")
    return result


def compile_graph(*, classifier: Classifier | None = None):
    """Classify, then copy. Phase 6 mounts this compiled graph as the Facts node."""
    corpus = load()

    def classify(state: FactsState) -> dict[str, str | None]:
        scheme_id, section_title = _pick(state["redacted_text"], classifier, corpus)
        return {"scheme_id": scheme_id, "section_title": section_title}

    def copy(state: FactsState) -> dict[str, FactsResult]:
        scheme_id = state.get("scheme_id")
        section_title = state.get("section_title")
        if scheme_id and section_title:
            result = _covered(corpus, scheme_id, section_title)
        else:
            result = _not_covered(corpus)
        logger.info("facts %r", result)
        return {"result": result}

    graph = StateGraph(FactsState)
    graph.add_node("classify", classify)
    graph.add_node("copy", copy)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "copy")
    graph.add_edge("copy", END)
    return graph.compile()


def _pick(
    text: str,
    classifier: Classifier | None,
    corpus: Corpus,
) -> tuple[str | None, str | None]:
    scheme_ids = tuple(item.scheme_id for item in corpus.schemes)
    payload = classify_or_none(
        task=TASK_FACTS,
        redacted_text=text,
        allowed={"scheme_id": scheme_ids, "section_title": SECTION_TITLES},
        labels={item.scheme_id: item.scheme_name for item in corpus.schemes},
        classifier=classifier,
    )
    if payload is None or ("scheme_id" not in payload and "section_title" not in payload):
        scheme_id = _match_scheme(text)
        section_title = _match_section(text) if scheme_id else None
    else:
        scheme_id = match_allowed(payload.get("scheme_id"), scheme_ids)
        section_title = match_allowed(payload.get("section_title"), SECTION_TITLES)
    scheme_id = _accept_scheme(text, scheme_id, scheme_ids)
    if scheme_id is None:
        return None, None
    return scheme_id, _accept_section(text, section_title)


def _accept_scheme(
    text: str,
    scheme_id: str | None,
    scheme_ids: tuple[str, ...],
) -> str | None:
    """A different AMC, or two schemes in one complaint, is no match."""
    if scheme_id not in scheme_ids:
        return None
    if _OTHER_AMC.search(text) and not re.search(r"\bhdfc\b", text, re.I):
        return None
    found = [item for item, cue in _SCHEME_CUES if cue.search(text)]
    if len(found) > 1:
        return None
    return scheme_id


def _accept_section(text: str, section_title: str | None) -> str | None:
    """NAV history is not the current NAV paragraph."""
    if section_title not in SECTION_TITLES:
        return None
    if section_title == "Key fund facts" and _NAV_HISTORY.search(text):
        return None
    return section_title


def _match_scheme(text: str) -> str | None:
    found = [scheme_id for scheme_id, cue in _SCHEME_CUES if cue.search(text)]
    if len(found) != 1:
        return None
    if _OTHER_AMC.search(text) and not re.search(r"\bhdfc\b", text, re.I):
        return None
    return found[0]


def _match_section(text: str) -> str | None:
    searchable = _NAV_HISTORY.sub(" ", text)
    for title, cue in _SECTION_CUES:
        if cue.search(searchable):
            return title
    return None


def _covered(corpus: Corpus, scheme_id: str, section_title: str) -> FactsResult:
    scheme = corpus.scheme(scheme_id)
    block = scheme.section(section_title)
    return FactsResult(
        status="covered",
        scheme_id=scheme_id,
        section_title=section_title,
        fact=block.text,
        citation=scheme.source_url,
        last_checked=scheme.nav_date,
        searched=tuple(item.scheme_id for item in corpus.schemes),
    )


def _not_covered(corpus: Corpus) -> FactsResult:
    opened = tuple(item.scheme_id for item in corpus.schemes) + SECTION_TITLES
    return FactsResult(
        status="not_covered",
        scheme_id=None,
        section_title=None,
        fact=None,
        citation=None,
        last_checked=None,
        searched=opened,
    )
