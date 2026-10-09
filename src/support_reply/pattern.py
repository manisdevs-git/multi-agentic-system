"""Pattern Agent. Reads the redacted complaint, `allThemes`, and review dates.

LangGraph subgraph: classify a theme, then count that theme's review dates.
Groq classifies when `GROQ_API_KEY` is set. With no key, or when the call
fails, the cue table assigns the theme. An unknown id is `other`.
The model does not supply `count` or `signal`.
It does not read Facts or Compliance output.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from support_reply.contracts import FULL_WINDOW, RECENT_WINDOW, PatternResult, Signal
from support_reply.llm import TASK_PATTERN, Classifier, classify_or_none, match_allowed
from support_reply.themes import ThemeIndex, load

logger = logging.getLogger(__name__)

# Day before the recent slice. Inclusive, so the two slices partition the full window.
_RECENT_START = date.fromisoformat(RECENT_WINDOW.start)
_RECENT_END = date.fromisoformat(RECENT_WINDOW.end)
_EARLIER_START = date.fromisoformat(FULL_WINDOW.start)
_EARLIER_END = _RECENT_START - timedelta(days=1)
_EARLIER_DAYS = (_EARLIER_END - _EARLIER_START).days + 1

_RISING_RECENT_COUNT = 40
_RECURRING_MIN_COUNT = 20

# First matching cue wins, so a fee complaint is fees_brokerage even when the
# same sentence also mentions an order, a chart, or a crash.
_THEME_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "fees_brokerage",
        re.compile(
            r"\bexit[\s-]?loads?\b|\bbrokerage\b|\bfees?\b|\bcharg(?:e|es|ed)\b"
            r"|\bcommissions?\b|\bexpense ratio\b",
            re.I,
        ),
    ),
    (
        "money_movement",
        re.compile(
            r"\bwithdr(?:aw\w*|ew)\b|\bdeposits?\b|\bpayments?\b|\brefunds?\b|\bupi\b|\bbanks?\b",
            re.I,
        ),
    ),
    (
        "trading_charts",
        re.compile(
            r"\bcharts?\b|\borders?\b|\btrades?\b|\bgtt\b|\bbuy/sell(?:\s+button)?\b",
            re.I,
        ),
    ),
    (
        "reliability",
        re.compile(
            r"\bcrash(?:es|ed|ing)?\b|\bslow\b|\blags?\b|\bfreez(?:e|es|ing|en)\b"
            r"|\bbugs?\b|\blog[\s-]?in\b",
            re.I,
        ),
    ),
)


class PatternState(TypedDict, total=False):
    redacted_text: str
    theme_id: str
    result: PatternResult


def answer(
    redacted_text: str,
    *,
    index: ThemeIndex | None = None,
    classifier: Classifier | None = None,
) -> PatternResult:
    """Classify one complaint. `index` defaults to the loaded theme and date index."""
    if not isinstance(redacted_text, str):
        raise TypeError("pattern expects a str")

    themes = load() if index is None else index
    state = compile_graph(index=themes, classifier=classifier).invoke(
        {"redacted_text": redacted_text}
    )
    result = state["result"]
    if not isinstance(result, PatternResult):
        raise TypeError("pattern graph did not return a PatternResult")
    return result


def compile_graph(*, index: ThemeIndex | None = None, classifier: Classifier | None = None):
    """Classify, then count. Phase 6 mounts this compiled graph as the Pattern node."""
    themes = load() if index is None else index

    def classify(state: PatternState) -> dict[str, str]:
        return {"theme_id": _theme_id(state["redacted_text"], classifier, themes)}

    def count(state: PatternState) -> dict[str, PatternResult]:
        theme = themes.theme(state["theme_id"])
        dates = themes.dates_for(theme.review_ids)
        recent_count = sum(1 for day in dates if _RECENT_START <= day <= _RECENT_END)
        earlier_count = sum(1 for day in dates if _EARLIER_START <= day <= _EARLIER_END)
        result = PatternResult(
            theme_id=theme.id,
            theme_name=theme.name,
            signal=signal_for(
                theme.id,
                count=theme.count,
                recent_count=recent_count,
                earlier_count=earlier_count,
            ),
            count=theme.count,
            recent_count=recent_count,
            window=FULL_WINDOW,
            recent_window=RECENT_WINDOW,
        )
        logger.info("pattern %r", result)
        return {"result": result}

    graph = StateGraph(PatternState)
    graph.add_node("classify", classify)
    graph.add_node("count", count)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "count")
    graph.add_edge("count", END)
    return graph.compile()


def _theme_id(text: str, classifier: Classifier | None, index: ThemeIndex) -> str:
    allowed = tuple(item.id for item in index.all_themes)
    payload = classify_or_none(
        task=TASK_PATTERN,
        redacted_text=text,
        allowed={"theme_id": allowed},
        labels={item.id: item.name for item in index.all_themes},
        classifier=classifier,
    )
    if payload is None or "theme_id" not in payload:
        return _match_theme(text)
    matched = match_allowed(payload.get("theme_id"), allowed)
    if matched is not None:
        return matched
    return "other"


def signal_for(
    theme_id: str,
    *,
    count: int,
    recent_count: int,
    earlier_count: int,
) -> Signal:
    """Label the theme from the full-window count and the recent slice.

    Weekly rate is `earlier_count / inclusive_days * 7`. Rising uses `>=`.
    """
    if theme_id == "other" or count < _RECURRING_MIN_COUNT:
        return "one-off"
    weekly_rate = earlier_count / _EARLIER_DAYS * 7
    if recent_count >= _RISING_RECENT_COUNT or recent_count >= 2 * weekly_rate:
        return "rising"
    return "recurring"


def _match_theme(text: str) -> str:
    for theme_id, cue in _THEME_CUES:
        if cue.search(text):
            return theme_id
    return "other"
