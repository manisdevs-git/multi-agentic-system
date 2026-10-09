"""Phase 3 Pattern Agent exit: theme cues, date counts, and the rising bar of 40."""

import ast
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from support_reply.contracts import FULL_WINDOW, RECENT_WINDOW
from support_reply.pattern import answer, compile_graph, signal_for
from support_reply.themes import ThemeIndex, load as load_themes

EXIT_LOAD = "Why was I charged an exit load on HDFC Large Cap Fund?"


def test_exit_load_maps_to_fees_brokerage_outside_top_themes():
    themes = load_themes()
    result = answer(EXIT_LOAD)

    assert result.theme_id == "fees_brokerage"
    assert result.theme_name == "Fees & brokerage"
    assert result.count == 136
    assert result.recent_count == 12
    assert result.signal == "recurring"
    assert result.theme_id not in themes.top_theme_ids
    assert result.window == FULL_WINDOW
    assert result.recent_window == RECENT_WINDOW
    assert result.to_dict()["window"] == {"start": "2026-04-12", "end": "2026-07-04"}
    assert result.to_dict()["recent_window"] == {"start": "2026-06-28", "end": "2026-07-04"}


def test_other_is_always_one_off():
    result = answer("The mutual fund page looks fine")

    assert result.theme_id == "other"
    assert result.theme_name == "Other feedback"
    assert result.count == 709
    assert result.recent_count == 86
    assert result.signal == "one-off"
    assert signal_for("other", count=709, recent_count=40, earlier_count=669) == "one-off"


def test_recent_slice_of_40_is_rising():
    """A fee theme whose recent slice holds 40 reviews clears the rising bar."""
    real = load_themes().theme("fees_brokerage")
    review_ids = tuple(f"fixture-{i}" for i in range(real.count))
    recent = 40
    dates = {
        review_id: date(2026, 6, 28) if i < recent else date(2026, 4, 12)
        for i, review_id in enumerate(review_ids)
    }
    index = ThemeIndex(
        review_count=real.count,
        all_themes=(replace(real, review_ids=review_ids),),
        top_theme_ids=(),
        review_dates=dates,
    )

    result = answer(EXIT_LOAD, index=index)

    assert result.theme_id == "fees_brokerage"
    assert result.count == 136
    assert result.recent_count == 40
    assert result.signal == "rising"


def test_rising_also_trips_on_twice_the_earlier_weekly_rate():
    # Earlier window is 77 inclusive days. 77 reviews there is a weekly rate of 7,
    # so 14 in the recent slice is exactly 2x and 13 is just under.
    assert signal_for("fees_brokerage", count=136, recent_count=14, earlier_count=77) == "rising"
    assert signal_for("fees_brokerage", count=136, recent_count=13, earlier_count=77) == "recurring"
    assert signal_for("fees_brokerage", count=19, recent_count=40, earlier_count=0) == "one-off"


@pytest.mark.parametrize(
    "text,theme_id,count,recent_count,signal",
    [
        ("The expense ratio went up", "fees_brokerage", 136, 12, "recurring"),
        ("My UPI payment is stuck", "money_movement", 162, 17, "recurring"),
        ("The buy/sell button does nothing", "trading_charts", 503, 50, "rising"),
        ("The app crashed on login", "reliability", 140, 11, "recurring"),
    ],
)
def test_cue_table_uses_real_date_counts(text, theme_id, count, recent_count, signal):
    result = answer(text)

    assert result.theme_id == theme_id
    assert result.count == count
    assert result.recent_count == recent_count
    assert result.signal == signal


def test_first_cue_wins_when_a_complaint_matches_two_themes():
    assert answer("I was charged a fee and then the app crashed").theme_id == "fees_brokerage"
    assert answer("My withdrawal failed and the app crashed").theme_id == "money_movement"
    assert answer("The chart froze").theme_id == "trading_charts"


def test_pattern_log_is_the_result_repr(caplog):
    caplog.set_level(logging.INFO, logger="support_reply.pattern")
    result = answer("CASE-9f3a " + EXIT_LOAD)

    assert "fees_brokerage" in caplog.text
    assert result.signal in caplog.text
    assert "CASE-9f3a" not in caplog.text
    assert "CASE-9f3a" not in repr(result)


def test_pattern_module_does_not_import_facts_or_compliance():
    import support_reply.pattern as pattern

    tree = ast.parse(Path(pattern.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)

    assert not any(name in {"facts", "compliance"} or name.endswith((".facts", ".compliance")) for name in imported)


def test_pattern_rejects_non_text():
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


def test_pattern_graph_classifies_then_counts():
    edges = {(edge.source, edge.target) for edge in compile_graph().get_graph().edges}

    assert edges == {
        ("__start__", "classify"),
        ("classify", "count"),
        ("count", "__end__"),
    }


def test_fake_classifier_matches_the_cue_path_and_does_not_supply_the_count():
    spy = _Spy({"theme_id": "fees_brokerage"})

    assert answer(EXIT_LOAD, classifier=spy) == answer(EXIT_LOAD)
    assert spy.calls[0]["task"] == "pattern"
    assert "fees_brokerage" in spy.calls[0]["allowed"]["theme_id"]
    assert "136" not in str(spy.calls)
    assert answer(EXIT_LOAD, classifier=spy).count == 136


def test_unknown_theme_id_is_other():
    spy = _Spy({"theme_id": "not_a_theme"})

    result = answer(EXIT_LOAD, classifier=spy)

    assert result.theme_id == "other"
    assert result.theme_name == "Other feedback"
    assert result.count == 709
    assert result.signal == "one-off"


def test_classifier_failure_uses_the_cue_table():
    spy = _Spy({}, raises=True)

    assert answer(EXIT_LOAD, classifier=spy) == answer(EXIT_LOAD)
