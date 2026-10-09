"""Phase 0 loader exit: five schemes, seven sections, 1,650 review dates."""

from datetime import date

import pytest

from support_reply.corpus import SECTION_TITLES, load as load_corpus
from support_reply.themes import Theme, load as load_themes

REVIEW_PHRASE = "groww is the bestest app"


def test_corpus_has_five_schemes_with_seven_sections():
    corpus = load_corpus()

    assert load_corpus() is corpus
    assert len(corpus.schemes) == 5
    assert corpus.scheme_ids == {
        "large_cap",
        "mid_cap",
        "small_cap",
        "gold_fof",
        "silver_fof",
    }
    assert len(corpus.source_urls) == 5
    for scheme in corpus.schemes:
        titles = tuple(block.section_title for block in scheme.text_blocks)
        assert titles == SECTION_TITLES
        assert scheme.nav_date == "25-Jun-2026"
        exit_load = scheme.section("Exit load")
        assert exit_load.text
        assert exit_load.text not in repr(scheme)
        assert exit_load.text not in repr(corpus)


def test_theme_index_has_1650_review_dates():
    index = load_themes()

    assert load_themes() is index
    assert index.review_count == 1650
    assert len(index.review_dates) == 1650
    assert all(isinstance(value, date) for value in index.review_dates.values())
    assert min(index.review_dates.values()) == date(2026, 4, 12)
    assert max(index.review_dates.values()) == date(2026, 7, 4)

    fees = index.theme("fees_brokerage")
    assert fees.name == "Fees & brokerage"
    assert fees.count == 136
    assert len(index.dates_for(fees.review_ids)) == 136
    assert index.top_theme_ids == ("trading_charts", "money_movement", "reliability")
    assert "fees_brokerage" not in index.top_theme_ids
    assert {theme.id for theme in index.all_themes} == {
        "other",
        "trading_charts",
        "money_movement",
        "reliability",
        "fees_brokerage",
    }


def test_theme_index_does_not_keep_review_text():
    index = load_themes()

    assert set(Theme.__dataclass_fields__) == {
        "id",
        "name",
        "description",
        "review_ids",
        "count",
        "avg_rating",
        "score",
    }
    assert REVIEW_PHRASE not in repr(index)
    for theme in index.all_themes:
        assert REVIEW_PHRASE not in theme.description
        assert REVIEW_PHRASE not in repr(theme)


def test_review_dates_are_read_only():
    index = load_themes()

    with pytest.raises(TypeError):
        index.review_dates["new-id"] = date(2026, 4, 12)
