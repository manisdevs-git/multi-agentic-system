"""Read-only theme index and review-date lookup for the Pattern Agent."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from types import MappingProxyType

from support_reply.corpus import data_dir


@dataclass(frozen=True, slots=True, repr=False)
class Theme:
    id: str
    name: str
    description: str
    review_ids: tuple[str, ...]
    count: int
    avg_rating: float
    score: float

    def __repr__(self) -> str:
        return f"Theme(id={self.id!r}, count={self.count!r})"


@dataclass(frozen=True, slots=True, repr=False)
class ThemeIndex:
    """Themes plus `review_id -> date`. Review text is not kept."""

    review_count: int
    all_themes: tuple[Theme, ...]
    top_theme_ids: tuple[str, ...]
    review_dates: Mapping[str, date] = field(hash=False)

    def theme(self, theme_id: str) -> Theme:
        for item in self.all_themes:
            if item.id == theme_id:
                return item
        raise KeyError(theme_id)

    def dates_for(self, review_ids: Iterable[str]) -> tuple[date, ...]:
        return tuple(self.review_dates[review_id] for review_id in review_ids)

    def __repr__(self) -> str:
        ids = [item.id for item in self.all_themes]
        return f"ThemeIndex(review_count={self.review_count!r}, themes={ids!r})"


@lru_cache(maxsize=1)
def load() -> ThemeIndex:
    """Load `allThemes` and the review-date index. Cached. Does not keep review text."""
    root = data_dir()
    reviews = _read_list(root / "normalized_reviews.json")
    dates: dict[str, date] = {}
    for review in reviews:
        review_id = review["id"]
        if not isinstance(review_id, str) or not review_id:
            raise ValueError("every review needs an id")
        if review_id in dates:
            raise ValueError(f"duplicate review id {review_id}")
        raw_date = review["date"]
        if not isinstance(raw_date, str):
            raise ValueError(f"review {review_id} has no date")
        dates[review_id] = _parse_date(raw_date)

    document = _read_object(root / "analyzed_themes.json")
    if document["reviewCount"] != len(dates):
        raise ValueError("reviewCount does not match normalized_reviews.json")

    themes = tuple(_theme(item) for item in document["allThemes"])
    seen: set[str] = set()
    for theme in themes:
        overlap = seen.intersection(theme.review_ids)
        if overlap:
            raise ValueError(f"review id {next(iter(overlap))} is in more than one theme")
        missing = [review_id for review_id in theme.review_ids if review_id not in dates]
        if missing:
            raise ValueError(f"theme {theme.id} cites unknown review id {missing[0]}")
        if theme.count != len(theme.review_ids):
            raise ValueError(f"theme {theme.id} count does not match its reviewIds")
        seen.update(theme.review_ids)

    if sum(theme.count for theme in themes) != len(dates):
        raise ValueError("theme review ids must cover every normalized review")

    top_theme_ids = tuple(item["id"] for item in document["topThemes"])
    known = {theme.id for theme in themes}
    if any(theme_id not in known for theme_id in top_theme_ids):
        raise ValueError("topThemes contains an id that is not in allThemes")

    return ThemeIndex(
        review_count=len(dates),
        all_themes=themes,
        top_theme_ids=top_theme_ids,
        review_dates=MappingProxyType(dates),
    )


def _theme(payload: dict[str, object]) -> Theme:
    review_ids = payload["reviewIds"]
    if not isinstance(review_ids, list) or any(not isinstance(item, str) for item in review_ids):
        raise ValueError(f"theme {payload.get('id')} reviewIds must be strings")
    return Theme(
        id=str(payload["id"]),
        name=str(payload["name"]),
        description=str(payload["description"]),
        review_ids=tuple(review_ids),
        count=int(payload["count"]),
        avg_rating=float(payload["avgRating"]),
        score=float(payload["score"]),
    )


def _parse_date(value: str) -> date:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).date()


def _read_object(path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _read_list(path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must be a JSON array")
    return payload
