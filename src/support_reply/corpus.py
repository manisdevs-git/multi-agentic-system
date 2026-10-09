"""Read-only loader for the Milestone 1 scheme corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SECTION_TITLES: tuple[str, ...] = (
    "Key fund facts",
    "Exit load",
    "Benchmark",
    "Investment objective",
    "Stamp duty and tax",
    "Fund management",
    "Minimum investments",
)


def data_dir() -> Path:
    """Return the repository `data/` directory."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "parsed" / "parse_manifest.json"
        if candidate.is_file():
            return candidate.parents[1]
    raise FileNotFoundError("data/parsed/parse_manifest.json not found")


@dataclass(frozen=True, slots=True, repr=False)
class TextBlock:
    section_title: str
    text: str

    def __repr__(self) -> str:
        return f"TextBlock(section_title={self.section_title!r})"


@dataclass(frozen=True, slots=True, repr=False)
class Scheme:
    scheme_id: str
    scheme_name: str
    source_url: str
    nav_date: str
    text_blocks: tuple[TextBlock, ...]

    def section(self, title: str) -> TextBlock:
        for block in self.text_blocks:
            if block.section_title == title:
                return block
        raise KeyError(title)

    def __repr__(self) -> str:
        return f"Scheme(scheme_id={self.scheme_id!r}, source_url={self.source_url!r})"


@dataclass(frozen=True, slots=True, repr=False)
class Corpus:
    schemes: tuple[Scheme, ...]

    def scheme(self, scheme_id: str) -> Scheme:
        for item in self.schemes:
            if item.scheme_id == scheme_id:
                return item
        raise KeyError(scheme_id)

    def scheme_for_url(self, source_url: str) -> Scheme:
        for item in self.schemes:
            if item.source_url == source_url:
                return item
        raise KeyError(source_url)

    @property
    def source_urls(self) -> frozenset[str]:
        return frozenset(item.source_url for item in self.schemes)

    @property
    def scheme_ids(self) -> frozenset[str]:
        return frozenset(item.scheme_id for item in self.schemes)

    def __repr__(self) -> str:
        ids = [item.scheme_id for item in self.schemes]
        return f"Corpus(schemes={ids!r})"


@lru_cache(maxsize=1)
def load() -> Corpus:
    """Load `parse_manifest.json` and the five scheme files. Cached and read-only."""
    parsed = data_dir() / "parsed"
    manifest = _read_json(parsed / "parse_manifest.json")
    entries = manifest["schemes"]
    if manifest["scheme_count"] != 5 or len(entries) != 5:
        raise ValueError("parse_manifest.json must list exactly five schemes")

    schemes: list[Scheme] = []
    for entry in entries:
        payload = _read_json(parsed / entry["file"])
        blocks = tuple(
            TextBlock(section_title=block["section_title"], text=block["text"])
            for block in payload["text_blocks"]
        )
        scheme = Scheme(
            scheme_id=payload["scheme_id"],
            scheme_name=payload["scheme_name"],
            source_url=payload["source_url"],
            nav_date=payload["nav_date"],
            text_blocks=blocks,
        )
        _check_scheme(scheme, entry, payload)
        schemes.append(scheme)

    ids = [item.scheme_id for item in schemes]
    urls = [item.source_url for item in schemes]
    if len(set(ids)) != 5 or len(set(urls)) != 5:
        raise ValueError("scheme ids and source_url values must be unique")
    return Corpus(schemes=tuple(schemes))


def _check_scheme(scheme: Scheme, entry: dict[str, object], payload: dict[str, object]) -> None:
    if scheme.scheme_id != entry["scheme_id"] or scheme.scheme_name != entry["scheme_name"]:
        raise ValueError(f"{entry['file']} does not match the manifest scheme identity")
    if scheme.nav_date != entry["nav_date"]:
        raise ValueError(f"{scheme.scheme_id} nav_date does not match the manifest")
    titles = tuple(block.section_title for block in scheme.text_blocks)
    if titles != SECTION_TITLES or payload["text_block_count"] != 7 or entry["text_block_count"] != 7:
        raise ValueError(f"{scheme.scheme_id} must contain the seven corpus sections")
    if any(not block.text.strip() for block in scheme.text_blocks):
        raise ValueError(f"{scheme.scheme_id} has an empty text block")
    if not scheme.source_url.startswith("https://groww.in/mutual-funds/"):
        raise ValueError(f"{scheme.scheme_id} source_url is outside the Groww scheme pages")


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload
