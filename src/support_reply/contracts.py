"""JSON contracts for agent results, the orchestrator draft, and session state.

`to_dict` is the stored shape. `repr` is the log shape: redacted complaint text and ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

FactsStatus = Literal["covered", "not_covered"]
Signal = Literal["recurring", "rising", "one-off"]
ComplianceFlag = Literal["advice", "performance_claim", "pii"]
Disposition = Literal["reply", "escalate", "handoff", "refuse", "disagree"]
Priority = Literal["P1", "P2", "P3"]
ReviewStatus = Literal["pass", "fail", "blocked"]
ReviewCheck = Literal["citation", "advice", "performance", "pii", "shape"]
AgentName = Literal["facts", "pattern", "compliance", "orchestrator"]
FiredKind = Literal["refinement", "reviewer_rework"]

PII_TYPES: frozenset[str] = frozenset(
    {"PAN", "AADHAAR", "ACCOUNT", "PHONE", "EMAIL", "OTP"}
)
COMPLIANCE_FLAGS: frozenset[str] = frozenset({"advice", "performance_claim", "pii"})
FACTS_STATUSES: frozenset[str] = frozenset({"covered", "not_covered"})
SIGNALS: frozenset[str] = frozenset({"recurring", "rising", "one-off"})
DISPOSITIONS: frozenset[str] = frozenset(
    {"reply", "escalate", "handoff", "refuse", "disagree"}
)
COVERED_DISPOSITIONS: frozenset[str] = frozenset({"reply", "escalate", "disagree"})
PRIORITIES: frozenset[str] = frozenset({"P1", "P2", "P3"})
REVIEW_STATUSES: frozenset[str] = frozenset({"pass", "fail", "blocked"})
REVIEW_CHECKS: frozenset[str] = frozenset(
    {"citation", "advice", "performance", "pii", "shape"}
)
AGENT_NAMES: frozenset[str] = frozenset(
    {"facts", "pattern", "compliance", "orchestrator"}
)
FIRED_KINDS: frozenset[str] = frozenset({"refinement", "reviewer_rework"})


@dataclass(frozen=True, slots=True, repr=False)
class DateWindow:
    start: str
    end: str

    def __post_init__(self) -> None:
        start = _iso_date(self.start, "window.start")
        end = _iso_date(self.end, "window.end")
        if start > end:
            raise ValueError("window start must be on or before window end")

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end}

    def __repr__(self) -> str:
        return f"DateWindow(start={self.start!r}, end={self.end!r})"


@dataclass(frozen=True, slots=True, repr=False)
class FactsResult:
    status: FactsStatus
    scheme_id: str | None
    section_title: str | None
    fact: str | None
    citation: str | None
    last_checked: str | None
    searched: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "searched", _string_tuple(self.searched, "searched"))
        if self.status not in FACTS_STATUSES:
            raise ValueError("status must be covered or not_covered")
        if self.citation is not None and self.citation not in _corpus().source_urls:
            raise ValueError("citation must be one of the five corpus source_url values")
        if self.status == "not_covered":
            _require_nulls(
                scheme_id=self.scheme_id,
                section_title=self.section_title,
                fact=self.fact,
                citation=self.citation,
                last_checked=self.last_checked,
            )
            if not self.searched:
                raise ValueError("not_covered searched must name the schemes and sections opened")
            return
        if not isinstance(self.scheme_id, str):
            raise ValueError("covered scheme_id is required")
        try:
            scheme = _corpus().scheme(self.scheme_id)
        except KeyError as exc:
            raise ValueError(f"unknown scheme_id {self.scheme_id!r}") from exc
        if self.citation != scheme.source_url:
            raise ValueError("citation must be the source_url for scheme_id " + self.scheme_id)
        if self.last_checked != scheme.nav_date:
            raise ValueError("last_checked must be the scheme nav_date")
        try:
            block = scheme.section(self.section_title or "")
        except KeyError as exc:
            raise ValueError("section_title must be one of the scheme sections") from exc
        if self.fact != block.text:
            raise ValueError("fact must be copied from that section text_block")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scheme_id": self.scheme_id,
            "section_title": self.section_title,
            "fact": self.fact,
            "citation": self.citation,
            "last_checked": self.last_checked,
            "searched": list(self.searched),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FactsResult:
        _reject_unknown(
            payload,
            {"status", "scheme_id", "section_title", "fact", "citation", "last_checked", "searched"},
            "FactsResult",
        )
        return cls(
            status=payload["status"],
            scheme_id=payload.get("scheme_id"),
            section_title=payload.get("section_title"),
            fact=payload.get("fact"),
            citation=payload.get("citation"),
            last_checked=payload.get("last_checked"),
            searched=payload.get("searched", ()),
        )

    def __repr__(self) -> str:
        return (
            f"FactsResult(status={self.status!r}, scheme_id={self.scheme_id!r}, "
            f"citation={self.citation!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PatternResult:
    theme_id: str
    theme_name: str
    signal: Signal
    count: int
    recent_count: int
    window: DateWindow
    recent_window: DateWindow

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError("signal must be recurring, rising, or one-off")
        if self.window != FULL_WINDOW or self.recent_window != RECENT_WINDOW:
            raise ValueError("window must be the review file and the last seven days of it")
        try:
            theme = _themes().theme(self.theme_id)
        except KeyError as exc:
            raise ValueError(f"unknown theme_id {self.theme_id!r}") from exc
        if self.theme_name != theme.name:
            raise ValueError("theme_name must match the theme id")
        if type(self.count) is not int or self.count != theme.count:
            raise ValueError("count must be the theme full-window count")
        if type(self.recent_count) is not int or not 0 <= self.recent_count <= self.count:
            raise ValueError("recent_count must be an int from 0 through count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "signal": self.signal,
            "count": self.count,
            "recent_count": self.recent_count,
            "window": self.window.to_dict(),
            "recent_window": self.recent_window.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PatternResult:
        _reject_unknown(
            payload,
            {
                "theme_id",
                "theme_name",
                "signal",
                "count",
                "recent_count",
                "window",
                "recent_window",
            },
            "PatternResult",
        )
        return cls(
            theme_id=payload["theme_id"],
            theme_name=payload["theme_name"],
            signal=payload["signal"],
            count=payload["count"],
            recent_count=payload["recent_count"],
            window=_window(payload["window"]),
            recent_window=_window(payload["recent_window"]),
        )

    def __repr__(self) -> str:
        return f"PatternResult(theme_id={self.theme_id!r}, signal={self.signal!r})"


@dataclass(frozen=True, slots=True, repr=False)
class ComplianceResult:
    flags: tuple[ComplianceFlag, ...]
    pii_types: tuple[str, ...]
    advice_refusal: bool
    redacted_text: str

    def __post_init__(self) -> None:
        flags = _string_tuple(self.flags, "flags")
        pii_types = _pii_types(self.pii_types)
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "pii_types", pii_types)
        unknown = [flag for flag in flags if flag not in COMPLIANCE_FLAGS]
        if unknown:
            raise ValueError(f"flags must be advice, performance_claim, or pii: {unknown}")
        if len(flags) != len(set(flags)):
            raise ValueError("flags lists each flag once")
        if type(self.advice_refusal) is not bool:
            raise TypeError("advice_refusal must be a bool")
        expects_refusal = "advice" in flags or "performance_claim" in flags
        if self.advice_refusal != expects_refusal:
            raise ValueError("advice_refusal must match advice and performance_claim flags")
        if not isinstance(self.redacted_text, str):
            raise TypeError("redacted_text must be a str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": list(self.flags),
            "pii_types": list(self.pii_types),
            "advice_refusal": self.advice_refusal,
            "redacted_text": self.redacted_text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComplianceResult:
        _reject_unknown(
            payload,
            {"flags", "pii_types", "advice_refusal", "redacted_text"},
            "ComplianceResult",
        )
        return cls(
            flags=payload.get("flags", ()),
            pii_types=payload.get("pii_types", ()),
            advice_refusal=payload["advice_refusal"],
            redacted_text=payload["redacted_text"],
        )

    def __repr__(self) -> str:
        return (
            f"ComplianceResult(flags={list(self.flags)!r}, pii_types={list(self.pii_types)!r}, "
            f"advice_refusal={self.advice_refusal!r}, redacted_text={self.redacted_text!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class RoutingNote:
    team: str
    priority: Priority
    why: str

    def __post_init__(self) -> None:
        if not isinstance(self.team, str) or not self.team.strip():
            raise ValueError("team must be non-empty")
        if self.priority not in PRIORITIES:
            raise ValueError("priority must be P1, P2, or P3")
        if not isinstance(self.why, str) or not self.why.strip():
            raise ValueError("why must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"team": self.team, "priority": self.priority, "why": self.why}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoutingNote:
        _reject_unknown(payload, {"team", "priority", "why"}, "RoutingNote")
        return cls(team=payload["team"], priority=payload["priority"], why=payload["why"])

    def __repr__(self) -> str:
        return f"RoutingNote(priority={self.priority!r})"


@dataclass(frozen=True, slots=True, repr=False)
class Draft:
    """Orchestrator output: user-facing reply plus the internal routing note."""

    reply: str
    last_checked: str | None
    citation: str | None
    disposition: Disposition
    disagreement: bool
    routing_note: RoutingNote

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                "disposition must be reply, escalate, handoff, refuse, or disagree"
            )
        if type(self.disagreement) is not bool or self.disagreement != (
            self.disposition == "disagree"
        ):
            raise ValueError("disagreement is true only when disposition is disagree")
        if not isinstance(self.routing_note, RoutingNote):
            raise TypeError("routing_note must be a RoutingNote")
        if not isinstance(self.reply, str) or not self.reply.strip():
            raise ValueError("reply must be non-empty text")
        if self.disposition in COVERED_DISPOSITIONS:
            if self.citation not in _corpus().source_urls:
                raise ValueError("citation must be one of the five corpus source_url values")
            scheme = _corpus().scheme_for_url(self.citation or "")
            if self.last_checked != scheme.nav_date:
                raise ValueError("last_checked must be the scheme nav_date")
            return
        if self.citation is not None or self.last_checked is not None:
            raise ValueError("handoff and refuse carry no citation and no last-checked date")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "last_checked": self.last_checked,
            "citation": self.citation,
            "disposition": self.disposition,
            "disagreement": self.disagreement,
            "routing_note": self.routing_note.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Draft:
        _reject_unknown(
            payload,
            {"reply", "last_checked", "citation", "disposition", "disagreement", "routing_note"},
            "Draft",
        )
        note = payload["routing_note"]
        if isinstance(note, dict):
            note = RoutingNote.from_dict(note)
        return cls(
            reply=payload["reply"],
            last_checked=payload.get("last_checked"),
            citation=payload.get("citation"),
            disposition=payload["disposition"],
            disagreement=payload["disagreement"],
            routing_note=note,
        )

    def __repr__(self) -> str:
        return (
            f"Draft(disposition={self.disposition!r}, citation={self.citation!r}, "
            f"disagreement={self.disagreement!r}, priority={self.routing_note.priority!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ReviewFailure:
    check: ReviewCheck
    detail: str

    def __post_init__(self) -> None:
        if self.check not in REVIEW_CHECKS:
            raise ValueError("check must be citation, advice, performance, pii, or shape")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("detail must name the offending span")

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewFailure:
        _reject_unknown(payload, {"check", "detail"}, "ReviewFailure")
        return cls(check=payload["check"], detail=payload["detail"])

    def __repr__(self) -> str:
        return f"ReviewFailure(check={self.check!r})"


@dataclass(frozen=True, slots=True, repr=False)
class ReviewerResult:
    status: ReviewStatus
    round: int
    failures: tuple[ReviewFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failures",
            _instance_tuple(self.failures, ReviewFailure, "failures"),
        )
        if self.status not in REVIEW_STATUSES:
            raise ValueError("status must be pass, fail, or blocked")
        if type(self.round) is not int:
            raise ValueError("round must be an int")
        if self.status == "fail" and self.round != 1:
            raise ValueError("fail is review round 1")
        if self.status == "blocked" and self.round != 2:
            raise ValueError("blocked is review round 2")
        if self.status == "pass" and self.round not in (1, 2):
            raise ValueError("pass is review round 1 or 2")
        if self.status == "pass" and self.failures:
            raise ValueError("pass cannot include failures")
        if self.status in ("fail", "blocked") and not self.failures:
            raise ValueError("a failed review names at least one check")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "round": self.round,
            "failures": [failure.to_dict() for failure in self.failures],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewerResult:
        _reject_unknown(payload, {"status", "round", "failures"}, "ReviewerResult")
        failures = payload.get("failures", ())
        if isinstance(failures, list):
            failures = tuple(
                item if isinstance(item, ReviewFailure) else ReviewFailure.from_dict(item)
                for item in failures
            )
        return cls(status=payload["status"], round=payload["round"], failures=failures)

    def __repr__(self) -> str:
        checks = [failure.check for failure in self.failures]
        return (
            f"ReviewerResult(status={self.status!r}, round={self.round!r}, checks={checks!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AgentsFired:
    """One log entry per refinement and per reviewer rework.

    `round` is the refinement round the entry belongs to. Refinements start at 1.
    A reviewer rework carries the round of the draft it fixed, so a rework on the
    first complaint, before any refinement, is round 0.
    """

    round: int
    kind: FiredKind
    ran: tuple[AgentName, ...]
    skipped: tuple[AgentName, ...]

    def __post_init__(self) -> None:
        ran = _agent_tuple(self.ran, "ran")
        skipped = _agent_tuple(self.skipped, "skipped")
        object.__setattr__(self, "ran", ran)
        object.__setattr__(self, "skipped", skipped)
        if self.kind not in FIRED_KINDS:
            raise ValueError("kind must be refinement or reviewer_rework")
        lowest = 0 if self.kind == "reviewer_rework" else 1
        if type(self.round) is not int or self.round < lowest:
            raise ValueError(f"{self.kind} round must be an int >= {lowest}")
        if set(ran) & set(skipped):
            raise ValueError("ran and skipped must be disjoint")
        if set(ran) | set(skipped) != AGENT_NAMES:
            raise ValueError("agents_fired must name every agent as ran or skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "kind": self.kind,
            "ran": list(self.ran),
            "skipped": list(self.skipped),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentsFired:
        _reject_unknown(payload, {"round", "kind", "ran", "skipped"}, "AgentsFired")
        return cls(
            round=payload["round"],
            kind=payload["kind"],
            ran=payload["ran"],
            skipped=payload["skipped"],
        )

    def __repr__(self) -> str:
        return (
            f"AgentsFired(round={self.round!r}, kind={self.kind!r}, "
            f"ran={list(self.ran)!r}, skipped={list(self.skipped)!r})"
        )


@dataclass(slots=True, repr=False)
class Session:
    """One complaint thread. The raw complaint is not a field."""

    redacted_text: str = ""
    pii_types: tuple[str, ...] = ()
    facts: FactsResult | None = None
    pattern: PatternResult | None = None
    compliance: ComplianceResult | None = None
    draft: Draft | None = None
    routing_note: RoutingNote | None = None
    review: ReviewerResult | None = None
    advice_refusal: bool = False
    refinement_count: int = 0
    agents_fired: tuple[AgentsFired, ...] = ()

    def __setattr__(self, name: str, value: object) -> None:
        if name not in self.__dataclass_fields__:
            raise AttributeError(
                f"Session cannot store {name!r}. The raw complaint is not a session field; "
                "only redacted_text is."
            )
        if isinstance(value, str) and name != "redacted_text":
            raise TypeError(
                f"Session cannot store a raw string on {name!r}. "
                "Only redacted_text holds complaint text."
            )
        value = _coerce_session_value(name, value)
        if (
            name == "advice_refusal"
            and value is False
            and getattr(self, "advice_refusal", False) is True
        ):
            raise ValueError("advice_refusal is sticky once set")
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "redacted_text": self.redacted_text,
            "pii_types": list(self.pii_types),
            "facts": None if self.facts is None else self.facts.to_dict(),
            "pattern": None if self.pattern is None else self.pattern.to_dict(),
            "compliance": None if self.compliance is None else self.compliance.to_dict(),
            "draft": None if self.draft is None else self.draft.to_dict(),
            "routing_note": None if self.routing_note is None else self.routing_note.to_dict(),
            "review": None if self.review is None else self.review.to_dict(),
            "advice_refusal": self.advice_refusal,
            "refinement_count": self.refinement_count,
            "agents_fired": [entry.to_dict() for entry in self.agents_fired],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Session:
        if not isinstance(payload, Mapping):
            raise TypeError("Session payload must be a mapping")
        _reject_unknown(payload, set(cls.__dataclass_fields__), "Session")
        data = dict(payload)
        data["facts"] = _nested(data.get("facts"), FactsResult)
        data["pattern"] = _nested(data.get("pattern"), PatternResult)
        data["compliance"] = _nested(data.get("compliance"), ComplianceResult)
        data["draft"] = _nested(data.get("draft"), Draft)
        data["routing_note"] = _nested(data.get("routing_note"), RoutingNote)
        data["review"] = _nested(data.get("review"), ReviewerResult)
        fired = data.get("agents_fired", ())
        if isinstance(fired, list):
            data["agents_fired"] = tuple(
                item if isinstance(item, AgentsFired) else AgentsFired.from_dict(item)
                for item in fired
            )
        return cls(**data)

    def __repr__(self) -> str:
        scheme_id = None if self.facts is None else self.facts.scheme_id
        theme_id = None if self.pattern is None else self.pattern.theme_id
        disposition = None if self.draft is None else self.draft.disposition
        review_status = None if self.review is None else self.review.status
        fired = [
            {"round": entry.round, "kind": entry.kind, "ran": list(entry.ran)}
            for entry in self.agents_fired
        ]
        return (
            "Session("
            f"redacted_text={self.redacted_text!r}, "
            f"pii_types={list(self.pii_types)!r}, "
            f"scheme_id={scheme_id!r}, "
            f"theme_id={theme_id!r}, "
            f"disposition={disposition!r}, "
            f"review_status={review_status!r}, "
            f"advice_refusal={self.advice_refusal!r}, "
            f"refinement_count={self.refinement_count!r}, "
            f"agents_fired={fired!r})"
        )


def _corpus():
    from support_reply.corpus import load

    return load()


def _themes():
    from support_reply.themes import load

    return load()


def _iso_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


FULL_WINDOW = DateWindow(start="2026-04-12", end="2026-07-04")
RECENT_WINDOW = DateWindow(start="2026-06-28", end="2026-07-04")


def _window(value: object) -> DateWindow:
    if isinstance(value, DateWindow):
        return value
    if isinstance(value, Mapping):
        return DateWindow(start=value["start"], end=value["end"])
    raise TypeError("window must be a DateWindow")


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple of strings")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must be a tuple of strings")
    return items


def _pii_types(value: object) -> tuple[str, ...]:
    items = _string_tuple(value, "pii_types")
    unknown = [item for item in items if item not in PII_TYPES]
    if unknown:
        raise ValueError(f"pii_types must be token types, not values: {unknown}")
    if len(items) != len(set(items)):
        raise ValueError("pii_types lists each type once")
    return items


def _agent_tuple(value: object, field_name: str) -> tuple[AgentName, ...]:
    items = _string_tuple(value, field_name)
    unknown = [item for item in items if item not in AGENT_NAMES]
    if unknown:
        raise ValueError(f"{field_name} has unknown agent {unknown[0]!r}")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} lists each agent once")
    return tuple(items)  # type: ignore[return-value]


def _instance_tuple(value: object, kind: type[Any], field_name: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple of {kind.__name__}")
    items = tuple(value)
    for item in items:
        if not isinstance(item, kind):
            raise TypeError(f"{field_name} must be a tuple of {kind.__name__}")
    return items


def _require_nulls(**fields: object) -> None:
    present = [name for name, value in fields.items() if value is not None]
    if present:
        raise ValueError("not_covered sets " + ", ".join(present) + " to null")


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{label} cannot store {', '.join(unknown)}. "
            "The raw complaint is not a field; only redacted_text is, on Session and ComplianceResult."
        )


def _nested(value: object, kind: type[Any]) -> object:
    if value is None or isinstance(value, kind):
        return value
    if isinstance(value, Mapping):
        return kind.from_dict(value)
    raise TypeError(f"expected {kind.__name__} or null")


def _coerce_session_value(name: str, value: object) -> object:
    if name == "redacted_text":
        if not isinstance(value, str):
            raise TypeError("redacted_text must be a str")
        return value
    if name == "pii_types":
        return _pii_types(value)
    if name == "facts":
        return _optional(value, FactsResult, name)
    if name == "pattern":
        return _optional(value, PatternResult, name)
    if name == "compliance":
        return _optional(value, ComplianceResult, name)
    if name == "draft":
        return _optional(value, Draft, name)
    if name == "routing_note":
        return _optional(value, RoutingNote, name)
    if name == "review":
        return _optional(value, ReviewerResult, name)
    if name == "advice_refusal":
        if type(value) is not bool:
            raise TypeError("advice_refusal must be a bool")
        return value
    if name == "refinement_count":
        if type(value) is not int or not 0 <= value <= 5:
            raise ValueError("refinement_count must be an int from 0 to 5")
        return value
    if name == "agents_fired":
        return _instance_tuple(value, AgentsFired, name)
    raise AssertionError(name)


def _optional(value: object, kind: type[Any], name: str) -> object:
    if value is None or isinstance(value, kind):
        return value
    raise TypeError(f"{name} must be a {kind.__name__} or None")
