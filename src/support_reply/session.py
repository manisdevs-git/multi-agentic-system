"""Session refinement. The human pushes back; only the agents the pushback names re-run.

One session is one complaint thread (`contracts.Session`). After the first reviewed
draft the human may push back up to `MAX_REFINEMENTS` times. `refine` classifies the
pushback in code, starts only the named agent nodes through the one compiled graph,
copies every other result from the session, and then runs draft and reviewer as on a
complaint, including the reviewer's one rework. That rework is not one of the five.

| Pushback | Runs | Skipped |
|---|---|---|
| "Shorter", or any wording or length change | orchestrator | facts, pattern, compliance |
| A different scheme | facts, then orchestrator | pattern, compliance |
| A fee issue, last week, a theme or a window | pattern, then orchestrator | facts, compliance |
| Both a scheme and a theme | facts, pattern, then orchestrator | compliance |
| The complaint text itself changed (`complaint=`) | all three, then orchestrator | — |

The pushback is redacted before anything reads it and is not stored. Facts and
Pattern receive redacted text only. Compliance re-runs only when the complaint text
changes, because its flags are about what the user wrote, not about the draft.

`advice_refusal` stays sticky through every refinement: the graph takes the session's
flag in, so a later "just tell me if large cap is worth it" is still a refusal even
when Facts would now cover an exit load.

On the sixth pushback `refine` raises `RefinementRefused` and leaves the session, and
its last reviewed draft, unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from support_reply.contracts import AGENT_NAMES, AgentName, AgentsFired, Session
from support_reply.graph import run, store_turn
from support_reply.ingress import scan
from support_reply.llm import Classifier
from support_reply.trace import AGENT_NODES, Trace

logger = logging.getLogger(__name__)

MAX_REFINEMENTS = 5
ORCHESTRATOR: AgentName = "orchestrator"

PushbackKind = Literal["wording", "scheme", "theme", "scheme_and_theme", "complaint"]
PUSHBACK_KINDS: frozenset[str] = frozenset(
    {"wording", "scheme", "theme", "scheme_and_theme", "complaint"}
)

# --- scheme cues ------------------------------------------------------------------------
# The name written into the complaint when the pushback moves it to another scheme.
# Each one trips the Facts scheme cue for that id.
SCHEME_NAMES: dict[str, str] = {
    "large_cap": "HDFC Large Cap Fund",
    "mid_cap": "HDFC Mid Cap Fund",
    "small_cap": "HDFC Small Cap Fund",
    "gold_fof": "HDFC Gold ETF Fund of Fund",
    "silver_fof": "HDFC Silver ETF FoF",
}
_FOF = r"(?:\s+etf)?(?:\s+(?:fund\s+of\s+funds?|fof))?"
_TAIL = r"(?:\s+fund)?(?:\s+direct(?:\s+plan)?)?(?:\s+growth)?"
# One alternation, so a pass over the text finds each mention once. Gold and silver
# come first; their phrases never contain an equity-cap word.
_SCHEME_MENTION = re.compile(
    r"(?:\bhdfc\s+)?(?:"
    rf"(?P<gold_fof>\bgold{_FOF}{_TAIL})"
    rf"|(?P<silver_fof>\bsilver{_FOF}{_TAIL})"
    rf"|(?P<large_cap>\blarge[\s-]?cap{_TAIL})"
    rf"|(?P<mid_cap>\bmid[\s-]?cap{_TAIL})"
    rf"|(?P<small_cap>\bsmall[\s-]?cap{_TAIL})"
    r")\b",
    re.I,
)
# "a different scheme", "wrong fund", "not that scheme": Facts re-runs, but the
# pushback names no scheme, so the complaint goes in without one.
_OTHER_SCHEME = re.compile(
    r"\b(?:different|another|other|wrong|not\s+(?:the|that|this)(?:\s+same)?)\s+"
    r"(?:scheme|fund|fof|etf)\b",
    re.I,
)

# --- theme and window cues --------------------------------------------------------------
# The Phase 3 theme words. When the pushback carries one, Pattern classifies the
# pushback; otherwise it recounts the complaint's theme.
_THEME_WORD = re.compile(
    r"\bexit[\s-]?loads?\b|\bbrokerage\b|\bfees?\b|\bcharg(?:e|es|ed|ing)\b"
    r"|\bcommissions?\b|\bexpense\s+ratio\b"
    r"|\bwithdr(?:aw\w*|ew)\b|\bdeposits?\b|\bpayments?\b|\brefunds?\b|\bupi\b|\bbanks?\b"
    r"|\bcharts?\b|\borders?\b|\btrades?\b|\bgtt\b|\bbuy/sell\b"
    r"|\bcrash(?:es|ed|ing)?\b|\bslow\b|\blags?\b|\bfreez(?:e|es|ing)\b|\bbugs?\b"
    r"|\blog[\s-]?in\b",
    re.I,
)
# A window, a trend, or the theme itself: "last week", "is this a pattern", "others".
_WINDOW_WORD = re.compile(
    r"\blast\s+(?:week|month|\d+\s+days)\b|\bthis\s+week\b|\brecent(?:ly)?\b"
    r"|\bpatterns?\b|\bthemes?\b|\btrends?\b|\bspikes?\b|\bsurges?\b|\bwindow\b"
    r"|\brecurring\b|\brising\b|\bone[\s-]?off\b|\bwidespread\b|\bwider\b"
    r"|\bother\s+(?:users|customers|people|reviews|complaints)\b|\bhow\s+many\b",
    re.I,
)


class RefinementRefused(RuntimeError):
    """Raised when a pushback is rejected and the last reviewed draft stands."""


@dataclass(frozen=True, slots=True, repr=False)
class Pushback:
    """One classified pushback. Holds redacted text only.

    `rerun` is the agent nodes to start, in node order. `facts_text` and
    `pattern_text` are the redacted text those nodes classify this round, or None
    when they should read the complaint. `scheme_id` is the scheme the pushback
    named, when it named one.
    """

    kind: PushbackKind
    rerun: tuple[str, ...]
    redacted_pushback: str
    pii_types: tuple[str, ...]
    facts_text: str | None = None
    pattern_text: str | None = None
    scheme_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in PUSHBACK_KINDS:
            raise ValueError("kind must be wording, scheme, theme, scheme_and_theme, or complaint")
        unknown = sorted(set(self.rerun) - set(AGENT_NODES))
        if unknown:
            raise ValueError(f"rerun names unknown agent nodes: {unknown}")
        object.__setattr__(self, "rerun", tuple(n for n in AGENT_NODES if n in self.rerun))
        if self.facts_text is not None and "facts" not in self.rerun:
            raise ValueError("facts_text needs facts in rerun")
        if self.pattern_text is not None and "pattern" not in self.rerun:
            raise ValueError("pattern_text needs pattern in rerun")

    @property
    def ran(self) -> tuple[AgentName, ...]:
        return (*self.rerun, ORCHESTRATOR)  # type: ignore[return-value]

    @property
    def skipped(self) -> tuple[AgentName, ...]:
        return tuple(n for n in AGENT_NODES if n not in self.rerun)  # type: ignore[return-value]

    @property
    def agent_text(self) -> dict[str, str]:
        """The `agent_text` argument for `graph.run`."""
        texts: dict[str, str] = {}
        if self.facts_text is not None:
            texts["facts"] = self.facts_text
        if self.pattern_text is not None:
            texts["pattern"] = self.pattern_text
        return texts

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ran": list(self.ran),
            "skipped": list(self.skipped),
            "scheme_id": self.scheme_id,
            "redacted_pushback": self.redacted_pushback,
            "pii_types": list(self.pii_types),
        }

    def __repr__(self) -> str:
        return (
            f"Pushback(kind={self.kind!r}, ran={list(self.ran)!r}, "
            f"skipped={list(self.skipped)!r}, scheme_id={self.scheme_id!r}, "
            f"redacted_pushback={self.redacted_pushback!r})"
        )


def refine(
    session: Session,
    pushback: str,
    *,
    complaint: str | None = None,
    classifier: Classifier | None = None,
    review_classifier: Classifier | None = None,
    trace: Trace | None = None,
) -> Session:
    """Apply one human pushback to a reviewed session and return it.

    `pushback` is what the human said about the draft. `complaint` is the user's
    message when the human replaced it; it goes through ingress like a first
    complaint and all three agents re-run on it. Without it, Facts and Pattern
    re-run only when the pushback names a scheme or a theme, and Compliance is
    copied from the session.

    The session changes only after the graph has produced a reviewed draft:
    `refinement_count` goes up by one, a `refinement` entry joins `agents_fired`,
    then the new results, draft, and review replace the old ones. A reviewer
    rework during this round appends its own entry for the same round.
    """
    if not isinstance(session, Session):
        raise TypeError("refine expects a Session")
    if not isinstance(pushback, str):
        raise TypeError("refine expects the pushback as a str")
    if not pushback.strip():
        raise ValueError("pushback is empty")
    if complaint is not None and not isinstance(complaint, str):
        raise TypeError("complaint must be a str or None")
    if session.draft is None or session.review is None:
        raise RefinementRefused("nothing to refine: the session has no reviewed draft")
    if session.refinement_count >= MAX_REFINEMENTS:
        raise RefinementRefused(
            f"refinement limit reached: {MAX_REFINEMENTS} pushbacks already applied; "
            "the last reviewed draft stands"
        )

    changed = complaint is not None and scan(complaint).redacted_text != session.redacted_text
    plan = classify_pushback(pushback, session, complaint_changed=changed)
    del pushback
    logger.info("pushback %r", plan)

    if changed:
        state = run(
            complaint,
            advice_refusal=session.advice_refusal,
            classifier=classifier,
            review_classifier=review_classifier,
            trace=trace,
        )
    else:
        state = run(
            None,
            redacted_text=session.redacted_text,
            pii_types=session.pii_types,
            advice_refusal=session.advice_refusal,
            rerun=plan.rerun,
            reuse={node: getattr(session, node) for node in plan.skipped},
            agent_text=plan.agent_text,
            classifier=classifier,
            review_classifier=review_classifier,
            trace=trace,
        )
    del complaint

    session.refinement_count = session.refinement_count + 1
    session.agents_fired = (
        *session.agents_fired,
        AgentsFired(
            round=session.refinement_count,
            kind="refinement",
            ran=plan.ran,
            skipped=plan.skipped,
        ),
    )
    return store_turn(session, state)


def classify_pushback(
    pushback: str,
    session: Session,
    *,
    complaint_changed: bool = False,
) -> Pushback:
    """Decide which agents a pushback re-runs. Plain code over cue tables.

    The pushback is redacted first; the raw string is not kept. With
    `complaint_changed` every agent re-runs on the new complaint and the cues are
    not consulted. Otherwise a named scheme (or "a different scheme") re-runs
    Facts, a theme or window word re-runs Pattern, both re-run both, and anything
    else is a wording change the orchestrator handles alone.
    """
    if not isinstance(pushback, str):
        raise TypeError("classify_pushback expects the pushback as a str")
    if not isinstance(session, Session):
        raise TypeError("classify_pushback expects a Session")
    redacted = scan(pushback)
    del pushback
    text = redacted.redacted_text

    if complaint_changed:
        return Pushback(
            kind="complaint",
            rerun=AGENT_NODES,
            redacted_pushback=text,
            pii_types=redacted.pii_types,
        )

    complaint = session.redacted_text
    current = None if session.facts is None else session.facts.scheme_id
    named = schemes_named(text)
    other_scheme = _OTHER_SCHEME.search(text) is not None

    scheme_id: str | None = None
    facts_text: str | None = None
    if named:
        candidates = [item for item in named if item != current]
        scheme_id = candidates[0] if candidates else named[0]
        facts_text = _complaint_about(complaint, scheme_id)
    elif other_scheme:
        facts_text = _complaint_without_scheme(complaint)

    pattern_text: str | None = None
    theme_word = _THEME_WORD.search(text) is not None
    window_word = _WINDOW_WORD.search(text) is not None
    if theme_word:
        pattern_text = text
    elif window_word:
        pattern_text = complaint

    rerun: list[str] = []
    if facts_text is not None:
        rerun.append("facts")
    if pattern_text is not None:
        rerun.append("pattern")
    kind: PushbackKind
    if rerun == ["facts", "pattern"]:
        kind = "scheme_and_theme"
    elif rerun == ["facts"]:
        kind = "scheme"
    elif rerun == ["pattern"]:
        kind = "theme"
    else:
        kind = "wording"
    return Pushback(
        kind=kind,
        rerun=tuple(rerun),
        redacted_pushback=text,
        pii_types=redacted.pii_types,
        facts_text=facts_text,
        pattern_text=pattern_text,
        scheme_id=scheme_id,
    )


def schemes_named(text: str) -> tuple[str, ...]:
    """Scheme ids mentioned in the text, in order of first mention, each once."""
    if not isinstance(text, str):
        raise TypeError("schemes_named expects a str")
    found: list[str] = []
    for match in _SCHEME_MENTION.finditer(text):
        scheme_id = match.lastgroup
        if scheme_id and scheme_id not in found:
            found.append(scheme_id)
    return tuple(found)


def _complaint_about(complaint: str, scheme_id: str) -> str:
    """The complaint with every scheme mention replaced by the named scheme.

    The question words stay, so Facts picks the same section for the new scheme.
    A complaint that names no scheme gets one appended.
    """
    name = SCHEME_NAMES[scheme_id]
    if _SCHEME_MENTION.search(complaint):
        return _SCHEME_MENTION.sub(name, complaint)
    return f"{complaint.rstrip()} The scheme is {name}."


def _complaint_without_scheme(complaint: str) -> str:
    """The complaint with its scheme mentions removed, for "a different scheme"."""
    stripped = _SCHEME_MENTION.sub(" ", complaint)
    stripped = re.sub(r"\s+([?.!,;:])", r"\1", stripped)
    return " ".join(stripped.split())


assert set(AGENT_NODES) | {ORCHESTRATOR} == AGENT_NAMES
