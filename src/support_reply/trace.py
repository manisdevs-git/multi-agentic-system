"""Start/return trace for one graph run.

The orchestrator records one `start` and one `return` event per graph node. The
trace holds node names, a sequence number, and elapsed time only. It never holds
node input or output, so it cannot carry the raw complaint.

`violations()` is the parallel proof. It fails when an agent node returns before
all three agent nodes have started, or when the reviewer starts before all three
have returned.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

TraceKind = Literal["start", "return"]

AGENT_NODES: tuple[str, ...] = ("facts", "pattern", "compliance")
REVIEWER_NODE = "reviewer"
TRACE_KINDS: frozenset[str] = frozenset({"start", "return"})


@dataclass(frozen=True, slots=True)
class TraceEvent:
    seq: int
    node: str
    kind: TraceKind
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "node": self.node,
            "kind": self.kind,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class Trace:
    """Ordered node events for one `handle_complaint` call."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

    def record(self, node: str, kind: str) -> TraceEvent:
        if not isinstance(node, str) or not node.strip():
            raise ValueError("trace node must be a non-empty name")
        if kind not in TRACE_KINDS:
            raise ValueError("trace kind must be start or return")
        with self._lock:
            event = TraceEvent(
                seq=len(self._events),
                node=node,
                kind=kind,  # type: ignore[arg-type]
                elapsed_ms=(time.perf_counter() - self._t0) * 1000.0,
            )
            self._events.append(event)
        return event

    def start(self, node: str) -> TraceEvent:
        return self.record(node, "start")

    def returned(self, node: str) -> TraceEvent:
        return self.record(node, "return")

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def starts(self, node: str | None = None) -> tuple[TraceEvent, ...]:
        return self._select("start", node)

    def returns(self, node: str | None = None) -> tuple[TraceEvent, ...]:
        return self._select("return", node)

    def first(self, kind: str, node: str) -> TraceEvent | None:
        for event in self.events:
            if event.kind == kind and event.node == node:
                return event
        return None

    def sequence(self) -> tuple[tuple[str, str], ...]:
        """`(kind, node)` pairs in order. Convenient for asserting a whole run."""
        return tuple((event.kind, event.node) for event in self.events)

    def violations(self, agents: tuple[str, ...] = AGENT_NODES) -> list[str]:
        """Return the parallel-rule breaks for this run. An empty list is a pass.

        Rules, over the first start and first return of each named agent:
        1. every agent started and returned;
        2. no agent returned before the last agent start;
        3. the reviewer started, and only after the last agent return.
        """
        problems: list[str] = []
        starts = {node: self.first("start", node) for node in agents}
        returns = {node: self.first("return", node) for node in agents}
        for node in agents:
            if starts[node] is None:
                problems.append(f"{node} never started")
            if returns[node] is None:
                problems.append(f"{node} never returned")

        started = [event.seq for event in starts.values() if event is not None]
        if started:
            last_start = max(started)
            for node in agents:
                event = returns[node]
                if event is not None and event.seq < last_start:
                    problems.append(f"{node} returned before all agents started")

        reviewer = self.first("start", REVIEWER_NODE)
        if reviewer is None:
            problems.append("reviewer never started")
        else:
            for node in agents:
                event = returns[node]
                if event is not None and event.seq > reviewer.seq:
                    problems.append(f"reviewer started before {node} returned")
        return problems

    def render(self) -> str:
        """One line per event: sequence, elapsed ms, kind, node."""
        return "\n".join(
            f"{event.seq:03d} {event.elapsed_ms:9.3f}ms {event.kind:<6} {event.node}"
            for event in self.events
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"Trace(events={list(self.sequence())!r})"

    def _select(self, kind: str, node: str | None) -> tuple[TraceEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.kind == kind and (node is None or event.node == node)
        )
