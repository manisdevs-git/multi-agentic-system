"""LangGraph orchestrator: one `StateGraph` for one complaint turn.

Nodes: `ingress`, `facts`, `pattern`, `compliance`, `draft`, `reviewer`.

- `ingress` runs first. It writes `redacted_text` and `pii_types`. The raw
  complaint arrives on an ephemeral channel that LangGraph clears after this
  step, so the fan-out nodes and the final state never see it.
- `facts`, `pattern`, and `compliance` are the Phase 2–4 subgraphs mounted as
  nodes. The graph starts all three in one superstep. Each node's input schema
  is the redacted text (plus `pii_types` for Compliance), so none receives
  another node's result. Each writes only its own channel.
- `draft` joins after all three channels are set and calls `draft.compose`.
- `reviewer` runs after the draft. A round-1 `fail` takes the conditional edge
  to `rewrite`, which rebuilds the draft from the three channels already on the
  state with the offending spans excluded, then returns to `reviewer` for round
  2. The rewrite edge never re-enters `facts`, `pattern`, or `compliance`. Round
  2 ends `pass` or `blocked`; both go to END.

`handle_complaint(session, raw_text)` streams the graph, records a `Trace`
event per node start and return, and copies the output onto the session. A
rework appends one `agents_fired` entry with `kind: "reviewer_rework"` and
`ran: ["orchestrator"]`; it does not touch `refinement_count`. The graph is the
only caller of the three agents and the reviewer.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.channels import EphemeralValue
from langgraph.graph import END, START, StateGraph

from support_reply import compliance, facts, pattern
from support_reply.contracts import (
    AgentsFired,
    ComplianceResult,
    Draft,
    FactsResult,
    PatternResult,
    ReviewerResult,
    Session,
)
from support_reply.draft import compose
from support_reply.ingress import redact
from support_reply.llm import Classifier
from support_reply.reviewer import review, spans
from support_reply.trace import AGENT_NODES, Trace

logger = logging.getLogger(__name__)

INGRESS_NODE = "ingress"
DRAFT_NODE = "draft"
REVIEWER_NODE = "reviewer"
REWRITE_NODE = "rewrite"
NODES: tuple[str, ...] = (
    INGRESS_NODE,
    *AGENT_NODES,
    DRAFT_NODE,
    REVIEWER_NODE,
    REWRITE_NODE,
)
MAX_REVIEW_ROUNDS = 2


class ComplaintInput(TypedDict):
    """What a caller hands the graph. `advice_refusal` is the session's sticky flag."""

    raw_text: Annotated[str, EphemeralValue]
    advice_refusal: NotRequired[bool]


class GraphState(TypedDict, total=False):
    raw_text: Annotated[str, EphemeralValue]
    redacted_text: str
    pii_types: tuple[str, ...]
    advice_refusal: bool
    facts: FactsResult
    pattern: PatternResult
    compliance: ComplianceResult
    draft: Draft
    review: ReviewerResult
    review_round: int
    reworked: bool


class GraphOutput(TypedDict, total=False):
    redacted_text: str
    pii_types: tuple[str, ...]
    advice_refusal: bool
    facts: FactsResult
    pattern: PatternResult
    compliance: ComplianceResult
    draft: Draft
    review: ReviewerResult
    review_round: int
    reworked: bool


class RedactedInput(TypedDict):
    """Facts and Pattern see the redacted complaint and nothing else."""

    redacted_text: str


class ComplianceInput(TypedDict):
    """Compliance also sees the token types ingress reported."""

    redacted_text: str
    pii_types: tuple[str, ...]


class DraftInput(TypedDict):
    redacted_text: str
    advice_refusal: bool
    facts: FactsResult
    pattern: PatternResult
    compliance: ComplianceResult


class ReviewerInput(TypedDict, total=False):
    """The reviewer sees the draft, the Facts it was built from, and the record to store."""

    redacted_text: str
    pii_types: tuple[str, ...]
    facts: FactsResult
    pattern: PatternResult
    compliance: ComplianceResult
    draft: Draft
    review_round: int


class RewriteInput(TypedDict):
    """The rewrite sees the three results already on the state and the failed review."""

    redacted_text: str
    advice_refusal: bool
    facts: FactsResult
    pattern: PatternResult
    compliance: ComplianceResult
    review: ReviewerResult


def build_graph(
    *,
    classifier: Classifier | None = None,
    review_classifier: Classifier | None = None,
) -> StateGraph:
    """Assemble the uncompiled graph. `compile_graph` is what callers run.

    `classifier` serves the three agent nodes (Groq by default). `review_classifier`
    serves the reviewer's advice and performance scan (Gemini by default).
    """
    facts_graph = facts.compile_graph(classifier=classifier)
    pattern_graph = pattern.compile_graph(classifier=classifier)
    compliance_graph = compliance.compile_graph(classifier=classifier)

    def ingress_node(state: ComplaintInput) -> dict[str, Any]:
        result = redact(state["raw_text"])
        return {"redacted_text": result.redacted_text, "pii_types": result.pii_types}

    def facts_node(state: RedactedInput) -> dict[str, FactsResult]:
        out = facts_graph.invoke({"redacted_text": state["redacted_text"]})
        return {"facts": _subgraph_result(out, "facts", FactsResult)}

    def pattern_node(state: RedactedInput) -> dict[str, PatternResult]:
        out = pattern_graph.invoke({"redacted_text": state["redacted_text"]})
        return {"pattern": _subgraph_result(out, "pattern", PatternResult)}

    def compliance_node(state: ComplianceInput) -> dict[str, ComplianceResult]:
        out = compliance_graph.invoke(
            {
                "redacted_text": state["redacted_text"],
                "pii_types": tuple(state.get("pii_types", ())),
            }
        )
        return {"compliance": _subgraph_result(out, "compliance", ComplianceResult)}

    def draft_node(state: DraftInput) -> dict[str, Any]:
        sticky = bool(state.get("advice_refusal", False))
        drafted = compose(
            state["facts"],
            state["pattern"],
            state["compliance"],
            state["redacted_text"],
            advice_refusal=sticky,
        )
        logger.info("draft %r", drafted)
        return {
            "draft": drafted,
            "advice_refusal": sticky or state["compliance"].advice_refusal,
        }

    def reviewer_node(state: ReviewerInput) -> dict[str, Any]:
        round_number = int(state.get("review_round", 1))
        record = {
            "redacted_text": state["redacted_text"],
            "pii_types": list(state.get("pii_types", ())),
            "facts": state["facts"].to_dict(),
            "pattern": state["pattern"].to_dict(),
            "compliance": state["compliance"].to_dict(),
            "draft": state["draft"].to_dict(),
        }
        reviewed = review(
            state["draft"],
            facts=state["facts"],
            round_number=round_number,
            record=record,
            classifier=review_classifier,
        )
        return {"review": reviewed, "review_round": round_number}

    def rewrite_node(state: RewriteInput) -> dict[str, Any]:
        """Rebuild the draft from the stored results. No agent runs here."""
        failed = state["review"]
        if failed.status != "fail" or failed.round != 1:
            raise RuntimeError("rewrite runs once, after a round-1 reviewer fail")
        drafted = compose(
            state["facts"],
            state["pattern"],
            state["compliance"],
            state["redacted_text"],
            advice_refusal=bool(state.get("advice_refusal", False)),
            exclude=spans(failed.failures),
        )
        logger.info("rewrite %r", drafted)
        return {"draft": drafted, "review_round": MAX_REVIEW_ROUNDS, "reworked": True}

    def after_review(state: GraphState) -> Literal["rewrite", "__end__"]:
        reviewed = state["review"]
        if reviewed.status == "fail" and int(state.get("review_round", 1)) < MAX_REVIEW_ROUNDS:
            return REWRITE_NODE
        return END

    graph = StateGraph(GraphState, input_schema=ComplaintInput, output_schema=GraphOutput)
    graph.add_node(INGRESS_NODE, ingress_node, input_schema=ComplaintInput)
    graph.add_node("facts", facts_node, input_schema=RedactedInput)
    graph.add_node("pattern", pattern_node, input_schema=RedactedInput)
    graph.add_node("compliance", compliance_node, input_schema=ComplianceInput)
    graph.add_node(DRAFT_NODE, draft_node, input_schema=DraftInput)
    graph.add_node(REVIEWER_NODE, reviewer_node, input_schema=ReviewerInput)
    graph.add_node(REWRITE_NODE, rewrite_node, input_schema=RewriteInput)

    graph.add_edge(START, INGRESS_NODE)
    for node in AGENT_NODES:
        graph.add_edge(INGRESS_NODE, node)
    graph.add_edge(list(AGENT_NODES), DRAFT_NODE)
    graph.add_edge(DRAFT_NODE, REVIEWER_NODE)
    graph.add_conditional_edges(
        REVIEWER_NODE, after_review, {REWRITE_NODE: REWRITE_NODE, END: END}
    )
    graph.add_edge(REWRITE_NODE, REVIEWER_NODE)
    return graph


def compile_graph(
    *,
    classifier: Classifier | None = None,
    review_classifier: Classifier | None = None,
):
    """Compile the complaint graph. One compiled graph serves many invocations."""
    return build_graph(classifier=classifier, review_classifier=review_classifier).compile()


def run(
    raw_text: str,
    *,
    advice_refusal: bool = False,
    classifier: Classifier | None = None,
    review_classifier: Classifier | None = None,
    trace: Trace | None = None,
    compiled: Any = None,
) -> dict[str, Any]:
    """Stream one turn through the graph and return its output state.

    Each LangGraph task event becomes one trace event: a task without a result is
    a node start, a task with one is a node return. The trace keeps node names
    only. The returned state has no raw-text key.
    """
    if not isinstance(raw_text, str):
        raise TypeError("handle_complaint expects the complaint as a str")
    if type(advice_refusal) is not bool:
        raise TypeError("advice_refusal must be a bool")
    app = (
        compiled
        if compiled is not None
        else compile_graph(classifier=classifier, review_classifier=review_classifier)
    )
    recorder = trace if trace is not None else Trace()

    final: dict[str, Any] | None = None
    payload: ComplaintInput = {"raw_text": raw_text, "advice_refusal": advice_refusal}
    for mode, event in app.stream(payload, stream_mode=["tasks", "values"]):
        if mode == "tasks":
            kind = "return" if ("result" in event or "error" in event) else "start"
            recorded = recorder.record(event["name"], kind)
            logger.info("trace %03d %s %s", recorded.seq, recorded.kind, recorded.node)
            # A node error is recorded as that node's return; LangGraph re-raises it.
        else:
            final = event
    if final is None:
        raise RuntimeError("graph produced no state")
    final.pop("raw_text", None)
    return final


def handle_complaint(
    session: Session,
    raw_text: str,
    *,
    classifier: Classifier | None = None,
    review_classifier: Classifier | None = None,
    trace: Trace | None = None,
) -> Session:
    """Run one complaint through the graph and store the output on the session.

    The session's `advice_refusal` goes in as the sticky flag and comes back true
    if this turn's Compliance flagged advice or a performance ask. It is never
    reset here. A reviewer rework appends a `reviewer_rework` entry to
    `agents_fired` for the current refinement round and leaves `refinement_count`
    alone. Pass a `Trace` to inspect node order after the call.
    """
    if not isinstance(session, Session):
        raise TypeError("handle_complaint expects a Session")
    if not isinstance(raw_text, str):
        raise TypeError("handle_complaint expects the complaint as a str")

    state = run(
        raw_text,
        advice_refusal=session.advice_refusal,
        classifier=classifier,
        review_classifier=review_classifier,
        trace=trace,
    )
    del raw_text

    facts_result = _result(state, "facts", FactsResult)
    pattern_result = _result(state, "pattern", PatternResult)
    compliance_result = _result(state, "compliance", ComplianceResult)
    drafted = _result(state, "draft", Draft)
    reviewed = _result(state, "review", ReviewerResult)

    session.redacted_text = state["redacted_text"]
    session.pii_types = tuple(state.get("pii_types", ()))
    session.facts = facts_result
    session.pattern = pattern_result
    session.compliance = compliance_result
    session.draft = drafted
    session.routing_note = drafted.routing_note
    session.review = reviewed
    if state.get("advice_refusal", False) and not session.advice_refusal:
        session.advice_refusal = True
    if state.get("reworked", False):
        session.agents_fired = (
            *session.agents_fired,
            AgentsFired(
                round=session.refinement_count,
                kind="reviewer_rework",
                ran=("orchestrator",),
                skipped=AGENT_NODES,
            ),
        )
    logger.info("session %r", session)
    return session


def _subgraph_result(out: Any, node: str, kind: type) -> Any:
    """Read the `result` channel an agent subgraph returned. Missing is an error."""
    value = out.get("result") if isinstance(out, dict) else None
    if value is None:
        raise RuntimeError(f"{node} node returned no result")
    if not isinstance(value, kind):
        raise TypeError(f"{node} must return a {kind.__name__}, not {type(value).__name__}")
    return value


def _result(state: Any, key: str, kind: type) -> Any:
    """Read one channel of the parent graph output. Missing is an orchestrator error."""
    value = state.get(key) if isinstance(state, dict) else None
    if value is None:
        if key in AGENT_NODES:
            raise RuntimeError(f"{key} node returned no result")
        raise RuntimeError(f"graph state has no {key}")
    if not isinstance(value, kind):
        raise TypeError(f"{key} must be a {kind.__name__}, not {type(value).__name__}")
    return value
