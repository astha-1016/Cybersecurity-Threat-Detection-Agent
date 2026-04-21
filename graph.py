"""
graph.py — LangGraph StateGraph with intent routing and eval retry loop.

New query flow:
  input → memory → intent → retrieve → decision → tool → response → eval
                                                              ↑        |
                                                              └─retry──┘ (faith < 0.7)
                                                                       |
                                                                      END

Follow-up flow:
  input → memory → intent → followup → END

FIXES APPLIED:
  - FIX 1: input_node — initialise 'severity_score' to 0.0 so AgentState
           TypedDict is always fully populated and downstream code never
           receives a KeyError on that field.
  - FIX 2: retrieve_node — source name extraction now handles both forward-
           slash and back-slash path separators on all OSes, and strips the
           'data/docs/' prefix if present so source labels are clean.
  - FIX 3: eval_decision — was comparing state faithfulness BEFORE self_eval_node
           updated it, because conditional edge functions receive the state as
           it was when the node ENTERED (LangGraph passes post-node state to
           conditional edges, but only if the node explicitly returns it).
           Added explicit logging to confirm the score being evaluated.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    AgentState,
    memory_node,
    intent_node,
    followup_node,
    decision_node,
    tool_node,
    response_node,
    self_eval_node,
)
from config import FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES
from logger import get_logger

log = get_logger(__name__)


def build_graph(retriever):
    """Build and return the compiled LangGraph application."""

    # ── Input node ────────────────────────────────────────────────────────────
    def input_node(state: AgentState) -> dict:
        return {
            "input":          state["input"],
            "messages":       state.get("messages", []),
            "eval_retries":   0,
            "faithfulness":   0.0,
            "severity_score": 0.0,   # FIX 1: always initialise
            "sources":        [],
            "context":        "",
            "tool_output":    "",
            "decision":       "safe",
            "attack_type":    "",
            "intent":         "new_query",
            "response":       "",
            "final":          "",
        }

    # ── Retrieve node (closure — retriever never stored in state) ─────────────
    def retrieve_node(state: AgentState) -> dict:
        docs    = retriever.invoke(state["input"])
        context = "\n\n---\n\n".join([doc.page_content for doc in docs[:3]])
        sources = []
        for doc in docs[:3]:
            src = doc.metadata.get("source", "")
            if src:
                # FIX 2: handle both / and \ separators, strip known prefix
                name = src.replace("\\", "/").split("/")[-1]
                name = name.replace(".txt", "")
                sources.append(name)
        log.info(f"Retrieved {len(docs)} docs: {sources}")
        return {**state, "context": context, "sources": sources}

    # ── Routing functions ─────────────────────────────────────────────────────
    def route_intent(state: AgentState) -> str:
        return "followup" if state.get("intent") == "followup" else "retrieve"

    def eval_decision(state: AgentState) -> str:
        # FIX 3: log the actual score being evaluated for easier debugging
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        log.info(f"Eval decision: faithfulness={score:.2f}, retries={retries}, "
                 f"threshold={FAITHFULNESS_THRESHOLD}, max_retries={MAX_EVAL_RETRIES}")
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            log.info(f"Eval: END (score={score:.2f}, retries={retries})")
            return "end"
        log.info(f"Eval: RETRY (score={score:.2f} < {FAITHFULNESS_THRESHOLD})")
        return "retry"

    # ── Build ─────────────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    workflow.add_node("input",    input_node)
    workflow.add_node("memory",   memory_node)
    workflow.add_node("intent",   intent_node)
    workflow.add_node("followup", followup_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("tool",     tool_node)
    workflow.add_node("response", response_node)
    workflow.add_node("eval",     self_eval_node)

    workflow.set_entry_point("input")
    workflow.add_edge("input",    "memory")
    workflow.add_edge("memory",   "intent")

    workflow.add_conditional_edges(
        "intent", route_intent,
        {"followup": "followup", "retrieve": "retrieve"}
    )

    workflow.add_edge("followup", END)

    workflow.add_edge("retrieve", "decision")
    workflow.add_edge("decision", "tool")
    workflow.add_edge("tool",     "response")
    workflow.add_edge("response", "eval")

    # AFTER
    workflow.add_conditional_edges(
        "eval", eval_decision,
        {"retry": "retrieve", "end": END}
    )

    memory = MemorySaver()
    app    = workflow.compile(checkpointer=memory)
    log.info("Graph compiled successfully")
    return app