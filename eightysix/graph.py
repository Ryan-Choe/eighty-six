"""The graph. Read this file to understand the whole control flow."""

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from eightysix.nodes import (
    cancel_po,
    deflect,
    draft_po,
    human_approval,
    inventory_agent,
    policy_qa,
    route,
    send_po,
)
from eightysix.state import AgentState
from eightysix.tools import INVENTORY_TOOLS

log = logging.getLogger(__name__)

# Where each intent goes. A test asserts this stays in sync with the Intent
# Literal in state.py.
INTENT_EDGES = {
    "inventory": "inventory_agent",
    "reorder": "draft_po",
    "policy": "policy_qa",
    "off_topic": "deflect",
}


def pick_branch(state: AgentState) -> str:
    return INTENT_EDGES.get(state.get("intent"), "inventory_agent")


def after_draft(state: AgentState) -> str:
    # no draft means nothing to approve: either nothing was low, or the model
    # couldn't produce a valid order and already said so
    return "human_approval" if state.get("po_draft") else END


def after_approval(state: AgentState) -> str:
    return "send_po" if state.get("approved") else "cancel_po"


def _tool_error(e: Exception) -> str:
    # langgraph 1.x ToolNode only converts schema errors by default; a real
    # exception inside a tool would kill the whole turn. Log it for us,
    # describe it for the model.
    log.exception("tool raised during execution")
    return (f"The tool failed internally ({type(e).__name__}: {e}). Answer with "
            "what you already know, or tell the owner what you couldn't check.")


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("route", route)
    g.add_node("inventory_agent", inventory_agent)
    g.add_node("inventory_tools", ToolNode(INVENTORY_TOOLS, handle_tool_errors=_tool_error))
    g.add_node("policy_qa", policy_qa)
    g.add_node("draft_po", draft_po)
    g.add_node("human_approval", human_approval)
    g.add_node("send_po", send_po)
    g.add_node("cancel_po", cancel_po)
    g.add_node("deflect", deflect)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", pick_branch, sorted(set(INTENT_EDGES.values())))

    # the agent loop, drawn explicitly: the model either asks for a tool or is done
    g.add_conditional_edges(
        "inventory_agent", tools_condition, {"tools": "inventory_tools", END: END}
    )
    g.add_edge("inventory_tools", "inventory_agent")

    # the reorder path: draft -> (pause for a human) -> send or cancel.
    # human_approval calls interrupt(); the graph checkpoints there and resumes
    # only on Command(resume={"approved": ...}) from the interface.
    g.add_conditional_edges("draft_po", after_draft, ["human_approval", END])
    g.add_conditional_edges("human_approval", after_approval, ["send_po", "cancel_po"])
    g.add_edge("send_po", END)
    g.add_edge("cancel_po", END)

    g.add_edge("policy_qa", END)
    g.add_edge("deflect", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
