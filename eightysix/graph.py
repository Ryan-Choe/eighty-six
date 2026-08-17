"""The graph. Read this file to understand the whole control flow."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from eightysix.nodes import deflect, inventory_agent, route
from eightysix.state import AgentState
from eightysix.tools import INVENTORY_TOOLS

# Where each intent goes. Day 3 points "reorder" and "policy" at their own
# nodes; until then they land on deflect and say so.
INTENT_EDGES = {
    "inventory": "inventory_agent",
    "reorder": "deflect",
    "policy": "deflect",
    "off_topic": "deflect",
}


def pick_branch(state: AgentState) -> str:
    return INTENT_EDGES.get(state.get("intent"), "inventory_agent")


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("route", route)
    g.add_node("inventory_agent", inventory_agent)
    g.add_node("inventory_tools", ToolNode(INVENTORY_TOOLS))
    g.add_node("deflect", deflect)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", pick_branch, sorted(set(INTENT_EDGES.values())))

    # The agent loop, drawn explicitly rather than hidden inside a prebuilt
    # react agent: the model either asks for a tool or it's done.
    g.add_conditional_edges(
        "inventory_agent", tools_condition, {"tools": "inventory_tools", END: END}
    )
    g.add_edge("inventory_tools", "inventory_agent")
    g.add_edge("deflect", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
