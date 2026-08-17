"""Structural tests for the graph. No API keys: building a graph doesn't call
a model, so these run in CI alongside the arithmetic tests."""

from typing import get_args

from langgraph.graph import END

from eightysix.graph import INTENT_EDGES, build_graph
from eightysix.state import Intent
from eightysix.tools import INVENTORY_TOOLS


def test_graph_has_exactly_the_nodes_the_readme_claims():
    nodes = set(build_graph().get_graph().nodes)
    assert nodes == {
        "__start__",
        "__end__",
        "route",
        "inventory_agent",
        "inventory_tools",
        "deflect",
    }


def test_every_intent_has_a_branch():
    # The Intent Literal and the edge map must stay in sync — adding an intent
    # without an edge would route to the fallback silently.
    assert set(get_args(Intent)) == set(INTENT_EDGES)


def test_agent_can_loop_back_from_tools():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert ("inventory_tools", "inventory_agent") in edges
    assert ("inventory_agent", "inventory_tools") in edges
    assert ("inventory_agent", END) in edges


def test_tools_are_the_five_read_only_ones():
    assert [t.name for t in INVENTORY_TOOLS] == [
        "get_stock",
        "get_low_stock",
        "list_ingredients",
        "check_feasibility",
        "get_usage",
    ]


def test_mermaid_renders_for_the_readme():
    mermaid = build_graph().get_graph().draw_mermaid()
    assert "route" in mermaid and "inventory_tools" in mermaid
