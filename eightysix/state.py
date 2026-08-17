"""Graph state. Nodes return partial updates; LangGraph merges them."""

from typing import Literal, Optional

from langgraph.graph import MessagesState

# The router picks exactly one of these. Adding a value here means adding an
# edge in graph.py — keeping them in sync is the point of the Literal.
Intent = Literal["inventory", "reorder", "policy", "off_topic"]


class AgentState(MessagesState):
    """Inherits `messages` (append reducer) from MessagesState.

    Every other field is last-write-wins. One invocation is one user turn, so
    nothing here needs merging.
    """

    intent: Optional[Intent]
    # Day 3 adds: low_stock, po_draft, approved, citations
