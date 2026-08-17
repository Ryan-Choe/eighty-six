"""Graph state. Nodes return partial updates; LangGraph merges them."""

from typing import Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# The router picks exactly one of these. Adding a value here means adding an
# edge in graph.py — a test asserts they stay in sync.
Intent = Literal["inventory", "reorder", "policy", "off_topic"]


class POLineChoice(BaseModel):
    """One line of the model's proposed order. Only names and counts —
    prices come from the catalog when price_po() re-prices the draft."""

    ingredient: str
    packs: int = Field(ge=1)


class POChoice(BaseModel):
    """The model's full proposal: which supplier, what, and why."""

    supplier_name: str
    lines: list[POLineChoice]
    expected_delivery: str = Field(
        description="When the goods arrive and why that works, citing the terms."
    )
    rationale: str = Field(
        description="Two or three sentences on why this supplier over the "
        "alternatives, citing terms like [source § section]."
    )


class AgentState(MessagesState):
    """Inherits `messages` (append reducer). Everything else is
    last-write-wins; one invocation is one owner turn."""

    intent: Optional[Intent]
    low_stock: Optional[list[dict]]
    candidates: Optional[list[dict]]     # supplier options, deterministic math
    po_draft: Optional[dict]             # price_po() output + delivery + rationale
    approved: Optional[bool]             # written by the human via the interrupt
    citations: Optional[list[dict]]      # [{"source", "section"}] from chunk metadata
