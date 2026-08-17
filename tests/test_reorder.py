"""Day 3 deterministic layer: purchasing math, KB chunking, graph wiring,
and the interrupt mechanics (with the LLM node stubbed out). No API keys."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from eightysix import db, purchasing
from eightysix.nodes import repair_tool_history, human_approval, cancel_po
from eightysix.rag import load_chunks
from eightysix.state import AgentState

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    db.seed_from_csvs(c, SEED_DIR)
    # the demo state: Friday rush already applied
    c.execute("UPDATE ingredients SET on_hand = 1160 WHERE name = 'fresh mozzarella'")
    c.execute("UPDATE ingredients SET on_hand = 840 WHERE name = 'pepperoni'")
    return c


# --- purchasing math ---------------------------------------------------------

def test_candidates_round_packs_up_to_par(conn):
    cands = purchasing.build_candidates(conn, db.get_low_stock(conn))
    moz_roma = next(c for c in cands
                    if c["ingredient"] == "fresh mozzarella" and c["supplier"] == "Roma Foods")
    # need 4000 - 1160 = 2840g; packs of 2270g -> 2 packs, never 1.25
    assert moz_roma["need"] == 2840
    assert moz_roma["packs_to_par"] == 2


def test_candidates_offer_both_suppliers_for_the_trap(conn):
    cands = purchasing.build_candidates(conn, db.get_low_stock(conn))
    moz = {c["supplier"]: c for c in cands if c["ingredient"] == "fresh mozzarella"}
    # Valco must be cheaper -- the trap only works if the bait is real
    assert moz["Valco Cash & Carry"]["price_cents"] < moz["Roma Foods"]["price_cents"]


def test_price_po_reprices_from_catalog_only(conn):
    po = purchasing.price_po(conn, "roma foods", [   # case-insensitive supplier
        {"ingredient": "fresh mozzarella", "packs": 2},
        {"ingredient": "pepperoni", "packs": 1},
    ])
    assert po["supplier_name"] == "Roma Foods"
    assert po["subtotal_cents"] == 2 * 5400 + 5800
    assert po["meets_minimum"] is False        # $166 < $250 -- the fee scenario

def test_price_po_rejects_items_the_supplier_does_not_carry(conn):
    with pytest.raises(purchasing.POValidationError):
        purchasing.price_po(conn, "Roma Foods", [{"ingredient": "basil", "packs": 1}])


def test_price_po_rejects_unknown_supplier_and_bad_packs(conn):
    with pytest.raises(purchasing.POValidationError):
        purchasing.price_po(conn, "Sysco", [{"ingredient": "pepperoni", "packs": 1}])
    with pytest.raises(purchasing.POValidationError):
        purchasing.price_po(conn, "Roma Foods", [{"ingredient": "pepperoni", "packs": 0}])


def test_send_po_writes_and_marks_sent(conn, capsys):
    po = purchasing.price_po(conn, "Roma Foods", [{"ingredient": "pepperoni", "packs": 1}])
    po_id = purchasing.create_and_send_po(conn, po)
    row = conn.execute("SELECT status, total_cents FROM purchase_orders WHERE id = ?",
                       (po_id,)).fetchone()
    assert row["status"] == "sent"
    assert row["total_cents"] == 5800
    assert f"PO-{po_id}" in capsys.readouterr().out


# --- knowledge base ----------------------------------------------------------

def test_kb_chunks_carry_metadata_and_the_trap_exists():
    chunks = load_chunks()
    assert len(chunks) >= 15
    types = {c.metadata["doc_type"] for c in chunks}
    assert types == {"supplier", "policy"}
    # the planted trap must survive editing: Valco's pickup-only section
    trap = [c for c in chunks
            if c.metadata["source"] == "valco_cash_carry"
            and "deliver" in c.page_content.lower()]
    assert trap, "Valco's no-delivery section is the demo trap -- don't lose it"


# --- graph wiring ------------------------------------------------------------

def test_intent_edges_cover_every_intent():
    from typing import get_args
    from eightysix.graph import INTENT_EDGES
    from eightysix.state import Intent
    assert set(get_args(Intent)) == set(INTENT_EDGES)


def test_reorder_path_is_wired_through_approval():
    from eightysix.graph import build_graph
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert ("draft_po", "human_approval") in edges
    assert ("human_approval", "send_po") in edges
    assert ("human_approval", "cancel_po") in edges
    assert ("send_po", "__end__") in edges


# --- interrupt mechanics, LLM stubbed ----------------------------------------

def _tiny_approval_graph():
    """The real human_approval/cancel_po nodes behind a stub drafter, so the
    pause/resume mechanics are tested without a model call."""
    def fake_draft(state: AgentState) -> dict:
        return {"po_draft": {"supplier_name": "Roma Foods", "lines": [],
                             "subtotal_cents": 100, "min_order_cents": 0,
                             "meets_minimum": True, "supplier_id": 1,
                             "expected_delivery": "Saturday"}}

    def fake_send(state: AgentState) -> dict:
        return {"messages": [AIMessage("sent!")]}

    g = StateGraph(AgentState)
    g.add_node("draft", fake_draft)
    g.add_node("human_approval", human_approval)
    g.add_node("send_po", fake_send)
    g.add_node("cancel_po", cancel_po)
    g.add_edge(START, "draft")
    g.add_edge("draft", "human_approval")
    g.add_conditional_edges(
        "human_approval",
        lambda s: "send_po" if s.get("approved") else "cancel_po",
        ["send_po", "cancel_po"],
    )
    g.add_edge("send_po", END)
    g.add_edge("cancel_po", END)
    return g.compile(checkpointer=InMemorySaver())


def test_interrupt_pauses_with_payload_then_resumes_on_approval():
    graph = _tiny_approval_graph()
    config = {"configurable": {"thread_id": "t-approve"}}
    result = graph.invoke({"messages": []}, config)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["po_draft"]["supplier_name"] == "Roma Foods"

    resumed = graph.invoke(Command(resume={"approved": True}), config)
    assert resumed["approved"] is True
    assert resumed["messages"][-1].content == "sent!"


def test_interrupt_reject_routes_to_cancel():
    graph = _tiny_approval_graph()
    config = {"configurable": {"thread_id": "t-reject"}}
    graph.invoke({"messages": []}, config)
    resumed = graph.invoke(Command(resume={"approved": False}), config)
    assert resumed["approved"] is False
    assert "Cancelled" in resumed["messages"][-1].content
    assert resumed["po_draft"] is None


# --- checkpoint repair -------------------------------------------------------

def _dangling(call_id="call_1"):
    return AIMessage(content="", tool_calls=[
        {"name": "get_stock", "args": {"ingredient": "basil"}, "id": call_id},
    ])


def test_repair_splices_result_immediately_after_dangling_call():
    from langchain_core.messages import HumanMessage
    # the REALISTIC shape: the next turn's HumanMessage already sits after
    # the dangling AIMessage -- Anthropic requires the tool result to come
    # immediately after the tool_use, so end-appending would still 400
    history = [HumanMessage("q1"), _dangling(), HumanMessage("q2")]
    repaired = repair_tool_history(history)
    assert [type(m).__name__ for m in repaired] == [
        "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage",
    ]
    assert repaired[2].tool_call_id == "call_1"


def test_repair_is_idempotent_and_skips_answered_calls():
    from langchain_core.messages import HumanMessage, ToolMessage
    answered = [
        _dangling("call_9"),
        ToolMessage(content="ok", tool_call_id="call_9", name="get_stock"),
        HumanMessage("next"),
    ]
    assert repair_tool_history(answered) == answered
    twice = repair_tool_history(repair_tool_history([_dangling()]))
    assert sum(1 for m in twice if type(m).__name__ == "ToolMessage") == 1


def test_repair_leaves_clean_history_alone():
    assert repair_tool_history([AIMessage(content="hi")]) == [AIMessage(content="hi")]
    assert repair_tool_history([]) == []


# --- review-fix pins ---------------------------------------------------------

def test_price_po_rejects_empty_orders(conn):
    with pytest.raises(purchasing.POValidationError):
        purchasing.price_po(conn, "Roma Foods", [])


def test_supplier_terms_are_source_filtered():
    from eightysix.rag import CHROMA_DIR, retrieve_supplier_terms
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        pytest.skip("Chroma index not built -- run `make seed` first")
    docs = retrieve_supplier_terms("Valco Cash & Carry")
    assert docs, "index must contain Valco terms"
    assert {d.metadata["source"] for d in docs} == {"valco_cash_carry"}
    # the trap chunk itself must be in the retrieved set, by construction
    assert any("deliver" in d.page_content.lower() for d in docs)


def test_every_seed_supplier_has_a_docs_mapping(conn):
    from eightysix.rag import SUPPLIER_DOCS
    names = {r["name"] for r in conn.execute("SELECT name FROM suppliers")}
    assert names == set(SUPPLIER_DOCS)
