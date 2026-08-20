"""The LLM seam, faked: what the nodes do when the model misbehaves.

config.chat_model() is the only place a real model is constructed, so these
tests monkeypatch it at the nodes' import site and script the replies. The
nodes, the graph, and the purchasing math under test are the real ones; only
the model is fake. No API keys, no network.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

import eightysix.nodes as nodes
from eightysix import db, rag
from eightysix.graph import build_graph
from eightysix.nodes import Route
from eightysix.state import POChoice, POLineChoice

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


class FakeStructured:
    """Stands in for with_structured_output(schema): pops one scripted reply
    per invoke; a reply that is an Exception raises instead. Keeps every
    prompt it saw so tests can assert what the model was actually told."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeChatModel:
    """Duck-types the slice of ChatAnthropic the nodes touch."""

    def __init__(self, structured=None, replies=()):
        self.structured = structured or {}   # schema class -> FakeStructured
        self.plain = FakeStructured(replies)

    def with_structured_output(self, schema):
        return self.structured[schema]

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return self.plain.invoke(prompt)


@pytest.fixture()
def low_pepperoni_db(tmp_path, monkeypatch):
    """A real db file with pepperoni below threshold. Nodes open their own
    connections, so db.connect itself is pointed at the fixture."""
    path = tmp_path / "fixture.db"
    conn = db.connect(path)
    db.init_schema(conn)
    db.seed_from_csvs(conn, SEED_DIR)
    conn.execute("UPDATE ingredients SET on_hand = 840 WHERE name = 'pepperoni'")
    conn.commit()
    conn.close()
    real_connect = db.connect
    monkeypatch.setattr(db, "connect", lambda p=None: real_connect(path))
    return path


@pytest.fixture()
def offline_terms(monkeypatch):
    """Keeps draft_po off the Chroma index: supplier terms come from here."""
    doc = Document(
        page_content="Roma Foods - Delivery days:\nTrucks run Tue/Thu/Sat.",
        metadata={"source": "roma_foods", "section": "Delivery days"},
    )
    monkeypatch.setattr(rag, "retrieve_supplier_terms", lambda supplier, k=3: [doc])


def _fake(monkeypatch, **kwargs):
    fake = FakeChatModel(**kwargs)
    monkeypatch.setattr(nodes, "chat_model", lambda *a, **k: fake)
    return fake


# --- the router's failure direction ------------------------------------------

def test_router_falls_back_to_inventory_on_model_failure(monkeypatch):
    _fake(monkeypatch, structured={Route: FakeStructured([RuntimeError("api down")])})
    update = nodes.route({"messages": [HumanMessage("we're almost out of pepperoni")]})
    # the safe default is the read-only branch, never reorder
    assert update["intent"] == "inventory"
    # and the per-turn reset must still happen on the fallback path
    assert update["po_draft"] is None
    assert update["approved"] is None


# --- draft_po: propose, reject, retry, re-price -------------------------------

def test_draft_po_feeds_rejection_back_then_reprices_from_catalog(
        low_pepperoni_db, offline_terms, monkeypatch):
    bad = POChoice(supplier_name="Sysco",
                   lines=[POLineChoice(ingredient="pepperoni", packs=1)],
                   expected_delivery="tomorrow", rationale="invented supplier")
    good = POChoice(supplier_name="Roma Foods",
                    lines=[POLineChoice(ingredient="pepperoni", packs=1)],
                    expected_delivery="Saturday morning", rationale="per the terms")
    drafts = FakeStructured([bad, good])
    _fake(monkeypatch, structured={POChoice: drafts})

    update = nodes.draft_po({"messages": [HumanMessage("draft a reorder for whatever's low")]})

    po = update["po_draft"]
    assert po["supplier_name"] == "Roma Foods"
    assert po["subtotal_cents"] == 5800          # the catalog's price, nobody else's
    # the second attempt must know why the first was rejected
    assert len(drafts.prompts) == 2
    assert "previous proposal was rejected" in drafts.prompts[1]
    assert "Sysco" in drafts.prompts[1]
    assert "previous proposal was rejected" not in drafts.prompts[0]


def test_draft_po_gives_up_honestly_after_two_bad_proposals(
        low_pepperoni_db, offline_terms, monkeypatch):
    # Roma doesn't carry basil, so this proposal can never price
    bad = POChoice(supplier_name="Roma Foods",
                   lines=[POLineChoice(ingredient="basil", packs=1)],
                   expected_delivery="x", rationale="y")
    _fake(monkeypatch, structured={POChoice: FakeStructured([bad, bad])})

    update = nodes.draft_po({"messages": [HumanMessage("draft a reorder")]})

    assert update["po_draft"] is None
    assert "couldn't put together a valid order" in update["messages"][-1].content
    # the terms it read still surface as citations on the failure message
    assert update["citations"] == [{"source": "roma_foods", "section": "Delivery days"}]


def test_draft_po_api_failure_aborts_without_a_draft(
        low_pepperoni_db, offline_terms, monkeypatch):
    _fake(monkeypatch, structured={POChoice: FakeStructured([RuntimeError("overloaded")])})
    update = nodes.draft_po({"messages": [HumanMessage("draft a reorder")]})
    assert update["po_draft"] is None
    assert "Nothing was sent" in update["messages"][-1].content


# --- inventory_agent: truncation must not reach the tool node -----------------

def test_truncated_agent_turn_never_reaches_the_tool_node(monkeypatch):
    truncated = AIMessage(
        content="",
        tool_calls=[{"name": "get_stock", "args": {"ingredient": "moz"}, "id": "c1"}],
        response_metadata={"stop_reason": "max_tokens"},
    )
    _fake(monkeypatch, replies=[truncated])
    update = nodes.inventory_agent({"messages": [HumanMessage("how much mozzarella?")]})
    reply = update["messages"][-1]
    # tools_condition routes on tool_calls; the replacement must carry none
    assert not reply.tool_calls
    assert "ran out of room" in reply.content


# --- the stale-draft semantics, pinned through the real graph -----------------

def test_new_question_abandons_a_paused_draft(low_pepperoni_db, offline_terms, monkeypatch):
    """The behavior the UI's approval card relies on (app.py drops the card on
    a new question): new input on an interrupted thread discards the pending
    approval, and route's per-turn reset nulls the draft in the checkpoint --
    so a late or replayed approve can never send a stale PO."""
    good = POChoice(supplier_name="Roma Foods",
                    lines=[POLineChoice(ingredient="pepperoni", packs=1)],
                    expected_delivery="Saturday", rationale="the terms say so")
    _fake(monkeypatch, structured={
        Route: FakeStructured([Route(intent="reorder"), Route(intent="off_topic")]),
        POChoice: FakeStructured([good]),
    })
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-abandon"}}

    paused = graph.invoke({"messages": [("user", "draft a reorder for whatever's low")]}, config)
    assert "__interrupt__" in paused
    assert graph.get_state(config).next == ("human_approval",)

    # the owner asks something else instead of deciding
    graph.invoke({"messages": [("user", "who won the game last night?")]}, config)
    state = graph.get_state(config)
    assert state.values["po_draft"] is None      # route's reset killed the draft
    assert not state.next                        # nothing left pending

    # a late approve (stale card, replayed request) is a quiet no-op: the
    # thread has no pending task, so the invoke returns the checkpoint as-is
    before = len(state.values["messages"])
    late = graph.invoke(Command(resume={"approved": True}), config)
    assert late["po_draft"] is None
    assert len(late["messages"]) == before       # nothing ran, nothing answered
    check = db.connect()
    sent = check.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
    check.close()
    assert sent == 0                             # and above all: nothing sent
