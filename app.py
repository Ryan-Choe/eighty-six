"""Streamlit UI for eighty-six.

The graph is interface-agnostic: this file renders state and forwards
decisions, nothing more. Two Streamlit traps shape the structure -- the whole
script reruns on every interaction, so the compiled graph lives behind
st.cache_resource (a fresh graph per rerun would orphan the checkpointer and
the interrupt could never resume), and the pending approval lives in
st.session_state so the card survives the rerun a button click triggers.
"""

import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# anchor to this file so `streamlit run` works from any directory
REPO = Path(__file__).resolve().parent
load_dotenv(REPO / ".env")

from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.types import Command  # noqa: E402

from eightysix import db  # noqa: E402
from eightysix.config import RECURSION_LIMIT  # noqa: E402
from eightysix.graph import build_graph  # noqa: E402
from eightysix.ingest import ingest_file  # noqa: E402
from eightysix.redaction import redact  # noqa: E402

st.set_page_config(page_title="eighty-six", page_icon="🍕", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []      # [(role, text)] for re-rendering
    st.session_state.pending_po = None
# outside the guard: a live session from before this key existed (the dev
# server hot-reloads across source edits) must not AttributeError on it
st.session_state.setdefault("pending_decision", None)

CONFIG = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "recursion_limit": RECURSION_LIMIT,
    "run_name": "ui_turn",
    "tags": ["streamlit"],
    "metadata": {"thread_id": st.session_state.thread_id, "interface": "streamlit"},
}


def _md(text: str) -> str:
    # st.markdown treats $...$ as LaTeX; a sentence with two prices turns into
    # garbled math. Escape at display time only -- history keeps the raw text.
    return text.replace("$", "\\$")


def _flat(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def run_turn(graph_input) -> str:
    """Stream one graph run into the current chat message; returns final text."""
    is_resume = isinstance(graph_input, Command)
    text_area = st.empty()
    collected = []
    citations = None
    failed = False
    with st.status("thinking...", expanded=False) as status:
        # same guards the CLI has: a blip mid-demo should end the turn with a
        # sentence, not a traceback
        try:
            for mode, chunk in graph.stream(
                graph_input, CONFIG, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        st.session_state.pending_po = chunk["__interrupt__"][0].value["po_draft"]
                        continue
                    for node, update in chunk.items():
                        if node == "route":
                            status.update(label=f"route -> {update.get('intent')}")
                        elif node == "inventory_tools":
                            for m in update.get("messages", []):
                                status.write(f"tool {m.name}: {str(m.content)[:80]}")
                        if update and update.get("citations"):
                            citations = update["citations"]
                else:
                    message_chunk, meta = chunk
                    if meta.get("langgraph_node") in (
                        "inventory_agent", "deflect", "policy_qa", "draft_po",
                        "send_po", "cancel_po",
                    ):
                        token = _flat(message_chunk.content)
                        if token:
                            collected.append(token)
                            text_area.markdown(_md("".join(collected)))
            status.update(label="done", state="complete")
        except GraphRecursionError:
            failed = True
            status.update(label="hit the tool budget", state="error")
            collected.append("\n\nI hit my tool-call budget on that one without "
                             "landing an answer. Try a narrower question.")
        except Exception as e:
            failed = True
            status.update(label="error", state="error")
            if is_resume:
                # send_po commits before the stream finishes, so "nothing was
                # ordered" would be a guess here -- don't claim it
                collected.append(f"\n\nSomething went wrong ({type(e).__name__}) "
                                 "while finalizing the decision. Check the sent "
                                 "POs before re-ordering.")
            else:
                collected.append(f"\n\nSomething went wrong ({type(e).__name__}). "
                                 "Nothing was ordered -- ask again.")
    answer = "".join(collected).strip()
    if citations and not failed:
        # the state's citations come from retrieved chunk metadata, never from
        # the model's text -- this line is the UI's sources list
        sources = " · ".join(f"{c['source']} § {c['section']}" for c in citations)
        answer += f"\n\n*Sources: {sources}*"
    if answer:
        text_area.markdown(_md(answer))
    return answer


# ---- sidebar: the 86 board ---------------------------------------------------

with st.sidebar:
    st.title("🍕 eighty-six")
    st.caption("Sal's Slice House — inventory copilot")

    conn = db.connect()
    rows = db.get_all_stock(conn)
    conn.close()
    flagged = [r for r in rows if r["flagged"]]

    st.subheader(f"86 board ({len(flagged)} flagged)")
    st.dataframe(
        [{"ingredient": r["name"],
          "on hand": f"{r['on_hand']}{r['unit']}",
          "threshold": f"{r['reorder_threshold']}{r['unit']}"}
         for r in rows],
        hide_index=True, height=320,
    )

    if st.button("Simulate Friday rush", use_container_width=True):
        # stock is about to change, so a draft priced against pre-rush stock
        # is stale -- abandon it, same as asking a new question would
        st.session_state.pending_po = None
        conn = db.connect()
        summary = ingest_file(conn, REPO / "data" / "pos_orders" / "orders_friday.json")
        conn.close()
        st.toast(f"{summary['applied']} orders in, "
                 f"{summary['skipped_duplicates']} duplicates skipped, "
                 f"{len(summary['low_stock'])} items flagged")
        st.rerun()

# ---- main: chat --------------------------------------------------------------

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(_md(text))

# an approve/reject click lands here one rerun later, so this pass renders no
# card and its buttons can't fire twice. Streamlit keeps the PREVIOUS pass's
# widgets clickable until this run finishes, so a stray click can still abort
# the stream -- but the decision is consumed before the run starts, so an
# aborted resume can never replay, and nothing can double-send.
if (decision := st.session_state.pending_decision) is not None:
    st.session_state.pending_decision = None
    with st.chat_message("assistant"):
        outcome = run_turn(Command(resume={"approved": decision}))
    if outcome:
        st.session_state.history.append(("assistant", outcome))

if question := st.chat_input("Ask about stock, reorders, or kitchen policy"):
    question = redact(question)   # before invoke: traces record inputs verbatim
    st.session_state.pending_po = None   # a new question abandons the paused draft; the card is dead
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(_md(question))
    with st.chat_message("assistant"):
        answer = run_turn({"messages": [("user", question)]})
    if answer:
        st.session_state.history.append(("assistant", answer))
    if st.session_state.pending_po:
        st.rerun()   # render the approval card immediately

# ---- the approval card -------------------------------------------------------

if po := st.session_state.pending_po:
    with st.container(border=True):
        st.subheader(f"Purchase order — {po['supplier_name']}")
        st.dataframe(
            [{"item": l["ingredient"], "packs": l["packs"], "pack": l["pack_desc"],
              "line total": f"${l['line_total_cents'] / 100:.2f}"}
             for l in po["lines"]],
            hide_index=True,
        )
        subtotal = f"${po['subtotal_cents'] / 100:.2f}"
        if po["meets_minimum"]:
            st.markdown(_md(f"**Total: {subtotal}**"))
        else:
            st.markdown(_md(f"**Total: {subtotal}** — under the "
                            f"${po['min_order_cents'] / 100:.0f} minimum, expect a fee"))
        st.caption(_md(po["expected_delivery"]))

        approve, reject = st.columns(2)
        decision = None
        if approve.button("Approve & send", type="primary", use_container_width=True):
            decision = True
        if reject.button("Reject", use_container_width=True):
            decision = False
        if decision is not None:
            # don't resume here: this run would keep rendering the card under
            # the stream. Kill the card, hand the decision to the next pass.
            st.session_state.pending_po = None
            st.session_state.pending_decision = decision
            st.rerun()
