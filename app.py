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
    text_area = st.empty()
    collected = []
    with st.status("thinking...", expanded=False) as status:
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
    return "".join(collected)


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

if question := st.chat_input("Ask about stock, reorders, or kitchen policy"):
    question = redact(question)   # before invoke: traces record inputs verbatim
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
            st.markdown(f"**Total: {subtotal}**")
        else:
            st.markdown(f"**Total: {subtotal}** — under the "
                        f"${po['min_order_cents'] / 100:.0f} minimum, expect a fee")
        st.caption(po["expected_delivery"])

        approve, reject = st.columns(2)
        decision = None
        if approve.button("Approve & send", type="primary", use_container_width=True):
            decision = True
        if reject.button("Reject", use_container_width=True):
            decision = False
        if decision is not None:
            st.session_state.pending_po = None
            with st.chat_message("assistant"):
                outcome = run_turn(Command(resume={"approved": decision}))
            if outcome:
                st.session_state.history.append(("assistant", outcome))
            st.rerun()
