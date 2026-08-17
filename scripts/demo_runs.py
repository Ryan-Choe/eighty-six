"""Populate the eighty-six-demo LangSmith project with a clean, curated set of
runs -- the reviewer opens ~a dozen named runs telling one story, not hundreds
of dev traces. Run via `make demo` (which sets LANGSMITH_PROJECT)."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

import os  # noqa: E402

os.environ["EIGHTYSIX_DEMO_NOW"] = "Friday August 14, 09:30 PM"

from langgraph.types import Command  # noqa: E402

from eightysix import db  # noqa: E402
from eightysix.graph import build_graph  # noqa: E402
from eightysix.ingest import ingest_file  # noqa: E402

BEATS = [
    ("demo:stock_question", "how much fresh mozzarella do we have?"),
    ("demo:feasibility", "can we make 12 margheritas?"),
    ("demo:weekend_judgment", "do we have enough fresh mozzarella for the weekend?"),
    ("demo:policy_citation", "how long can thawed dough sit in the walk-in?"),
    ("demo:honesty_case", "what's our gluten-free cross-contamination policy?"),
]

if __name__ == "__main__":
    print("resetting world: seed + friday rush")
    os.system(f"{REPO}/.venv/bin/python {REPO}/scripts/seed_db.py > /dev/null")
    conn = db.connect()
    ingest_file(conn, REPO / "data" / "pos_orders" / "orders_friday.json")
    conn.close()

    graph = build_graph()
    for run_name, question in BEATS:
        config = {"configurable": {"thread_id": f"demo-{uuid.uuid4()}"},
                  "recursion_limit": 12, "run_name": run_name, "tags": ["demo"]}
        state = graph.invoke({"messages": [("user", question)]}, config)
        print(f"  {run_name}: ok")

    # the hero run: reorder -> interrupt -> approve, one thread, two traces
    config = {"configurable": {"thread_id": f"demo-{uuid.uuid4()}"},
              "recursion_limit": 12, "run_name": "demo:reorder_draft", "tags": ["demo"]}
    state = graph.invoke(
        {"messages": [("user", "we got slammed tonight, draft a reorder for whatever's low")]},
        config,
    )
    assert state.get("po_draft"), "demo reorder produced no draft"
    config["run_name"] = "demo:reorder_approved"
    graph.invoke(Command(resume={"approved": True}), config)
    print("  demo:reorder_draft + demo:reorder_approved: ok (paused, then resumed)")

    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    print("done -- open the eighty-six-demo project in LangSmith")
