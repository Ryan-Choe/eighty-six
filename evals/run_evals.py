"""Run the three experiments. Every target builds fresh state per example --
evals never touch the demo database."""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402

os.environ.setdefault("EIGHTYSIX_DEMO_NOW", "Friday August 14, 09:30 PM")

from langsmith import evaluate  # noqa: E402

import eightysix.nodes as nodes  # noqa: E402
from eightysix import db  # noqa: E402
from eightysix.graph import build_graph  # noqa: E402
from eightysix.ingest import apply_orders, minimize_order  # noqa: E402
from evals.evaluators import (  # noqa: E402
    answer_quality, cited_expected_doc, correct_route, reorder_decision, stock_exact,
)

REPO = Path(__file__).resolve().parent.parent


def routing_target(inputs: dict) -> dict:
    # the real route node, not a copy of its prompt. Direct node calls skip
    # LangGraph's tuple->message coercion, so build the HumanMessage ourselves.
    from langchain_core.messages import HumanMessage
    return nodes.route({"messages": [HumanMessage(inputs["message"])]})


def math_target(inputs: dict) -> dict:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    db.seed_from_csvs(conn, REPO / "data" / "seed")
    result = apply_orders(conn, inputs["orders"])
    tracked = ["fresh mozzarella", "low-moisture mozzarella", "pepperoni",
               "00 flour", "garlic", "tomato sauce"]
    return {
        "applied": result["applied"],
        "errors": len(result["errors"]),
        "stock_after": {n: db.get_stock(conn, n)["on_hand"] for n in tracked},
        "low_stock": [i["name"] for i in result["low_stock"]],
    }


def _fixture_db(kind: str, path: Path):
    conn = db.connect(path)
    db.init_schema(conn)
    db.seed_from_csvs(conn, REPO / "data" / "seed")
    if kind == "friday_rush":
        raw = json.loads((REPO / "data" / "pos_orders" / "orders_friday.json").read_text())
        apply_orders(conn, [minimize_order(o) for o in raw["orders"]])
    elif kind == "pepperoni_only":
        conn.execute("UPDATE ingredients SET on_hand = 840 WHERE name = 'pepperoni'")
        conn.commit()
    conn.close()


def graph_target(inputs: dict) -> dict:
    # each example gets its own database file; the graph's nodes open
    # connections themselves, so db.connect is patched for the duration
    fixture = Path(f"/tmp/eightysix-eval-{uuid.uuid4().hex}.db")
    _fixture_db(inputs.get("fixture", "seeded"), fixture)
    real_connect = db.connect
    try:
        with patch.object(db, "connect", lambda path=None: real_connect(fixture)):
            graph = build_graph()
            state = graph.invoke(
                {"messages": [("user", inputs["message"])]},
                {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"},
                 "recursion_limit": 12,
                 "run_name": "eval_turn"},
            )
        def _flat(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(b.get("text", "") for b in content if isinstance(b, dict))
            return ""

        answer = ""
        for message in reversed(state.get("messages", [])):
            if message.type == "ai" and _flat(message.content):
                answer = _flat(message.content)
                break
        return {"answer": answer,
                "intent": state.get("intent"),
                "po_draft": state.get("po_draft"),
                "citations": state.get("citations")}
    finally:
        fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    evaluate(routing_target, data="eighty-six-routing-v2",
             evaluators=[correct_route],
             experiment_prefix="routing-haiku", max_concurrency=1)
    # same node, big model: the on-screen evidence for the Haiku cost lever
    from eightysix.config import MODEL
    with patch.object(nodes, "ROUTER_MODEL", MODEL):
        evaluate(routing_target, data="eighty-six-routing-v2",
                 evaluators=[correct_route],
                 experiment_prefix="routing-opus", max_concurrency=1)
    evaluate(math_target, data="eighty-six-inventory-math-v1",
             evaluators=[stock_exact],
             experiment_prefix="inventory-math", max_concurrency=1)
    evaluate(graph_target, data="eighty-six-qa-reorder-v1",
             evaluators=[answer_quality, cited_expected_doc, reorder_decision],
             experiment_prefix="qa-reorder", max_concurrency=1)
    print("three experiments submitted -- see the Experiments tab per dataset")
