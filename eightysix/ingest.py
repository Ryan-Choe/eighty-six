"""Order ingestion. Deliberately NOT part of the agent graph: a batch of POS
orders is structured data, so this is a plain pipeline — deterministic,
unit-testable, no LLM anywhere in it."""

import json
import sys
import sqlite3

from langsmith import traceable

from eightysix import db

# The raw POS export carries customer_name / phone / notes. The inventory
# pipeline needs none of that, so we allowlist instead of redact — you can't
# leak what you never ingest.
ALLOWED_ORDER_FIELDS = ("order_id", "timestamp", "items")


def minimize_order(raw: dict) -> dict:
    return {
        "order_id": raw["order_id"],
        "timestamp": raw["timestamp"],
        "items": [{"menu_item": i["menu_item"], "qty": int(i["qty"])} for i in raw["items"]],
    }


@traceable(name="apply_orders")
def apply_orders(conn: sqlite3.Connection, orders: list[dict]) -> dict:
    """Deplete stock for a batch of minimized orders. Returns a summary.

    Replay-safe: order ids already in the applied_orders ledger are skipped,
    so running the same POS file twice is a no-op, not a double decrement.
    """
    recipe_map = db.get_recipe_map(conn)
    applied, skipped, errors = [], [], []

    for order in orders:
        oid = order["order_id"]
        if db.was_applied(conn, oid):
            skipped.append(oid)
            continue

        unknown = [i["menu_item"] for i in order["items"] if i["menu_item"] not in recipe_map]
        if unknown:
            # whole order rejected: partial application would make the ledger lie
            errors.append({"order_id": oid, "unknown_items": unknown})
            continue

        db.record_order(conn, oid, order["timestamp"], order["items"])
        for item in order["items"]:
            for ingredient_id, qty_per_item in recipe_map[item["menu_item"]]:
                db.apply_depletion(
                    conn, ingredient_id, qty_per_item * item["qty"], order["timestamp"], oid
                )
        applied.append(oid)

    conn.commit()
    return {
        "applied": len(applied),
        "skipped_duplicates": len(skipped),
        "errors": errors,
        "low_stock": db.get_low_stock(conn),
    }


def ingest_file(conn: sqlite3.Connection, path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    orders = [minimize_order(o) for o in raw["orders"]]
    return apply_orders(conn, orders)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    conn = db.connect()
    summary = ingest_file(conn, sys.argv[1])
    print(f"applied {summary['applied']} orders, "
          f"skipped {summary['skipped_duplicates']} duplicates, "
          f"{len(summary['errors'])} errors")
    for item in summary["low_stock"]:
        print(f"  86 WATCH: {item['name']} at {item['on_hand']}{item['unit']} "
              f"(threshold {item['reorder_threshold']}{item['unit']})")

    # traces submit on a background thread; short-lived scripts must flush
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
