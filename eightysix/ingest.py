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


class MalformedOrder(Exception):
    """The raw order is missing fields or has non-whole quantities."""


def _whole_qty(value) -> int:
    # The reject-don't-clamp rule from apply_orders applies here too: int()
    # would silently turn a POS export's 2.9 into 2, so a fractional quantity
    # is malformed, not roundable. (bool is an int subclass; a JSON true here
    # is garbage, not a quantity of one.)
    if isinstance(value, bool):
        raise ValueError(f"quantity must be a number, got {value!r}")
    if isinstance(value, int):
        return value
    qty = int(value)           # floats truncate; garbage strings raise here
    if qty != float(value):    # ...and the truncated ones get rejected here
        raise ValueError(f"quantity {value!r} is not a whole number")
    return qty


def minimize_order(raw: dict) -> dict:
    try:
        return {
            "order_id": raw["order_id"],
            "timestamp": raw["timestamp"],
            "items": [{"menu_item": i["menu_item"], "qty": _whole_qty(i["qty"])} for i in raw["items"]],
        }
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        # OverflowError: python's json parser accepts Infinity; int() and
        # float() both refuse it
        raise MalformedOrder(f"{type(e).__name__}: {e}") from e


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

        # POS data is external input. A negative qty would INCREASE stock and
        # poison the usage ledger; zero writes noise. Reject, don't clamp.
        bad_qty = [i for i in order["items"] if not isinstance(i["qty"], int) or i["qty"] < 1]
        if bad_qty:
            errors.append({"order_id": oid, "invalid_qty": bad_qty})
            continue

        # one transaction per order: a failure mid-order rolls back that order
        # completely (no half-applied depletion, no lying ledger entry) and
        # the rest of the batch still lands
        try:
            db.record_order(conn, oid, order["timestamp"], order["items"])
            for item in order["items"]:
                for ingredient_id, qty_per_item in recipe_map[item["menu_item"]]:
                    db.apply_depletion(
                        conn, ingredient_id, qty_per_item * item["qty"], order["timestamp"], oid
                    )
            conn.commit()
            applied.append(oid)
        except Exception as e:
            conn.rollback()
            errors.append({"order_id": oid, "failed": f"{type(e).__name__}: {e}"})
    return {
        "applied": len(applied),
        "skipped_duplicates": len(skipped),
        "errors": errors,
        "low_stock": db.get_low_stock(conn),
    }


def _rebase_timestamps(orders: list[dict]) -> list[dict]:
    """Shift the batch so its latest order lands now, keeping relative spacing.

    The committed demo file is dated Friday 2026-08-14. Usage history windows
    on the real clock, so without rebasing, the "how fast are we burning it"
    numbers go silently empty a week after that date -- fine for us, wrong for
    a reviewer cloning later. Order content stays byte-identical; only the
    clock moves.
    """
    from datetime import datetime

    latest = max(datetime.fromisoformat(o["timestamp"]) for o in orders)
    delta = datetime.now() - latest
    return [
        {**o, "timestamp": (datetime.fromisoformat(o["timestamp"]) + delta)
                            .strftime("%Y-%m-%dT%H:%M:%S")}
        for o in orders
    ]


def ingest_file(conn: sqlite3.Connection, path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # minimize per order so one malformed order is reported, not fatal to the file
    orders, malformed = [], []
    for i, raw_order in enumerate(raw["orders"]):
        try:
            orders.append(minimize_order(raw_order))
        except MalformedOrder as e:
            # raw_order may not even be a dict -- the report must not crash
            # on the same garbage the parse just rejected
            oid = raw_order.get("order_id", f"#{i}") if isinstance(raw_order, dict) else f"#{i}"
            malformed.append({"order_id": oid, "malformed": str(e)})
    if orders:
        orders = _rebase_timestamps(orders)
    summary = apply_orders(conn, orders)
    summary["errors"] = malformed + summary["errors"]
    return summary


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
