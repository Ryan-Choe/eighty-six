"""Deterministic purchasing math. Nothing in this file calls a model.

The LLM chooses a supplier and pack counts; price_po() is the authority
that re-prices the choice against the catalog and rejects anything that
doesn't exist. Model output is a proposal, never a fact.
"""

import json
import math
from datetime import datetime

from eightysix import db


def build_candidates(conn, low_stock: list[dict]) -> list[dict]:
    """For each low ingredient, every supplier option with packs-to-par math."""
    if not low_stock:
        return []
    options = db.get_supplier_options(conn, [i["name"] for i in low_stock])
    candidates = []
    for option in options:
        need = option["par_level"] - option["on_hand"]
        packs = max(1, math.ceil(need / option["pack_qty"]))
        candidates.append({
            "ingredient": option["ingredient"],
            "supplier": option["supplier"],
            "supplier_id": option["supplier_id"],
            "need": need,
            "pack_desc": option["pack_desc"],
            "pack_qty": option["pack_qty"],
            "price_cents": option["price_cents"],
            "packs_to_par": packs,
            "cost_to_par_cents": packs * option["price_cents"],
            "supplier_min_order_cents": option["min_order_cents"],
        })
    return candidates


class POValidationError(Exception):
    """The model proposed a line the catalog can't back."""


def price_po(conn, supplier_name: str, lines: list[dict]) -> dict:
    """Re-price a proposed PO from the catalog. Raises on anything invented.

    lines: [{"ingredient": str, "packs": int}, ...]
    """
    supplier = conn.execute(
        "SELECT id, name, min_order_cents FROM suppliers WHERE name = ? COLLATE NOCASE",
        (supplier_name,),
    ).fetchone()
    if supplier is None:
        raise POValidationError(f"no supplier named {supplier_name!r}")
    if not lines:
        raise POValidationError("order has no lines")

    priced_lines = []
    for line in lines:
        if not isinstance(line.get("packs"), int) or line["packs"] < 1:
            raise POValidationError(f"bad pack count for {line.get('ingredient')!r}")
        row = conn.execute(
            """SELECT i.name AS ingredient, si.pack_qty, si.pack_desc, si.price_cents
               FROM supplier_items si
               JOIN ingredients i ON i.id = si.ingredient_id
               WHERE si.supplier_id = ? AND lower(i.name) = lower(?)""",
            (supplier["id"], line["ingredient"]),
        ).fetchone()
        if row is None:
            raise POValidationError(
                f"{supplier['name']} does not carry {line['ingredient']!r}"
            )
        priced_lines.append({
            "ingredient": row["ingredient"],
            "packs": line["packs"],
            "pack_desc": row["pack_desc"],
            "unit_price_cents": row["price_cents"],       # catalog, never the model
            "line_total_cents": line["packs"] * row["price_cents"],
        })

    subtotal = sum(l["line_total_cents"] for l in priced_lines)
    return {
        "supplier_id": supplier["id"],
        "supplier_name": supplier["name"],
        "lines": priced_lines,
        "subtotal_cents": subtotal,
        "min_order_cents": supplier["min_order_cents"],
        "meets_minimum": subtotal >= supplier["min_order_cents"],
    }


def create_and_send_po(conn, po: dict) -> int:
    """Write the PO and 'send' it. The only side effect in the reorder path,
    and it runs strictly after the human approval interrupt."""
    po_id = db.create_po(
        conn, po["supplier_id"], po["subtotal_cents"], json.dumps(po["lines"])
    )
    db.mark_po_sent(conn, po_id)
    conn.commit()
    # the stub: a real system would email/EDI this
    print(f"[send_po stub] PO-{po_id} -> {po['supplier_name']} "
          f"(${po['subtotal_cents'] / 100:.2f}) at {datetime.now():%H:%M}")
    return po_id
