"""SQLite access for eighty-six. Every SQL string in the project lives in this file."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "pizzeria.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingredients (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    unit TEXT NOT NULL,                -- 'g' | 'ml' | 'each'; quantities are integers
    on_hand INTEGER NOT NULL,
    par_level INTEGER NOT NULL,
    reorder_threshold INTEGER NOT NULL,
    storage TEXT NOT NULL              -- 'walk-in' | 'dry' | 'freezer'
);

CREATE TABLE IF NOT EXISTS menu_items (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    price_cents INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS recipes (
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    qty_per_item INTEGER NOT NULL,
    PRIMARY KEY (menu_item_id, ingredient_id)
);

CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    contact TEXT NOT NULL,
    min_order_cents INTEGER NOT NULL,
    lead_time_days INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS supplier_items (
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    pack_qty INTEGER NOT NULL,         -- in the ingredient's unit
    pack_desc TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    PRIMARY KEY (supplier_id, ingredient_id)
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,               -- POS order id
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL REFERENCES orders(id),
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id),
    qty INTEGER NOT NULL
);

-- ledger of every stock change; get_usage() is a SUM over this
CREATE TABLE IF NOT EXISTS stock_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    delta INTEGER NOT NULL,
    reason TEXT NOT NULL,              -- 'order' | 'delivery' | 'adjustment'
    ref TEXT
);

-- idempotency: an order id in here has already been applied, so replaying
-- the same POS file is a no-op instead of a double decrement
CREATE TABLE IF NOT EXISTS applied_orders (
    order_id TEXT PRIMARY KEY,
    applied_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts TEXT NOT NULL,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    status TEXT NOT NULL,              -- 'draft' | 'sent' | 'cancelled'
    total_cents INTEGER NOT NULL,
    lines_json TEXT NOT NULL           -- JSON lines are fine at demo scale
);
"""


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


# --- reads -------------------------------------------------------------------


# Name resolution: exact -> unique substring -> ambiguous list -> full catalog.
# Deterministic code does the resolving so the model never has to guess a name.
RESOLVABLE_TABLES = {"ingredients", "menu_items"}


def resolve_name(conn, table: str, name: str) -> tuple[str, str | list[str]]:
    """Returns ("ok", canonical) | ("ambiguous", candidates) | ("unknown", all_names)."""
    if table not in RESOLVABLE_TABLES:  # keeps the f-string un-injectable
        raise ValueError(f"unresolvable table: {table}")
    names = [r["name"] for r in conn.execute(f"SELECT name FROM {table}")]
    wanted = name.strip().lower()
    exact = [n for n in names if n.lower() == wanted]
    if exact:
        return ("ok", exact[0])
    subs = [n for n in names if wanted in n.lower() or n.lower() in wanted]
    if len(subs) == 1:
        return ("ok", subs[0])
    if subs:
        return ("ambiguous", sorted(subs))
    return ("unknown", sorted(names))


def get_stock(conn, ingredient: str) -> dict | None:
    row = conn.execute(
        """SELECT name, unit, on_hand, par_level, reorder_threshold, storage
           FROM ingredients WHERE name = ? COLLATE NOCASE""",
        (ingredient,),
    ).fetchone()
    return dict(row) if row else None


def get_low_stock(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT name, unit, on_hand, par_level, reorder_threshold
           FROM ingredients WHERE on_hand <= reorder_threshold
           ORDER BY on_hand * 1.0 / reorder_threshold"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_stock(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT name, unit, on_hand, par_level, reorder_threshold, storage,
                  on_hand <= reorder_threshold AS flagged
           FROM ingredients ORDER BY flagged DESC, name"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_recipe_map(conn) -> dict[str, list[tuple[int, int]]]:
    """menu item name -> [(ingredient_id, qty_per_item), ...]"""
    rows = conn.execute(
        """SELECT m.name AS menu_item, r.ingredient_id, r.qty_per_item
           FROM recipes r JOIN menu_items m ON m.id = r.menu_item_id"""
    ).fetchall()
    out: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        out.setdefault(r["menu_item"], []).append((r["ingredient_id"], r["qty_per_item"]))
    return out


def get_usage(conn, ingredient: str, days: int = 7) -> dict | None:
    row = conn.execute(
        """SELECT i.name, i.unit, i.on_hand,
                  COALESCE(SUM(-m.delta), 0) AS total_used,
                  COUNT(DISTINCT date(m.ts)) AS data_days
           FROM ingredients i
           LEFT JOIN stock_movements m
             ON m.ingredient_id = i.id AND m.reason = 'order'
             AND m.ts >= datetime('now', ?)
           WHERE i.name = ? COLLATE NOCASE
           GROUP BY i.id""",
        (f"-{days} days", ingredient),
    ).fetchone()
    if row is None:
        return None
    total, data_days = row["total_used"], row["data_days"]
    # Burn rate per day that actually has sales. Dividing by the requested
    # window reports a 7x-too-low rate when only one day of history exists;
    # days-with-data treats missing days as "no data", not "no sales".
    avg_daily = total / data_days if data_days else 0
    return {
        "name": row["name"],
        "unit": row["unit"],
        "on_hand": row["on_hand"],
        "window_days": days,
        "data_days": data_days,
        "total_used": total,
        "avg_daily": round(avg_daily, 1),
        "days_of_cover": round(max(row["on_hand"], 0) / avg_daily, 1) if avg_daily else None,
    }


def check_feasibility(conn, menu_item: str, servings: int) -> dict | None:
    """How many of this menu item can we make right now, and what runs out first."""
    rows = conn.execute(
        """SELECT i.name, i.unit, i.on_hand, r.qty_per_item
           FROM recipes r
           JOIN menu_items m ON m.id = r.menu_item_id
           JOIN ingredients i ON i.id = r.ingredient_id
           WHERE m.name = ? COLLATE NOCASE""",
        (menu_item,),
    ).fetchall()
    if not rows:
        return None

    # integer division: you can't plate a partial pizza. max(0, ...) because
    # oversold stock can go negative in the ledger, but "-87 servings" is
    # not an answer -- zero is.
    possible = {r["name"]: max(0, r["on_hand"]) // r["qty_per_item"] for r in rows}
    max_servings = min(possible.values())
    limiting = min(possible, key=possible.get)
    shortfalls = [
        {
            "ingredient": r["name"],
            "needed": r["qty_per_item"] * servings,
            "on_hand": r["on_hand"],
            "short_by": r["qty_per_item"] * servings - r["on_hand"],
            "unit": r["unit"],
        }
        for r in rows
        if r["qty_per_item"] * servings > r["on_hand"]
    ]
    return {
        "menu_item": menu_item,
        "requested": servings,
        "feasible": max_servings >= servings,
        "max_servings": max_servings,
        "limiting_ingredient": limiting,
        "shortfalls": shortfalls,
    }


def get_supplier_options(conn, ingredient_names: list[str]) -> list[dict]:
    qmarks = ",".join("?" for _ in ingredient_names)
    rows = conn.execute(
        f"""SELECT s.id AS supplier_id, s.name AS supplier, s.min_order_cents,
                   s.lead_time_days, i.name AS ingredient, i.on_hand, i.par_level,
                   si.pack_qty, si.pack_desc, si.price_cents
            FROM supplier_items si
            JOIN suppliers s ON s.id = si.supplier_id
            JOIN ingredients i ON i.id = si.ingredient_id
            WHERE lower(i.name) IN ({qmarks})
            ORDER BY i.name, si.price_cents""",
        [n.lower() for n in ingredient_names],
    ).fetchall()
    return [dict(r) for r in rows]


def was_applied(conn, order_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM applied_orders WHERE order_id = ?", (order_id,)
    ).fetchone() is not None


# --- writes (called only from deterministic pipeline code, never from LLM tools)

def record_order(conn, order_id: str, ts: str, items: list[dict]) -> None:
    conn.execute("INSERT INTO orders (id, ts) VALUES (?, ?)", (order_id, ts))
    for item in items:
        conn.execute(
            """INSERT INTO order_items (order_id, menu_item_id, qty)
               SELECT ?, id, ? FROM menu_items WHERE name = ?""",
            (order_id, item["qty"], item["menu_item"]),
        )
    conn.execute(
        "INSERT INTO applied_orders (order_id, applied_ts) VALUES (?, datetime('now'))",
        (order_id,),
    )


def apply_depletion(conn, ingredient_id: int, qty: int, ts: str, ref: str) -> None:
    conn.execute(
        "UPDATE ingredients SET on_hand = on_hand - ? WHERE id = ?",
        (qty, ingredient_id),
    )
    conn.execute(
        """INSERT INTO stock_movements (ts, ingredient_id, delta, reason, ref)
           VALUES (?, ?, ?, 'order', ?)""",
        (ts, ingredient_id, -qty, ref),
    )


def create_po(conn, supplier_id: int, total_cents: int, lines_json: str) -> int:
    cur = conn.execute(
        """INSERT INTO purchase_orders (created_ts, supplier_id, status, total_cents, lines_json)
           VALUES (datetime('now'), ?, 'draft', ?, ?)""",
        (supplier_id, total_cents, lines_json),
    )
    return cur.lastrowid


def mark_po_sent(conn, po_id: int) -> None:
    conn.execute("UPDATE purchase_orders SET status = 'sent' WHERE id = ?", (po_id,))


# --- seeding (used by scripts/seed_db.py and the test fixtures) ---------------

def seed_from_csvs(conn, seed_dir: str | Path) -> None:
    import csv

    seed_dir = Path(seed_dir)

    def rows(name: str):
        with open(seed_dir / name, newline="") as f:
            yield from csv.DictReader(f)

    for r in rows("ingredients.csv"):
        conn.execute(
            """INSERT INTO ingredients (name, unit, on_hand, par_level, reorder_threshold, storage)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (r["name"], r["unit"], int(r["on_hand"]), int(r["par_level"]),
             int(r["reorder_threshold"]), r["storage"]),
        )
    for r in rows("menu_items.csv"):
        conn.execute(
            "INSERT INTO menu_items (name, price_cents) VALUES (?, ?)",
            (r["name"], int(r["price_cents"])),
        )
    for r in rows("recipes.csv"):
        conn.execute(
            """INSERT INTO recipes (menu_item_id, ingredient_id, qty_per_item)
               SELECT m.id, i.id, ? FROM menu_items m, ingredients i
               WHERE m.name = ? AND i.name = ?""",
            (int(r["qty_per_item"]), r["menu_item"], r["ingredient"]),
        )
    for r in rows("suppliers.csv"):
        conn.execute(
            """INSERT INTO suppliers (name, contact, min_order_cents, lead_time_days)
               VALUES (?, ?, ?, ?)""",
            (r["name"], r["contact"], int(r["min_order_cents"]), int(r["lead_time_days"])),
        )
    for r in rows("supplier_items.csv"):
        conn.execute(
            """INSERT INTO supplier_items (supplier_id, ingredient_id, pack_qty, pack_desc, price_cents)
               SELECT s.id, i.id, ?, ?, ? FROM suppliers s, ingredients i
               WHERE s.name = ? AND i.name = ?""",
            (int(r["pack_qty"]), r["pack_desc"], int(r["price_cents"]),
             r["supplier"], r["ingredient"]),
        )
    conn.commit()
