"""Deterministic core tests. No API keys, no network — pure math against the
real seed CSVs, so a seed-data change that breaks the demo story fails here."""

from pathlib import Path

import pytest

from eightysix import db
from eightysix.ingest import apply_orders, minimize_order

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    db.seed_from_csvs(c, SEED_DIR)
    return c


def order(oid, items):
    return {"order_id": oid, "timestamp": "2026-08-14T18:00:00", "items": items}


def test_single_margherita_depletes_exact_amounts(conn):
    apply_orders(conn, [order("T-1", [{"menu_item": "Margherita", "qty": 1}])])
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000 - 150
    assert db.get_stock(conn, "00 flour")["on_hand"] == 25000 - 250
    assert db.get_stock(conn, "tomato sauce")["on_hand"] == 12000 - 100
    assert db.get_stock(conn, "basil")["on_hand"] == 200 - 5


def test_multi_item_order_multiplies_by_qty(conn):
    apply_orders(conn, [order("T-2", [
        {"menu_item": "Margherita", "qty": 2},
        {"menu_item": "Pepperoni", "qty": 2},
    ])])
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000 - 2 * 150
    assert db.get_stock(conn, "pepperoni")["on_hand"] == 1400 - 2 * 80
    assert db.get_stock(conn, "00 flour")["on_hand"] == 25000 - 4 * 250


def test_replay_is_a_noop(conn):
    batch = [order("T-3", [{"menu_item": "Calzone", "qty": 1}])]
    first = apply_orders(conn, batch)
    after_first = db.get_stock(conn, "pepperoni")["on_hand"]
    second = apply_orders(conn, batch)
    assert first["applied"] == 1
    assert second["applied"] == 0
    assert second["skipped_duplicates"] == 1
    assert db.get_stock(conn, "pepperoni")["on_hand"] == after_first


def test_unknown_menu_item_rejects_whole_order_and_touches_nothing(conn):
    result = apply_orders(conn, [order("T-4", [
        {"menu_item": "Margherita", "qty": 1},
        {"menu_item": "Hawaiian", "qty": 1},
    ])])
    assert result["applied"] == 0
    assert result["errors"] == [{"order_id": "T-4", "unknown_items": ["Hawaiian"]}]
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000


def test_empty_batch_changes_nothing_and_flags_nothing(conn):
    result = apply_orders(conn, [])
    assert result["applied"] == 0
    assert result["low_stock"] == []


def test_friday_rush_flags_exactly_mozzarella_and_pepperoni(conn):
    # the committed demo batch must produce exactly this outcome — the
    # recording depends on it, so this test pins the demo story to the data
    import json
    raw = json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "pos_orders" / "orders_friday.json").read_text()
    )
    orders = [minimize_order(o) for o in raw["orders"]]
    result = apply_orders(conn, orders)
    assert result["applied"] == 20
    assert result["errors"] == []
    flagged = {i["name"] for i in result["low_stock"]}
    assert flagged == {"fresh mozzarella", "pepperoni"}
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000 - 840   # 4 margherita + 2 white pie
    assert db.get_stock(conn, "pepperoni")["on_hand"] == 1400 - 560          # 6 pepperoni + 2 calzone


def test_minimize_strips_customer_fields():
    raw = {
        "order_id": "F-1", "timestamp": "2026-08-14T18:00:00",
        "customer_name": "Sarah Kim", "phone": "555-867-5309", "notes": "ring side door",
        "items": [{"menu_item": "Margherita", "qty": 1}],
    }
    clean = minimize_order(raw)
    assert set(clean.keys()) == {"order_id", "timestamp", "items"}
    assert "555-867-5309" not in str(clean)


def test_feasibility_counts_whole_servings_and_names_the_limiter(conn):
    apply_orders(conn, [order("T-5", [{"menu_item": "Margherita", "qty": 5}])])
    # 2000 - 5*150 = 1250 g mozzarella left, at 150 g per pie -> 8 more pies
    result = db.check_feasibility(conn, "Margherita", 12)
    assert result["feasible"] is False
    assert result["max_servings"] == 8
    assert result["limiting_ingredient"] == "fresh mozzarella"
    assert result["shortfalls"][0]["short_by"] == 12 * 150 - 1250


def test_feasibility_on_unknown_menu_item_returns_none(conn):
    assert db.check_feasibility(conn, "Hawaiian", 1) is None
