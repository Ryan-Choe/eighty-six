"""Pins for the four P1 fixes from the Day-2 review, plus the later review
fixes on the same ingestion seam. Each test names the bug it guards against;
if one fails, that bug is back."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from eightysix import db
from eightysix.ingest import MalformedOrder, apply_orders, ingest_file, minimize_order

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    db.seed_from_csvs(c, SEED_DIR)
    return c


def order(oid, items, ts="2026-08-14T18:00:00"):
    return {"order_id": oid, "timestamp": ts, "items": items}


# --- P1: usage denominator was the requested window, reporting 7x-low burn ---

def test_usage_divides_by_days_with_data_not_window(conn):
    # one service day of history: 4 margheritas = 600 g of fresh mozzarella
    today = datetime.now().strftime("%Y-%m-%dT18:00:00")
    apply_orders(conn, [order("U-1", [{"menu_item": "Margherita", "qty": 4}], ts=today)])
    u = db.get_usage(conn, "fresh mozzarella", 7)
    assert u["data_days"] == 1
    assert u["window_days"] == 7
    assert u["avg_daily"] == 600.0          # per day WITH data — not 600/7
    assert u["days_of_cover"] == round((2000 - 600) / 600, 1)   # ~2.3, not ~16


def test_usage_two_days_of_data(conn):
    d1 = datetime.now().strftime("%Y-%m-%dT18:00:00")
    d2 = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT18:00:00")
    apply_orders(conn, [
        order("U-2", [{"menu_item": "Margherita", "qty": 2}], ts=d1),
        order("U-3", [{"menu_item": "Margherita", "qty": 4}], ts=d2),
    ])
    u = db.get_usage(conn, "fresh mozzarella", 7)
    assert u["data_days"] == 2
    assert u["avg_daily"] == 450.0          # 900 g over 2 days with data


def test_usage_with_no_history_reports_none_not_infinity(conn):
    u = db.get_usage(conn, "fresh mozzarella", 7)
    assert u["data_days"] == 0
    assert u["avg_daily"] == 0
    assert u["days_of_cover"] is None


# --- P1: name lookups were exact-match dead ends ---

def test_resolve_unique_substring(conn):
    assert db.resolve_name(conn, "ingredients", "sauce") == ("ok", "tomato sauce")
    assert db.resolve_name(conn, "menu_items", "pepperoni pizza") == ("ok", "Pepperoni")


def test_resolve_ambiguous_lists_candidates(conn):
    status, names = db.resolve_name(conn, "ingredients", "mozzarella")
    assert status == "ambiguous"
    assert names == ["fresh mozzarella", "low-moisture mozzarella"]


def test_resolve_unknown_returns_full_catalog(conn):
    status, names = db.resolve_name(conn, "ingredients", "truffle oil")
    assert status == "unknown"
    assert len(names) == 16


def test_resolve_rejects_arbitrary_table():
    with pytest.raises(ValueError):
        db.resolve_name(None, "sqlite_master", "x")


# --- P1: order quantities were unvalidated external input ---

def test_negative_qty_rejected_and_stock_untouched(conn):
    result = apply_orders(conn, [order("Q-1", [{"menu_item": "Margherita", "qty": -3}])])
    assert result["applied"] == 0
    assert result["errors"][0]["invalid_qty"] == [{"menu_item": "Margherita", "qty": -3}]
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000


def test_zero_qty_rejected_no_ledger_noise(conn):
    result = apply_orders(conn, [order("Q-2", [{"menu_item": "Margherita", "qty": 0}])])
    assert result["applied"] == 0
    assert conn.execute("SELECT COUNT(*) FROM stock_movements").fetchone()[0] == 0


def test_rejected_order_can_be_fixed_and_replayed(conn):
    # rejection must NOT mark the order as applied, or the corrected file
    # would be skipped as a duplicate
    apply_orders(conn, [order("Q-3", [{"menu_item": "Margherita", "qty": -1}])])
    result = apply_orders(conn, [order("Q-3", [{"menu_item": "Margherita", "qty": 1}])])
    assert result["applied"] == 1
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000 - 150


# --- review fix: fractional quantities were truncated, not rejected ---

def test_fractional_qty_is_malformed_not_truncated():
    # int() would quietly turn 2.9 into 2 -- the reject-don't-clamp rule
    # has to hold at minimize time too, before the apply gate ever sees it
    with pytest.raises(MalformedOrder):
        minimize_order(order("F-1", [{"menu_item": "Margherita", "qty": 2.9}]))


def test_boolean_qty_is_malformed():
    with pytest.raises(MalformedOrder):
        minimize_order(order("F-2", [{"menu_item": "Margherita", "qty": True}]))


def test_infinite_qty_is_malformed_not_fatal():
    # python's json parser accepts Infinity; without OverflowError in the
    # except tuple this would crash the whole batch instead of one order
    with pytest.raises(MalformedOrder):
        minimize_order(order("F-3", [{"menu_item": "Margherita", "qty": float("inf")}]))


def test_whole_number_qtys_still_normalize():
    m = minimize_order(order("F-4", [{"menu_item": "Margherita", "qty": 3.0},
                                     {"menu_item": "Margherita", "qty": "2"}]))
    assert [i["qty"] for i in m["items"]] == [3, 2]


def test_garbage_order_entry_is_reported_not_fatal(conn, tmp_path):
    # a non-dict entry in the orders list must cost one error line, not the
    # file -- the valid order behind it still applies
    import json
    batch = {"orders": ["garbage", order("G-1", [{"menu_item": "Margherita", "qty": 1}])]}
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch))
    summary = ingest_file(conn, path)
    assert summary["applied"] == 1
    assert summary["errors"][0]["order_id"] == "#0"


# --- P1 adjacent: oversell may go negative, but feasibility clamps at zero ---

def test_oversell_is_honest_but_feasibility_clamps(conn):
    apply_orders(conn, [order("O-1", [{"menu_item": "Margherita", "qty": 100}])])
    assert db.get_stock(conn, "fresh mozzarella")["on_hand"] == 2000 - 15000  # ledger tells the truth
    f = db.check_feasibility(conn, "Margherita", 1)
    assert f["max_servings"] == 0           # not -87
    assert f["feasible"] is False
