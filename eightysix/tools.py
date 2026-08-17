"""The agent's tools. All read-only, all thin wrappers over db.py.

Nothing here writes. The model can look at the pantry; it cannot change it.
Writes happen in the ingestion pipeline and (from Day 3) behind the approval
interrupt, both of which are plain Python the model never calls directly.

Name lookups resolve deterministically (exact -> unique substring), so
"mozzarella" never forces the model to guess between the two tracked
varieties -- the tool reports both and the model asks the owner.
"""

from contextlib import contextmanager

from langchain_core.tools import tool

from eightysix import db


@contextmanager
def _conn():
    c = db.connect()
    try:
        yield c
    finally:
        c.close()


def _resolved(c, table: str, name: str):
    """Returns (canonical_name, None) or (None, guidance_string)."""
    status, payload = db.resolve_name(c, table, name)
    if status == "ok":
        return payload, None
    kind = "ingredient" if table == "ingredients" else "menu item"
    if status == "ambiguous":
        return None, (
            f"Several tracked {kind}s match {name!r}: {', '.join(payload)}. "
            "Ask the owner which one they mean."
        )
    return None, (
        f"No tracked {kind} matches {name!r}. Tracked {kind}s: {', '.join(payload)}."
    )


@tool
def get_stock(ingredient: str) -> dict | str:
    """Current stock for one ingredient, with its par level and reorder threshold.

    Partial names resolve automatically ("sauce" finds "tomato sauce"). If
    several ingredients match, the tool lists them instead of picking one.
    """
    with _conn() as c:
        name, guidance = _resolved(c, "ingredients", ingredient)
        return guidance if guidance else db.get_stock(c, name)


@tool
def get_low_stock() -> list[dict] | str:
    """Every ingredient at or below its reorder threshold, most urgent first.

    This is the "86 board" -- what is about to run out.
    """
    with _conn() as c:
        rows = db.get_low_stock(c)
    return rows or "Nothing is below its reorder threshold right now."


@tool
def list_ingredients() -> list[dict]:
    """All tracked ingredients: name, unit, current stock, storage location,
    and whether each is flagged for reorder. Use this to see what exists
    before asking about something specific, or for "what's in the walk-in?"
    """
    with _conn() as c:
        return db.get_all_stock(c)


@tool
def check_feasibility(menu_item: str, servings: int) -> dict | str:
    """Whether we can make N of a menu item right now, and what runs out first.

    Returns max_servings, the limiting ingredient, and any shortfalls.
    Partial menu-item names resolve automatically ("pepperoni pizza" finds
    "Pepperoni").
    """
    with _conn() as c:
        name, guidance = _resolved(c, "menu_items", menu_item)
        return guidance if guidance else db.check_feasibility(c, name, servings)


@tool
def get_usage(ingredient: str, days: int = 7) -> dict | str:
    """Consumption history for an ingredient over the last N days.

    avg_daily is per day WITH recorded sales; data_days says how many such
    days back the number. When data_days is small, say so in your answer --
    one busy Friday is thin evidence for a weekly rate.
    """
    with _conn() as c:
        name, guidance = _resolved(c, "ingredients", ingredient)
        return guidance if guidance else db.get_usage(c, name, days)


INVENTORY_TOOLS = [get_stock, get_low_stock, list_ingredients, check_feasibility, get_usage]
