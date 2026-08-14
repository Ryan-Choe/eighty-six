"""Drop and rebuild pizzeria.db from data/seed/. Doubles as the demo reset button."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eightysix import db

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

if __name__ == "__main__":
    db.DB_PATH.unlink(missing_ok=True)
    conn = db.connect()
    db.init_schema(conn)
    db.seed_from_csvs(conn, SEED_DIR)
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("ingredients", "menu_items", "recipes", "suppliers", "supplier_items")
    }
    print(f"seeded {db.DB_PATH.name}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
