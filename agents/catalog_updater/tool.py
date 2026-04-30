import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.logger import log_event


class SaveDbInput(BaseModel):
    entries: str = Field(..., description="JSON-encoded list of pricing dicts")
    db_path: str = Field(default="catalog.db", description="Path to the SQLite database file")


class SaveDbOutput(BaseModel):
    rows_saved: int
    db_path: str
    status: str


@tool
def save_to_local_db(entries: str, db_path: str = "catalog.db") -> str:
    """Persist a batch of pricing results to a local SQLite database.

    Parses the JSON-encoded ``entries`` list, creates the ``catalog``
    table if it does not already exist, and inserts every entry in a
    single atomic transaction.  The transaction is rolled back entirely
    if any insert fails, ensuring the database is never left in a partial
    state.  Uses parameterised queries throughout, providing full
    protection against SQL injection.

    Args:
        entries: A JSON-encoded list of pricing dicts.  Each dict must
            contain the keys ``product_name``, ``cost``,
            ``competitor_price``, ``suggested_price``, ``margin_percent``,
            and ``pricing_strategy``.
        db_path: File path for the SQLite database.  The file and the
            ``catalog`` table are created automatically if absent.
            Defaults to ``"catalog.db"``.

    Returns:
        JSON string conforming to ``SaveDbOutput``:
        ``{"rows_saved": <int>, "db_path": "<path>",
           "status": "success" | "error: <reason>"}``

    Raises:
        Does not raise — all exceptions are caught, the transaction is
        rolled back, and the error is returned inside ``status``.

    Example:
        >>> import json
        >>> entry = {"product_name": "Laptop", "cost": 750.0,
        ...          "competitor_price": 999.99, "suggested_price": 900.0,
        ...          "margin_percent": 16.67, "pricing_strategy": "capped_at_competitor"}
        >>> raw = save_to_local_db.invoke(
        ...     {"entries": json.dumps([entry]), "db_path": "catalog.db"})
        >>> data = json.loads(raw)
        >>> data["rows_saved"]
        1
        >>> data["status"]
        'success'
    """
    log_event("TOOL_CALL", "CatalogUpdater",
              f"save_to_local_db(db_path='{db_path}', entries_len={len(entries)})")
    conn: sqlite3.Connection | None = None
    try:
        entries_list: List[Dict[str, Any]] = json.loads(entries)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS catalog (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name     TEXT    NOT NULL,
                cost             REAL    NOT NULL,
                competitor_price REAL    NOT NULL,
                suggested_price  REAL    NOT NULL,
                margin_percent   REAL    NOT NULL,
                pricing_strategy TEXT,
                saved_at         TEXT    NOT NULL
            )
        """)

        saved_at = datetime.now().isoformat()
        rows_saved = 0

        conn.execute("BEGIN")
        for entry in entries_list:
            cur.execute(
                """
                INSERT INTO catalog
                  (product_name, cost, competitor_price, suggested_price,
                   margin_percent, pricing_strategy, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.get("product_name", ""),
                    float(entry.get("cost", 0)),
                    float(entry.get("competitor_price", 0)),
                    float(entry.get("suggested_price", 0)),
                    float(entry.get("margin_percent", 0)),
                    entry.get("pricing_strategy", ""),
                    saved_at,
                ),
            )
            rows_saved += 1

        conn.commit()
        conn.close()

        out = SaveDbOutput(rows_saved=rows_saved, db_path=db_path, status="success")
        log_event("TOOL_RESULT", "CatalogUpdater", f"Saved {rows_saved} rows → {db_path}")
        return out.model_dump_json()

    except Exception as exc:
        if conn:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        log_event("TOOL_ERROR", "CatalogUpdater", f"save_to_local_db failed: {exc}")
        return SaveDbOutput(rows_saved=0, db_path=db_path, status=f"error: {exc}").model_dump_json()
