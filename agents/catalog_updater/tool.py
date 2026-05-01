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
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_unchanged: int = 0
    db_path: str
    status: str


@tool
def save_to_local_db(entries: str, db_path: str = "catalog.db") -> str:
    """Persist a batch of pricing results to a local SQLite database.

    Parses the JSON-encoded ``entries`` list, creates the ``catalog``
    table if it does not already exist, and then performs change-aware
    writes: new products are inserted, existing products are updated only
    when one or more tracked fields changed, and unchanged products are
    skipped. The transaction is rolled back entirely if any write fails,
    ensuring the database is never left in a partial state. Uses
    parameterised queries throughout, providing full protection against
    SQL injection.

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
        ``{"rows_saved": <int>, "rows_inserted": <int>,
           "rows_updated": <int>, "rows_unchanged": <int>,
           "db_path": "<path>", "status": "success" | "error: <reason>"}``

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
        parsed_entries = json.loads(entries)
        if isinstance(parsed_entries, dict):
            parsed_entries = [parsed_entries]
        elif not isinstance(parsed_entries, list):
            raise ValueError("entries must decode to a dict or list of dicts")

        entries_list: List[Dict[str, Any]] = []
        for raw_entry in parsed_entries:
            if isinstance(raw_entry, dict):
                entries_list.append(raw_entry)
                continue
            if isinstance(raw_entry, str):
                try:
                    decoded_entry = json.loads(raw_entry)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid entry JSON string: {exc}") from exc
                if isinstance(decoded_entry, dict):
                    entries_list.append(decoded_entry)
                    continue
            raise ValueError("Each entry must be a dict or JSON string of a dict")

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
        rows_inserted = 0
        rows_updated = 0
        rows_unchanged = 0

        conn.execute("BEGIN")
        for entry in entries_list:
            product_name = entry.get("product_name", "")
            cost = float(entry.get("cost", 0))
            competitor_price = float(entry.get("competitor_price", 0))
            suggested_price = float(entry.get("suggested_price", 0))
            margin_percent = float(entry.get("margin_percent", 0))
            pricing_strategy = entry.get("pricing_strategy", "")

            cur.execute(
                """
                SELECT id, cost, competitor_price, suggested_price,
                       margin_percent, pricing_strategy
                FROM catalog
                WHERE product_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (product_name,),
            )
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO catalog
                      (product_name, cost, competitor_price, suggested_price,
                       margin_percent, pricing_strategy, saved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        product_name,
                        cost,
                        competitor_price,
                        suggested_price,
                        margin_percent,
                        pricing_strategy,
                        saved_at,
                    ),
                )
                rows_inserted += 1
                continue

            existing_id = existing[0]
            current_values = (
                float(existing[1]),
                float(existing[2]),
                float(existing[3]),
                float(existing[4]),
                existing[5] or "",
            )
            new_values = (
                cost,
                competitor_price,
                suggested_price,
                margin_percent,
                pricing_strategy,
            )
            if current_values == new_values:
                rows_unchanged += 1
                continue

            cur.execute(
                """
                UPDATE catalog
                SET cost = ?,
                    competitor_price = ?,
                    suggested_price = ?,
                    margin_percent = ?,
                    pricing_strategy = ?,
                    saved_at = ?
                WHERE id = ?
                """,
                (
                    cost,
                    competitor_price,
                    suggested_price,
                    margin_percent,
                    pricing_strategy,
                    saved_at,
                    existing_id,
                ),
            )
            rows_updated += 1

        conn.commit()
        conn.close()
        rows_saved = rows_inserted + rows_updated

        out = SaveDbOutput(
            rows_saved=rows_saved,
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
            rows_unchanged=rows_unchanged,
            db_path=db_path,
            status="success",
        )
        log_event(
            "TOOL_RESULT",
            "CatalogUpdater",
            (
                f"Saved {rows_saved} rows (inserted={rows_inserted}, "
                f"updated={rows_updated}, unchanged={rows_unchanged}) → {db_path}"
            ),
        )
        return out.model_dump_json()

    except Exception as exc:
        if conn:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        log_event("TOOL_ERROR", "CatalogUpdater", f"save_to_local_db failed: {exc}")
        return SaveDbOutput(
            rows_saved=0,
            rows_inserted=0,
            rows_updated=0,
            rows_unchanged=0,
            db_path=db_path,
            status=f"error: {exc}",
        ).model_dump_json()
