"""Tests for dashboard data parsing and metrics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ui.data_access import (
    compute_agent_durations,
    compute_agent_status,
    compute_metrics,
    load_catalog_rows,
    load_log_events,
    parse_log_line,
    strategy_breakdown,
)


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                cost REAL NOT NULL,
                competitor_price REAL NOT NULL,
                suggested_price REAL NOT NULL,
                margin_percent REAL NOT NULL,
                pricing_strategy TEXT,
                saved_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            INSERT INTO catalog
            (product_name, cost, competitor_price, suggested_price, margin_percent, pricing_strategy, saved_at)
            VALUES
            ('Laptop', 700, 1000, 840, 16.67, 'below_competitor_with_margin', '2026-05-01T10:00:00'),
            ('Laptop', 700, 950, 830, 15.66, 'standard_markup', '2026-05-01T11:00:00'),
            ('Mouse', 10, 19, 12, 16.67, 'below_competitor_with_margin', '2026-05-01T11:05:00')
            """
        )
        conn.commit()


def test_load_catalog_rows_returns_latest_per_product(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    _create_db(db_path)

    rows = load_catalog_rows(str(db_path))

    assert len(rows) == 2
    laptop = next(item for item in rows if item.product_name == "Laptop")
    assert laptop.competitor_price == 950
    assert laptop.pricing_strategy == "standard_markup"


def test_log_parsing_status_metrics_and_breakdown(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-05-01 10:00:00 | INFO         | [AGENT_START     ] agent=InventoryManager     Started inventory",
                "2026-05-01 10:00:01 | INFO         | [AGENT_END       ] agent=InventoryManager     Completed inventory, duration=1.230s",
                "2026-05-01 10:00:02 | INFO         | [TOOL_RESULT     ] agent=WebScraper           fallback_used: unknown product",
            ]
        ),
        encoding="utf-8",
    )

    events = load_log_events(str(log_path))
    assert len(events) == 3
    assert parse_log_line("invalid line") is None

    statuses = compute_agent_status(events)
    assert statuses["InventoryManager"] == "completed"
    assert statuses["WebScraper"] == "not_started"
    durations = compute_agent_durations(events)
    assert durations["InventoryManager"] == 1.23
    assert durations["WebScraper"] is None

    db_path = tmp_path / "catalog.db"
    _create_db(db_path)
    rows = load_catalog_rows(str(db_path))

    metrics = compute_metrics(rows, events)
    assert metrics["rows_processed"] == 2.0
    assert metrics["fallback_count"] == 1.0
    assert metrics["avg_margin_percent"] > 0

    breakdown = strategy_breakdown(rows)
    assert "standard_markup" in breakdown
