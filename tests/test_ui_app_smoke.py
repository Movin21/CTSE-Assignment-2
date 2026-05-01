"""Smoke tests for dashboard orchestration helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ui.app import _rows_for_display, _rows_to_csv, load_dashboard_data, reset_demo_artifacts


def _seed_catalog(path: Path) -> None:
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
            VALUES ('Keyboard', 40, 70, 48, 16.67, 'below_competitor_with_margin', '2026-05-01T09:00:00')
            """
        )
        conn.commit()


def test_load_dashboard_data_and_reset(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    log_path = tmp_path / "trace.log"
    _seed_catalog(db_path)
    log_path.write_text(
        "2026-05-01 10:00:01 | INFO         | [AGENT_END       ] agent=InventoryManager     Completed inventory",
        encoding="utf-8",
    )

    rows, events = load_dashboard_data(str(db_path), str(log_path))
    assert len(rows) == 1
    assert len(events) == 1

    reset_demo_artifacts(str(db_path), str(log_path))
    assert not db_path.exists()
    assert not log_path.exists()


def test_display_rows_include_deltas_and_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    log_path = tmp_path / "trace.log"
    _seed_catalog(db_path)
    log_path.write_text("", encoding="utf-8")

    rows, _ = load_dashboard_data(str(db_path), str(log_path))
    display_rows = _rows_for_display(rows)
    assert len(display_rows) == 1
    assert display_rows[0]["price_delta"] == -22.0
    assert display_rows[0]["price_delta_percent"] == -31.43

    csv_text = _rows_to_csv(display_rows)
    assert "price_delta" in csv_text
    assert "Keyboard" in csv_text
