"""Data access helpers for dashboard reads and metrics."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

from ui.models import CatalogRow, LogEvent

AGENT_NAMES = (
    "InventoryManager",
    "WebScraper",
    "PriceStrategist",
    "CatalogUpdater",
)

_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>[^|]+)\s+\|\s+"
    r"(?P<level>[^|]+)\s+\|\s+"
    r"\[(?P<event_type>[^\]]+)\]\s+"
    r"agent=(?P<agent>\S+)\s+"
    r"(?P<details>.*)$"
)
_DURATION_PATTERN = re.compile(r"duration=(?P<seconds>\d+(?:\.\d+)?)s")


def load_catalog_rows(db_path: str) -> list[CatalogRow]:
    """Read latest catalog rows by product from SQLite."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    query = """
        SELECT product_name, cost, competitor_price, suggested_price,
               margin_percent, pricing_strategy, saved_at
        FROM catalog
        ORDER BY saved_at DESC
    """
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='catalog'"
        )
        if cur.fetchone() is None:
            return []
        cur.execute(query)
        rows = cur.fetchall()

    latest_by_product: dict[str, CatalogRow] = {}
    for row in rows:
        model = CatalogRow(
            product_name=row[0],
            cost=row[1],
            competitor_price=row[2],
            suggested_price=row[3],
            margin_percent=row[4],
            pricing_strategy=row[5] or "",
            saved_at=row[6],
        )
        if model.product_name not in latest_by_product:
            latest_by_product[model.product_name] = model

    return sorted(latest_by_product.values(), key=lambda item: item.product_name)


def parse_log_line(line: str) -> LogEvent | None:
    """Parse one ``trace.log`` line into a typed event."""
    match = _LOG_PATTERN.match(line.strip())
    if not match:
        return None
    parsed = match.groupdict()
    return LogEvent(
        timestamp=parsed["timestamp"].strip(),
        level=parsed["level"].strip(),
        event_type=parsed["event_type"].strip(),
        agent=parsed["agent"].strip(),
        details=parsed["details"].strip(),
    )


def load_log_events(log_path: str) -> list[LogEvent]:
    """Read and parse all valid log events from ``trace.log``."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    events: list[LogEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = parse_log_line(line)
        if event is not None:
            events.append(event)
    return events


def compute_agent_status(events: list[LogEvent]) -> dict[str, str]:
    """Infer per-agent status from parsed logs."""
    status = {name: "not_started" for name in AGENT_NAMES}
    for event in events:
        agent_key = event.agent.strip()
        if agent_key not in status:
            continue
        if event.event_type == "AGENT_START":
            status[agent_key] = "running"
        elif event.event_type == "AGENT_END":
            status[agent_key] = "completed"
        elif event.event_type in {"AGENT_WARN", "TOOL_ERROR"}:
            status[agent_key] = "warning"
    return status


def compute_metrics(rows: list[CatalogRow], events: list[LogEvent]) -> dict[str, float]:
    """Build high-level metrics displayed in cards."""
    strategy_counts = Counter(row.pricing_strategy for row in rows if row.pricing_strategy)
    fallback_count = sum(
        1
        for event in events
        if event.event_type in {"TOOL_RESULT", "TOOL_ERROR"}
        and "fallback" in event.details.lower()
    )
    avg_margin = round(sum(row.margin_percent for row in rows) / len(rows), 2) if rows else 0.0
    return {
        "rows_processed": float(len(rows)),
        "avg_margin_percent": avg_margin,
        "fallback_count": float(fallback_count),
        "strategy_count": float(len(strategy_counts)),
    }


def compute_agent_durations(events: list[LogEvent]) -> dict[str, float | None]:
    """Extract latest AGENT_END duration in seconds for each agent."""
    durations: dict[str, float | None] = {name: None for name in AGENT_NAMES}
    for event in events:
        if event.event_type != "AGENT_END" or event.agent not in durations:
            continue
        match = _DURATION_PATTERN.search(event.details)
        if match:
            durations[event.agent] = float(match.group("seconds"))
    return durations


def strategy_breakdown(rows: list[CatalogRow]) -> dict[str, int]:
    """Return count by pricing strategy."""
    counts = Counter(row.pricing_strategy or "unknown" for row in rows)
    return dict(sorted(counts.items(), key=lambda item: item[0]))
