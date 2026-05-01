"""Streamlit dashboard for swarm monitoring and reporting."""

from __future__ import annotations

from datetime import datetime
import subprocess
import sys
from pathlib import Path
import csv
import io

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from ui.data_access import (
    AGENT_NAMES,
    compute_agent_durations,
    compute_agent_status,
    compute_metrics,
    load_catalog_rows,
    load_log_events,
    strategy_breakdown,
)
from ui.models import CatalogRow, LogEvent

DEFAULT_DB_PATH = str(ROOT / "catalog.db")
DEFAULT_LOG_PATH = str(ROOT / "trace.log")


def run_pipeline() -> tuple[bool, str, str]:
    """Execute ``graph.py`` and return status + message + combined output."""
    proc = subprocess.run(
        [sys.executable, "graph.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return (
            False,
            proc.stderr.strip() or "Pipeline failed without stderr output.",
            f"{proc.stdout}\n{proc.stderr}".strip(),
        )
    return True, "Pipeline completed successfully.", f"{proc.stdout}\n{proc.stderr}".strip()


def reset_demo_artifacts(db_path: str, log_path: str) -> None:
    """Delete generated demo artifacts when they exist."""
    for path in (Path(db_path), Path(log_path)):
        if path.exists():
            path.unlink()


def load_dashboard_data(db_path: str, log_path: str) -> tuple[list[CatalogRow], list[LogEvent]]:
    """Load dashboard data with graceful fallback for missing files."""
    rows: list[CatalogRow] = []
    events: list[LogEvent] = []
    try:
        rows = load_catalog_rows(db_path)
    except FileNotFoundError:
        rows = []
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Could not read database: {exc}")

    try:
        events = load_log_events(log_path)
    except FileNotFoundError:
        events = []
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Could not read trace log: {exc}")
    return rows, events


def _status_badge(status: str) -> str:
    mapping = {
        "completed": "SUCCESS",
        "running": "RUNNING",
        "warning": "WARNING",
        "not_started": "PENDING",
    }
    return mapping.get(status, "UNKNOWN")


def _rows_for_display(rows: list[CatalogRow]) -> list[dict[str, float | str]]:
    """Create display rows with additional price delta fields."""
    display_rows: list[dict[str, float | str]] = []
    for row in rows:
        competitor = row.competitor_price
        delta_value = round(row.suggested_price - competitor, 2)
        delta_percent = round((delta_value / competitor) * 100, 2) if competitor else 0.0
        display_rows.append(
            {
                "product_name": row.product_name,
                "cost": row.cost,
                "competitor_price": competitor,
                "suggested_price": row.suggested_price,
                "price_delta": delta_value,
                "price_delta_percent": delta_percent,
                "margin_percent": row.margin_percent,
                "pricing_strategy": row.pricing_strategy,
                "saved_at": row.saved_at,
            }
        )
    return display_rows


def _rows_to_csv(rows: list[dict[str, float | str]]) -> str:
    """Serialize table rows into CSV text."""
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    """Render dashboard UI."""
    st.set_page_config(page_title="Swarm Dashboard", layout="wide")
    st.title("E-Commerce Competitor Intelligence Dashboard")
    st.caption("Read-only analytics over catalog and trace outputs.")

    if "last_refresh_at" not in st.session_state:
        st.session_state["last_refresh_at"] = "Not yet refreshed"
    if "last_run_output" not in st.session_state:
        st.session_state["last_run_output"] = ""

    with st.sidebar:
        st.header("Controls")
        db_path = st.text_input("Catalog DB path", value=DEFAULT_DB_PATH)
        log_path = st.text_input("Trace log path", value=DEFAULT_LOG_PATH)
        run_now = st.button("Run Pipeline", type="primary", use_container_width=True)
        reset_and_run = st.button("Demo Reset + Run", use_container_width=True)
        refresh = st.button("Refresh Data", use_container_width=True)
        st.caption(f"Last refresh: {st.session_state['last_refresh_at']}")

    if reset_and_run:
        reset_demo_artifacts(db_path, log_path)
        with st.spinner("Running pipeline with clean artifacts..."):
            ok, message, output = run_pipeline()
            st.session_state["last_run_output"] = output
        if ok:
            st.success(message)
        else:
            st.error(message)
    elif run_now:
        with st.spinner("Running pipeline..."):
            ok, message, output = run_pipeline()
            st.session_state["last_run_output"] = output
        if ok:
            st.success(message)
        else:
            st.error(message)
    elif refresh:
        st.toast("Dashboard refreshed")

    st.session_state["last_refresh_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows, events = load_dashboard_data(db_path, log_path)
    status = compute_agent_status(events)
    durations = compute_agent_durations(events)
    metrics = compute_metrics(rows, events)

    tabs = st.tabs(["Overview", "Results", "Audit"])

    with tabs[0]:
        st.subheader("Pipeline Status")
        status_cols = st.columns(len(AGENT_NAMES))
        for index, agent in enumerate(AGENT_NAMES):
            raw_status = status.get(agent, "not_started")
            agent_status = _status_badge(raw_status)
            duration = durations.get(agent)
            duration_label = f"{duration:.3f}s" if duration is not None else "N/A"
            status_cols[index].metric(agent, agent_status, delta=duration_label)

        st.subheader("Key Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Rows Processed", int(metrics["rows_processed"]))
        metric_cols[1].metric("Average Margin %", f"{metrics['avg_margin_percent']:.2f}")
        metric_cols[2].metric("Fallback Count", int(metrics["fallback_count"]))
        metric_cols[3].metric("Strategy Types", int(metrics["strategy_count"]))

        if st.session_state["last_run_output"]:
            with st.expander("Last pipeline console output", expanded=False):
                st.code(st.session_state["last_run_output"], language="text")

    with tabs[1]:
        st.subheader("Results")
        if not rows:
            st.info("No pricing rows found. Run the pipeline to populate `catalog.db`.")
        else:
            all_strategies = sorted({row.pricing_strategy for row in rows if row.pricing_strategy})
            control_cols = st.columns([2, 2, 1])
            strategy_filter = control_cols[0].multiselect(
                "Filter by strategy", all_strategies, default=all_strategies
            )
            product_search = control_cols[1].text_input("Search product name", value="")
            reverse = control_cols[2].toggle("Descending", value=False)
            sort_by = st.selectbox(
                "Sort rows by",
                options=[
                    "product_name",
                    "cost",
                    "competitor_price",
                    "suggested_price",
                    "price_delta",
                    "price_delta_percent",
                    "margin_percent",
                ],
                index=0,
            )

            filtered = [row for row in rows if not strategy_filter or row.pricing_strategy in strategy_filter]
            if product_search.strip():
                term = product_search.strip().lower()
                filtered = [row for row in filtered if term in row.product_name.lower()]

            display_rows = _rows_for_display(filtered)
            display_rows.sort(key=lambda item: item[sort_by], reverse=reverse)
            df = pd.DataFrame(display_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv_data = _rows_to_csv(display_rows)
            st.download_button(
                label="Download Filtered Results (CSV)",
                data=csv_data,
                file_name="filtered_pricing_results.csv",
                mime="text/csv",
                disabled=not bool(display_rows),
                use_container_width=True,
            )

            st.subheader("Pricing Strategy Breakdown")
            breakdown = strategy_breakdown(filtered)
            if breakdown:
                st.bar_chart(breakdown)

    with tabs[2]:
        st.subheader("Audit View")
        if not events:
            st.warning("No trace events found. Run pipeline to generate `trace.log`.")
        else:
            left, right = st.columns(2)
            selected_agent = left.selectbox("Agent filter", options=["All"] + list(AGENT_NAMES), index=0)
            event_types = sorted({event.event_type for event in events})
            selected_types = right.multiselect("Event type filter", event_types, default=event_types)
            show_count = st.slider("Recent events", min_value=10, max_value=200, value=40, step=10)
            filtered_events = [
                event
                for event in events
                if (selected_agent == "All" or event.agent == selected_agent)
                and event.event_type in selected_types
            ]
            recent_events = filtered_events[-show_count:]
            st.dataframe([event.model_dump() for event in recent_events], use_container_width=True)


if __name__ == "__main__":
    main()
