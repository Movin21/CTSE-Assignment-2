"""
graph.py — LangGraph orchestration for the E-Commerce Competitor Intelligence Swarm.

Pipeline:
  START → inventory_agent → (conditional) → scraper_agent
       → pricing_agent → (conditional) → updater_agent → END

Conditional edges short-circuit the pipeline on failure so downstream
agents are never invoked with empty state.

Run:  python graph.py
"""

from langgraph.graph import END, StateGraph

from agents import (
    run_inventory_agent,
    run_pricing_agent,
    run_scraper_agent,
    run_updater_agent,
)
from shared.logger import log_event
from state import GlobalState


# ─── Routing Functions ────────────────────────────────────────────────────────

def _route_after_inventory(state: GlobalState) -> str:
    """Skip remaining agents if inventory failed to load."""
    if not state.get("inventory"):
        log_event("SYSTEM", "Orchestrator",
                  "Inventory empty after load — short-circuiting pipeline")
        return END
    return "scraper_agent"


def _route_after_pricing(state: GlobalState) -> str:
    """Skip catalog update if no pricing decisions were produced."""
    if not state.get("pricing_logic"):
        log_event("SYSTEM", "Orchestrator",
                  "No pricing results produced — short-circuiting pipeline")
        return END
    return "updater_agent"


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph with conditional routing."""
    graph = StateGraph(GlobalState)

    graph.add_node("inventory_agent", run_inventory_agent)
    graph.add_node("scraper_agent", run_scraper_agent)
    graph.add_node("pricing_agent", run_pricing_agent)
    graph.add_node("updater_agent", run_updater_agent)

    graph.set_entry_point("inventory_agent")

    graph.add_conditional_edges(
        "inventory_agent",
        _route_after_inventory,
        {"scraper_agent": "scraper_agent", END: END},
    )

    graph.add_edge("scraper_agent", "pricing_agent")

    graph.add_conditional_edges(
        "pricing_agent",
        _route_after_pricing,
        {"updater_agent": "updater_agent", END: END},
    )

    graph.add_edge("updater_agent", END)

    return graph.compile()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run_swarm() -> GlobalState:
    """Initialise state, execute the swarm, and print the final report."""
    log_event("SYSTEM", "Orchestrator",
              "══════ E-Commerce Competitor Intelligence Swarm STARTED ══════")

    initial_state: GlobalState = {
        "inventory": [],
        "competitor_data": {},
        "pricing_logic": {},
        "logs": [],
        "messages": [],
        "current_agent": "none",
        "catalog_saved": False,
        "errors": [],
        "execution_times": {},
    }

    app = build_graph()
    final_state: GlobalState = app.invoke(initial_state)

    _print_report(final_state)

    log_event("SYSTEM", "Orchestrator", "══════ Swarm COMPLETE ══════")
    return final_state


def _print_report(state: GlobalState) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  COMPETITOR INTELLIGENCE SWARM — FINAL REPORT")
    print(sep)
    print(f"  Products in inventory  : {len(state['inventory'])}")
    print(f"  Competitor prices found: {len(state['competitor_data'])}")
    print(f"  Pricing decisions made : {len(state['pricing_logic'])}")
    print(f"  Catalog persisted      : {state['catalog_saved']}")

    if state.get("execution_times"):
        print(f"\n  Agent Execution Times:")
        for agent, secs in state["execution_times"].items():
            print(f"    {agent:<22} {secs:.3f}s")

    if state.get("errors"):
        print(f"\n  Non-Fatal Errors ({len(state['errors'])}):")
        for err in state["errors"]:
            print(f"    ⚠  {err}")

    if state["pricing_logic"]:
        print(f"\n  {'Product':<25} {'Cost':>8} {'Competitor':>12} "
              f"{'Suggested':>10} {'Margin':>8} {'Strategy'}")
        print("  " + "-" * 78)
        for p, d in state["pricing_logic"].items():
            print(
                f"  {p:<25} "
                f"${d.get('cost', 0):>7.2f} "
                f"${d.get('competitor_price', 0):>11.2f} "
                f"${d.get('suggested_price', 0):>9.2f} "
                f"{d.get('margin_percent', 0):>7.2f}% "
                f"{d.get('pricing_strategy', '')}"
            )

    if state["logs"]:
        print(f"\n  Audit Log ({len(state['logs'])} entries):")
        for entry in state["logs"]:
            print(f"    • {entry}")

    print(sep + "\n")


if __name__ == "__main__":
    run_swarm()
