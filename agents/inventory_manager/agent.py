"""LangGraph node for the Inventory Manager agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from shared.llm import llm
from shared.logger import log_agent_start, log_agent_end
from .tool import read_inventory_csv
from state import GlobalState
import time

SYSTEM_PROMPT = """You are a detail-oriented Data Clerk responsible for loading and validating the product inventory.

1. You MUST call the `read_inventory_csv` tool — never skip this step.
2. Report ONLY products from the CSV — do NOT invent product names or prices.
3. If the CSV is missing or malformed, report the EXACT error message returned by the tool.
4. Do NOT alter or round any cost values returned by the tool.
5. Flag any row missing `product_name` or `cost` explicitly in your response.

Response format rules:
- Call the tool exactly once with the provided `csv_path`
- Do not add prose, commentary, or explanation beyond what the tool returns
- Your output will be consumed by the next agent — keep it structured and factual"""

llm_with_tools = llm.bind_tools([read_inventory_csv])


def inventory_manager_node(state: GlobalState) -> GlobalState:
    """Run the Inventory Manager: load inventory from CSV via tool (with LLM fallback).

    Invokes the bound LLM with system and human messages so it may call
    ``read_inventory_csv``. If the model issues no tool call, the tool is
    invoked directly with ``state['csv_path']`` so the pipeline never stalls.

    Args:
        state: LangGraph global state; must include ``csv_path`` and is updated
            with ``inventory`` (list of product dicts), ``messages`` (appends the
            model ``AIMessage``), ``logs``, ``errors`` (on missing CSV),
            ``current_agent``, and ``execution_times``.

    Returns:
        The same state object with audit fields aligned with other swarm agents.
    """
    log_agent_start("InventoryManager")
    start_time = time.time()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Load the inventory from this CSV file: {state['csv_path']}"
            )
        ),
    ]

    response: AIMessage = llm_with_tools.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    used_fallback = not tool_calls
    path_arg = state["csv_path"]
    inventory: list[dict] = []

    try:
        if used_fallback:
            inventory = read_inventory_csv.invoke({"csv_path": state["csv_path"]})
        else:
            tc = tool_calls[0]
            if isinstance(tc, dict):
                args = tc.get("args", {}) or {}
            else:
                args = getattr(tc, "args", None) or {}
            if not isinstance(args, dict):
                args = {}
            path_arg = args.get("csv_path", state["csv_path"])
            inventory = read_inventory_csv.invoke({"csv_path": path_arg})
    except FileNotFoundError as exc:
        err_msg = str(exc)
        inventory = []
        state["errors"].append(f"InventoryManager: {err_msg}")
        state["logs"].append(
            f"[InventoryManager] read_inventory_csv → error: {err_msg}"
        )
    else:
        if used_fallback:
            state["logs"].append(
                f"[InventoryManager] Fallback direct call → {len(inventory)} products"
            )
        else:
            state["logs"].append(
                f"[InventoryManager] read_inventory_csv(csv_path={path_arg!r}) "
                f"→ {len(inventory)} products"
            )

    state["inventory"] = inventory
    state["current_agent"] = "InventoryManager"
    state["messages"] = state.get("messages", []) + [response]

    elapsed = time.time() - start_time
    state["execution_times"]["InventoryManager"] = round(elapsed, 3)
    log_agent_end(
        "InventoryManager",
        elapsed,
        {"products_loaded": len(inventory)},
    )
    return state


run_inventory_agent = inventory_manager_node
