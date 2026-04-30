import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from shared.logger import log_event
from shared.llm import _llm
from state import GlobalState
from agents.inventory_manager.tool import read_inventory_csv

_SYSTEM_PROMPT = """You are a Detail-oriented Data Clerk in an e-commerce intelligence system.

ROLE: Load and validate product inventory from a CSV file using the read_inventory_csv tool.

HARD CONSTRAINTS — violating any of these is a critical failure:
  1. You MUST call read_inventory_csv. Never skip this step.
  2. Report only products that appear in the CSV. Do NOT invent product names or prices.
  3. If the CSV is missing or malformed, report the exact error message from the tool output.
  4. Do not alter cost values returned by the tool.
  5. If a required field (product_name or cost) is absent in a row, flag it explicitly.

RESPONSE FORMAT RULES:
  - Your ONLY action is to call read_inventory_csv exactly once with the given csv_path.
  - Do NOT write any explanation or prose before or after the tool call.
  - Do NOT call the tool more than once.
  - Do NOT invent or modify any values."""


def run_inventory_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", "InventoryManager", "→ entering node")
    start = time.time()

    llm_with_tools = _llm.bind_tools([read_inventory_csv])

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content="Load the inventory now using read_inventory_csv with csv_path='inventory.csv'."),
    ]

    response: AIMessage = llm_with_tools.invoke(messages)
    inventory_data: list[dict[str, Any]] = []

    if response.tool_calls:
        for tc in response.tool_calls:
            raw = read_inventory_csv.invoke(tc["args"])
            result = json.loads(raw)
            inventory_data = result.get("items", [])
            count = result.get("count", 0)
            status = result.get("status", "")
            state["logs"].append(
                f"[InventoryManager] read_inventory_csv → {count} products, status={status}"
            )
            log_event("TOOL_INVOKE", "InventoryManager", f"args={tc['args']} → count={count}")
            if status.startswith("error:"):
                state["errors"].append(f"InventoryManager: {status}")
    else:
        log_event("AGENT_WARN", "InventoryManager", "No tool call issued by LLM — forcing direct call")
        raw = read_inventory_csv.invoke({"csv_path": "inventory.csv"})
        result = json.loads(raw)
        inventory_data = result.get("items", [])
        state["logs"].append(
            f"[InventoryManager] Fallback direct call → {len(inventory_data)} products"
        )

    state["inventory"] = inventory_data
    state["current_agent"] = "InventoryManager"
    state["messages"] = state.get("messages", []) + [response]

    elapsed = round(time.time() - start, 3)
    state["execution_times"]["InventoryManager"] = elapsed
    log_event("AGENT_END", "InventoryManager",
              f"← exiting node, loaded {len(inventory_data)} products, duration={elapsed}s")
    return state
