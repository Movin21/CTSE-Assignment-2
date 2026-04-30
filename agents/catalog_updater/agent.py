import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from shared.logger import log_event
from shared.llm import _llm
from state import GlobalState
from agents.catalog_updater.tool import save_to_local_db

_SYSTEM_PROMPT = """You are a System Administrator responsible for data persistence and catalog integrity.

ROLE: Save all verified pricing results to the local SQLite database using the save_to_local_db tool.

HARD CONSTRAINTS:
  1. Only save data that has passed through the Price Strategist — do not save raw or partial data.
  2. Do NOT modify any field values before saving.
  3. Report the exact rows_saved count returned by the tool. Never claim more rows than were saved.
  4. If the tool returns an error status, report the error verbatim.
  5. Confirm the db_path where the data was persisted.

RESPONSE FORMAT RULES:
  - Your ONLY action is to call save_to_local_db exactly once with the provided entries and db_path.
  - Do NOT write any explanation or prose before or after the tool call.
  - Do NOT call the tool more than once.
  - Pass the entries JSON string exactly as provided — do not reformat or reorder fields."""


def run_updater_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", "CatalogUpdater", "→ entering node")
    start = time.time()

    pricing_logic = state.get("pricing_logic", {})
    if not pricing_logic:
        log_event("AGENT_WARN", "CatalogUpdater", "No pricing data — skipping save")
        state["catalog_saved"] = False
        state["current_agent"] = "CatalogUpdater"
        state["execution_times"]["CatalogUpdater"] = round(time.time() - start, 3)
        return state

    llm_with_tools = _llm.bind_tools([save_to_local_db])
    entries = list(pricing_logic.values())
    entries_json = json.dumps(entries)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Save the following {len(entries)} pricing entries to the database "
                f"(db_path='catalog.db') using the save_to_local_db tool.\n"
                f"Entries (JSON): {entries_json}"
            )
        ),
    ]

    response: AIMessage = llm_with_tools.invoke(messages)
    catalog_saved = False

    if response.tool_calls:
        for tc in response.tool_calls:
            args = dict(tc["args"])
            if isinstance(args.get("entries"), list):
                args["entries"] = json.dumps(args["entries"])
            raw = save_to_local_db.invoke(args)
            result = json.loads(raw)
            rows = result.get("rows_saved", 0)
            status = result.get("status", "")
            catalog_saved = status == "success"
            state["logs"].append(
                f"[CatalogUpdater] save_to_local_db → {rows} rows, "
                f"db={result.get('db_path')}, status={status}"
            )
            if not catalog_saved:
                state["errors"].append(f"CatalogUpdater: {status}")
    else:
        log_event("AGENT_WARN", "CatalogUpdater",
                  "LLM did not call tool — forcing direct call")
        raw = save_to_local_db.invoke({"entries": entries_json, "db_path": "catalog.db"})
        result = json.loads(raw)
        catalog_saved = result.get("status") == "success"
        state["logs"].append(
            f"[CatalogUpdater] fallback direct save → {result.get('rows_saved')} rows"
        )

    state["catalog_saved"] = catalog_saved
    state["current_agent"] = "CatalogUpdater"
    state["messages"] = state.get("messages", []) + [response]

    elapsed = round(time.time() - start, 3)
    state["execution_times"]["CatalogUpdater"] = elapsed
    log_event("AGENT_END", "CatalogUpdater",
              f"← exiting node, catalog_saved={catalog_saved}, duration={elapsed}s")
    return state
