"""LangGraph node for the Web Scraper agent."""

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .constants import AGENT_NAME, DEFAULT_HTML_SOURCE
from .models import ScrapeOutput
from .tool import scrape_competitor_price
from shared.llm import _llm
from shared.logger import log_event
from state import GlobalState

_SYSTEM_PROMPT = """You are a Market Researcher specialised in ethical, structured web data collection.

ROLE: Retrieve competitor prices for each product using the scrape_competitor_price tool.

HARD CONSTRAINTS:
  1. Only scrape products that exist in the inventory — never fabricate a product list.
  2. Never invent or estimate a competitor price. Use only the value returned by the tool.
  3. If the tool returns status='fallback_used:...', clearly label the price as UNVERIFIED FALLBACK.
  4. Always include the source field in your summary so the price can be audited.
  5. Do not modify the numeric price value in any way.

RESPONSE FORMAT RULES:
  - Your ONLY action is to call scrape_competitor_price exactly once with the given product_name and html_source.
  - Do NOT write any explanation or prose before or after the tool call.
  - Do NOT call the tool more than once per message.
  - Pass product_name verbatim — do not alter spelling, capitalisation, or encoding."""


def _parse_tool_result(raw: str) -> ScrapeOutput:
    return ScrapeOutput.model_validate_json(raw)


def run_scraper_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", AGENT_NAME, "→ entering node")
    start = time.perf_counter()

    inventory = state.get("inventory") or []
    if not inventory:
        log_event("AGENT_WARN", AGENT_NAME, "Inventory is empty — skipping scrape")
        state["competitor_data"] = {}
        state["current_agent"] = AGENT_NAME
        state["execution_times"][AGENT_NAME] = round(time.perf_counter() - start, 3)
        return state

    llm_with_tools = _llm.bind_tools([scrape_competitor_price])
    competitor_data: dict[str, float] = {}
    last_response: AIMessage | None = None

    for item in inventory:
        product_name = str(item["product_name"])

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Scrape the competitor price for {product_name!r} from {DEFAULT_HTML_SOURCE!r} "
                    "using the scrape_competitor_price tool."
                )
            ),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)
        last_response = response

        if response.tool_calls:
            for tc in response.tool_calls:
                raw = scrape_competitor_price.invoke(tc["args"])
                result = _parse_tool_result(raw)
                competitor_data[product_name] = result.competitor_price
                state["logs"].append(
                    f"[{AGENT_NAME}] {result.product_name!r} → ${result.competitor_price} "
                    f"(source={result.source}, status={result.status})"
                )
        else:
            log_event(
                "AGENT_WARN",
                AGENT_NAME,
                f"LLM did not call tool for {product_name!r} — forcing direct call",
            )
            raw = scrape_competitor_price.invoke(
                {"product_name": product_name, "html_source": DEFAULT_HTML_SOURCE}
            )
            result = _parse_tool_result(raw)
            competitor_data[product_name] = result.competitor_price
            state["logs"].append(
                f"[{AGENT_NAME}] {result.product_name!r} fallback direct → ${result.competitor_price}"
            )

    state["competitor_data"] = competitor_data
    state["current_agent"] = AGENT_NAME
    if last_response:
        state["messages"] = state.get("messages", []) + [last_response]

    elapsed = round(time.perf_counter() - start, 3)
    state["execution_times"][AGENT_NAME] = elapsed
    log_event(
        "AGENT_END",
        AGENT_NAME,
        f"← exiting node, scraped {len(competitor_data)} prices, duration={elapsed}s",
    )
    return state
