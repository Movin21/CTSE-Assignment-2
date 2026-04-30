import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from shared.logger import log_event
from shared.llm import _llm
from state import GlobalState
from agents.web_scraper.tool import scrape_competitor_price

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


def run_scraper_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", "WebScraper", "→ entering node")
    start = time.time()

    inventory = state.get("inventory", [])
    if not inventory:
        log_event("AGENT_WARN", "WebScraper", "Inventory is empty — skipping scrape")
        state["competitor_data"] = {}
        state["current_agent"] = "WebScraper"
        state["execution_times"]["WebScraper"] = round(time.time() - start, 3)
        return state

    llm_with_tools = _llm.bind_tools([scrape_competitor_price])
    competitor_data: dict[str, float] = {}
    last_response: AIMessage | None = None

    for item in inventory:
        product_name = item["product_name"]

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Scrape the competitor price for '{product_name}' from 'competitor.html' "
                    f"using the scrape_competitor_price tool."
                )
            ),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)
        last_response = response

        if response.tool_calls:
            for tc in response.tool_calls:
                raw = scrape_competitor_price.invoke(tc["args"])
                result = json.loads(raw)
                price = result.get("competitor_price", 0.0)
                competitor_data[product_name] = price
                state["logs"].append(
                    f"[WebScraper] '{product_name}' → ${price} "
                    f"(source={result.get('source')}, status={result.get('status')})"
                )
        else:
            log_event("AGENT_WARN", "WebScraper",
                      f"LLM did not call tool for '{product_name}' — forcing direct call")
            raw = scrape_competitor_price.invoke({
                "product_name": product_name,
                "html_source": "competitor.html",
            })
            result = json.loads(raw)
            price = result.get("competitor_price", 0.0)
            competitor_data[product_name] = price
            state["logs"].append(
                f"[WebScraper] '{product_name}' fallback direct → ${price}"
            )

    state["competitor_data"] = competitor_data
    state["current_agent"] = "WebScraper"
    if last_response:
        state["messages"] = state.get("messages", []) + [last_response]

    elapsed = round(time.time() - start, 3)
    state["execution_times"]["WebScraper"] = elapsed
    log_event("AGENT_END", "WebScraper",
              f"← exiting node, scraped {len(competitor_data)} prices, duration={elapsed}s")
    return state
