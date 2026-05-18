"""LangGraph node for the Web Scraper agent."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .constants import AGENT_NAME, DEFAULT_HTML_SOURCE, FALLBACK_STATUS_PREFIX
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


def _build_scrape_messages(product_name: str, html_source: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Scrape the competitor price for {product_name!r} from {html_source!r} "
                "using the scrape_competitor_price tool."
            )
        ),
    ]


def _invoke_scrape_tool(*, product_name: str, html_source: str) -> ScrapeOutput:
    raw = scrape_competitor_price.invoke(
        {"product_name": product_name, "html_source": html_source}
    )
    return ScrapeOutput.model_validate_json(raw)


def _tool_args_from_call(tool_call: Any, *, product_name: str) -> dict[str, str]:
    raw_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
    args = raw_args if isinstance(raw_args, dict) else {}
    return {
        "product_name": str(args.get("product_name") or product_name),
        "html_source": str(args.get("html_source") or DEFAULT_HTML_SOURCE),
    }


def _record_scrape_result(
    state: GlobalState,
    result: ScrapeOutput,
    *,
    via_direct: bool,
) -> None:
    prefix = "direct tool call" if via_direct else "scraped"
    state["logs"].append(
        f"[{AGENT_NAME}] {result.product_name!r} {prefix} → "
        f"${result.competitor_price} "
        f"(source={result.source}, status={result.status})"
    )
    if result.status.startswith(FALLBACK_STATUS_PREFIX):
        state["errors"].append(
            f"{AGENT_NAME}: unverified fallback for {result.product_name!r} — {result.status}"
        )


def _scrape_via_llm(
    llm_with_tools: Any,
    *,
    product_name: str,
    html_source: str,
) -> tuple[ScrapeOutput, AIMessage]:
    response: AIMessage = llm_with_tools.invoke(_build_scrape_messages(product_name, html_source))

    if not response.tool_calls:
        log_event(
            "AGENT_WARN",
            AGENT_NAME,
            f"LLM did not call tool for {product_name!r} — forcing direct call",
        )
        result = _invoke_scrape_tool(product_name=product_name, html_source=html_source)
        return result, response

    tool_call = response.tool_calls[0]
    args = _tool_args_from_call(tool_call, product_name=product_name)
    result = _invoke_scrape_tool(**args)
    return result, response


def web_scraper_node(state: GlobalState) -> GlobalState:
    """Scrape competitor prices for every product in inventory."""
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
    messages: list[Any] = list(state.get("messages") or [])

    for item in inventory:
        product_name = item.get("product_name")
        if not product_name:
            log_event("AGENT_WARN", AGENT_NAME, f"Skipping inventory row without product_name: {item!r}")
            state["errors"].append(f"{AGENT_NAME}: inventory row missing product_name")
            continue
        product_name = str(product_name)
        result, response = _scrape_via_llm(
            llm_with_tools,
            product_name=product_name,
            html_source=DEFAULT_HTML_SOURCE,
        )
        competitor_data[product_name] = result.competitor_price
        _record_scrape_result(state, result, via_direct=not response.tool_calls)
        messages.append(response)

    state["competitor_data"] = competitor_data
    state["current_agent"] = AGENT_NAME
    state["messages"] = messages

    elapsed = round(time.perf_counter() - start, 3)
    state["execution_times"][AGENT_NAME] = elapsed
    log_event(
        "AGENT_END",
        AGENT_NAME,
        f"← exiting node, scraped {len(competitor_data)} prices, duration={elapsed}s",
    )
    return state


run_scraper_agent = web_scraper_node
