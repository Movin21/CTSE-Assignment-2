import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from shared.logger import log_event
from shared.llm import _llm
from state import GlobalState
from agents.price_strategist.tool import calculate_margin_price

_SYSTEM_PROMPT = """You are a Financial Analyst responsible for competitive pricing strategy.

ROLE: Calculate the optimal selling price for every product using the calculate_margin_price tool.

HARD CONSTRAINTS:
  1. Apply EXACTLY 20% markup (markup_percent=20.0). Never change this value.
  2. A suggested price MUST NOT fall below cost. This is a non-negotiable floor.
  3. Use only the values returned by the tool — do not manually adjust or round the output.
  4. If competitor_price was flagged as UNVERIFIED FALLBACK, note that in your analysis.
  5. Report the pricing_strategy field returned by the tool for each product.
  6. You MUST always pass product_name, cost, competitor_price, and markup_percent to the tool.

RESPONSE FORMAT RULES:
  - Your ONLY action is to call calculate_margin_price exactly once with all four arguments.
  - Do NOT write any explanation or prose before or after the tool call.
  - Do NOT call the tool more than once per message.
  - Do NOT invent, round, or modify any numeric value given to you."""


def run_pricing_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", "PriceStrategist", "→ entering node")
    start = time.time()

    inventory = state.get("inventory", [])
    competitor_data = state.get("competitor_data", {})

    if not inventory:
        log_event("AGENT_WARN", "PriceStrategist", "No inventory — skipping pricing")
        state["pricing_logic"] = {}
        state["current_agent"] = "PriceStrategist"
        state["execution_times"]["PriceStrategist"] = round(time.time() - start, 3)
        return state

    llm_with_tools = _llm.bind_tools([calculate_margin_price])
    pricing_logic: dict[str, dict[str, Any]] = {}
    last_response: AIMessage | None = None

    for item in inventory:
        product_name = item["product_name"]
        cost = item["cost"]
        competitor_price = competitor_data.get(product_name, round(cost * 1.5, 2))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Calculate the margin price for '{product_name}' with "
                    f"cost={cost} and competitor_price={competitor_price} "
                    f"and markup_percent=20.0 using the calculate_margin_price tool."
                )
            ),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)
        last_response = response

        if response.tool_calls:
            for tc in response.tool_calls:
                args = {**tc["args"], "product_name": product_name}
                raw = calculate_margin_price.invoke(args)
                result = json.loads(raw)
                if "error" not in result:
                    pricing_logic[product_name] = result
                    state["logs"].append(
                        f"[PriceStrategist] '{product_name}' → "
                        f"suggested=${result.get('suggested_price')}, "
                        f"margin={result.get('margin_percent')}%, "
                        f"strategy={result.get('pricing_strategy')}"
                    )
                else:
                    state["errors"].append(
                        f"PriceStrategist: error for '{product_name}': {result['error']}"
                    )
                    state["logs"].append(
                        f"[PriceStrategist] ERROR for '{product_name}': {result['error']}"
                    )
        else:
            log_event("AGENT_WARN", "PriceStrategist",
                      f"LLM did not call tool for '{product_name}' — forcing direct call")
            raw = calculate_margin_price.invoke({
                "product_name": product_name,
                "cost": cost,
                "competitor_price": competitor_price,
                "markup_percent": 20.0,
            })
            result = json.loads(raw)
            if "error" not in result:
                pricing_logic[product_name] = result

    state["pricing_logic"] = pricing_logic
    state["current_agent"] = "PriceStrategist"
    if last_response:
        state["messages"] = state.get("messages", []) + [last_response]

    elapsed = round(time.time() - start, 3)
    state["execution_times"]["PriceStrategist"] = elapsed
    log_event("AGENT_END", "PriceStrategist",
              f"← exiting node, priced {len(pricing_logic)} products, duration={elapsed}s")
    return state
