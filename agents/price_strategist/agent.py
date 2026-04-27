import time
from typing import Any

from shared.logger import log_event
from state import GlobalState


def run_pricing_agent(state: GlobalState) -> GlobalState:
    log_event("AGENT_START", "PriceStrategist", "-> entering node")
    start = time.time()

    state["pricing_logic"] = {}
    state["current_agent"] = "PriceStrategist"
    state["execution_times"]["PriceStrategist"] = round(time.time() - start, 3)
    return state