"""
state.py — Global shared state and Pydantic models for the E-Commerce Competitor Intelligence Swarm.
"""

from datetime import datetime
from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, Field


# ─── Pydantic Domain Models ───────────────────────────────────────────────────

class InventoryItem(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    cost: float = Field(..., gt=0, description="Our cost price (must be positive)")


class CompetitorPrice(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    competitor_price: float = Field(..., gt=0, description="Price found on competitor site")
    source: str = Field(..., description="URL or file path of the competitor data")


class PricingResult(BaseModel):
    product_name: str
    cost: float
    competitor_price: float
    suggested_price: float
    margin_percent: float
    pricing_strategy: str


class CatalogEntry(BaseModel):
    product_name: str
    cost: float
    competitor_price: float
    suggested_price: float
    margin_percent: float
    pricing_strategy: str = ""
    saved_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ─── LangGraph Global State ───────────────────────────────────────────────────

class GlobalState(TypedDict):
    """
    Shared state object passed between all agent nodes in the LangGraph.

    Fields:
        csv_path         — path to the merchant inventory CSV for the swarm
        inventory        — raw product records loaded from CSV
        competitor_data  — map of product_name → competitor_price
        pricing_logic    — map of product_name → full pricing analysis dict
        logs             — ordered audit trail of agent actions
        messages         — LangChain message history (for context)
        current_agent    — name of the currently active agent
        catalog_saved    — True after CatalogUpdater successfully persists data
        errors           — non-fatal errors collected across agents
        execution_times  — map of agent_name → elapsed seconds
    """
    csv_path: str
    inventory: List[Dict[str, Any]]
    competitor_data: Dict[str, float]
    pricing_logic: Dict[str, Dict[str, Any]]
    logs: List[str]
    messages: List[Any]
    current_agent: str
    catalog_saved: bool
    errors: List[str]
    execution_times: Dict[str, float]
