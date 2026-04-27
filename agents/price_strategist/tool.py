import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.logger import log_event


class MarginInput(BaseModel):
    product_name: str
    cost: float = Field(..., gt=0, description="Our cost price â€” must be positive")
    competitor_price: float = Field(..., gt=0, description="Competitor's selling price â€” must be positive")
    markup_percent: float = Field(default=20.0, ge=0, le=100,
                                  description="Markup applied to cost (not margin). Default 20%.")


class MarginOutput(BaseModel):
    product_name: str
    cost: float
    competitor_price: float
    suggested_price: float
    margin_percent: float
    pricing_strategy: str