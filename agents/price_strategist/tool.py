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


@tool
def calculate_margin_price(
    product_name: str,
    cost: float,
    competitor_price: float,
    markup_percent: float = 20.0,
) -> str:
    """Apply a percentage markup to cost and choose a pricing strategy.

    Selects one of three strategies based on how markup price compares to competitor:
    - below_competitor_with_margin: markup price is already > 5% below competitor
    - capped_at_competitor: markup price exceeds competitor by > 10%; cap at 99%
    - standard_markup: markup price is within normal range of competitor
    """
    log_event("TOOL_CALL", "PriceStrategist",
              f"calculate_margin_price(product='{product_name}', cost={cost}, "
              f"competitor={competitor_price})")
    try:
        markup_price = round(cost * (1 + markup_percent / 100), 2)

        if markup_price < competitor_price * 0.95:
            suggested_price = markup_price
            strategy = "below_competitor_with_margin"
        elif markup_price > competitor_price * 1.10:
            suggested_price = round(competitor_price * 0.99, 2)
            strategy = "capped_at_competitor"
        else:
            suggested_price = markup_price
            strategy = "standard_markup"

        actual_margin = round(((suggested_price - cost) / suggested_price) * 100, 2)

        out = MarginOutput(
            product_name=product_name,
            cost=cost,
            competitor_price=competitor_price,
            suggested_price=suggested_price,
            margin_percent=actual_margin,
            pricing_strategy=strategy,
        )
        log_event("TOOL_RESULT", "PriceStrategist",
                  f"'{product_name}' -> suggested=${suggested_price}, "
                  f"margin={actual_margin}%, strategy={strategy}")
        return out.model_dump_json()

    except Exception as exc:
        log_event("TOOL_ERROR", "PriceStrategist", f"calculate_margin_price failed: {exc}")
        return json.dumps({"error": str(exc), "product_name": product_name})