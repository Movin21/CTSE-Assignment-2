import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.logger import log_event


class MarginInput(BaseModel):
    product_name: str
    cost: float = Field(..., gt=0, description="Our cost price — must be positive")
    competitor_price: float = Field(..., gt=0, description="Competitor's selling price — must be positive")
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

    Computes ``markup_price = cost * (1 + markup_percent / 100)`` then
    selects one of four strategies based on how that price compares to the
    competitor:

    - ``below_competitor_with_margin``: markup_price is more than 5% below
      competitor — it is already competitive, so use it directly.
    - ``capped_at_competitor``: markup_price exceeds competitor by more than
      10% — cap at 99% of competitor to remain competitive.
    - ``standard_markup``: markup_price is within ±5–10% of competitor —
      use the markup price.
    - ``floor_applied``: any strategy would push the price below cost —
      apply a hard floor at cost * 1.01 (1% above cost).

    Note: ``markup_percent`` and ``margin_percent`` are *different*:
    - Markup % = (price − cost) / cost × 100  (applied to cost, e.g. 20%)
    - Margin % = (price − cost) / price × 100 (gross margin, e.g. 16.67%)

    Args:
        product_name: Name of the product being priced.
        cost: Our purchase/production cost. Must be positive.
        competitor_price: The competitor's current selling price. Must be positive.
        markup_percent: Percentage markup to apply to ``cost``.
            Must be between 0 and 100. Defaults to ``20.0``.

    Returns:
        JSON string conforming to ``MarginOutput``:
        ``{"product_name": str, "cost": float, "competitor_price": float,
           "suggested_price": float, "margin_percent": float,
           "pricing_strategy": str}``
        On error: ``{"error": "<reason>", "product_name": str}``

    Raises:
        Does not raise — all exceptions are caught and returned as an
        error dict so the pipeline can log and continue.

    Example:
        >>> raw = calculate_margin_price.invoke({
        ...     "product_name": "Laptop", "cost": 500.0,
        ...     "competitor_price": 650.0, "markup_percent": 20.0})
        >>> import json; data = json.loads(raw)
        >>> data["suggested_price"]
        600.0
        >>> data["pricing_strategy"]
        'below_competitor_with_margin'
        >>> data["margin_percent"]  # (600-500)/600*100
        16.67
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

        # Hard constraint: price must never go below cost
        if suggested_price < cost:
            suggested_price = round(cost * 1.01, 2)
            strategy = "floor_applied"

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
                  f"'{product_name}' → suggested=${suggested_price}, "
                  f"margin={actual_margin}%, strategy={strategy}")
        return out.model_dump_json()

    except Exception as exc:
        log_event("TOOL_ERROR", "PriceStrategist", f"calculate_margin_price failed: {exc}")
        return json.dumps({"error": str(exc), "product_name": product_name})
