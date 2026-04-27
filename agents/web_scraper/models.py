"""Pydantic schemas for scraper tool inputs and outputs."""

from pydantic import BaseModel, Field, field_validator

from .constants import (
    FALLBACK_PRICE,
    FALLBACK_SOURCE,
    FALLBACK_STATUS_PREFIX,
    STATUS_SUCCESS,
)


class ScrapeInput(BaseModel):
    product_name: str = Field(..., min_length=1, description="Exact product name to find")
    html_source: str = Field(
        default="competitor.html",
        min_length=1,
        description="Local HTML path or HTTP/HTTPS URL",
    )


class ScrapeOutput(BaseModel):
    product_name: str
    competitor_price: float = Field(..., gt=0)
    source: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)

    @field_validator("competitor_price")
    @classmethod
    def price_must_be_finite(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("competitor_price must be a positive number")
        return value

    @classmethod
    def success(
        cls,
        *,
        product_name: str,
        competitor_price: float,
        source: str,
    ) -> "ScrapeOutput":
        return cls(
            product_name=product_name,
            competitor_price=competitor_price,
            source=source,
            status=STATUS_SUCCESS,
        )

    @classmethod
    def fallback(cls, *, product_name: str, reason: str) -> "ScrapeOutput":
        return cls(
            product_name=product_name,
            competitor_price=FALLBACK_PRICE,
            source=FALLBACK_SOURCE,
            status=f"{FALLBACK_STATUS_PREFIX}: {reason}",
        )
