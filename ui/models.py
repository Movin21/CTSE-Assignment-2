"""Typed view models for dashboard data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CatalogRow(BaseModel):
    """Validated row loaded from the ``catalog`` SQLite table."""

    product_name: str = Field(min_length=1)
    cost: float = Field(ge=0)
    competitor_price: float = Field(ge=0)
    suggested_price: float = Field(ge=0)
    margin_percent: float
    pricing_strategy: str = ""
    saved_at: str


class LogEvent(BaseModel):
    """Validated event parsed from a ``trace.log`` line."""

    timestamp: str
    level: str
    event_type: str
    agent: str
    details: str
