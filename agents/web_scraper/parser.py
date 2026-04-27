"""HTML parsing utilities for competitor price extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, Tag

_PRICE_CLASS = "price"
_DATA_NAME_ATTR = "data-name"
_CURRENCY_PATTERN = re.compile(r"[^\d.\-]")


def normalize_price_text(raw_text: str) -> float:
    """Convert a price string such as ``$1,299.99`` into a float."""
    cleaned = _CURRENCY_PATTERN.sub("", raw_text.strip())
    if not cleaned:
        raise ValueError(f"Could not parse price from text: {raw_text!r}")
    return float(cleaned)


def find_price_by_data_name(soup: BeautifulSoup, product_name: str) -> float | None:
    """Strategy 1: locate ``data-name`` element with a ``.price`` child."""
    element: Tag | None = soup.find(attrs={_DATA_NAME_ATTR: product_name})
    if element is None:
        return None

    price_element = element.find(class_=_PRICE_CLASS)
    if price_element is None:
        return None

    return normalize_price_text(price_element.get_text(strip=True))


def find_price_by_table_row(soup: BeautifulSoup, product_name: str) -> float | None:
    """Strategy 2: match product name in the first ``<td>`` of a table row."""
    target = product_name.casefold()

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        cell_text = cells[0].get_text(strip=True).casefold()
        if target not in cell_text:
            continue

        return normalize_price_text(cells[1].get_text(strip=True))

    return None


def extract_competitor_price(soup: BeautifulSoup, product_name: str) -> float:
    """Run parsing strategies in order until a price is found."""
    for strategy in (find_price_by_data_name, find_price_by_table_row):
        price = strategy(soup, product_name)
        if price is not None:
            return price

    raise ValueError(f"Product '{product_name}' not found in HTML source")
