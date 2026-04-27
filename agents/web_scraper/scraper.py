"""Core scraping logic (framework-agnostic)."""

from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .constants import HTTP_TIMEOUT_SECONDS
from .models import ScrapeOutput
from .parser import extract_competitor_price


def _is_remote_source(html_source: str) -> bool:
    return html_source.startswith(("http://", "https://"))


def load_html(html_source: str) -> tuple[str, str]:
    """Load HTML from a URL or local file.

    Returns:
        Tuple of ``(html_content, source_label)`` for audit logging.
    """
    if _is_remote_source(html_source):
        response = requests.get(html_source, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text, html_source

    path = Path(html_source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HTML file not found: {html_source}")

    return path.read_text(encoding="utf-8"), str(path)


def scrape_product(product_name: str, html_source: str) -> ScrapeOutput:
    """Scrape a competitor price for one product.

    Never raises — failures are encoded as a fallback ``ScrapeOutput``.
    """
    try:
        html_content, source_label = load_html(html_source)
        soup = BeautifulSoup(html_content, "html.parser")
        price = extract_competitor_price(soup, product_name)
        return ScrapeOutput.success(
            product_name=product_name,
            competitor_price=price,
            source=source_label,
        )
    except Exception as exc:
        return ScrapeOutput.fallback(product_name=product_name, reason=str(exc))
