"""LangChain tool binding for competitor price scraping."""

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .constants import AGENT_NAME, HTTP_TIMEOUT_SECONDS
from .models import ScrapeInput, ScrapeOutput
from .parser import extract_competitor_price
from shared.logger import log_event


def _load_html(html_source: str) -> tuple[str, str]:
    if html_source.startswith(("http://", "https://")):
        response = requests.get(html_source, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text, html_source

    path = Path(html_source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HTML file not found: {html_source}")

    return path.read_text(encoding="utf-8"), str(path)


@tool
def scrape_competitor_price(product_name: str, html_source: str = "competitor.html") -> str:
    """Scrape a single competitor price for ``product_name`` using BeautifulSoup.

    Accepts either a local HTML file path or an HTTP/HTTPS URL as
    ``html_source``. Two parsing strategies are attempted in order:

    1. Element with a ``data-name`` attribute matching the product name,
       containing a child with ``class="price"``.
    2. ``<tr>`` whose first ``<td>`` contains the product name
       (case-insensitive), with the price in the second ``<td>``.

    If both strategies fail, a deterministic fallback price of ``$100.00``
    is returned with ``status="fallback_used: <reason>"`` so the pipeline
    never stalls.

    Args:
        product_name: Exact product name to search for in the HTML.
        html_source: Local file path or HTTP/HTTPS URL (default ``competitor.html``).

    Returns:
        JSON string conforming to ``ScrapeOutput``.

    Raises:
        Does not raise — failures are returned in the ``status`` field.
    """
    request = ScrapeInput(product_name=product_name, html_source=html_source)
    log_event(
        "TOOL_CALL",
        AGENT_NAME,
        (
            f"scrape_competitor_price(product={request.product_name!r}, "
            f"source={request.html_source!r})"
        ),
    )
    try:
        html_content, source_label = _load_html(request.html_source)
        soup = BeautifulSoup(html_content, "html.parser")
        price = extract_competitor_price(soup, request.product_name)
        out = ScrapeOutput.success(
            product_name=request.product_name,
            competitor_price=price,
            source=source_label,
        )
        log_event(
            "TOOL_RESULT",
            AGENT_NAME,
            f"{request.product_name!r} → ${price} (source: {source_label})",
        )
        return out.model_dump_json()

    except Exception as exc:
        log_event(
            "TOOL_ERROR",
            AGENT_NAME,
            f"scrape_competitor_price failed: {exc} — using fallback",
        )
        out = ScrapeOutput.fallback(product_name=request.product_name, reason=str(exc))
        return out.model_dump_json()
