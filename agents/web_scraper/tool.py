"""LangChain tool binding for competitor price scraping."""

from langchain_core.tools import tool

from .constants import AGENT_NAME
from .models import ScrapeInput, ScrapeOutput
from .scraper import scrape_product
from shared.logger import log_event


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

    result = scrape_product(request.product_name, request.html_source)

    if result.status.startswith("fallback_used"):
        log_event(
            "TOOL_ERROR",
            AGENT_NAME,
            f"scrape_competitor_price failed: {result.status} — using fallback",
        )
    else:
        log_event(
            "TOOL_RESULT",
            AGENT_NAME,
            f"{request.product_name!r} → ${result.competitor_price} (source: {result.source})",
        )

    return result.model_dump_json()
