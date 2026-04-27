"""LangChain tool binding for competitor price scraping."""

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .constants import AGENT_NAME, FALLBACK_PRICE, FALLBACK_SOURCE, HTTP_TIMEOUT_SECONDS
from .models import ScrapeInput, ScrapeOutput
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
    try:
        if request.html_source.startswith(("http://", "https://")):
            resp = requests.get(request.html_source, timeout=HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            html_content = resp.text
            source_label = request.html_source
        else:
            path = Path(request.html_source).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"HTML file not found: {request.html_source}")
            html_content = path.read_text(encoding="utf-8")
            source_label = str(path)

        soup = BeautifulSoup(html_content, "html.parser")
        price: float | None = None

        elem = soup.find(attrs={"data-name": request.product_name})
        if elem:
            price_el = elem.find(class_="price")
            if price_el:
                price = float(price_el.get_text(strip=True).replace("$", "").replace(",", ""))

        if price is None:
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if cells and request.product_name.lower() in cells[0].get_text(strip=True).lower():
                    if len(cells) > 1:
                        price = float(
                            cells[1].get_text(strip=True).replace("$", "").replace(",", "")
                        )
                        break

        if price is None:
            raise ValueError(f"Product '{request.product_name}' not found in HTML source")

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
        out = ScrapeOutput(
            product_name=request.product_name,
            competitor_price=FALLBACK_PRICE,
            source=FALLBACK_SOURCE,
            status=f"fallback_used: {exc}",
        )
        return out.model_dump_json()
