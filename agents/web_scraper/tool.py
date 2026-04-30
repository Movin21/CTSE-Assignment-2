from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.logger import log_event


class ScrapeInput(BaseModel):
    product_name: str = Field(..., description="Exact product name to find")
    html_source: str = Field(default="competitor.html", description="Local HTML path or HTTP URL")


class ScrapeOutput(BaseModel):
    product_name: str
    competitor_price: float
    source: str
    status: str


@tool
def scrape_competitor_price(product_name: str, html_source: str = "competitor.html") -> str:
    """Scrape a single competitor price for ``product_name`` using BeautifulSoup.

    Accepts either a local HTML file path or an HTTP/HTTPS URL as
    ``html_source``.  Two parsing strategies are attempted in order:

    1. Element with a ``data-name`` attribute matching the product name,
       containing a child with ``class="price"``.
    2. ``<tr>`` whose first ``<td>`` contains the product name
       (case-insensitive), with the price in the second ``<td>``.

    If both strategies fail, a deterministic fallback price of ``$100.00``
    is returned with ``status="fallback_used: <reason>"`` so the pipeline
    never stalls.

    Args:
        product_name: Exact product name to search for in the HTML.
            Passed verbatim — do not URL-encode or escape.
        html_source: A local file path (default ``"competitor.html"``) or a
            fully-qualified HTTP/HTTPS URL to the competitor price page.

    Returns:
        JSON string conforming to ``ScrapeOutput``:
        ``{"product_name": str, "competitor_price": float,
           "source": str, "status": "success" | "fallback_used: <reason>"}``

    Raises:
        Does not raise — all exceptions trigger the fallback branch and are
        encoded in the ``status`` field.

    Example:
        >>> raw = scrape_competitor_price.invoke(
        ...     {"product_name": "Laptop", "html_source": "competitor.html"})
        >>> import json; data = json.loads(raw)
        >>> data["status"]
        'success'
        >>> data["competitor_price"]
        999.99
    """
    log_event("TOOL_CALL", "WebScraper",
              f"scrape_competitor_price(product='{product_name}', source='{html_source}')")
    try:
        if html_source.startswith(("http://", "https://")):
            resp = requests.get(html_source, timeout=10)
            resp.raise_for_status()
            html_content = resp.text
            source_label = html_source
        else:
            path = Path(html_source).resolve()
            if not path.exists():
                raise FileNotFoundError(f"HTML file not found: {html_source}")
            html_content = path.read_text(encoding="utf-8")
            source_label = str(path)

        soup = BeautifulSoup(html_content, "html.parser")
        price: float | None = None

        # Strategy 1 — element with data-name attribute
        elem = soup.find(attrs={"data-name": product_name})
        if elem:
            price_el = elem.find(class_="price")
            if price_el:
                price = float(price_el.get_text(strip=True).replace("$", "").replace(",", ""))

        # Strategy 2 — table row: first cell = product name, second = price
        if price is None:
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if cells and product_name.lower() in cells[0].get_text(strip=True).lower():
                    if len(cells) > 1:
                        price = float(
                            cells[1].get_text(strip=True).replace("$", "").replace(",", "")
                        )
                        break

        if price is None:
            raise ValueError(f"Product '{product_name}' not found in HTML source")

        out = ScrapeOutput(
            product_name=product_name,
            competitor_price=price,
            source=source_label,
            status="success",
        )
        log_event("TOOL_RESULT", "WebScraper",
                  f"'{product_name}' → ${price} (source: {source_label})")
        return out.model_dump_json()

    except Exception as exc:
        log_event("TOOL_ERROR", "WebScraper",
                  f"scrape_competitor_price failed: {exc} — using fallback")
        fallback_price = 100.0
        out = ScrapeOutput(
            product_name=product_name,
            competitor_price=fallback_price,
            source="fallback_mock",
            status=f"fallback_used: {exc}",
        )
        return out.model_dump_json()
