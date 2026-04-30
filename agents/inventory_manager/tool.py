"""Inventory CSV loading tool for the Inventory Manager agent."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from shared.logger import log_tool_call

KEY_COST = "cost"
KEY_PRODUCT_NAME = "product_name"

_logger = logging.getLogger(__name__)


def _strip_row(row: dict[str | None, Any]) -> dict[str, Any]:
    """Return a copy of the row with stripped keys and stripped string values."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        sk = key.strip()
        if isinstance(value, str):
            out[sk] = value.strip()
        elif value is None:
            out[sk] = ""
        else:
            out[sk] = value
    return out


@tool
def read_inventory_csv(csv_path: str) -> list[dict]:
    """Read a CSV file from a given path and return a list of validated product dicts.

    Parses the file with ``csv.DictReader``, normalizes keys and string values by
    stripping whitespace, and keeps only rows with a non-empty ``product_name`` and
    a strictly positive ``cost`` that parses as a float. Rows that fail validation
    are skipped and a warning is logged.

    Args:
        csv_path: Path to the inventory CSV file. Expected columns include
            ``product_name`` and ``cost`` (after key stripping).

    Returns:
        A list of product records. Each dict contains at least ``product_name`` (str)
        and ``cost`` (float) taken from the CSV without modification beyond
        stripping whitespace and parsing ``cost`` as a float.

    Raises:
        FileNotFoundError: If ``csv_path`` does not refer to an existing file.

    Example:
        >>> read_inventory_csv.invoke({"csv_path": "inventory.csv"})
        [{'product_name': 'Laptop', 'cost': 750.0}, ...]
    """
    try:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        result: list[dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cleaned = _strip_row(row)
                name = cleaned.get(KEY_PRODUCT_NAME, "")
                if not isinstance(name, str):
                    name = str(name).strip()
                if not name:
                    _logger.warning(
                        "Skipping row: missing or empty %s in %s",
                        KEY_PRODUCT_NAME,
                        csv_path,
                    )
                    continue

                cost_raw = cleaned.get(KEY_COST)
                try:
                    cost_val = float(cost_raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    _logger.warning(
                        "Skipping row: invalid %s for product %r in %s",
                        KEY_COST,
                        name,
                        csv_path,
                    )
                    continue

                if cost_val <= 0:
                    _logger.warning(
                        "Skipping row: %s must be > 0 for product %r in %s",
                        KEY_COST,
                        name,
                        csv_path,
                    )
                    continue

                result.append({KEY_PRODUCT_NAME: name, KEY_COST: cost_val})

        log_tool_call("read_inventory_csv", {"csv_path": csv_path}, result)
        return result

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from None
    except Exception:
        _logger.exception("read_inventory_csv failed for csv_path=%s", csv_path)
        result: list[dict[str, Any]] = []
        log_tool_call("read_inventory_csv", {"csv_path": csv_path}, result)
        return result
