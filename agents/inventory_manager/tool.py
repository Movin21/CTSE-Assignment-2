import csv
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.logger import log_event


class ReadInventoryInput(BaseModel):
    csv_path: str = Field(default="inventory.csv", description="Path to the inventory CSV file")


class ReadInventoryOutput(BaseModel):
    items: List[Dict[str, Any]] = Field(..., description="List of {product_name, cost} dicts")
    count: int = Field(..., description="Number of items loaded")
    status: str = Field(..., description="'success' or 'error: <reason>'")


@tool
def read_inventory_csv(csv_path: str = "inventory.csv") -> str:
    """Load product inventory from a CSV file and validate every row.

    Reads the CSV at ``csv_path``, validates that the required columns
    ``product_name`` and ``cost`` are present, ensures each cost is a
    positive number, and returns the result as a JSON-encoded
    ``ReadInventoryOutput``.  All exceptions are caught and surfaced as
    ``status="error: <reason>"`` so the pipeline never crashes.

    Args:
        csv_path: Relative or absolute path to the inventory CSV file.
            Must contain columns ``product_name`` and ``cost``.
            Defaults to ``"inventory.csv"`` in the working directory.

    Returns:
        JSON string conforming to ``ReadInventoryOutput``:
        ``{"items": [...], "count": <int>, "status": "success" | "error: <reason>"}``

    Raises:
        Does not raise — all exceptions are caught and returned inside
        the ``status`` field to keep the pipeline non-blocking.

    Example:
        >>> raw = read_inventory_csv.invoke({"csv_path": "inventory.csv"})
        >>> import json; data = json.loads(raw)
        >>> data["status"]
        'success'
        >>> data["count"]
        8
    """
    log_event("TOOL_CALL", "InventoryManager", f"read_inventory_csv(csv_path='{csv_path}')")
    try:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        items: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "product_name" not in row or "cost" not in row:
                    raise ValueError("CSV must contain 'product_name' and 'cost' columns")
                cost_val = float(row["cost"])
                if cost_val <= 0:
                    raise ValueError(
                        f"Cost for '{row['product_name']}' must be positive, got {cost_val}"
                    )
                items.append({
                    "product_name": row["product_name"].strip(),
                    "cost": cost_val,
                })

        out = ReadInventoryOutput(items=items, count=len(items), status="success")
        log_event("TOOL_RESULT", "InventoryManager", f"Loaded {len(items)} products OK")
        return out.model_dump_json()

    except Exception as exc:
        log_event("TOOL_ERROR", "InventoryManager", f"read_inventory_csv failed: {exc}")
        return ReadInventoryOutput(items=[], count=0, status=f"error: {exc}").model_dump_json()
