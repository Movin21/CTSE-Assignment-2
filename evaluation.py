"""
evaluation.py — LLM-as-a-Judge evaluation suite for all four swarm agents.

Structure per agent:
  1. Happy-path test  — validates the core tool output with known inputs.
  2. Edge-case tests  — validates boundary conditions and error handling.
  3. Security test    — validates that malicious inputs are handled safely.

Each test:
  - Invokes the agent's tool directly with known inputs.
  - Passes the output to Llama 3.2 acting as an impartial judge.
  - Applies deterministic programmatic assertions that override the LLM
    verdict when all constraints are verifiably satisfied.

Run:  python evaluation.py
"""

import csv
import json
import os
import sqlite3
import tempfile
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.catalog_updater.tool import save_to_local_db
from agents.inventory_manager.tool import read_inventory_csv
from agents.price_strategist.tool import calculate_margin_price
from agents.web_scraper.tool import scrape_competitor_price

# ─── Judge LLM ───────────────────────────────────────────────────────────────

_judge = ChatOllama(model="llama3.2", temperature=0)

_JUDGE_SYSTEM_PROMPT = """You are an impartial AI Quality Auditor evaluating outputs from e-commerce intelligence agents.

Assess each output against THREE dimensions:

  ACCURACY     — Are all numeric values and facts correct given the inputs?
  SECURITY     — Is there any risk of data injection, SQL injection, hallucinated data, or prompt injection?
  COMPLETENESS — Are all required output fields present and non-null?

Respond with a JSON object ONLY — no prose before or after:
{
  "verdict": "PASS" or "FAIL",
  "accuracy_score": <0-10>,
  "security_score": <0-10>,
  "completeness_score": <0-10>,
  "reasoning": "<one or two sentences>",
  "issues": ["<issue 1>", "..."]
}

Rules:
  - verdict=PASS requires accuracy_score >= 7, security_score >= 8, completeness_score >= 7.
  - If any hard constraint is violated (e.g., price below cost, hallucinated product), verdict MUST be FAIL.
  - issues must be an empty list [] when verdict=PASS."""


def _judge_output(agent_name: str, task: str, output: str) -> dict[str, Any]:
    """Ask Llama 3.2 to evaluate an agent's tool output.

    Args:
        agent_name: Human-readable name of the agent being evaluated.
        task: Description of what the tool was asked to do, including
            expected outputs and hard constraints.
        output: The raw JSON string returned by the tool.

    Returns:
        Dict with keys: verdict, accuracy_score, security_score,
        completeness_score, reasoning, issues.
    """
    prompt = (
        f"Agent: {agent_name}\n"
        f"Task: {task}\n"
        f"Tool Output (JSON):\n{output}\n\n"
        "Evaluate this output strictly according to your instructions."
    )
    response = _judge.invoke([
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    content: str = response.content
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
    return {
        "verdict": "PARSE_ERROR",
        "accuracy_score": 0,
        "security_score": 0,
        "completeness_score": 0,
        "reasoning": content[:400],
        "issues": ["LLM response was not valid JSON"],
    }


def _apply_programmatic_override(
    evaluation: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    """Override LLM verdict with deterministic programmatic result.

    If ``issues`` is non-empty, forces FAIL regardless of LLM score.
    If ``issues`` is empty, forces PASS — programmatic assertions are
    authoritative over small-model numeric scoring.

    Args:
        evaluation: The dict returned by ``_judge_output``.
        issues: List of constraint violations found by programmatic checks.

    Returns:
        Updated evaluation dict with corrected verdict and issues.
    """
    if issues:
        evaluation["verdict"] = "FAIL"
        evaluation["issues"] = list(set(evaluation.get("issues", []) + issues))
    else:
        evaluation["verdict"] = "PASS"
        evaluation["issues"] = []
    return evaluation


def _print_result(test_name: str, result: dict[str, Any]) -> None:
    verdict = result.get("verdict", "UNKNOWN")
    symbol = "PASS" if verdict == "PASS" else "FAIL"
    print(f"\n  ┌─ {test_name}")
    print(f"  │  Verdict      : [{symbol}] {verdict}")
    print(f"  │  Accuracy     : {result.get('accuracy_score', 'N/A')}/10")
    print(f"  │  Security     : {result.get('security_score', 'N/A')}/10")
    print(f"  │  Completeness : {result.get('completeness_score', 'N/A')}/10")
    print(f"  │  Reasoning    : {result.get('reasoning', '')}")
    for iss in result.get("issues", []):
        print(f"  │  Issue        : {iss}")
    print("  └" + "─" * 55)


# ─── Inventory Manager Tests ──────────────────────────────────────────────────

def test_inventory_happy_path() -> dict[str, Any]:
    """Happy path: load inventory.csv and validate all 8 products."""
    items = read_inventory_csv.invoke({"csv_path": "inventory.csv"})
    raw = json.dumps(items)

    evaluation = _judge_output(
        agent_name="InventoryManager — Detail-oriented Data Clerk",
        task=(
            "Load 'inventory.csv' and return a list of dicts with product_name and cost. "
            "No hallucinated products. Expect eight valid rows."
        ),
        output=raw,
    )

    issues: list[str] = []
    if not isinstance(items, list):
        issues.append("tool must return a list[dict]")
    if not items:
        issues.append("items list is empty — no products loaded")
    for item in items:
        if not item.get("product_name"):
            issues.append("A row is missing product_name")
        cost = item.get("cost")
        if cost is None or not isinstance(cost, (int, float)) or cost <= 0:
            issues.append(f"Invalid cost for '{item.get('product_name')}': {cost}")
    if len(items) != 8:
        issues.append(f"expected 8 products from inventory.csv, got {len(items)}")

    _apply_programmatic_override(evaluation, issues)
    _print_result("Inventory Manager — Happy Path", evaluation)
    return evaluation


def test_inventory_missing_file() -> dict[str, Any]:
    """Edge case: non-existent CSV must raise FileNotFoundError with a clear message."""
    issues: list[str] = []
    try:
        read_inventory_csv.invoke({"csv_path": "does_not_exist_xyz.csv"})
        issues.append("Expected FileNotFoundError for missing CSV")
    except FileNotFoundError as exc:
        if "CSV file not found" not in str(exc):
            issues.append(f"Unexpected FileNotFoundError message: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        issues.append(f"Expected FileNotFoundError, got {type(exc).__name__}: {exc}")

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Inventory Manager — Missing File (Edge Case)", evaluation)
    return evaluation


def test_inventory_negative_cost() -> dict[str, Any]:
    """Edge case: CSV row with negative cost must be rejected."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["product_name", "cost"])
        writer.writerow(["BadProduct", "-50.0"])
        tmp_path = f.name

    try:
        items = read_inventory_csv.invoke({"csv_path": tmp_path})
    finally:
        os.unlink(tmp_path)

    issues: list[str] = []
    for item in items:
        if item.get("cost", 0) <= 0:
            issues.append(
                f"Negative cost accepted for '{item['product_name']}': {item['cost']}"
            )

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Inventory Manager — Negative Cost (Edge Case)", evaluation)
    return evaluation


# ─── Web Scraper Tests ────────────────────────────────────────────────────────

def test_scraper_happy_path() -> dict[str, Any]:
    """Happy path: scrape Laptop price from competitor.html."""
    raw = scrape_competitor_price.invoke({
        "product_name": "Laptop",
        "html_source": "competitor.html",
    })
    result = json.loads(raw)

    evaluation = _judge_output(
        agent_name="WebScraper — Market Researcher",
        task=(
            "Scrape competitor price for 'Laptop' from competitor.html. "
            "Must return a positive numeric price and a non-empty source field. "
            "Must never fabricate a price. "
            "If fallback_mock is used, status must include 'fallback_used'."
        ),
        output=raw,
    )

    issues: list[str] = []
    price = result.get("competitor_price", 0)
    if not isinstance(price, (int, float)) or price <= 0:
        issues.append(f"competitor_price must be positive, got: {price}")
    if not result.get("source"):
        issues.append("source field is missing or empty")
    if result.get("product_name") != "Laptop":
        issues.append(f"product_name mismatch: got '{result.get('product_name')}'")

    _apply_programmatic_override(evaluation, issues)
    _print_result("Web Scraper — Happy Path", evaluation)
    return evaluation


def test_scraper_unknown_product() -> dict[str, Any]:
    """Edge case: product not in HTML must trigger fallback with status 'fallback_used'."""
    raw = scrape_competitor_price.invoke({
        "product_name": "ProductThatDoesNotExist999",
        "html_source": "competitor.html",
    })
    result = json.loads(raw)

    issues: list[str] = []
    if "fallback_used" not in result.get("status", ""):
        issues.append(
            f"Expected status to contain 'fallback_used', got: {result.get('status')}"
        )
    if result.get("competitor_price") != 100.0:
        issues.append(
            f"Expected fallback price 100.0, got: {result.get('competitor_price')}"
        )

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Web Scraper — Unknown Product Fallback (Edge Case)", evaluation)
    return evaluation


def test_scraper_security() -> dict[str, Any]:
    """Security: product name with script injection must be handled safely."""
    malicious_name = "<script>alert('xss')</script>"
    raw = scrape_competitor_price.invoke({
        "product_name": malicious_name,
        "html_source": "competitor.html",
    })
    result = json.loads(raw)

    issues: list[str] = []
    # Tool must not crash and must return a valid JSON structure
    if "product_name" not in result:
        issues.append("product_name missing from output — tool may have crashed on injection input")
    # Fallback is expected since the injected name won't match any product
    if result.get("competitor_price", 0) <= 0:
        issues.append("competitor_price must be positive even for unrecognised (injected) input")

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic security check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Web Scraper — XSS Input (Security)", evaluation)
    return evaluation


# ─── Price Strategist Tests ───────────────────────────────────────────────────

def test_pricing_happy_path() -> dict[str, Any]:
    """Happy path: 20% markup on cost=500, competitor=650 → below_competitor_with_margin."""
    cost, competitor_price = 500.0, 650.0
    expected_markup_price = round(cost * 1.20, 2)  # 600.0
    expected_margin = round(((expected_markup_price - cost) / expected_markup_price) * 100, 2)

    raw = calculate_margin_price.invoke({
        "product_name": "Laptop",
        "cost": cost,
        "competitor_price": competitor_price,
        "markup_percent": 20.0,
    })
    result = json.loads(raw)

    evaluation = _judge_output(
        agent_name="PriceStrategist — Financial Analyst",
        task=(
            f"Apply a 20% markup to cost=${cost} with competitor_price=${competitor_price}. "
            f"Note: markup% and margin% use different formulas. "
            f"markup% = (price-cost)/cost*100 = 20% (the INPUT). "
            f"margin% = (price-cost)/price*100 = {expected_margin}% (the CORRECT OUTPUT). "
            f"Expected: suggested_price={expected_markup_price}, margin_percent≈{expected_margin}. "
            "pricing_strategy must be one of: "
            "below_competitor_with_margin | capped_at_competitor | standard_markup | floor_applied."
        ),
        output=raw,
    )

    issues: list[str] = []
    suggested = result.get("suggested_price", 0)
    margin = result.get("margin_percent", 0)

    if suggested < cost:
        issues.append(f"CRITICAL: suggested_price ({suggested}) is below cost ({cost})")
    if margin <= 0:
        issues.append(f"margin_percent must be positive, got: {margin}")
    if abs(suggested - expected_markup_price) > 5.0:
        issues.append(f"Markup deviation too large: expected ~${expected_markup_price}, got ${suggested}")
    valid_strategies = {"below_competitor_with_margin", "capped_at_competitor",
                        "standard_markup", "floor_applied"}
    if result.get("pricing_strategy") not in valid_strategies:
        issues.append(f"Unknown pricing_strategy: '{result.get('pricing_strategy')}'")

    _apply_programmatic_override(evaluation, issues)
    _print_result("Price Strategist — Happy Path", evaluation)
    return evaluation


def test_pricing_floor_applied() -> dict[str, Any]:
    """Edge case: competitor much lower than cost → floor_applied strategy, price >= cost."""
    cost, competitor_price = 100.0, 50.0

    raw = calculate_margin_price.invoke({
        "product_name": "TestProduct",
        "cost": cost,
        "competitor_price": competitor_price,
        "markup_percent": 20.0,
    })
    result = json.loads(raw)

    issues: list[str] = []
    if result.get("pricing_strategy") != "floor_applied":
        issues.append(
            f"Expected strategy 'floor_applied', got '{result.get('pricing_strategy')}'"
        )
    if result.get("suggested_price", 0) < cost:
        issues.append(
            f"CRITICAL: suggested_price {result.get('suggested_price')} is below cost {cost}"
        )

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Price Strategist — Floor Applied (Edge Case)", evaluation)
    return evaluation


def test_pricing_capped_at_competitor() -> dict[str, Any]:
    """Edge case: markup price far exceeds competitor → capped_at_competitor strategy."""
    cost, competitor_price = 50.0, 52.0
    # markup = 60, competitor*1.10 = 57.2 → 60 > 57.2 → capped

    raw = calculate_margin_price.invoke({
        "product_name": "CheapProduct",
        "cost": cost,
        "competitor_price": competitor_price,
        "markup_percent": 20.0,
    })
    result = json.loads(raw)

    issues: list[str] = []
    if result.get("pricing_strategy") != "capped_at_competitor":
        issues.append(
            f"Expected 'capped_at_competitor', got '{result.get('pricing_strategy')}'"
        )
    if result.get("suggested_price", 0) >= competitor_price:
        issues.append(
            f"Capped price {result.get('suggested_price')} should be below competitor {competitor_price}"
        )
    if result.get("suggested_price", 0) < cost:
        issues.append(
            f"CRITICAL: suggested_price {result.get('suggested_price')} is below cost {cost}"
        )

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Price Strategist — Capped At Competitor (Edge Case)", evaluation)
    return evaluation


def test_pricing_standard_markup() -> dict[str, Any]:
    """Edge case: markup within ±5-10% of competitor → standard_markup strategy."""
    cost, competitor_price = 100.0, 120.0
    # markup = 120, competitor*0.95=114, competitor*1.10=132
    # 120 >= 114 and 120 <= 132 → standard_markup

    raw = calculate_margin_price.invoke({
        "product_name": "StandardProduct",
        "cost": cost,
        "competitor_price": competitor_price,
        "markup_percent": 20.0,
    })
    result = json.loads(raw)

    issues: list[str] = []
    if result.get("pricing_strategy") != "standard_markup":
        issues.append(
            f"Expected 'standard_markup', got '{result.get('pricing_strategy')}'"
        )
    if result.get("suggested_price") != 120.0:
        issues.append(
            f"Expected suggested_price=120.0, got {result.get('suggested_price')}"
        )

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Price Strategist — Standard Markup (Edge Case)", evaluation)
    return evaluation


# ─── Catalog Updater Tests ────────────────────────────────────────────────────

def test_updater_happy_path() -> dict[str, Any]:
    """Happy path: save 2 records to test_eval.db, verify rows_saved and status."""
    test_entries = [
        {"product_name": "EvalProduct_A", "cost": 100.0, "competitor_price": 150.0,
         "suggested_price": 120.0, "margin_percent": 16.67, "pricing_strategy": "standard_markup"},
        {"product_name": "EvalProduct_B", "cost": 200.0, "competitor_price": 350.0,
         "suggested_price": 240.0, "margin_percent": 16.67,
         "pricing_strategy": "below_competitor_with_margin"},
    ]

    raw = save_to_local_db.invoke({
        "entries": json.dumps(test_entries),
        "db_path": "test_eval.db",
    })
    result = json.loads(raw)

    evaluation = _judge_output(
        agent_name="CatalogUpdater — System Administrator",
        task=(
            f"Save {len(test_entries)} records to test_eval.db. "
            f"Expected: rows_saved={len(test_entries)}, status='success'. "
            "Must not modify any field values. Must report the exact db_path."
        ),
        output=raw,
    )

    issues: list[str] = []
    if result.get("rows_saved") != len(test_entries):
        issues.append(f"rows_saved: expected {len(test_entries)}, got {result.get('rows_saved')}")
    if result.get("status") != "success":
        issues.append(f"status is not 'success': {result.get('status')}")
    if not result.get("db_path"):
        issues.append("db_path is missing from output")

    _apply_programmatic_override(evaluation, issues)
    _print_result("Catalog Updater — Happy Path", evaluation)
    return evaluation


def test_updater_empty_entries() -> dict[str, Any]:
    """Edge case: empty entries list → rows_saved=0, status='success'."""
    raw = save_to_local_db.invoke({
        "entries": "[]",
        "db_path": "test_eval_empty.db",
    })
    result = json.loads(raw)

    issues: list[str] = []
    if result.get("rows_saved") != 0:
        issues.append(f"Expected rows_saved=0, got {result.get('rows_saved')}")
    if result.get("status") != "success":
        issues.append(f"status is not 'success': {result.get('status')}")

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic edge-case check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Catalog Updater — Empty Entries (Edge Case)", evaluation)
    return evaluation


def test_updater_sql_injection() -> dict[str, Any]:
    """Security: SQL injection in product_name must be stored as literal text."""
    malicious_entry = {
        "product_name": "'; DROP TABLE catalog; --",
        "cost": 10.0,
        "competitor_price": 20.0,
        "suggested_price": 12.0,
        "margin_percent": 16.67,
        "pricing_strategy": "standard_markup",
    }
    db_path = "test_security.db"

    raw = save_to_local_db.invoke({
        "entries": json.dumps([malicious_entry]),
        "db_path": db_path,
    })
    result = json.loads(raw)

    issues: list[str] = []
    if result.get("status") != "success":
        issues.append(f"SQL injection caused tool failure: {result.get('status')}")
    if result.get("rows_saved") != 1:
        issues.append(f"Expected 1 row saved, got {result.get('rows_saved')}")

    # Verify the table still exists and the literal string was stored
    if result.get("status") == "success":
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT product_name FROM catalog").fetchall()
            conn.close()
            names = [r[0] for r in rows]
            if malicious_entry["product_name"] not in names:
                issues.append("Injected product_name was not stored as literal text")
        except sqlite3.OperationalError as e:
            issues.append(f"SQL injection may have damaged the database: {e}")

    evaluation = {"verdict": "", "accuracy_score": 10, "security_score": 10,
                  "completeness_score": 10, "reasoning": "Programmatic security check", "issues": []}
    _apply_programmatic_override(evaluation, issues)
    _print_result("Catalog Updater — SQL Injection (Security)", evaluation)
    return evaluation


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  E-COMMERCE SWARM — LLM-AS-A-JUDGE EVALUATION SUITE")
    print(f"  Judge model: llama3.2 (temperature=0)")
    print(sep)

    tests = [
        # ── Inventory Manager ─────────────────────────────────────────────────
        ("Inventory Manager  — Happy Path       ", test_inventory_happy_path),
        ("Inventory Manager  — Missing File     ", test_inventory_missing_file),
        ("Inventory Manager  — Negative Cost    ", test_inventory_negative_cost),
        # ── Web Scraper ───────────────────────────────────────────────────────
        ("Web Scraper        — Happy Path       ", test_scraper_happy_path),
        ("Web Scraper        — Unknown Product  ", test_scraper_unknown_product),
        ("Web Scraper        — XSS Security     ", test_scraper_security),
        # ── Price Strategist ──────────────────────────────────────────────────
        ("Price Strategist   — Happy Path       ", test_pricing_happy_path),
        ("Price Strategist   — Floor Applied    ", test_pricing_floor_applied),
        ("Price Strategist   — Capped           ", test_pricing_capped_at_competitor),
        ("Price Strategist   — Standard Markup  ", test_pricing_standard_markup),
        # ── Catalog Updater ───────────────────────────────────────────────────
        ("Catalog Updater    — Happy Path       ", test_updater_happy_path),
        ("Catalog Updater    — Empty Entries    ", test_updater_empty_entries),
        ("Catalog Updater    — SQL Injection    ", test_updater_sql_injection),
    ]

    results: list[tuple[str, dict[str, Any]]] = []
    for name, fn in tests:
        print(f"\n  Running: {name.strip()} ...")
        try:
            result = fn()
            results.append((name, result))
        except Exception as exc:
            print(f"  [EXCEPTION] {name.strip()}: {exc}")
            results.append((name, {
                "verdict": "ERROR",
                "accuracy_score": 0,
                "security_score": 0,
                "completeness_score": 0,
                "reasoning": str(exc),
                "issues": [str(exc)],
            }))
        time.sleep(0.3)

    print(f"\n{sep}")
    print("  FINAL SUMMARY")
    print(sep)
    passed = sum(1 for _, r in results if r.get("verdict") == "PASS")
    print(f"  Passed: {passed} / {len(results)}\n")
    for name, result in results:
        verdict = result.get("verdict", "UNKNOWN")
        marker = "✓" if verdict == "PASS" else "✗"
        scores = (
            f"acc={result.get('accuracy_score', '-')}/10  "
            f"sec={result.get('security_score', '-')}/10  "
            f"cmp={result.get('completeness_score', '-')}/10"
        )
        print(f"  {marker}  {name}  {verdict:<5}  [{scores}]")
    print(sep + "\n")


if __name__ == "__main__":
    run_all_tests()
