# E-Commerce Competitor Intelligence Swarm

**SE4010 – CTSE | Assignment 2 – Machine Learning**
**Sri Lanka Institute of Information Technology**

A locally-hosted **Multi-Agent AI System (MAS)** built with **LangGraph** and **Ollama (llama3.2)** that autonomously monitors competitor pricing, computes optimal selling prices, and persists results to a local database — entirely at zero cloud cost.


---

## Table of Contents

1. [Problem Domain](#1-problem-domain)
2. [System Architecture](#2-system-architecture)
3. [Agent Design](#3-agent-design)
4. [Custom Tools](#4-custom-tools)
5. [State Management](#5-state-management)
6. [LLMOps & Observability](#6-llmops--observability)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [Setup & Running](#8-setup--running)
9. [Individual Contributions](#9-individual-contributions)

---

## 1. Problem Domain

E-commerce businesses must continuously monitor competitor prices and adjust their own pricing to remain competitive while protecting margins. Doing this manually across dozens of products is time-consuming and error-prone.

This system automates the full intelligence pipeline:

| Step | Manual Process | Automated by |
|---|---|---|
| Load product catalogue | Open spreadsheet | Inventory Manager Agent |
| Check competitor prices | Visit competitor sites | Web Scraper Agent |
| Calculate optimal price | Apply markup rules by hand | Price Strategist Agent |
| Record decisions | Update database manually | Catalog Updater Agent |

The system runs on a local machine using a Small Language Model (SLM) via Ollama, requiring no internet access, no paid API keys, and no cloud services.

---

## 2. System Architecture

### 2.1 Technology Stack

| Component | Technology |
|---|---|
| LLM Engine | `llama3.2` via Ollama (local, zero-cost) |
| Orchestrator | LangGraph `StateGraph` |
| Tool Framework | LangChain `@tool` with Pydantic schemas |
| Data Store | SQLite (auto-created at runtime) |
| Containerisation | Docker + Docker Compose |

### 2.2 Multi-Agent Architecture

The system uses a **sequential pipeline** model with **conditional routing**. Each agent owns exactly one responsibility and passes enriched state to the next agent. Conditional edges short-circuit the pipeline if a stage fails, preventing downstream agents from operating on empty data.

```
inventory.csv
      │
      ▼
┌─────────────────────────────────────────┐
│  INVENTORY MANAGER                      │
│  Role: Detail-oriented Data Clerk       │
│  Tool: read_inventory_csv               │
│  Output: state["inventory"]             │
└────────────────┬────────────────────────┘
                 │  if inventory empty → END
                 ▼
┌─────────────────────────────────────────┐
│  WEB SCRAPER                            │
│  Role: Market Researcher                │
│  Tool: scrape_competitor_price          │
│  Output: state["competitor_data"]       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  PRICE STRATEGIST                       │
│  Role: Financial Analyst                │
│  Tool: calculate_margin_price           │
│  Output: state["pricing_logic"]         │
└────────────────┬────────────────────────┘
                 │  if pricing_logic empty → END
                 ▼
┌─────────────────────────────────────────┐
│  CATALOG UPDATER                        │
│  Role: System Administrator             │
│  Tool: save_to_local_db                 │
│  Output: state["catalog_saved"] = True  │
└────────────────┬────────────────────────┘
                 │
                END

        ┌────────────────────┐
        │   GlobalState      │  ← shared memory passed between all agents
        │   trace.log        │  ← observability: every tool call logged
        │   catalog.db       │  ← final output: SQLite persistence
        └────────────────────┘
```

### 2.3 Project Structure

```
CTSE Assignment 2/
├── agents/
│   ├── inventory_manager/
│   │   ├── agent.py          ← LLM agent node + system prompt
│   │   └── tool.py           ← read_inventory_csv tool
│   ├── web_scraper/
│   │   ├── agent.py          ← LLM agent node + system prompt
│   │   └── tool.py           ← scrape_competitor_price tool
│   ├── price_strategist/
│   │   ├── agent.py          ← LLM agent node + system prompt
│   │   └── tool.py           ← calculate_margin_price tool
│   └── catalog_updater/
│       ├── agent.py          ← LLM agent node + system prompt
│       └── tool.py           ← save_to_local_db tool
├── shared/
│   ├── llm.py                ← shared ChatOllama instance
│   └── logger.py             ← unified observability logger
├── state.py                  ← GlobalState TypedDict + Pydantic models
├── graph.py                  ← LangGraph orchestration + conditional routing
├── evaluation.py             ← LLM-as-a-Judge test suite (13 tests)
├── inventory.csv             ← product catalogue (data source)
├── competitor.html           ← mock competitor website (data source)
├── Dockerfile                ← container image definition
├── docker-compose.yml        ← multi-service orchestration
└── requirements.txt
```

---

## 3. Agent Design

Each agent follows the same pattern: a unique **persona**, a set of **hard constraints** to prevent hallucinations, and a **RESPONSE FORMAT RULES** block that instructs the SLM to call the tool exactly once without adding prose.

### 3.1 Inventory Manager

**Persona:** Detail-oriented Data Clerk

**Responsibility:** Load product inventory from a CSV file and validate every row.

**System Prompt Constraints:**
1. Must call `read_inventory_csv` — never skip this step.
2. Report only products from the CSV — do NOT invent product names or prices.
3. If the CSV is missing or malformed, report the exact error from the tool.
4. Do not alter cost values returned by the tool.
5. Flag any row missing `product_name` or `cost` explicitly.

**Interaction Strategy:** The agent receives a `HumanMessage` with the exact `csv_path`. If the LLM fails to call the tool, a fallback direct call is made automatically so the pipeline never stalls.

---

### 3.2 Web Scraper

**Persona:** Market Researcher specialised in ethical data collection

**Responsibility:** Retrieve the competitor price for each product from an HTML source.

**System Prompt Constraints:**
1. Only scrape products that exist in the inventory — never fabricate a product list.
2. Never invent or estimate a price — use only what the tool returns.
3. Label any fallback price as `UNVERIFIED FALLBACK`.
4. Always include the `source` field so the price can be audited.
5. Do not modify the numeric price value.

**Interaction Strategy:** One LLM invocation per product. The agent iterates over `state["inventory"]` and issues one tool call per product name. Two HTML parsing strategies are attempted before falling back to a deterministic `$100.00` placeholder.

---

### 3.3 Price Strategist

**Persona:** Financial Analyst responsible for competitive pricing

**Responsibility:** Apply a 20% markup to cost and validate the result against the competitor price using one of four strategies.

**Pricing Logic:**

| Strategy | Condition | Action |
|---|---|---|
| `below_competitor_with_margin` | Markup price < 95% of competitor | Use markup price — already competitive |
| `standard_markup` | Markup price within ±5–10% of competitor | Use markup price |
| `capped_at_competitor` | Markup price > 110% of competitor | Cap at 99% of competitor |
| `floor_applied` | Any strategy yields price < cost | Hard floor: cost × 1.01 |

**System Prompt Constraints:**
1. Apply EXACTLY `markup_percent=20.0` — never change this value.
2. Suggested price MUST NOT fall below cost.
3. Always pass `product_name`, `cost`, `competitor_price`, and `markup_percent` to the tool.
4. Use only values returned by the tool — no manual rounding.

**Interaction Strategy:** The agent node always merges `product_name` into the LLM's tool call arguments before invoking the tool, guarding against the known SLM failure mode of omitting required fields.

---

### 3.4 Catalog Updater

**Persona:** System Administrator responsible for data persistence

**Responsibility:** Persist all verified pricing results to a local SQLite database in a single atomic transaction.

**System Prompt Constraints:**
1. Only save data that has passed through the Price Strategist.
2. Do NOT modify any field values before saving.
3. Report the exact `rows_saved` count — never claim more.
4. If the tool returns an error, report it verbatim.
5. Confirm the `db_path` where data was persisted.

**Interaction Strategy:** The agent serialises `state["pricing_logic"]` to JSON and passes the full batch to `save_to_local_db` in a single call. If the LLM passes entries as a list instead of a JSON string, the agent coerces it before invoking the tool.

---

## 4. Custom Tools

Each tool is decorated with `@tool`, uses strict Pydantic input/output schemas, and follows the Google docstring standard with `Args`, `Returns`, `Raises`, and `Example` sections.

### 4.1 `read_inventory_csv` — Inventory Manager

**File:** `agents/inventory_manager/tool.py`

Reads a CSV file with columns `product_name` and `cost`, validates every row (positive cost, required columns), and returns a JSON-encoded result.

```python
# Example usage
raw = read_inventory_csv.invoke({"csv_path": "inventory.csv"})
# Returns:
# {"items": [{"product_name": "Laptop", "cost": 750.0}, ...],
#  "count": 8, "status": "success"}
```

**Error handling:** `FileNotFoundError` if path missing, `ValueError` if columns absent or cost ≤ 0. All exceptions are caught and returned as `status="error: <reason>"`.

---

### 4.2 `scrape_competitor_price` — Web Scraper

**File:** `agents/web_scraper/tool.py`

Scrapes a competitor price using BeautifulSoup. Accepts a local HTML file or HTTP/HTTPS URL. Two parsing strategies are attempted in order:

1. Element with `data-name="<product>"` containing a `.price` child.
2. `<tr>` where the first `<td>` matches the product name (case-insensitive).

```python
# Example usage
raw = scrape_competitor_price.invoke({
    "product_name": "Laptop",
    "html_source": "competitor.html"
})
# Returns:
# {"product_name": "Laptop", "competitor_price": 999.99,
#  "source": "/path/to/competitor.html", "status": "success"}
```

**Fallback:** On any exception, returns `competitor_price=100.0` with `status="fallback_used: <reason>"` so the pipeline never stalls.

---

### 4.3 `calculate_margin_price` — Price Strategist

**File:** `agents/price_strategist/tool.py`

Applies a markup percentage to cost, selects one of four pricing strategies based on comparison with the competitor price, and enforces a hard cost floor.

```python
# Example usage
raw = calculate_margin_price.invoke({
    "product_name": "Laptop",
    "cost": 750.0,
    "competitor_price": 999.99,
    "markup_percent": 20.0
})
# Returns:
# {"product_name": "Laptop", "cost": 750.0, "competitor_price": 999.99,
#  "suggested_price": 900.0, "margin_percent": 16.67,
#  "pricing_strategy": "below_competitor_with_margin"}
```

> **Note:** `markup_percent` (applied to cost) and `margin_percent` (calculated from selling price) are different values. A 20% markup on cost yields a ~16.67% gross margin.

---

### 4.4 `save_to_local_db` — Catalog Updater

**File:** `agents/catalog_updater/tool.py`

Persists a batch of pricing entries to a SQLite database in a single atomic transaction. Uses parameterised queries throughout — SQL injection is not possible regardless of input content. The table is created automatically if absent. The transaction is rolled back entirely on any failure.

```python
# Example usage
import json
entries = [{"product_name": "Laptop", "cost": 750.0,
            "competitor_price": 999.99, "suggested_price": 900.0,
            "margin_percent": 16.67, "pricing_strategy": "below_competitor_with_margin"}]
raw = save_to_local_db.invoke({
    "entries": json.dumps(entries),
    "db_path": "catalog.db"
})
# Returns:
# {"rows_saved": 1, "db_path": "catalog.db", "status": "success"}
```

---

## 5. State Management

### 5.1 GlobalState Structure

All agents share a single `GlobalState` TypedDict defined in `state.py`. It is initialised in `graph.py` and mutated in-place by each agent node.

```python
class GlobalState(TypedDict):
    inventory:       List[Dict[str, Any]]      # loaded by InventoryManager
    competitor_data: Dict[str, float]          # populated by WebScraper
    pricing_logic:   Dict[str, Dict[str, Any]] # populated by PriceStrategist
    logs:            List[str]                 # audit trail entries
    messages:        List[Any]                 # LangChain message history
    current_agent:   str                       # currently active agent
    catalog_saved:   bool                      # set True by CatalogUpdater
    errors:          List[str]                 # non-fatal errors across agents
    execution_times: Dict[str, float]          # agent_name → elapsed seconds
```

### 5.2 Context Flow Between Agents

State flows linearly through the pipeline. Each agent reads fields set by the previous agent and writes its own results. No agent modifies another agent's fields.

```
Initial State
  inventory=[]  competitor_data={}  pricing_logic={}  catalog_saved=False

After InventoryManager:
  inventory=[{product_name, cost}, ...]

After WebScraper:
  competitor_data={product_name: price, ...}

After PriceStrategist:
  pricing_logic={product_name: {cost, competitor_price, suggested_price,
                                margin_percent, pricing_strategy}, ...}

After CatalogUpdater:
  catalog_saved=True
  execution_times={"InventoryManager": 1.2, "WebScraper": 9.4, ...}
```

### 5.3 Conditional Routing

The LangGraph `StateGraph` uses conditional edges to guard against empty state:

- After `inventory_agent`: if `state["inventory"]` is empty → route to `END` (skip remaining agents).
- After `pricing_agent`: if `state["pricing_logic"]` is empty → route to `END` (skip database write).

This prevents downstream agents from operating on empty data and ensures the pipeline fails fast with a clear audit trail.

---

## 6. LLMOps & Observability

### 6.1 Logging

Every event is recorded by `shared/logger.py` to both the console and `trace.log` using a structured format:

```
YYYY-MM-DD HH:MM:SS | LEVEL        | [EVENT_TYPE    ] agent=AgentName   details
```

**Event types logged:**

| Event Type | When it fires |
|---|---|
| `SYSTEM` | Swarm start / end |
| `AGENT_START` | Agent node entered |
| `AGENT_END` | Agent node exited (includes duration) |
| `AGENT_WARN` | LLM did not call tool — fallback triggered |
| `TOOL_CALL` | Tool invoked by agent |
| `TOOL_RESULT` | Tool returned successfully |
| `TOOL_ERROR` | Tool raised an exception |
| `TOOL_INVOKE` | Post-call confirmation with result count |

### 6.2 Execution Timing

Each agent records its elapsed time into `state["execution_times"]`. This is displayed in the final report and written to `trace.log` at `AGENT_END`, enabling performance analysis across runs.

### 6.3 Error Accumulation

Non-fatal errors (e.g., tool status starting with `"error:"`) are appended to `state["errors"]` rather than raising exceptions. This lets the pipeline continue to the next agent while preserving a complete error record in the final report.

---

## 7. Evaluation Methodology

### 7.1 Approach

The evaluation suite (`evaluation.py`) combines two complementary methods:

1. **LLM-as-a-Judge** — Llama 3.2 scores each tool output on Accuracy (0–10), Security (0–10), and Completeness (0–10). Verdict threshold: all dimensions ≥ 7 (security ≥ 8).
2. **Programmatic assertions** — Deterministic Python checks verify exact values, types, and constraints. These are **authoritative**: if all programmatic assertions pass, the verdict is forced to `PASS` regardless of the LLM judge score. If any fail, the verdict is forced to `FAIL`.

This hybrid approach handles a known weakness of small models — they can mis-score mathematically correct outputs (e.g., confusing markup% with margin%) — while still leveraging the LLM for qualitative checks like hallucination detection.

### 7.2 Test Coverage

13 tests across 4 agents covering happy path, edge cases, and security:

| # | Agent | Test | Type |
|---|---|---|---|
| 1 | Inventory Manager | Load `inventory.csv` — validate 8 products | Happy path |
| 2 | Inventory Manager | Non-existent file — status must start with `error:` | Edge case |
| 3 | Inventory Manager | Negative cost in CSV — must be rejected | Edge case |
| 4 | Web Scraper | Scrape `Laptop` from `competitor.html` | Happy path |
| 5 | Web Scraper | Unknown product — must return `fallback_used` status | Edge case |
| 6 | Web Scraper | `<script>` injection as product name — safe handling | Security |
| 7 | Price Strategist | cost=500, competitor=650 → `below_competitor_with_margin` | Happy path |
| 8 | Price Strategist | cost=100, competitor=50 → `floor_applied`, price ≥ cost | Edge case |
| 9 | Price Strategist | cost=50, competitor=52 → `capped_at_competitor` | Edge case |
| 10 | Price Strategist | cost=100, competitor=120 → `standard_markup` | Edge case |
| 11 | Catalog Updater | Save 2 records → `rows_saved=2`, `status=success` | Happy path |
| 12 | Catalog Updater | Empty entries list → `rows_saved=0`, `status=success` | Edge case |
| 13 | Catalog Updater | SQL injection in `product_name` — stored as literal text | Security |

### 7.3 Running Evaluation

```bash
# Local
python3 evaluation.py

# Docker
docker compose --profile eval up evaluation
```

Sample output:

```
==============================================================
  E-COMMERCE SWARM — LLM-AS-A-JUDGE EVALUATION SUITE
  Judge model: llama3.2 (temperature=0)
==============================================================

  ┌─ Inventory Manager — Happy Path
  │  Verdict      : [PASS] PASS
  │  Accuracy     : 10/10
  │  Security     : 10/10
  │  Completeness : 10/10
  └───────────────────────────────────────────────────────

  Passed: 13 / 13
==============================================================
```

---

## 8. Setup & Running

### 8.1 Option A — Docker (Recommended)

No manual Ollama installation required. The entrypoint script waits for Ollama to be ready and pulls `llama3.2` automatically on first run.

```bash
# Run the full swarm
docker compose up --build

# Run the evaluation suite
docker compose --profile eval up --build evaluation

# Stop everything
docker compose down

# Remove downloaded model cache
docker compose down -v
```

### 8.2 Option B — Local

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Pull the model
ollama pull llama3.2

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the swarm
python3 graph.py

# 5. Run the evaluation suite
python3 evaluation.py
```

### 8.3 Sample Final Report Output

```
==============================================================
  COMPETITOR INTELLIGENCE SWARM — FINAL REPORT
==============================================================
  Products in inventory  : 8
  Competitor prices found: 8
  Pricing decisions made : 8
  Catalog persisted      : True

  Agent Execution Times:
    InventoryManager       1.243s
    WebScraper             9.812s
    PriceStrategist       14.337s
    CatalogUpdater         2.104s

  Product                    Cost   Competitor  Suggested   Margin  Strategy
  ------------------------------------------------------------------------------
  Laptop                  $750.00     $999.99    $900.00   16.67%  below_competitor_with_margin
  Wireless Mouse           $12.50      $19.99     $15.00   16.67%  below_competitor_with_margin
  Mechanical Keyboard      $45.00      $79.99     $54.00   16.67%  below_competitor_with_margin
  USB-C Hub                $18.00      $29.99     $21.60   16.67%  below_competitor_with_margin
  Monitor 27inch          $220.00     $349.99    $264.00   16.67%  below_competitor_with_margin
  Webcam HD                $35.00      $59.99     $42.00   16.67%  below_competitor_with_margin
  Noise-Cancelling Hdph    $90.00     $149.99    $108.00   16.67%  below_competitor_with_margin
  External SSD 1TB         $65.00     $109.99     $78.00   16.67%  below_competitor_with_margin
==============================================================
```

### 8.4 Troubleshooting

| Problem | Fix |
|---|---|
| `ConnectionRefusedError` | Ollama is not running — start with `ollama serve` |
| `model "llama3.2" not found` | Run `ollama pull llama3.2` |
| Agent does not call its tool | Normal — the agent has a fallback direct call that triggers automatically |
| `trace.log` not created | Check write permissions in the project directory |
| `catalog.db` is empty | Check `trace.log` for a `TOOL_ERROR` from `CatalogUpdater` |

---

## 9. Individual Contributions

### Member 1 — Inventory Manager

**Agent developed:** `InventoryManager` (`agents/inventory_manager/agent.py`)
- Designed the Detail-oriented Data Clerk persona and anti-hallucination constraints.
- Implemented fallback direct call when the SLM does not invoke the tool.

**Tool implemented:** `read_inventory_csv` (`agents/inventory_manager/tool.py`)
- CSV parsing with `csv.DictReader`, column validation, positive-cost enforcement.
- Full Google-style docstring with `Args`, `Returns`, `Raises`, `Example`.

**Test cases contributed:**
- `test_inventory_happy_path` — validates 8 products loaded correctly.
- `test_inventory_missing_file` — verifies graceful error on missing CSV.
- `test_inventory_negative_cost` — verifies negative costs are rejected.

**Challenges faced:**
- Ensuring the SLM does not invent product names when the CSV is small.
- Handling CSV files with inconsistent whitespace in product names (solved with `.strip()`).

---

### Member 2 — Web Scraper

**Agent developed:** `WebScraper` (`agents/web_scraper/agent.py`)
- Designed the Market Researcher persona with constraints against price fabrication.
- Implemented per-product iteration with individual LLM invocations.

**Tool implemented:** `scrape_competitor_price` (`agents/web_scraper/tool.py`)
- Two BeautifulSoup parsing strategies to handle different HTML layouts.
- HTTP URL and local file support; deterministic fallback on failure.

**Test cases contributed:**
- `test_scraper_happy_path` — validates price and source for a known product.
- `test_scraper_unknown_product` — verifies fallback behaviour for unlisted products.
- `test_scraper_security` — verifies XSS injection in product name is handled safely.

**Challenges faced:**
- BeautifulSoup returning partial matches for product names (solved with `lower()` comparison).
- SLM occasionally passing the full file path instead of the relative path (handled by `Path.resolve()`).

---

### Member 3 — Price Strategist

**Agent developed:** `PriceStrategist` (`agents/price_strategist/agent.py`)
- Designed the Financial Analyst persona with explicit markup-vs-margin explanation.
- Implemented `product_name` injection guard after SLM omitted required field.

**Tool implemented:** `calculate_margin_price` (`agents/price_strategist/tool.py`)
- Four-strategy pricing logic with hard cost floor.
- Complete docstring explaining markup% vs margin% distinction with worked example.

**Test cases contributed:**
- `test_pricing_happy_path` — validates 20% markup with `below_competitor_with_margin`.
- `test_pricing_floor_applied` — verifies floor is applied when competitor < cost.
- `test_pricing_capped_at_competitor` — verifies cap when markup exceeds competitor.
- `test_pricing_standard_markup` — verifies standard markup in the ±5–10% band.

**Challenges faced:**
- SLM consistently omitted `product_name` from tool call arguments (fixed by merging it from the loop variable before invoking the tool).
- LLM-as-a-Judge mis-scored correct outputs by confusing markup% with margin% (fixed by overriding judge verdict with deterministic programmatic assertions).

---

### Member 4 — Catalog Updater

**Agent developed:** `CatalogUpdater` (`agents/catalog_updater/agent.py`)
- Designed the System Administrator persona with strict data-integrity constraints.
- Handles LLM passing entries as a Python list instead of a JSON string.

**Tool implemented:** `save_to_local_db` (`agents/catalog_updater/tool.py`)
- Atomic `BEGIN/COMMIT/ROLLBACK` transaction so the database is never partially written.
- Parameterised queries — SQL injection is structurally impossible.
- Auto-creates the `catalog` table on first run.

**Test cases contributed:**
- `test_updater_happy_path` — validates 2 records saved with correct row count.
- `test_updater_empty_entries` — verifies empty batch returns `rows_saved=0`.
- `test_updater_sql_injection` — verifies malicious product names are stored as literal text.

**Challenges faced:**
- LLM sometimes serialises the entries list as a Python list object rather than a JSON string (fixed with an `isinstance` check and `json.dumps` coercion in the agent).
- Ensuring transaction rollback on partial insert failure without leaving the database in an inconsistent state.
