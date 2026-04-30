# E-Commerce Competitor Intelligence Swarm

A **Multi-Agent AI System** built with **LangGraph** and **Ollama (llama3.2)** that automates competitor price intelligence for an e-commerce business — entirely locally, at zero cost.

---

## What This System Does

Imagine you run an online store. Every day you need to know:
- What products do we carry and what did they cost us?
- What are competitors charging for the same products right now?
- What price should *we* charge to stay competitive while keeping a healthy margin?
- Where do we store that analysis so it can feed other systems?

This swarm answers all four questions automatically by passing a shared piece of state through four specialised AI agents, each doing exactly one job.

---

## How It Works — The Pipeline

```
START
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 1 — Inventory Manager  (Detail-oriented Clerk)   │
│  Reads inventory.csv → loads product names + costs      │
│  State updated: inventory = [{product_name, cost}, ...]  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 2 — Web Scraper  (Market Researcher)             │
│  Scrapes competitor.html with BeautifulSoup             │
│  State updated: competitor_data = {product: price}       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 3 — Price Strategist  (Financial Analyst)        │
│  Applies 20% markup logic, validates against competitor  │
│  State updated: pricing_logic = {product: full_analysis} │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 4 — Catalog Updater  (System Administrator)      │
│  Persists all results to catalog.db (SQLite)            │
│  State updated: catalog_saved = True                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                     END
```

Every agent transition and every tool call is recorded to `trace.log` in real time.

---

## File-by-File Explanation

| File | What it does |
|---|---|
| `state.py` | Defines `GlobalState` — the shared memory object that flows through every agent. Also holds Pydantic data models (`InventoryItem`, `PricingResult`, etc.) for strict type safety. |
| `tools.py` | The four actual working functions (`@tool` decorated). Each tool has a Pydantic input/output schema and writes to the observability logger. This is where the real I/O happens. |
| `agents.py` | Wraps each tool inside an LLM agent. Each agent has a unique **system prompt with hard constraints** to prevent hallucinations, then calls its tool and writes results back into state. |
| `graph.py` | Wires everything together using LangGraph's `StateGraph`. Defines the four nodes and the sequential edges. Run this file to execute the full swarm. |
| `evaluation.py` | An independent test suite. For each agent it runs the tool with known inputs, then asks **Llama 3.2 itself to act as a judge** — scoring Accuracy, Security, and Completeness. Adds a layer of programmatic assertions on top. |
| `inventory.csv` | Sample product catalogue (8 items) used as the inventory source. Edit this to add your own products. |
| `competitor.html` | Mock competitor website (local HTML file). Contains product prices in two HTML structures so the scraper works regardless of layout. |
| `trace.log` | Auto-generated at runtime. Contains a timestamped audit trail of every tool call, agent transition, and error. |
| `catalog.db` | Auto-generated at runtime. SQLite database where final pricing results are stored. |

---

## Pricing Strategy Logic (Agent 3)

Given `cost` and `competitor_price`, Agent 3 calculates `suggested_price` using one of three strategies:

| Strategy | Condition | Result |
|---|---|---|
| `below_competitor_with_margin` | Our markup price is already >5% below competitor | Use markup price — no adjustment needed |
| `standard_markup` | Our markup price is within ±5–10% of competitor | Use markup price — we're in the right zone |
| `capped_at_competitor` | Our markup price exceeds competitor by >10% | Cap at 99% of competitor price to stay competitive |

**Hard floor**: the suggested price can never fall below cost. If it would, strategy becomes `floor_applied`.

---

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running locally

---

## Setup & Run

**1. Install Ollama and pull the model**
```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
```

**2. Install Python dependencies**
```bash
pip3 install -r requirements.txt
```

**3. Run the full swarm**
```bash
python graph.py
```

You will see a live console log of each agent as it runs, followed by a formatted pricing report. Results are saved to `catalog.db` and the full audit trail is in `trace.log`.

**4. Run the evaluation suite**
```bash
python evaluation.py
```

Each agent is tested individually. Llama 3.2 acts as a judge and scores each output. Programmatic assertions are checked on top of the LLM verdict.

---

## Sample Output

```
══════════════════════════════════════════════════════════════
  COMPETITOR INTELLIGENCE SWARM — FINAL REPORT
══════════════════════════════════════════════════════════════
  Products in inventory  : 8
  Competitor prices found: 8
  Pricing decisions made : 8
  Catalog persisted      : True

  Product                   Cost   Competitor  Suggested   Margin  Strategy
  ──────────────────────────────────────────────────────────────────────────
  Laptop                  $750.00     $999.99    $900.00   16.67%  below_competitor_with_margin
  Wireless Mouse           $12.50      $19.99     $15.00   16.67%  below_competitor_with_margin
  Mechanical Keyboard      $45.00      $79.99     $54.00   16.67%  below_competitor_with_margin
  ...
```

---

## Project Architecture Diagram

```
inventory.csv ──► Agent 1 ──► Agent 2 ──► Agent 3 ──► Agent 4 ──► catalog.db
                    │           │           │           │
                    └───────────┴───────────┴───────────┘
                                      │
                               GlobalState (shared)
                                      │
                                  trace.log  (observability)
                                      │
                              evaluation.py  (LLM-as-Judge)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ConnectionRefusedError` when running | Ollama is not running. Start it with `ollama serve` |
| `model "llama3.2" not found` | Run `ollama pull llama3.2` first |
| Agent does not call its tool | Normal for smaller models — the agent nodes have a **fallback direct call** that kicks in automatically |
| `trace.log` not created | Check write permissions in the project directory |
| `catalog.db` is empty | Check `trace.log` for a `TOOL_ERROR` from `CatalogUpdater` |
