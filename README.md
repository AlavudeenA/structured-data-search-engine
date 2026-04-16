# Analytical Search Engine

A **database-agnostic, schema-agnostic** analytical question-answering system. Connect it to any SQL Server database and ask plain-English questions — the engine figures out the SQL, retrieves pre-computed insights, and delivers a natural-language answer.

No changes to the core engine are needed when switching databases. You only swap out the configuration folder.

---

## Quick Start

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.11 or 3.13 recommended |
| **SQL Server** | Any edition (Express works); ODBC Driver 17 must be installed |
| **Groq API key** | Free at [console.groq.com](https://console.groq.com) |
| **ODBC Driver 17** | [Download from Microsoft](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) |

### 1 — Clone the repo

```bash
git clone https://github.com/your-org/structured-data-search-engine
cd structured-data-search-engine
```

### 2 — Create a virtual environment (recommended)

```bash
py -3 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3 — Install all dependencies

```bash
py -3 -m pip install -r requirements.txt
```

**What gets installed:**

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.45.0 | Web UI |
| `pyodbc` | 5.3.0 | SQL Server connection |
| `qdrant-client` | 1.17.0 | Local vector database |
| `fastembed` | 0.7.4 | Embedding model (downloads ~100 MB on first run) |
| `groq` | latest | LLM API client |
| `pydantic` / `pydantic-settings` | latest | Settings management |
| `python-dotenv` | latest | Reads `.env` file |
| `numpy` / `scipy` | latest | Anomaly scoring and statistics |

> **Note:** On first run, `fastembed` will automatically download the `BAAI/bge-base-en-v1.5` embedding model (~100 MB). This only happens once and is cached locally.

### 4 — Create your `.env` file

Create a file named `.env` in the project root. Copy this template and fill in your values:

```env
# ── Required ──────────────────────────────────────────────────────────────────

# Get a free key at https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# Your SQL Server connection string
# Example for SQL Express with Windows auth:
SQLSERVER_CONN_STR=DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\SQLEXPRESS;DATABASE=YourDatabase;Trusted_Connection=yes;

# ── Optional (defaults shown) ─────────────────────────────────────────────────

# Qdrant vector store path (local folder, auto-created)
QDRANT_PATH=./qdrant_data

# Embedding model
EMBED_MODEL=BAAI/bge-base-en-v1.5

# LLM model slots (Groq model names)
GROQ_INTENT_MODEL=llama-3.1-8b-instant
GROQ_SQL_MODEL=llama-3.3-70b-versatile
GROQ_SQL_FIX_MODEL=llama-3.3-70b-versatile
GROQ_ANALYTICAL_MODEL=llama-3.1-8b-instant
GROQ_SUMMARY_MODEL=llama-3.1-8b-instant
GROQ_SIGNAL_MODEL=llama-3.1-8b-instant

# Set to 1 to print all executed SQL to the console
SQL_DEBUG=0
```

### 5 — Set up your database (first time only)

The SQL schema script is at `src/business_schema/dbscript.sql`. Run it against your SQL Server to create all tables and seed sample data. It includes safe `DROP` guards so it can be re-run cleanly.

### 6 — Run the app

```bash
py -3 -m streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

### 7 — Generate capsules (first time only)

In the browser, click the **Generate Capsules** tab → **Generate All Capsules**.  
This runs all SQL queries, embeds the results, and builds the vector knowledge base.  
Takes 1–3 minutes depending on DB size. The sidebar collection counts will update when complete.

---

## What It Does

You type a question like:
> *"Which broker dealer has the highest rejection rate?"*

The engine:
1. Understands what you're asking (intent detection)
2. Searches its pre-built knowledge base (vector search)
3. Either answers directly from cached insights **or** writes and runs a live SQL query
4. Returns a plain-English answer with the data behind it

---

## Key Concepts

### Capsules
A **capsule** is a pre-computed unit of knowledge. Before you ask a question, the engine runs all the SQL queries defined in your configuration, compresses the results into a summary ("signal"), and stores everything in a vector database. Capsules are the engine's long-term memory.

There are three types:

| Type | What it is | Count (example) |
|---|---|---|
| `analytical_capsules` | Pre-run SQL results + signals for common business questions | ~42 |
| `schema_context_capsules` | Maps of the database structure — which tables join to what | 5 |
| `related_capsules` | Auto-generated risk/anomaly alerts from capsule cross-analysis | ~17 |

### Two Pipelines
| Pipeline | Name | When it runs | What it does |
|---|---|---|---|
| `capsule_builder/` | Knowledge Builder | When you click "Generate Capsules" | Runs all SQL, computes signals, builds the graph, stores to Qdrant |
| `query_engine/` | Answer Engine | Every time a user asks a question | Searches Qdrant, routes the question, generates SQL if needed, returns an answer |

---

## Architecture

```
SQL Server Database
        │
        ▼
 ┌──────────────────────────────┐
 │       capsule_builder/       │  ← runs once (or on-demand)
 │  Execute SQL → Extract Signal│
 │  Embed → Store in Qdrant     │
 └──────────────┬───────────────┘
                │  writes to
                ▼
      ┌─────────────────┐
      │  Qdrant (local) │  ← 3 collections
      │  analytical     │
      │  schema_context │
      │  related        │
      └────────┬────────┘
               │  read by
               ▼
 ┌──────────────────────────────┐
 │        query_engine/         │  ← runs on every question
 │  Route → Retrieve → Answer   │
 │  or Generate + Run SQL       │
 └──────────────────────────────┘
                │
                ▼
         Plain-English Answer
```

---

## Business Schema Configuration

Everything specific to your database lives in one folder:

```
src/business_schema/
  capsule_definitions.py   ← all queries, priorities, and schema maps
  sql_schema.txt           ← your database creation script (reference)
  Sample_Questions.md      ← example questions for this domain
```

**To deploy against a different database:** delete this folder, drop in a new one for your new domain. The core engine does not need to change.

Inside `capsule_definitions.py` there are two lists:

- **`SCHEMA_DEFINITIONS`** — 5 entries describing how the tables relate to each other. Used to guide SQL generation.
- **`CAPSULE_DEFINITIONS`** — the full set of analytical queries the engine pre-runs and stores.

---

## Project Structure

```
streamlit_app.py            ← the web UI
requirements.txt
.env                        ← secrets (API keys, DB connection string)
README.md

src/
  app_constants.py          ← engine-wide settings (collection names, thresholds)
  config.py                 ← reads .env settings
  models.py                 ← data models shared across both pipelines
  prompts.py                ← all LLM system prompts (business-agnostic)
  database_connection.py    ← SQL Server connection + dynamic schema discovery
  llm_service.py            ← Groq API calls
  embedding.py              ← fastembed vector generation
  vector_store.py           ← Qdrant local read/write operations

  capsule_builder/          ← Pipeline 1: builds knowledge
    capsule_generator.py    ← runs SQL, extracts signal, returns GeneratedCapsule
    schema_capsule_generator.py  ← builds schema maps from SCHEMA_DEFINITIONS
    ml_enricher.py          ← anomaly scores, trend direction
    relationship_builder.py ← links capsules, generates anomaly alerts
    store_manager.py        ← orchestrates the full build + persists to Qdrant
    append_capsules.py      ← writes new capsules from UI to capsule_definitions.py

  query_engine/             ← Pipeline 2: answers questions
    query_router.py         ← classifies intent (structured / analytical / hybrid)
    context_searcher.py     ← searches Qdrant for relevant capsules
    context_packager.py     ← assembles context for the LLM
    analytical_retriever.py ← returns answer directly from capsule signal
    sql_generator.py        ← generates SQL using schema capsules as guides
    sql_executor.py         ← runs SQL against SQL Server
    sql_autofix.py          ← retries once with LLM-corrected SQL on failure
    result_summarizer.py    ← converts SQL rows into a natural-language answer
    orchestrator.py         ← coordinates all of the above end-to-end

  business_schema/          ← your domain-specific configuration (swap to change domains)
    capsule_definitions.py
    sql_schema.txt
    Sample_Questions.md

data/                       ← runtime cache (auto-created, safe to delete)
  .analytical_refresh_plan.json   ← tracks which capsules exist and when they expire
  .schema_fingerprint.json        ← fingerprint of DB schema for change detection
  .capsule_graph.json             ← capsule relationship graph for the Graph tab

qdrant_data/                ← local Qdrant vector store (auto-created)
```

---

## Setup

### 1. Install dependencies
```bash
py -3 -m pip install -r requirements.txt
```

### 2. Configure `.env`
```
GROQ_API_KEY=your_groq_key_here
SQLSERVER_CONN_STR=DRIVER={ODBC Driver 17 for SQL Server};SERVER=...;DATABASE=...;...
```

### 3. Run the app
```bash
py -3 -m streamlit run streamlit_app.py
```

### 4. Generate capsules (first time)
In the browser, go to the **Generate Capsules** tab and click **Generate All Capsules**. This runs all the SQL queries and builds the vector knowledge base. It takes a minute or two.

---

## The UI Tabs

| Tab | What it does |
|---|---|
| **Ask Question** | Type a plain-English question and get an answer |
| **Generate Capsules** | Build or refresh the engine's knowledge base |
| **Capsule Explorer** | Browse all stored capsules with filters; inspect any capsule's SQL, signal, and metadata |
| **Capsule Graph** | Visual graph of how capsules relate to each other and any anomaly alerts |
| **Telemetry** | Log of all questions asked, routes taken, confidence scores, and timing |
| **Insert Capsule** | Add a new custom SQL capsule through the UI — it gets vectorized and saved permanently |
| **Reset** | Wipe all Qdrant collections (collections sidebar updates instantly) |

---

## Generate Capsules — Three Buttons

| Button | What it does | When to use |
|---|---|---|
| **Generate All Capsules** | Full rebuild — wipes Qdrant and regenerates everything from scratch | First run, after changing `capsule_definitions.py`, or when something is broken |
| **Refresh Data** | Re-runs only the analytical SQL queries; leaves schema and related capsules untouched | Daily/routine refresh when DB data has changed but structure hasn't |
| **Schema Refresh** | Detects whether the DB schema has changed (via fingerprint) and rebuilds everything if it has | After adding or removing columns/tables in SQL Server |

---

## How a Question Gets Answered

1. **Intent detection** — Is this a structured fact lookup, an analytical insight question, a hybrid, or an operational check?
2. **Vector search** — Top matching capsules are retrieved from all three Qdrant collections.
3. **Routing decision:**
   - High-confidence analytical match → answer directly from the capsule's pre-computed signal
   - Low-confidence or structured question → generate and execute live SQL
   - Hybrid → deliver both
4. **SQL path** — LLM receives the question + schema context capsules (joining guidance) + FK relationships → generates a `SELECT` query → runs it → auto-fixes once if it fails → summarizes the rows in plain English
5. **Answer** is shown with the source SQL and the raw data expandable below it

---

## Sidebar — Collections Panel

The top of the sidebar shows live Qdrant collection counts:
```json
{
  "analytical_capsules": 42,
  "schema_context_capsules": 5,
  "related_capsules": 16
}
```
This updates automatically after Generate, Reset, or Insert operations.

---

## Capsule Explorer — Column Guide

The explorer grid shows every capsule with these columns:

| Column | Meaning |
|---|---|
| `capsule_id` | Unique identifier |
| `type` | aggregation / trend / violation / risk / pattern / operational / related / schema |
| `what` | Plain-English description of what this capsule measures |
| `how` | How the measurement is computed |
| `priority` | P1 (critical) → P4 (low) |
| `signal_method` | `rule_based` (fast formula) or `llm_summary` (AI-generated) |
| `tables` | Database tables involved |
| `tags` | Search tags |
| `staleness_trigger` | What event makes this capsule outdated |
| `ttl_hours` | How long this capsule is valid before needing a refresh |
| `anomaly` | 0.0–1.0 score; >0.7 = anomaly flag |
| `trend` | rising / falling / flat |
| `related` | Number of linked capsules |
| `expires_at` | Expiry timestamp |

Hover any cell to read full text in a tooltip. Click a capsule in the dropdown below the grid to inspect its full signal, embed text, SQL, and related links.

---

## Insert Capsule

Add your own SQL capsule through the UI:

1. Give it an ID, type, and priority
2. Write the SQL query
3. Fill in the description fields (or leave blank — the engine uses the LLM to fill gaps)
4. Click **Insert**

The capsule is immediately vectorized, stored in Qdrant, and permanently written to `capsule_definitions.py` so it survives a full rebuild.

---

## Technical Notes

- **Embedding model:** `BAAI/bge-base-en-v1.5` via fastembed, 768 dimensions
- **LLM provider:** Groq (configurable models per task in `.env`)
- **Vector database:** Qdrant running entirely locally (no cloud account needed)
- **SQL dialect:** Microsoft SQL Server — all generated queries are `SELECT` only
- **Schema discovery:** Dynamic — engine reads `INFORMATION_SCHEMA` at runtime, no hardcoded table lists
- **Data folder:** `data/` is always written to the repo root regardless of launch directory
