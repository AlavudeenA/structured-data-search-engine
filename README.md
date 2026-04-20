# Analytical Search Engine

## Capsule Context-Driven Intelligence — Beyond Query Generation

The simplest approach to answering questions from a database is to take a user's question, send it to an AI with the table structure, generate SQL, run it, and show the result. For ad-hoc exploration by a single user, that is the right choice.

This system is built for a different requirement — teams that need consistent, fast, and auditable answers at scale.

When many users ask similar questions throughout the day, a pure query-generation approach pays full AI cost and latency every single time. This system pre-computes answers to the most common analytical questions and serves them in milliseconds — no AI call needed for the majority of queries. Two identical questions asked on different days return the same verified answer, not whatever SQL the AI decides to write that day.

Beyond speed and consistency, query generation alone cannot detect whether a metric is unusual without historical context. Anomaly detection here uses statistical Z-scores across the full result set. Trend directions are computed from actual data. Neither requires the AI to guess. When the same entity appears in three separate analytical results simultaneously, that connection is invisible to a query tool unless someone thinks to ask exactly the right question. This system builds a relationship graph across all pre-computed results and surfaces those cross-signal risk patterns automatically.

Change awareness is an audit requirement in many domains, not a convenience. The system tracks what the data looked like at each refresh cycle, detects when something shifted, and can show a direct comparison between any two time periods.

The honest framing: it is a pre-computed analytical knowledge base with a natural language interface — the difference between a data store with pre-built intelligence versus an ad-hoc query editor. Both are valid. If the use case is one analyst occasionally exploring data, query generation is simpler and sufficient. If the use case is teams needing consistent answers, anomaly alerts, cross-entity risk signals, and auditable change history — this architecture earns its design.

The system connects to any SQLite database with no changes to the core engine. Only the `src/business_schema/` folder changes per deployment.

---

## What It Does

You type a question like:

> _"Which entity has the highest rejection rate?"_

The engine:

1. Understands what you're asking (intent detection)
2. Searches its pre-built knowledge base (vector search)
3. Checks whether the underlying data has changed since the last capsule build — and refreshes only the relevant capsules if so
4. Either answers directly from cached insights **or** writes and runs a live SQL query
5. Returns a plain-English answer with the data behind it

---

## Quick Start

### Prerequisites

| Requirement         | Details                                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Python**          | 3.11 or 3.13 recommended                                                                                    |
| **SLM provider**    | **Groq** (default — set `GROQ_API_KEY` in `.env`) **or** VS Code with GitHub Copilot (set `USE_GROQ=False`) |
| **fastembed_cache** | Pre-bundled in the repo — no internet download needed                                                       |

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

| Package                          | Purpose                                              |
| -------------------------------- | ---------------------------------------------------- |
| `streamlit`                      | Web UI                                               |
| `qdrant-client`                  | Local vector database                                |
| `fastembed`                      | Embedding model (bge-small-en-v1.5, 384-dim, ~63 MB) |
| `pydantic` / `pydantic-settings` | Settings and data models                             |
| `python-dotenv`                  | Reads `.env` file                                    |
| `numpy` / `scipy`                | Anomaly scoring and statistics                       |
| `groq`                           | Groq API client (used when `USE_GROQ=True`)          |

> **Note:** The embedding model (`BAAI/bge-small-en-v1.5`, 384-dim, ~63 MB) is pre-bundled in `fastembed_cache/` — no internet access or download required on first run.

### 4 — Create your `.env` file

Create a file named `.env` in the project root:

```env
# ── Required: choose one SLM provider ────────────────────────────────────────

# Groq (default — USE_GROQ=True in app_constants.py)
GROQ_API_KEY=your_groq_api_key_here

# VS Code LM API (only needed when USE_GROQ=False)
VSCODE_LM_PORT=50234
VSCODE_LM_SECRET=your_secret_here

# ── Optional (defaults shown) ─────────────────────────────────────────────────

# Qdrant vector store path (local folder, auto-created)
QDRANT_PATH=./qdrant_data

# SQLite database path
DB_PATH=./compliance.db

# Set to 1 to print all executed SQL to the console
SQL_DEBUG=0
```

### 5 — Set up your database (first time only)

The SQL schema and seed data script is at `src/business_schema/dbscript.sql`. It is automatically run when the app starts if the database file does not exist.

All tables include an `UpdatedAt TEXT DEFAULT (datetime('now'))` column used for data change detection.

### 6 — Choose your SLM provider

The `USE_GROQ` flag in `src/app_constants.py` controls which SLM backend is used:

```python
USE_GROQ: bool = True   # True = Groq API;  False = VS Code LM API
```

**Option A — Groq (default, `USE_GROQ=True`):**
Set `GROQ_API_KEY` in `.env`. No VS Code or extension required. Each SLM task uses the model configured for that slot in `.env` (`GROQ_INTENT_MODEL`, `GROQ_SQL_MODEL`, `GROQ_SQL_FIX_MODEL`, `GROQ_ANALYTICAL_MODEL`, `GROQ_SIGNAL_MODEL`, `GROQ_SUMMARY_MODEL`).

**Option B — VS Code LM API (`USE_GROQ=False`):**
Install the companion extension:

```bash
cd vscode-lm-extension
npm install
npm run compile
```

Then press **F5** in VS Code to run the extension host, or install the `.vsix` file if packaged. Requires **GitHub Copilot** active in VS Code.

### 7 — Run the app

```bash
py -3 -m streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

### 8 — Generate capsules (first time only)

In the browser, click the **Generate Capsules** tab → **Generate All Capsules**.
This runs all SQL queries, embeds the results, and builds the vector knowledge base.
Takes 1–3 minutes depending on DB size. The sidebar collection counts will update when complete.

---

## UI Tabs

### Ask Question

Type any plain-English question and press **Ask**. The engine classifies your intent, searches the knowledge base, checks whether the underlying data has changed, and returns a plain-English answer. If any capsules were automatically refreshed before answering, a **"Data refreshed"** banner appears. The answer includes the route taken (capsule or SQL), a confidence score, the source SQL if applicable, and expandable raw result rows.

### Generate Capsules

Builds or refreshes the engine's knowledge base. Four buttons:

| Button | What it does | When to use |
| --- | --- | --- |
| **Generate All Capsules** | Full rebuild — wipes Qdrant and regenerates everything from scratch (includes user capsules) | First run, after changing `capsule_definitions.py`, or when something is broken |
| **Refresh Data** | Re-runs only the analytical SQL queries; leaves schema and linked capsules untouched | Routine refresh when DB data changed but structure hasn't |
| **Schema Refresh** | Detects whether the DB schema changed (via fingerprint) and rebuilds everything if it has | After adding or removing columns/tables in the DB |
| **Generate Capsule Definitions** | Sends live DB schema + FK relationships to SLM; regenerates the entire `CAPSULE_DEFINITIONS` list; validated with `ast.parse()` before writing | Bootstrap a new domain or after major schema changes |

The sidebar shows live collection counts (analytical, schema, linked) and updates after each build.

### Insert Capsule

Lets you create your own analytical capsules without editing any Python files.

**Step 0 — Intent → SQL (optional):** Type a plain-English description of what you want to measure. The SLM reads the live database schema and FK relationships and generates a complete SQLite `SELECT` query pre-filled into the SQL box. You can edit it before proceeding, or skip this step and write SQL directly.

**Step 1 — Fill in the fields:** Provide a name, SQL query, a one-sentence description of what it measures, and a priority (P1–P4).

**Step 2 — Validate & Enrich:** Click **Validate & Enrich** to execute the SQL against the live database (previews up to 50 rows), then send the SQL and rows to the SLM to derive all remaining metadata — `capsule_type`, `how`, `tags`, `ttl_hours`, `signal_method`, `embed_text`, etc. An expandable panel shows the enriched metadata for review before saving.

**Step 3 — Save & Build:** Click **Save & Build Capsule**. The capsule is saved permanently to `data/user_capsules.json`, built immediately, upserted into Qdrant, and auto-linked to any related existing capsules. It is available for search in the next query.

**My Capsules:** The bottom of the tab lists all user-created capsules with a delete button per capsule. Deleting removes both the JSON definition and the Qdrant point.

### Data Activity

Compare what the data looked like across two different time periods.

1. Pick a **Baseline Period** (from/to dates) — the engine loads all unique capsule snapshots in that range, keeping the latest version per capsule per day.
2. Pick a **Comparison Period** (its own from/to dates) — loaded independently.
3. Select any combination of capsules via checkboxes on both sides.
4. Click **Compare Periods** — the SLM produces a per-capsule signal diff followed by an **Overall Summary** highlighting the highest-risk shifts.

Snapshots are created automatically on each successful targeted refresh. Folders older than 30 days are pruned automatically.

### Telemetry

A running log of every question asked in the current session. Shows the question, intent detected, route taken (capsule / SQL / fallback), confidence score, and timing. Useful for understanding how the engine is routing questions and where it is falling back to SQL.

### Capsule Explorer

Browse all capsules currently stored in Qdrant. Filter by collection, type, priority, or tags. Inspect the full payload of any capsule — SQL, signal text, result rows, anomaly score, trend direction, and linked capsule IDs. User-created capsules can be deleted directly from this tab.

### Capsule Graph

A visual graph of how capsules relate to each other. Nodes are capsules; edges are relationships (`corroborates`, `drills_down`, `aggregates_up`, `same_entity`). Linked (anomaly alert) capsules appear as distinct nodes. Use this tab to understand cross-entity risk patterns that span multiple analytical signals.

### Reset Capsules

Wipes all three Qdrant collections (analytical, schema, linked). All capsule definitions in `capsule_definitions.py` and `user_capsules.json` are preserved — only the indexed vectors are cleared. Use before a clean rebuild.

---

## Project Structure & What Each File Does

```
streamlit_app.py            ← Web UI entry point (all tabs rendered here)
requirements.txt            ← Python dependencies
.env                        ← Your config (DB path, Qdrant path, API keys) — never commit this
README.md
CLAUDE.md                   ← Agent design contract and guardrails

src/
  config.py                 ← Reads .env into a typed Settings object (pydantic-settings)
                               Groq model slots: GROQ_INTENT_MODEL, GROQ_SQL_MODEL, GROQ_SQL_FIX_MODEL,
                               GROQ_ANALYTICAL_MODEL, GROQ_SIGNAL_MODEL, GROQ_SUMMARY_MODEL
  app_constants.py          ← All shared constants: collection names, score thresholds, row limits,
                               intent names, USE_GROQ flag, history retention days
  models.py                 ← Pydantic data models: GeneratedCapsule, SchemaContextCapsule,
                               LinkedCapsule, CapsuleDefinition, BuildSummary, IntentResult, etc.
  llm_instructions.py       ← All SLM system + user prompt templates (neutral f-strings;
                               domain-specific vars injected from business_schema/domain.py)
  database_connection.py    ← SQLite connection, schema discovery via PRAGMA, FK relationships
  llm_service.py            ← Routes SLM calls to Groq or VS Code LM API via call_llm() / call_llm_json()
                               Holds module-level Groq client singleton
  embedding.py              ← Text → vector via fastembed (bge-small-en-v1.5, 384-dim)
                               Also manages schema fingerprint and refresh plan files
  vector_store.py           ← Qdrant read/write: upsert, scroll, search, delete, reset, set_payload
  data_fingerprint.py       ← Per-query data change detection: COUNT + MAX(UpdatedAt) per table → SHA-256
                               Calls save_capsule_snapshot() after every successful targeted refresh
  capsule_history.py        ← Save/load/prune dated capsule snapshots under data/capsule_history/
  user_capsules.py          ← CRUD for user-created capsule definitions in data/user_capsules.json

  capsule_builder/          ← Pipeline 1 — builds the knowledge base (runs on demand)
    store_manager.py        ← Master orchestrator: calls all generators, persists to Qdrant, saves plan
    capsule_generator.py    ← Runs each capsule's SQL, calls SLM for signal, applies ML enrichment
    user_capsule_builder.py ← Builds and upserts a single user-created capsule
                               generate_capsule_sql() — SLM writes SQL from plain-English intent
                               enrich_user_capsule_metadata() — SLM derives all metadata
                               build_single_capsule() — builds, persists, auto-links
    schema_capsule_generator.py ← Turns SCHEMA_DEFINITIONS into embedded schema context capsules
    ml_enricher.py          ← Anomaly score (numpy Z-score) and trend direction (moving average)
    relationship_builder.py ← Builds relationship graph; generates linked risk alert capsules;
                               link_user_capsule_to_existing() for bidirectional Qdrant linking
    append_capsules.py      ← Injects a new capsule dict into capsule_definitions.py
    definitions_generator.py ← SLM-driven full regeneration of CAPSULE_DEFINITIONS from live schema

  query_engine/             ← Pipeline 2 — runs on every user question
    orchestrator.py         ← Entry point: coordinates all steps, data fingerprint check, returns answer
    query_router.py         ← Classifies intent (text_to_sql / analytical); keyword fallback if SLM fails
    context_searcher.py     ← Vector searches all 3 Qdrant collections
    context_packager.py     ← Ranks and slots results; holds in-memory payload cache
    analytical_retriever.py ← Answers directly from capsule signal when confidence >= 0.65
    sql_generator.py        ← SLM generates a SELECT query from question + schema context
    sql_executor.py         ← Executes the SQL against SQLite; triggers autofix on error
    sql_autofix.py          ← One SLM repair attempt on broken SQL
    result_summarizer.py    ← Summarizes SQL result rows into a plain-English answer
    activity_comparator.py  ← Builds per-capsule diff blocks for Data Activity tab

  business_schema/          ← Domain configuration — swap this folder to change the database domain
    domain.py               ← Domain interface: DOMAIN_NAME, personas, DB_SCRIPT_PATH, SQL rules,
                               embed style, and capsule coverage requirements
    capsule_definitions.py  ← CAPSULE_DEFINITIONS list (SQL + metadata per capsule) + SCHEMA_DEFINITIONS
    dbscript.sql            ← Full DDL + sample data (all tables include UpdatedAt column)
    Sample_Questions.md     ← Example questions that work well with this schema

data/                       ← Runtime cache files (auto-created, safe to delete and rebuild)
  .analytical_refresh_plan.json  ← Last-run capsule plan (used by Refresh Data)
  .schema_fingerprint.json       ← Hash of DB schema (checked at startup)
  .capsule_graph.json            ← Capsule relationship graph (used by Capsule Graph tab)
  .data_fingerprint.json         ← Hash of table row counts + MAX(UpdatedAt) (checked per query)
  user_capsules.json             ← User-created capsule definitions (persisted across rebuilds)
  capsule_history/               ← Dated snapshots for the Data Activity tab
    MM-DD-YYYY/                  ← One folder per day a data change was detected
      {capsule_id}.json          ← Full capsule payload; same-day overwrites keep latest state

qdrant_data/                ← Local Qdrant vector store (auto-created)
fastembed_cache/            ← Pre-bundled bge-small-en-v1.5 model (~63 MB, no download needed)
```

---

## How Linked Capsules Are Established

Linked capsules surface risk patterns that span multiple independent analytical signals. For example, if the same entity appears in a high-anomaly aggregation capsule and a separate trend capsule, that co-occurrence is automatically detected and stored as a navigable connection — without anyone needing to write a specific query for it.

### Phase 1 — Relationship Graph

After all analytical capsules are generated, `build_graph()` maps how they relate to each other.

**Explicit edges** are declared in `capsule_definitions.py` via `linked_capsule_ids` and `relationship_types`.

**Inferred edges** are computed automatically at build time:

| Condition | Inferred relationship |
| --- | --- |
| Two capsules use the same tables but have different capsule types | `same_entity` |
| Capsule A's tables are a strict subset of capsule B's tables | `drills_down` |
| Capsule B's tables are a strict subset of capsule A's tables | `aggregates_up` |
| Two capsules share ≥ 2 tags, or a shared entity value appears in both result sets | `corroborates` |

Inferred edges are written back to the in-memory capsule objects before persisting to Qdrant, so the full relationship set is stored in the payload. The complete graph is saved to `data/.capsule_graph.json` and visualized in the **Capsule Graph** tab.

### Phase 2 — Anomaly Alert Capsules

Any capsule whose `anomaly_score >= 0.7` (Z-score on numeric result columns) automatically triggers creation of a linked alert capsule:

1. The first entity value from the capsule's top result row becomes the `entity_name`
2. An SLM writes a 2-sentence risk alert tying the anomaly to that entity
3. A `LinkedCapsule` is created with `risk_level = "critical"` (score ≥ 0.8) or `"high"`
4. The alert is embedded and stored in the `linked` Qdrant collection

These alert capsules appear in search results when a question is about that entity, surfacing the risk signal without the user needing to know it exists.

### Phase 3 — User Capsule Auto-Linking

When a user saves a capsule via the Insert Capsule tab, the same relationship inference runs against all existing analytical capsules (payload-only scan, no re-embedding). For each match:

- The existing capsule's `linked_capsule_ids` is updated in Qdrant via `set_payload` — no re-embedding needed
- The new capsule's links are flushed once at the end
- New edges are appended to `data/.capsule_graph.json`

---

## Where SLM Calls Are Made

The engine makes SLM calls in 5 workflows. Every call routes through `call_llm()` in `llm_service.py` — the same function regardless of provider (Groq or VS Code LM).

### Workflow 1 — Generate All Capsules (runs once, on demand)

| Call | Purpose | Model slot | Rows sent to SLM | If SLM fails |
|---|---|---|---|---|
| Signal — `llm_summary` capsules | SLM reads result rows and writes a 2–3 sentence plain-English insight describing what the data means | `groq_signal_model` | 20 rows | Falls back to rule-based signal automatically (top value + concentration % + trend) |
| Signal — `sample` capsules | SLM narrates patterns observed across a random wide-join sample (no numeric column to aggregate) | `groq_signal_model` | 15 rows | Falls back to a static placeholder string |
| Linked alert signal | For every capsule with anomaly score ≥ 0.7, SLM writes a 2-sentence risk alert naming the entity and the anomaly | `groq_signal_model` | No rows — uses signal text only | Alert capsule is created but stored with an empty signal |

> These calls fire **once per capsule** during a full build (~40 capsules = up to 40 SLM calls). Rule-based capsules (`signal_method = "rule_based"`) skip the SLM entirely — no call is made.

---

### Workflow 2 — User Asks a Question

| Call | Purpose | Model slot | Rows sent to SLM | If SLM fails |
|---|---|---|---|---|
| Intent detection | Classifies the question as `text_to_sql` or `analytical` | `groq_intent_model` (fast) | No rows | Keyword fallback: scans for trend/anomaly/pattern words; defaults to `text_to_sql` |
| SQL generation | Writes a SQLite SELECT query from the question + schema context + db_metadata | `groq_sql_model` (strong) | No rows | Returns no SQL; answer fails |
| SQL route explanation | Writes 2 sentences explaining why SQL was used and which tables mattered | `groq_summary_model` | No rows | Falls back to a static explanation string |
| SQL autofix | Fixes a broken SQL query using the error message — fires only if SQL execution fails | `groq_sql_fix_model` | No rows | Returns no fix; answer fails |
| Result summarizer | Converts raw SQL result rows into a 2–3 sentence plain-English answer | `groq_summary_model` | 25 rows | Falls back to a mechanical row preview string |
| Analytical answer | Writes an answer from pre-computed capsule signals — fires only on the capsule path | `groq_analytical_model` | No rows (uses signal text) | Falls back to first 800 chars of combined signal text |

> A typical `text_to_sql` question makes **4 SLM calls** (intent → SQL → explanation → summarize). An `analytical` capsule hit makes **2** (intent → answer). SQL error adds 1 more (autofix).

---

### Workflow 3 — Insert Capsule

| Call | Purpose | Model slot | Rows sent to SLM | If SLM fails |
|---|---|---|---|---|
| SQL generation from intent | SLM reads the live schema + FK relationships + db_metadata and writes an analytical SELECT query | `groq_sql_model` | No rows | Returns empty string; SQL box stays blank |
| Metadata enrichment | SLM derives `capsule_type`, `how`, `tags`, `ttl_hours`, `signal_method`, `embed_text` from the SQL and its result rows | default | 10 rows | Falls back to safe defaults (type=aggregation, ttl=24h, etc.) |

---

### Workflow 4 — Generate Capsule Definitions (bootstrap)

| Call | Purpose | Model slot | Rows sent to SLM | If SLM fails |
|---|---|---|---|---|
| Full definitions regen | SLM receives the live schema + FK relationships + db_metadata and regenerates the entire `CAPSULE_DEFINITIONS` list (40+ capsules) | `groq_sql_model` (strong, 8 000 max tokens) | No rows | Returns raw output; validated with `ast.parse()` before writing — rejected if invalid Python |

---

### Workflow 5 — Data Activity (Compare Periods)

| Call | Purpose | Model slot | Rows sent to SLM | If SLM fails |
|---|---|---|---|---|
| Period comparison | SLM receives capsule signal pairs from two time windows and writes a per-capsule diff + an Overall Summary | `groq_summary_model` | 3 rows per capsule (signal preview only) | No fallback — empty response shown |

---

### Row Limits at a Glance

| Stage | Rows fetched from DB | Rows stored in Qdrant | Rows sent to SLM |
|---|---|---|---|
| Capsule SQL execution (build) | 50 | 50 | — |
| SLM signal — `llm_summary` capsules | 50 | 50 | 20 |
| SLM signal — `sample` capsules | 50 | 50 | 15 |
| User capsule metadata enrichment | 500 (DB default) | 50 | 10 |
| Result summarizer (text_to_sql) | 500 (DB default) | — | 25 |
| SQL rows shown in UI | — | — | — (UI shows 50) |
| Data Activity comparison | — | — | 3 per capsule |

---

## Technical Notes

- **Embedding model:** `BAAI/bge-small-en-v1.5` via fastembed — 384 dimensions, bundled in `fastembed_cache/` (~63 MB, no download needed). Changing the model or dimension requires a full Qdrant wipe and rebuild.
- **SLM provider toggle:** `USE_GROQ` in `src/app_constants.py` — `True` (default) uses Groq API; `False` uses VS Code LM API (GitHub Copilot via extension). Switching the flag is the only change needed — all call sites use `call_llm()` / `call_llm_json()` which route automatically.
- **Groq model slots:** Each SLM task uses its own model slot (`GROQ_INTENT_MODEL`, `GROQ_SQL_MODEL`, `GROQ_SQL_FIX_MODEL`, `GROQ_ANALYTICAL_MODEL`, `GROQ_SIGNAL_MODEL`, `GROQ_SUMMARY_MODEL`). Defaults are set in `config.py` and can be overridden in `.env`.
- **Vector database:** Qdrant running entirely locally — no cloud account needed.
- **SQL database:** SQLite — no server or ODBC driver required.
- **SQL dialect:** SQLite only. All generated queries are `SELECT` or `WITH` only. `LIMIT` not `TOP`, `COALESCE` not `ISNULL`, no `dbo.` prefix, no `SELECT *`, always `ORDER BY`, always explicit `AS` aliases.
- **Schema discovery:** Fully dynamic — engine reads `PRAGMA table_info` and `PRAGMA foreign_key_list` at runtime; no table names hardcoded in engine files.
- **Data change detection:** Per-query `COUNT(*) + MAX(UpdatedAt)` per table → SHA-256 fingerprint. Only the 3–5 capsules relevant to the current question are refreshed (upsert, not clear).
- **Schema change detection:** At app startup, a PRAGMA-based schema fingerprint is compared against the saved hash. Full rebuild triggered automatically if different.
- **UpdatedAt requirement:** All tables must have an `UpdatedAt` column updated on every row change for data fingerprinting to detect modifications — not just insertions.
- **Capsule snapshot history:** On every successful targeted refresh, the refreshed capsules are saved to `data/capsule_history/MM-DD-YYYY/`. Same-day rebuilds overwrite the earlier snapshot for that day. Folders older than 30 days are pruned automatically.
- **Anomaly & trend detection:** Pure deterministic statistics via `numpy` — Z-scores for anomaly scoring, moving averages for trend direction. Minimum 4 rows required; returns `0.0` / `"flat"` for insufficient data. No SLM involved in data value scoring.
- **Domain deployment:** Replace `src/business_schema/` with a new folder implementing the same interface (`domain.py`, `capsule_definitions.py`, `dbscript.sql`). No engine file changes required.
- **Data folder:** Always resolved from `Path(__file__)` anchor — consistent regardless of launch directory.
