## Requirements

- Python 3.13+ (tested with `py -3`)
- SQL Server (default: `localhost\SQLEXPRESS`)
- ODBC Driver 17 for SQL Server
- Groq API key: https://console.groq.com/keys
- Dependencies from `requirements.txt`

## Installation

```powershell
py -3 -m pip install -r requirements.txt
```

## Run UI

```powershell
py -3 -m streamlit run streamlit_app.py
```

## Environment Variables

- Required for LLM features:
  - `GROQ_API_KEY`

- Optional model overrides:
  - `GROQ_INTENT_MODEL`
  - `GROQ_SQL_MODEL`
  - `GROQ_SUMMARY_MODEL`
  - `GROQ_SQL_FIX_MODEL`
  - `GROQ_ANALYTICAL_MODEL`

- Optional DB/vector settings:
  - `SQLSERVER_CONN_STR`
  - `EMBED_MODEL`
  - `QDRANT_PATH`

- Optional debug:
  - `SQL_DEBUG=1`

# Context-Driven Intelligence for Structured Enterprise Data

This project uses a hybrid architecture:

- Structured questions -> Text-to-SQL -> SQL Server -> concise business summary
- Analytical questions -> Vector retrieval over context capsules -> LLM reasoning

The goal is to avoid sending full relational tables to the LLM and instead use compact capsule context.

## Current High-Level Flow

1. User asks a question in Tab 1.
2. Intent routing decides structured vs analytical (or forced analytical).
3. Structured path:
   - Generate safe SQL from schema + relationships
   - Execute read-only SQL
   - Summarize result
4. Analytical path:
   - Embed question
   - Retrieve top capsules from Qdrant
   - Merge rows from retrieved hits
   - LLM-first answer with deterministic fallback
5. Capsule ingestion path:
   - Auto-generate schema-driven SQL plans (random/aggregation/distribution/trend/anomaly), or
   - Manually insert a single SQL-backed capsule
   - Embed capsule text
   - Upsert to vector store
   - Apply retention policies

## What Changed (Latest)

- Centralized operational constants into one file:
  - [app_constants.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\app_constants.py)
- Added dedicated schema-driven SQL planner:
  - [capsule_query_planner.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\capsule_query_planner.py)
- Removed console SQL plan printing; moved to UI expander in Tab 2.
- Removed `capsule_sanity_simulator` from runtime flow.
- Reset behavior is now hard clear:
  - purge local Qdrant storage + reset all collections.
- Collection name is fixed to default (`context_capsules`) in Tab 2/3/4 (not user-editable).
- Ingestion mode is fixed to `append_unique` in UI.
- Temporal plans are always generated in auto capsule generation.
- Manual insert UI simplified to minimal fields.
- Capsule retention controls in generation:
  - keep latest N random capsules per table
  - replace similar aggregation/distribution capsules

## UI Tabs (Current)

1. `Ask Question`
   - English question input
   - Optional `Force capsule retrieval`
   - Shows route, answer, confidence, and supporting capsules

2. `Ingest SQL to Vector DB`
   - Auto-generate and ingest Capsule
   - Inputs:
     - `Target capsules (count)`
     - `Rows per capsule (must be < 100)`
     - `Use LLM summaries`
     - `Max random capsules per table`
     - `Max group columns per table`
   - Shows generation result JSON
   - Shows `Planned SQL queries (N)` as last section (fresh per run)

3. `Manual Capsule Insert`
   - Minimal manual flow:
     - SQL query
     - Capsule type
     - Summary text
   - Enforces row limit: query must return `<= 100` rows

4. `Manage Capsules`
   - Load capsules
   - View capsule table
   - Delete selected capsule

5. `Reset Vector DB`
   - Full reset with confirmation
   - Purges local vector storage and resets collections

## Capsule Strategy

Capsule generation is schema-agnostic and metadata-driven:

- Random sample capsules
- Aggregation capsules
- Distribution capsules
- Trend capsules (temporal)
- Anomaly capsules (temporal spikes + numeric outliers)

Each capsule stores:

- `capsule_id`
- `capsule_type`
- `tables_used`
- `key_columns`
- `tags`
- `summary_text`
- `rows_json`
- `row_count`
- `created_at`
- `metrics`

## Retrieval Strategy

- Embed user query
- Search vector DB for top hits
- Dedupe and rerank hits
- Build answer from merged hit rows
- Return supporting capsule metadata for transparency

## Embedding + Vector Store

- Embedding library/model:
  - `fastembed`
  - default model: `BAAI/bge-base-en-v1.5`
- Vector DB:
  - Qdrant via `qdrant-client`
  - local embedded path (`qdrant_data` by default)

## Key Files

- [streamlit_app.py](C:\Users\alavu\Projects\Patent Structure Data Set\streamlit_app.py): UI tabs and actions
- [orchestrator.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\orchestrator.py): routing coordinator
- [query_router.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\query_router.py): intent classification
- [text_to_sql.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\text_to_sql.py): schema-grounded SQL generation
- [database_connection.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\database_connection.py): SQL Server access
- [capsule_query_planner.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\capsule_query_planner.py): plan generation logic
- [capsule_generator.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\capsule_generator.py): execute plans -> capsules
- [embedding.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding.py): generate+ingest pipeline
- [embedding_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding_service.py): embeddings
- [vector_store.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\vector_store.py): upsert/search/list/delete/reset
- [analytical_retriever.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\analytical_retriever.py): analytical retrieval + answering
- [result_summarizer.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\result_summarizer.py): structured result summary
- [llm_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\llm_service.py): shared Groq caller
- [app_constants.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\app_constants.py): centralized constants

## CLI

Run orchestrator:

```powershell
py -3 -m src.orchestrator "Which entities are most active?"
```

Run capsule generation/ingestion:

```powershell
py -3 -m src.embedding --generate- --limit 1000
```
