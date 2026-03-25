# Structured Data Search Engine

This project is a hybrid question-answering system for structured enterprise data.

It combines:
- Text-to-SQL for direct, executable data questions
- Vector retrieval over capsules for analytical context
- Schema-context guidance to improve SQL generation when analytical retrieval is weak

The goal is to answer business questions without sending full tables to the LLM, while still falling back to live SQL when exact or up-to-date answers are needed.

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

Required for LLM features:
- `GROQ_API_KEY`

Optional model overrides:
- `GROQ_INTENT_MODEL`
- `GROQ_SQL_MODEL`
- `GROQ_SUMMARY_MODEL`
- `GROQ_SQL_FIX_MODEL`
- `GROQ_ANALYTICAL_MODEL`

Optional DB and vector settings:
- `SQLSERVER_CONN_STR`
- `EMBED_MODEL`
- `QDRANT_PATH`

Optional debug:
- `SQL_DEBUG=1`

## How It Works

### Run Question

When the user clicks `Run Question`, the app:

1. Detects whether the question is structured or analytical.
2. For structured questions:
   - generates SQL from live schema and relationships
   - executes the SQL
   - summarizes the result
3. For analytical questions:
   - retrieves the most relevant capsules from Qdrant
   - if analytical capsules are strong enough, answers from capsule context
   - if retrieval is weak, empty, or the top guidance is `schema_context`, it switches to SQL planning mode
4. In SQL planning mode, the LLM receives:
   - the user question
   - live schema
   - foreign-key relationships
   - top retrieved `schema_context` capsules
5. The system generates SQL, executes it, and summarizes the result.
6. If SQL fails with an error like `Invalid column name ...`, it retries once using an autofix prompt that includes the SQL error plus schema guidance.

### Capsule Types

The system uses two capsule families.

Analytical capsules:
- `random_sample`
- `aggregation`
- `distribution`
- `trend`
- `anomaly`
- `summary`

Schema-context capsules:
- metadata-focused planning capsules built from:
  - tables
  - columns
  - foreign keys
  - join paths
  - situation patterns

Schema-context capsules help the LLM choose:
- which tables to join
- which columns matter
- which filters are typical
- which SQL pattern fits the question

## UI Tabs

### 1. Ask Question

Use this tab to ask questions in plain English.

The UI shows:
- route
- detected intent
- answer
- generated SQL when SQL was executed
- SQL reason
- returned rows
- supporting capsules
- SQL autofix notice when a retry was needed

### 2. Generate Capsules

This tab now has three actions:

- `Generate Capsules`
  - full build
  - generates analytical capsules and schema-context capsules
  - saves the analytical refresh plan
  - saves the schema fingerprint

- `Refresh Capsules`
  - data refresh only
  - deletes old analytical capsules
  - reloads the saved analytical SQL plan set
  - reruns that exact plan set against current data
  - rebuilds and re-indexes analytical capsules
  - keeps `schema_context` untouched

- `Schema Refresh`
  - schema-aware full rebuild
  - detects schema changes using a saved schema fingerprint
  - deletes old analytical capsules and old `schema_context` capsules
  - regenerates fresh `schema_context` capsules from the current schema
  - regenerates the analytical refresh plan for the current schema
  - reruns that new analytical plan
  - rebuilds and re-indexes everything together

### 3. Insert Capsule

Manual SQL-backed capsule insertion.

Used for:
- custom SQL
- custom capsule type
- manual summary text

### 4. Manage Capsules

Use this tab to:
- load capsules
- inspect stored capsule metadata
- delete a selected capsule

### 5. Reset Vector DB

Use this tab to clear local vector storage and reset collections.

## Refresh Model

The app separates data refresh from schema refresh.

### Data Refresh

Use `Refresh Capsules` when:
- row values changed
- counts changed
- trends changed
- schema did not change

This keeps analytical capsules current while preserving schema-context capsules.

### Schema Refresh

Use `Schema Refresh` when:
- a new table was added
- a column was added or removed
- a foreign key changed
- table relationships changed

This rebuilds both capsule families so `Run Question` uses one consistent semantic layer.

## What Gets Sent To the LLM in SQL Planning Mode

When the system switches from analytical retrieval to SQL planning, it sends:

- live database schema
- foreign-key relationships
- top retrieved `schema_context` capsules

For each retrieved schema-context capsule, the prompt includes:
- capsule name
- summary
- tables
- relevant columns
- recommended joins
- exact join columns
- recommended filters
- example questions
- SQL template

## Storage

Vector store:
- Qdrant via `qdrant-client`
- local path: `qdrant_data`

Embeddings:
- `fastembed`
- default model: `BAAI/bge-base-en-v1.5`

Refresh metadata files:
- analytical refresh plan: `.analytical_capsule_refresh_plan.json`
- schema fingerprint: `.schema_capsule_fingerprint.json`

## Key Files

- `streamlit_app.py`: Streamlit UI
- `src/orchestrator.py`: main routing and execution flow
- `src/query_router.py`: intent classification
- `src/text_to_sql.py`: SQL generation with schema-context guidance
- `src/sql_executor.py`: SQL execution and autofix retry
- `src/sql_autofix.py`: SQL correction on execution failure
- `src/database_connection.py`: SQL Server access and schema metadata
- `src/capsule_query_planner.py`: analytical SQL plan generation
- `src/capsule_generator.py`: build analytical capsules from SQL plans
- `src/schema_capsule_generator.py`: build schema-context capsules
- `src/embedding.py`: generation, refresh, ingestion, plan persistence
- `src/vector_store.py`: Qdrant operations
- `src/analytical_retriever.py`: analytical retrieval and answer building
- `src/result_summarizer.py`: SQL result summarization
- `src/llm_service.py`: shared Groq helper
- `src/app_constants.py`: shared constants

## CLI

Run orchestrator:

```powershell
py -3 -m src.orchestrator "Which broker-dealer has the most trade requests?"
```

Run capsule generation:

```powershell
py -3 -m src.embedding --generate- --limit 1000
```
