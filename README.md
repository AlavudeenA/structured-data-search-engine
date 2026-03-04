## Requirements

- Python 3.13+ (tested with `py -3`)
- SQL Server (default: `localhost\SQLEXPRESS`)
- ODBC Driver 17 for SQL Server installed
- SSMS. - Check tables scripts in dbscript.sql file. Insert dummy rows using LLM.
- Groq Key https://console.groq.com/keys (It's free).
- Install dependencies from requirements.txt

## Installation

```powershell
py -3 -m pip install -r requirements.txt
```

## Run UI

```powershell
py -3 -m streamlit run streamlit_app.py
```

## Environment variables

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
  - `SQL_DEBUG=1` (enables verbose Text-to-SQL debug logs)

# Context-Driven Intelligence for Structured Enterprise Data

This project implements a hybrid architecture for enterprise structured data:

- Structured/deterministic questions -> Text-to-SQL -> DB execution -> natural-language answer
- Analytical/non-deterministic questions -> Vector retrieval over context capsules

The core design avoids sending full tables to the LLM. Instead, the system works with compact, indexed capsules.

## Why this exists

Direct LLM reasoning over large tables causes:

- token window overflow
- high inference cost
- weaker numerical reliability
- loss of relational structure when tables are flattened into text

This solution pushes heavy computation to SQL/database and uses context capsules + retrieval for scalable analytical reasoning.

## High-level flow

1. User asks a question.
2. Intent router classifies as `structured_query` or `analytical_query`.
3. If router confidence is very low (`< 0.3`), orchestrator returns `invalid_query` with guidance.
4. Structured path:
   - English -> SQL (Groq + schema grounding)
   - Execute SQL in SQL Server
   - Summarize results in concise business English
5. Analytical path:
   - Embed user question
   - Retrieve relevant capsules from Qdrant
   - Dedup + rerank by priority
   - Reconstruct dataset context by loading all capsules with top `source_sql_hash`
   - LLM-first analytical summarization of merged rows (with deterministic fallback)
6. Ingestion path:
   - Run SQL
   - Build capsules (metadata + metrics + sampled rows)
   - Embed capsules
   - Upsert to Qdrant

## Key implementation details (latest)

- Groq integration uses **Responses API** (`/openai/v1/responses`) with robust response extraction.
- Text-to-SQL prompt is schema-driven and database-neutral (no domain-specific hardcoded SQL rules).
- SQL parser handles fenced JSON responses via `_clean_llm_json(...)`.
- Query router fallback detects comparative queries (`vs`, `versus`, `compared to/with`) as analytical.
- Capsule storage supports:
  - stable `content_hash` IDs (dedup)
  - `append_unique` and `replace_source` ingestion modes
  - metadata: `capsule_name`, `capsule_type`, `entity`, `capsule_topic`, `capsule_priority`, hashes
- Retrieval supports metadata filters (`capsule_type`, `entity`, `capsule_topic`), dedup, and priority-aware reranking.
- Analytical output includes grouped supporting metadata (single capsule vs grouped by `source_sql_hash`).
- UI is user-friendly: answer + confidence + supporting capsule info + capsule management.

## Project files and responsibilities

- [streamlit_app.py](C:\Users\alavu\Projects\Patent Structure Data Set\streamlit_app.py)
  - 4-tab UI:
    - Ask Question
    - Ingest SQL to Vector DB
    - Reset Vector DB
    - Manage Capsules (view all + delete selected capsule)

- [src/orchestrator.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\orchestrator.py)
  - Runtime coordinator for structured and analytical routes.
  - Includes low-confidence `invalid_query` guard.

- [src/query_router.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\query_router.py)
  - Groq + fallback intent detection.
  - Includes comparative-query heuristic in fallback.

- [src/intent_router_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\intent_router_service.py)
  - Normalized intent contract for orchestration.

- [src/text_to_sql.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\text_to_sql.py)
  - Schema-grounded SQL generation via Groq.
  - Safety validation and generic fallback query.
  - Optional debug logs controlled via `SQL_DEBUG`.

- [src/database_connection.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\database_connection.py)
  - SQL Server connection and read-only query execution.
  - Allows `SELECT` and `WITH` (CTE) query prefixes.
  - Dynamic schema/FK metadata retrieval.
  - Optional table auto-discovery when `ALLOWED_TABLES` is empty.

- [src/sql_executor.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\sql_executor.py)
  - SQL execution wrapper returning normalized payload.

- [src/result_summarizer.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\result_summarizer.py)
  - Business summary generation from SQL results (Groq + fallback).

- [src/embedding.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding.py)
  - Ingestion orchestrator: SQL/query/result -> capsules -> embeddings -> vector store.

- [src/capsule_builder.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\capsule_builder.py)
  - Capsule construction with:
    - numeric summaries
    - data-driven entity detection
    - insight-style summary text
    - sampled row payload (`rows_json` capped sample)
    - `max_capsules` guard (default 200)
    - capsule priority support

- [src/embedding_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding_service.py)
  - Text embedding generation (`BAAI/bge-base-en-v1.5` default).

- [src/vector_store.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\vector_store.py)
  - Qdrant upsert/search/delete/reset/list.
  - Source-level cleanup by `source_sql_hash`.
  - Source-hash scrolling (`scroll_capsules_by_sql_hash`) for full dataset reconstruction.
  - Pagination-safe delete and payload indexes for filter fields.

- [src/llm_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\llm_service.py)
  - Shared Groq caller utility used by analytical summarization.

- [src/analytical_retriever.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\analytical_retriever.py)
  - Analytical retrieval, dedup, rerank, type limits.
  - Rebuilds analytical context from all capsules sharing top `source_sql_hash`.
  - LLM-first summarization of merged rows with deterministic fallback.
  - Returns supporting capsule metadata for UI.

- [src/sql_autofix.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\sql_autofix.py)
  - Groq-based SQL typo correction during ingestion retry flow.

- [requirements.txt](C:\Users\alavu\Projects\Patent Structure Data Set\requirements.txt)
  - Python dependencies.

## UI usage

1. Tab: `Ingest SQL to Vector DB`
   - Enter SQL (`SELECT` only)
   - Optional metadata:
     - Capsule Name
     - Capsule Type
     - Entity
     - Capsule Topic
     - Capsule Priority
     - Metric Tags
   - Choose ingestion mode:
     - `append_unique`
     - `replace_source` (default)
   - Run ingestion

2. Tab: `Ask Question`
   - Ask in English
   - Optional analytical filters:
     - Capsule Type
     - Entity
     - Capsule Topic
   - Output shows:
     - Answer
     - Confidence
     - Supporting capsule name/group
     - Supporting capsule table (top retrieved capsules)

3. Tab: `Reset Vector DB`
   - Reset selected collection when needed

4. Tab: `Manage Capsules`
   - Load all capsules from a collection
   - Inspect capsule metadata table
   - Delete selected capsule by id from the loaded list

## Embedding and Vector DB stack

- Embedding technique:
  - Dense sentence embeddings (semantic vector embeddings).

- Embedding library:
  - `fastembed` (Qdrant ecosystem), used in [src/embedding_service.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding_service.py).
  - Default model: `BAAI/bge-base-en-v1.5`.
  - Called during ingestion by [src/embedding.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\embedding.py).

- Vector DB:
  - Qdrant, accessed via `qdrant-client` in [src/vector_store.py](C:\Users\alavu\Projects\Patent Structure Data Set\src\vector_store.py).
  - Running in local embedded mode using `QdrantClient(path=...)`.

- Local storage on your PC:
  - Default vector store directory: `qdrant_data` (project-relative).
  - Effective default path in this project: `C:\Users\alavu\Projects\Patent Structure Data Set\qdrant_data`.
  - You can override this with environment variable `QDRANT_PATH`.

## Optional

## CLI usage

Run orchestrator:

```powershell
py -3 -m src.orchestrator "Which entities are most active?"
```

Run ingestion:

```powershell
py -3 -m src.embedding --sql "SELECT TOP 100 * FROM SomeTable"
py -3 -m src.embedding --query "Show latest activity"
```

## Operational notes

- Retrieval quality depends on capsule coverage and refresh freshness.
- Recommended pattern:
  - scheduled refresh (daily/hourly) for critical SQL sources
  - incremental ingestion for newly introduced contexts
  - `replace_source` mode for deterministic rebuilds
