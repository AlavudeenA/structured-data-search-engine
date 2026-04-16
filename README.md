# Compliance Engine

A compliance-focused question answering and SQL planning system built for the `Compliance` SQL Server database.

It combines:
- Groq for intent detection, SQL generation, autofix, analytical synthesis, and summarization
- fastembed for embeddings
- Qdrant local mode for vector search
- SQL Server for live data retrieval
- Streamlit for the operator UI

## Architecture

```text
                +----------------------+
                |   Streamlit UI / CLI |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | pipeline2.orchestrator |
                +----------+-----------+
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+---------------+                    +-------------------+
| query_router  |                    | context_searcher  |
| intent detect |                    | Qdrant retrieval  |
+-------+-------+                    +---------+---------+
        |                                      |
        +------------------+-------------------+
                           |
                           v
                +----------------------+
                |  context_packager    |
                | primary + linked +   |
                | related + schema     |
                +----------+-----------+
                           |
              +------------+-------------+
              |                          |
              v                          v
   +---------------------+    +-----------------------+
   | analytical_retriever|    | sql_generator         |
   | answer from capsules|    | schema-guided SQL     |
   +----------+----------+    +-----------+-----------+
              |                           |
              v                           v
        +-----------+           +----------------------+
        | answer    |           | sql_executor         |
        | synthesis |           | execute + autofix    |
        +-----------+           +-----------+----------+
                                            |
                                            v
                                  +----------------------+
                                  | result_summarizer    |
                                  +----------------------+

pipeline1 builds the vector layer:
- capsule_generator
- schema_capsule_generator
- relationship_builder
- store_manager
```

## Capsule Families

### Analytical capsules
- Aggregation capsules for broker, department, employee, and security activity
- Violation capsules for active restriction overlap and repeat violators
- Trend capsules for alerts, escalations, rejections, and request volume
- Risk and pattern capsules for multi-alert entities and cross-entity risk
- Operational capsules for pending review and turnaround bottlenecks
- Distribution capsules for security concentration and restriction exposure
- Related capsules for systemic broker risk and employee risk

### Schema-context capsules
Metadata-focused planning capsules built from:
- tables
- columns
- foreign keys
- join paths
- situation patterns

These are not final evidence. They are guidance capsules used when the system needs to generate executable SQL.

## Query Flow

When you click **Run Question**, the app:

1. Detects whether the question is `structured`, `analytical`, `hybrid`, or `operational`.
2. Searches Qdrant across:
   - `analytical_capsules`
   - `schema_context_capsules`
   - `related_capsules`
3. Packages the top context into:
   - primary capsule
   - linked capsules
   - related capsules
   - schema capsules
4. Routes the question:
   - `structured` or `operational` -> live SQL
   - `analytical` with strong capsule confidence -> answer from analytical capsules
   - `analytical` with weak capsule confidence or schema-led support -> SQL planning mode
   - `hybrid` -> both capsule answer and SQL answer
5. In SQL planning mode, the LLM receives:
   - the user question
   - live schema
   - foreign-key relationships
   - top schema-context capsules
   - related risk signals when available
6. Generates SQL, executes it, retries once with autofix if needed, then summarizes actual SQL rows.

## Refresh Model

### Generate All Capsules
Full rebuild:
- analytical capsules
- schema-context capsules
- related capsules
- relationship graph
- saved analytical refresh plan
- saved schema fingerprint

### Refresh Data
Data-only refresh:
- loads the saved analytical plan
- reruns those analytical SQL definitions
- rebuilds analytical capsules only
- keeps schema-context and related collections unchanged

### Schema Refresh
Schema-aware rebuild:
- computes current schema fingerprint
- compares it with the saved fingerprint
- rebuilds analytical, schema-context, and related capsules
- regenerates the saved analytical refresh plan
- rewrites the schema fingerprint

## Project Structure

```text
streamlit_app.py
requirements.txt
.env
README.md
src/
  app_constants.py
  config.py
  models.py
  prompts.py
  database_connection.py
  llm_service.py
  embedding.py
  vector_store.py
  pipeline1/
    capsule_definitions.py
    capsule_generator.py
    schema_capsule_generator.py
    ml_enricher.py
    relationship_builder.py
    store_manager.py
  pipeline2/
    query_router.py
    context_searcher.py
    context_packager.py
    sql_generator.py
    sql_executor.py
    sql_autofix.py
    analytical_retriever.py
    result_summarizer.py
    orchestrator.py
data/
  .analytical_refresh_plan.json
  .schema_fingerprint.json
  .capsule_graph.json
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

3. Fill in `.env`:
   - `GROQ_API_KEY`
   - `SQLSERVER_CONN_STR`
4. Make sure SQL Server has the `Compliance` database and the expected tables.
5. Run the UI:

```bash
streamlit run streamlit_app.py
```

## CLI Usage

Run one question directly through the orchestrator:

```bash
py -3 -m src.pipeline2.orchestrator "Which broker has the most escalations?"
```

## Example Questions

1. Which broker dealer has the most trade requests?
2. Which department appears most active in trade requests?
3. Is buy more or sell more?
4. Which employees appear repeatedly in trade request activity?
5. Which broker dealer's trading activity is increasing over time?
6. Which requests violated an active restriction?
7. Which reviewer has the highest average turnaround time?
8. Which securities have both restriction history and alerts?
9. Which employees have zero alerts?
10. Which broker dealers are registered in the USA?

## Running Pipeline 1

Use the UI tab **Generate Capsules**.

Actions:
- **Generate All Capsules** for a full build
- **Refresh Data** for analytical-only refresh
- **Schema Refresh** for a full schema-aware rebuild

## Running Pipeline 2

Use the UI tab **Ask Question** or run the CLI module.

Pipeline 2 includes:
- query routing
- context retrieval
- context packaging
- analytical answering
- schema-guided SQL generation
- SQL autofix retry
- result summarization

## Capsule Categories

The analytical capsule set covers eight business categories:
- Volume and activity
- Violations
- Trends
- Risk patterns
- Approval workflow
- Security analysis
- Department and employee health
- Cross-entity risk

## Notes

- Qdrant runs in local mode at `QDRANT_PATH`.
- Embedding dimension is `768` using `BAAI/bge-base-en-v1.5`.
- SQL generation and SQL autofix are restricted to SQL Server `SELECT` queries.
- Schema-context capsules are planning aids, not final evidence.
