# Problem and Solution

## Problem

Modern LLMs are strong at natural language, but they struggle with large enterprise structured data (millions of rows, many columns).

Key issues:

- Token limits:
  - Full table serialization does not fit into LLM context windows.
- High cost:
  - Repeatedly sending large raw data to LLMs is expensive and slow.
- Loss of structure:
  - Flattened text loses relational meaning (joins, hierarchy, intersections).
- Weak numerical reliability:
  - Pure LLM reasoning is less reliable for aggregation/statistical questions.
- Poor scalability:
  - Direct LLM-over-table querying does not scale for production workloads.

## Our Solution

We use a **hybrid context-driven architecture**:

- Structured/deterministic questions -> **Text-to-SQL path**
- Analytical/non-deterministic questions -> **Vector retrieval over context capsules**

This avoids passing full raw tables to the LLM.

## How We Solve It

1. Intent routing

- Classify user query into `structured_query` or `analytical_query` (Groq + fallback rules).

2. Structured path (accurate, verifiable)

- Convert English question to SQL using schema + relationships.
- Execute SQL on SQL Server.
- Summarize verified SQL result in business English.

3. Context capsule ingestion

- Execute SQL once.
- Convert results into compact capsules (summary text + metrics + sampled rows + metadata).
- Generate embeddings.
- Store vectors + metadata in Qdrant.

4. Analytical path (scalable retrieval)

- Embed user query.
- Retrieve relevant capsules from vector DB.
- Deduplicate + rerank hits.
- Reconstruct broader context using shared `source_sql_hash`.
- Generate final analytical answer (LLM-first with deterministic fallback).

## Why This Works

- Reduces token usage by using compact capsules instead of full tables.
- Preserves structure through SQL, metadata, joins, and source grouping.
- Improves numerical trust by relying on database-computed results.
- Controls cost and latency via reusable indexed context in Qdrant.
- Scales better as data grows because retrieval is targeted, not full-table.

## Current Scope

Implemented now:

- Intent routing
- Text-to-SQL with safety validation
- SQL execution + summarization
- Capsule generation + embedding + Qdrant storage
- Analytical retrieval with dedup and grouped source reconstruction
- Streamlit UI for ask / ingest / manage / reset

Planned next (optional):

- Richer automated trend/anomaly/correlation capsule generation
- Scheduled refresh pipelines and incremental ingestion jobs
- Additional governance and evaluation metrics
