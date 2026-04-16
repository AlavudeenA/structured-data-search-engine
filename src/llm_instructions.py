"""
All LLM instruction templates for the engine. Two pipelines use this file:

  CAPSULE BUILDER  (Generate Capsules — runs offline)
    SIGNAL_GENERATION  → narrate SQL result rows into a plain-English signal
    RELATED_SIGNAL     → write a risk alert for a cross-capsule flagged entity

  QUERY ENGINE  (Ask Question — runs on every user question)
    INTENT_DETECTION   → classify the question before any retrieval
    SQL_GENERATION     → write a SELECT query using schema capsules as guidance
    SQL_AUTOFIX        → fix a broken SQL query from the database error
    ANALYTICAL_ANSWER  → answer using pre-computed capsule signals (no SQL)
    RESULT_SUMMARIZER  → summarize live SQL result rows into plain English
    SQL_REASON         → explain why SQL was chosen and what schema guidance helped
"""

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 1 — CAPSULE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

# Stage: Signal Writing (capsule_generator.py)
# When: after SQL executes and returns rows, if signal_method = "llm_summary"
# Job: compress raw SQL result rows into a 2-3 sentence human-readable insight
SIGNAL_GENERATION_SYSTEM = """You are a data analyst summarizing findings.
Write 2 to 3 sentences using specific numbers from the rows.
Do not invent values."""

SIGNAL_GENERATION_USER = """Capsule context: {capsule_what} - {capsule_how}
Rows:
{rows_json}

Write a concise signal summary."""

# ─────────────────────────────────────────────────────────────────────────────

# Stage: Risk Alert Writing (relationship_builder.py)
# When: after all capsules are built and anomaly scores are computed,
#       for each entity (broker, employee, security) that appears across
#       multiple high-anomaly capsules
# Job: write a 2-sentence risk alert tying the entity's cross-capsule signals together
RELATED_SIGNAL_SYSTEM = """You are an analyst generating a short relationship summary for a flagged entity.
Write exactly 2 sentences with specific numbers."""

RELATED_SIGNAL_USER = """Entity: {entity_type} - {entity_name}
Data signals:
{signals}

Write the intersection summary."""

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 2 — QUERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Stage: Intent Classification (query_router.py)
# When: first step on every user question, before any retrieval
# Job: decide whether the question needs structured SQL, analytical capsule retrieval,
#      a hybrid of both, or a live operational lookup
INTENT_DETECTION_SYSTEM = """You are a smart data assistant query classifier.
Classify the user question into exactly one intent and return valid JSON only, with no markdown.

Intents:
- structured: exact count, ranking, lookup, filter, date range, list, show, display
- analytical: trend, pattern, anomaly, comparison, "is X increasing", "which X is unusual", insight
- hybrid: question requires both a data lookup and trend or pattern analysis
- operational: live status needed, such as pending requests, open items, or active flags

Output format:
{
  "intent": "structured|analytical|hybrid|operational",
  "confidence": 0.0,
  "structured_parts": [],
  "analytical_parts": [],
  "reasoning": ""
}"""

INTENT_DETECTION_USER = "Question: {question}"

# ─────────────────────────────────────────────────────────────────────────────

# Stage: SQL Generation (sql_generator.py)
# When: question is structured/operational, or analytical confidence is too low
#       to answer from capsules alone
# Job: write a valid SQL Server SELECT query using schema context capsules
#      (table join paths, FK relationships, domain rules) as guidance
SQL_GENERATION_SYSTEM = """You are a senior SQL Server expert generating analytical queries.

Rules:
1. Output raw SQL only.
2. Use SQL Server syntax only.
3. Only use tables and columns present in the schema.
4. Use explicit aliases with AS for every selected column.
5. Always include ORDER BY.
6. Never use SELECT *.
7. Only generate SELECT or WITH queries.

Schema:
{schema}

Foreign Keys:
{fk_relationships}

Related Risk Context:
{related_context}

Schema Capsule Context (Use this to understand domain rules!):
{capsule_context}"""

SQL_GENERATION_USER = "Generate SQL for: {question}"

# ─────────────────────────────────────────────────────────────────────────────

# Stage: SQL Autofix (sql_autofix.py)
# When: sql_executor.py receives a database error from SQL Server
# Job: inspect the broken SQL + the exact error message and return a corrected query
#      The engine will retry exactly once with the fixed SQL
SQL_AUTOFIX_SYSTEM = """You are a SQL Server expert fixing a broken query.
Output only corrected SQL.

Rules:
- Keep the original business intent.
- Fix invalid column names, wrong joins, missing aliases, or SQL Server syntax issues.
- Use only schema-valid tables and columns.
- Never use SELECT *.
- Always include ORDER BY.

Schema:
{schema}

Foreign Keys:
{fk_relationships}"""

SQL_AUTOFIX_USER = """Original SQL:
{original_sql}

Error:
{error_message}

Return only corrected SQL."""

# ─────────────────────────────────────────────────────────────────────────────

# Stage: Capsule Answer (analytical_retriever.py)
# When: question intent is analytical AND the top-matching capsule confidence
#       is above the threshold (no SQL needed)
# Job: write a final answer using only the pre-computed capsule signal text —
#      the LLM narrates from capsule context, not from live database rows
ANALYTICAL_ANSWER_SYSTEM = """You are a senior data analyst answering questions from precomputed capsules.
Use the capsule context only.
Be specific, use numbers, and mention when the answer is based on multiple linked signals."""

ANALYTICAL_ANSWER_USER = """Question: {question}

Capsule Context:
{combined_context}

Answer using the context above."""

# ─────────────────────────────────────────────────────────────────────────────

# Stage: SQL Result Summarization (result_summarizer.py)
# When: SQL path was taken, query ran successfully, rows returned
# Job: convert the raw SQL result rows into a plain-English business answer
RESULT_SUMMARIZER_SYSTEM = """You are a senior data officer summarizing SQL query results.
Write 2 to 3 sentences in plain business English.
Use concrete numbers and highlight trends, risks, or key metrics where relevant."""

RESULT_SUMMARIZER_USER = """Question: {question}
SQL Result ({row_count} rows):
{rows_sample}

Summarize the result."""

# ─────────────────────────────────────────────────────────────────────────────

# Stage: SQL Planning Explanation (sql_generator.py)
# When: after SQL is generated, to show the user why SQL was chosen
# Job: write 2 sentences explaining the routing decision and what schema guidance was used
SQL_REASON_SYSTEM = """You explain SQL planning decisions for an analytics app.
Write 2 concise sentences explaining why SQL was used and what schema guidance mattered.
Mention relevant tables, joins, and rules when useful."""

SQL_REASON_USER = """Question: {question}
Intent payload:
{intent_payload}

Schema capsule guidance:
{schema_capsules}

Generated SQL:
{sql}

Explain the planning choice."""

# ─────────────────────────────────────────────────────────────────────────────

# Stage: Capsule Definition Regeneration (UI — "AI Rebuild Definitions" button)
# When: user clicks the button in Generate Capsules tab
# Job: given the live DB schema + FK relationships + a format example,
#      produce a full CAPSULE_DEFINITIONS list covering all useful analytical angles
#      for that schema — aggregation, trend, violation, risk, pattern, operational.
CAPSULE_REGEN_SYSTEM = """You are an expert data engineer and business analyst.
Given a SQL Server database schema, generate a comprehensive Python list called CAPSULE_DEFINITIONS.
Each capsule must be a dict with these exact keys:

  capsule_id         – unique snake_case string
  capsule_type       – one of: aggregation, trend, violation, risk, pattern, operational, distribution
  priority           – one of: P1, P2, P3, P4
  what               – one plain English sentence: what this capsule measures
  how                – one plain English sentence: how it is computed
  sql                – a valid SQL Server SELECT query (no SELECT *, always ORDER BY, always AS aliases)
  signal_method      – "rule_based" for aggregations/counts, "llm_summary" for complex patterns
  embed_text_template – rich paragraph describing what questions this answers + "Finding: {signal}"
  ttl_hours          – integer: how many hours until this becomes stale
  tags               – list of lowercase strings
  tables_used        – list of table names the SQL touches
  key_columns        – list of the most important output column aliases
  staleness_trigger  – short string: what event makes this capsule outdated
  related_capsule_ids – list of capsule_ids this naturally relates to (can be empty)
  relationship_types  – list matching related_capsule_ids (corroborates, drills_down, same_entity, aggregates_up)

Rules:
- Cover all major tables — do not skip any table in the schema.
- Include at least 2 capsules per major table.
- Cover a mix of all capsule_types.
- P1 = immediate action / violation / risk. P2 = trend / monitoring. P3-P4 = operational / informational.
- SQL must be valid SQL Server syntax. Use JOIN not comma joins. No STRING_AGG(DISTINCT ...).
- Output ONLY the Python list literal. No markdown. No explanation. No variable assignment."""

CAPSULE_REGEN_USER = """Database schema (tables and columns):
{schema}

Foreign key relationships:
{fk_relationships}

Format example (follow this structure exactly):
{format_example}

Generate a comprehensive CAPSULE_DEFINITIONS list for this schema."""

