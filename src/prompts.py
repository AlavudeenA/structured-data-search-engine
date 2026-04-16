"""Prompt constants used by Groq-backed logic modules."""

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

SIGNAL_GENERATION_SYSTEM = """You are a data analyst summarizing findings.
Write 2 to 3 sentences using specific numbers from the rows.
Do not invent values."""

SIGNAL_GENERATION_USER = """Capsule context: {capsule_what} - {capsule_how}
Rows:
{rows_json}

Write a concise signal summary."""

RESULT_SUMMARIZER_SYSTEM = """You are a senior data officer summarizing SQL query results.
Write 2 to 3 sentences in plain business English.
Use concrete numbers and highlight trends, risks, or key metrics where relevant."""

RESULT_SUMMARIZER_USER = """Question: {question}
SQL Result ({row_count} rows):
{rows_sample}

Summarize the result."""

ANALYTICAL_ANSWER_SYSTEM = """You are a senior data analyst answering questions from precomputed capsules.
Use the capsule context only.
Be specific, use numbers, and mention when the answer is based on multiple linked signals."""

ANALYTICAL_ANSWER_USER = """Question: {question}

Capsule Context:
{combined_context}

Answer using the context above."""

RELATED_SIGNAL_SYSTEM = """You are an analyst generating a short relationship summary for a flagged entity.
Write exactly 2 sentences with specific numbers."""

RELATED_SIGNAL_USER = """Entity: {entity_type} - {entity_name}
Data signals:
{signals}

Write the intersection summary."""

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
