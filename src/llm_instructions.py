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
# Job: given the live DB schema + FK relationships + two format examples,
#      produce a full CAPSULE_DEFINITIONS list (35+ capsules, 8 categories)
#      for the Compliance database with correct SQL Server syntax and
#      compliance domain rules baked into every query.
CAPSULE_REGEN_SYSTEM = """You are a senior compliance data engineer generating analytical capsule \
definitions for a SQL Server Compliance database.

═══════════════════════════════════════
COMPLIANCE DOMAIN RULES — apply in every relevant SQL
═══════════════════════════════════════
1. ACTIVE RESTRICTION: EndDate IS NULL means permanently active (no end set).
   Check: rs.EndDate IS NULL OR rs.EndDate >= GETDATE()
   Safe combined form: ISNULL(rs.EndDate, '9999-12-31') >= GETDATE()

2. VIOLATION DATE OVERLAP — trade happened while restriction was active:
   JOIN condition: tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')

3. ApprovalWorkflow.ReviewerID references Employee.EmployeeID.
   The REVIEWER is an employee (compliance/risk staff), NOT the trade requester.

4. TurnaroundDays is already stored in ApprovalWorkflow — do NOT recalculate it.

5. Escalated = sent to senior compliance review. It is NOT a rejection.

6. Repeat violators = employees with COUNT(ComplianceAlert) >= 2.

7. High severity = ca.Severity IN ('Critical', 'High')

8. Unresolved alerts = ca.Status IN ('Open', 'Investigating')

═══════════════════════════════════════
SQL RULES — strictly enforced
═══════════════════════════════════════
- No SELECT *. Every column must have an AS alias.
- Every query must have ORDER BY.
- SQL Server 2019+ syntax: TOP N, FORMAT(date_col, 'yyyy-MM'), ISNULL(), DATEDIFF(), DATEADD(), CAST().
- Use explicit JOIN ... ON (no comma joins, no implicit joins).
- Do NOT use STRING_AGG(DISTINCT ...) — invalid in SQL Server; omit DISTINCT from STRING_AGG.
- Monthly grouping: FORMAT(date_col, 'yyyy-MM')
- Weekly grouping: FORMAT(date_col, 'yyyy-') + CAST(DATEPART(ISO_WEEK, date_col) AS VARCHAR)
- Percentages: CAST(100.0 * numerator / NULLIF(denominator, 0) AS DECIMAL(5,2))
- Only SELECT or WITH queries. No INSERT, UPDATE, DELETE, DDL.
- All tables listed in the schema are in dbo schema. No schema prefix needed.

═══════════════════════════════════════
CAPSULE STRUCTURE — every capsule must have ALL keys
═══════════════════════════════════════
  capsule_id           unique snake_case string (no spaces, no hyphens)
  capsule_type         aggregation | trend | violation | risk | pattern | operational | distribution
  priority             P1 (violation/critical) | P2 (monitoring/trend) | P3 (operational) | P4 (info)
  what                 one sentence: what entity/metric this capsule measures
  how                  one sentence: how it is computed or joined
  sql                  complete valid SQL Server SELECT query — single-line string, no triple-quotes
  signal_method        "rule_based" for counts/aggregations | "llm_summary" for complex patterns
  embed_text_template  compliance officer search query style — see requirements below
  ttl_hours            P1 violation=6, pending/operational=2, trend=12, other=24-48 (integer)
  tags                 list of lowercase snake_case strings (5-10 per capsule)
  tables_used          list of exact table names the SQL queries
  key_columns          list of the most important output column AS aliases
  staleness_trigger    short string: what event/time makes this stale
  related_capsule_ids  list of capsule_ids this relates to (empty list [] if none applies yet)
  relationship_types   list matching related_capsule_ids:
                         corroborates | drills_down | aggregates_up | same_entity

═══════════════════════════════════════
EMBED_TEXT_TEMPLATE REQUIREMENTS
═══════════════════════════════════════
- Write as if a compliance officer is typing a natural search query.
- First sentence: what this capsule is about and its compliance relevance.
- Middle section: list 3-5 specific questions this capsule answers (not bullet points, embed in prose).
- Include domain synonyms inline:
    violation = breach = non-compliant = policy break
    restriction = ban = blackout = insider list = watch list
    alert = incident = compliance flag = issue
- End with exactly this text (no variation): "Finding: {signal}"
- Total length: 150-250 words.

═══════════════════════════════════════
COVERAGE REQUIREMENTS — generate at least these capsules
═══════════════════════════════════════
CATEGORY 1 — Volume & Activity (type=aggregation): generate 5 capsules
  trade requests by broker dealer (total, approved, rejected, escalated, rejection_rate_pct)
  trade requests by department
  trade requests by security symbol TOP 10
  trade requests by trade type BUY vs SELL
  monthly request volume last 6 months

CATEGORY 2 — Violations (type=violation, ALL priority=P1): generate 6 capsules
  trades made while security was on active restriction (date overlap join) — TTL=6h
  violations by restriction type (Blackout/Insider List/Watch List)
  violations by broker dealer
  violations by department
  repeat violators: employees with 2+ alerts
  active restrictions with trade attempts in last 30 days — TTL=6h

CATEGORY 3 — Trends (type=trend): generate 5 capsules
  monthly alert volume last 6 months by severity
  weekly trade request volume last 8 weeks
  monthly rejection rate trend by broker dealer
  escalation trend by department monthly
  alert severity trend over time

CATEGORY 4 — Risk Patterns (type=pattern or risk): generate 4 capsules
  employees with multiple distinct alert types (HAVING COUNT(DISTINCT AlertType) >= 2)
  high severity open alerts (Critical+High, Status Open or Investigating) — TTL=4h
  broker dealers with both high rejection rate AND high alert count
  escalation pattern: what dimensions (department, trade type) correlate with escalation

CATEGORY 5 — Approval Workflow (type=operational): generate 5 capsules
  reviewer decision distribution per reviewer (approved%, rejected%, escalated%)
  average turnaround by requesting department
  average turnaround by reviewer
  pending requests with NO ApprovalWorkflow row — TTL=2h (most urgent)
  requests pending more than 3 days

CATEGORY 6 — Security Analysis (type=distribution): generate 4 capsules
  most traded securities by total quantity TOP 20
  securities with restriction history (count of past restrictions)
  currently active restrictions (EndDate IS NULL or future)
  securities appearing in both restrictions and compliance alerts

CATEGORY 7 — Employee & Department Health (type=aggregation): generate 4 capsules
  department compliance scorecard (requests + rejections + escalations + alerts combined)
  employees with zero alerts (clean record, active employees only)
  alert rate by job title (alerts per employee per title)
  new employee compliance (HireDate >= 2 years ago, how many already have alerts)

CATEGORY 8 — Cross-Entity Risk (type=risk, P1): generate 4 capsules
  employees with compliance alerts across 2+ different broker dealers
  department + restriction type concentration (which department hits which restriction most)
  reviewer coverage gaps (departments with high % of unreviewed requests)
  full five-table risk profile join: Employee + TradeRequest + ComplianceAlert +
    ApprovalWorkflow + RestrictedSecurity (TOP 50, ordered by severity DESC)

═══════════════════════════════════════
OUTPUT FORMAT — critical
═══════════════════════════════════════
- Return ONLY a Python list literal: starts with [ ends with ]
- No variable assignment (no CAPSULE_DEFINITIONS =)
- No markdown fences, no explanation text, no comments
- Every string value properly escaped for Python single or double quotes
- SQL values must be single-line strings (no line breaks inside the string value)
- Produce all 37+ capsules across all 8 categories. Do not truncate or summarize."""

CAPSULE_REGEN_USER = """Database schema (all tables and columns):
{schema}

Foreign key relationships:
{fk_relationships}

Format examples — follow this exact dict structure for every capsule:
{format_example}

Generate the complete CAPSULE_DEFINITIONS list for this Compliance database.
Cover all 8 categories. Minimum 35 capsules. Return the Python list only."""

