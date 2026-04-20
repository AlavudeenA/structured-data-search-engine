"""Domain configuration for the Financial Compliance database.

Swap this file (and the rest of business_schema/) to deploy against a different domain.
The engine reads these constants at module load time and injects them into LLM prompts,
database initialisation, and file paths — so the engine itself stays fully neutral.

Interface contract (new domains must export all of these):
    DOMAIN_NAME               str   — human-readable name shown in prompts
    DOMAIN_ANALYST_PERSONA    str   — role the LLM plays when writing signals/summaries
    DOMAIN_DATA_ENG_PERSONA   str   — role the LLM plays when writing SQL / metadata
    DOMAIN_SQL_RULES          str   — business rules the LLM must follow when writing SQL
    EMBED_TEXT_STYLE          str   — instructions for writing embed_text_template values
    REGEN_COVERAGE            str   — category + capsule list for the "Generate Definitions" button
    DB_SCRIPT_PATH            Path  — absolute path to the database DDL + seed script
"""

from pathlib import Path

_DB_METADATA_PATH = Path(__file__).parent / "db_metadata.md"


def load_db_metadata() -> str:
    """Return the contents of db_metadata.md — column descriptions, enum values, join patterns.
    Called at SQL-generation time so the LLM understands column semantics, not just column names.
    Update db_metadata.md whenever dbscript.sql changes (new tables, columns, or enum values).
    """
    try:
        return _DB_METADATA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ── Identity ───────────────────────────────────────────────────────────────────

DOMAIN_NAME = "Financial Compliance"
DOMAIN_ANALYST_PERSONA = "compliance analyst"
DOMAIN_DATA_ENG_PERSONA = "compliance data engineer"
EMBED_SEARCH_PERSONA = "compliance officer"

DB_SCRIPT_PATH = Path(__file__).parent / "dbscript.sql"

# ── Business rules injected into SQL-generation and capsule-regen prompts ─────
# Add any rule the LLM needs to write correct queries for this schema.

DOMAIN_SQL_RULES = """
1. ACTIVE RESTRICTION: EndDate IS NULL means permanently active (no end set).
   Safe combined form: COALESCE(rs.EndDate, '9999-12-31') >= date('now')

2. VIOLATION DATE OVERLAP — trade happened while restriction was active:
   JOIN condition: tr.RequestDate BETWEEN rs.StartDate AND COALESCE(rs.EndDate, '9999-12-31')

3. ApprovalWorkflow.ReviewerID references Employee.EmployeeID.
   The REVIEWER is an employee (compliance/risk staff), NOT the trade requester.

4. TurnaroundDays is already stored in ApprovalWorkflow — do NOT recalculate it.

5. Escalated = sent to senior compliance review. It is NOT a rejection.

6. Repeat violators = employees with COUNT(ComplianceAlert) >= 2.

7. High severity = ca.Severity IN ('Critical', 'High')

8. Unresolved alerts = ca.Status IN ('Open', 'Investigating')
""".strip()

# ── Embed text style guidance (used in CAPSULE_REGEN_SYSTEM and CAPSULE_ENRICH_SYSTEM) ──

EMBED_TEXT_STYLE = f"""Write as if a {EMBED_SEARCH_PERSONA} is typing a natural search query.
First sentence: what this capsule is about and its compliance relevance.
Middle section: list 3-5 specific questions this capsule answers (not bullet points, embed in prose).
Include domain synonyms inline:
    violation = breach = non-compliant = policy break
    restriction = ban = blackout = insider list = watch list
    alert = incident = compliance flag = issue
End with exactly this text (no variation): "Finding: {{signal}}"
Total length: 150-250 words."""

# ── Coverage requirements for the "Generate Capsule Definitions" button ───────
# Defines the categories and capsules the LLM should produce for this domain.

REGEN_COVERAGE = """
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

CATEGORY 9 — Random Sample Views (type=sample, signal_method="sample"): generate 3 capsules
  Rules for sample capsules:
  - SQL must use ORDER BY random() LIMIT 50 to return random rows
  - Use LEFT JOIN so rows still appear even when related tables have no match
  - Select 12-16 columns spanning all joined tables — mix entity names, dates, statuses, and amounts
  - signal_method must be exactly "sample" (not "llm_summary" or "rule_based")
  - priority = P3, ttl_hours = 12
  - linked_capsule_ids must point to 2-3 aggregation/violation capsules from the same tables

  Capsule 1 — full five-table random sample:
    Join Employee + TradeRequest + BrokerDealer + ComplianceAlert (LEFT) + ApprovalWorkflow (LEFT) + RestrictedSecurity (LEFT, date-overlap)
    Select: employee_name, department, broker_dealer, security_symbol, trade_type, trade_status,
            request_date, alert_type, severity, alert_status, review_decision, turnaround_days,
            restriction_type, restriction_start
    Link to: violations_on_restricted_securities, trade_requests_by_broker_dealer, high_severity_open_alerts

  Capsule 2 — violation records sample:
    Join TradeRequest + RestrictedSecurity (INNER, date-overlap) + Employee + BrokerDealer
    Only records where an active restriction overlapped the trade date
    Select: employee_name, department, security_symbol, restriction_type, trade_type,
            trade_status, request_date, restriction_start, restriction_end, broker_dealer
    Link to: violations_on_restricted_securities, violations_by_broker_dealer, repeat_violators

  Capsule 3 — high risk records sample:
    Join TradeRequest + ComplianceAlert (INNER) + ApprovalWorkflow (LEFT) + Employee + BrokerDealer
    Only records where Severity IN ('Critical','High') AND alert Status IN ('Open','Investigating')
    Select: employee_name, department, broker_dealer, security_symbol, alert_type, severity,
            alert_status, trade_status, review_decision, turnaround_days, request_date, alert_date
    Link to: high_severity_open_alerts, repeat_violators, broker_dealers_high_rejection_and_alerts
""".strip()
