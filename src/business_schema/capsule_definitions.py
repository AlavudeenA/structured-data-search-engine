"""
Domain-specific capsule definitions for the Compliance database.
Every capsule is fully specified — no placeholders.
Import CAPSULE_DEFINITIONS and iterate to generate all capsules.
"""

from __future__ import annotations

SCHEMA_DEFINITIONS: list[dict] = [
    {
        "capsule_id": "schema_context_trade_request_backbone",
        "summary": "TradeRequest is the operational backbone for ranking, volume, broker, employee, and security analysis. Use it as the fact table and join to Employee and BrokerDealer first.",
        "tables": ["TradeRequest", "Employee", "BrokerDealer"],
        "relevant_columns": ["EmployeeName", "Department", "BrokerDealerName", "Country"],
        "join_columns": ["EmployeeID", "BrokerDealerID"],
        "sql_template": (
            "SELECT bd.BrokerDealerName AS broker_dealer, COUNT(tr.TradeRequestID) AS request_count "
            "FROM TradeRequest tr "
            "JOIN BrokerDealer bd ON tr.BrokerDealerID = bd.BrokerDealerID "
            "GROUP BY bd.BrokerDealerName "
            "ORDER BY request_count DESC"
        ),
        "tags": ["schema", "trade_request", "broker", "employee", "aggregation"]
    },
    {
        "capsule_id": "schema_context_restriction_overlap",
        "summary": "Use RestrictedSecurity with TradeRequest on SecuritySymbol plus date overlap when asking about violations, active restrictions, or restricted trading attempts. EndDate NULL means the restriction is still active.",
        "tables": ["TradeRequest", "RestrictedSecurity", "Employee", "BrokerDealer"],
        "relevant_columns": ["SecuritySymbol", "RestrictionType", "StartDate", "EndDate", "RequestDate", "EmployeeID", "BrokerDealerID"],
        "join_columns": ["SecuritySymbol", "EmployeeID", "BrokerDealerID"],
        "sql_template": (
            "SELECT tr.TradeRequestID AS trade_request_id, e.EmployeeName AS employee_name, tr.SecuritySymbol AS security_symbol, "
            "rs.RestrictionType AS restriction_type, tr.RequestDate AS request_date "
            "FROM TradeRequest tr "
            "JOIN RestrictedSecurity rs ON tr.SecuritySymbol = rs.SecuritySymbol "
            "AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31') "
            "JOIN Employee e ON tr.EmployeeID = e.EmployeeID "
            "ORDER BY tr.RequestDate DESC"
        ),
        "tags": ["schema", "restriction", "violation", "date_overlap", "active_restriction"]
    },
    {
        "capsule_id": "schema_context_alerts_and_reviews",
        "summary": "ComplianceAlert plus ApprovalWorkflow answers escalation, alert severity, reviewer performance, and unresolved issue questions. ReviewerID points back to Employee as the reviewer, not the requester.",
        "tables": ["ComplianceAlert", "TradeRequest", "ApprovalWorkflow", "Employee"],
        "relevant_columns": ["AlertType", "AlertDate", "Severity", "Status", "Description", "ResolvedDate", "Decision", "TurnaroundDays", "ReviewerID"],
        "join_columns": ["TradeRequestID", "EmployeeID", "ReviewerID"],
        "sql_template": (
            "SELECT ca.AlertType AS alert_type, ca.Severity AS severity, aw.Decision AS decision, aw.TurnaroundDays AS turnaround_days "
            "FROM ComplianceAlert ca "
            "LEFT JOIN ApprovalWorkflow aw ON ca.TradeRequestID = aw.TradeRequestID "
            "ORDER BY ca.AlertDate DESC"
        ),
        "tags": ["schema", "alerts", "workflow", "reviewer", "operational", "risk"]
    },
    {
        "capsule_id": "schema_context_employee_account_broker_path",
        "summary": "Use Account when the question is about employee brokerage registration, account coverage, or broker registration by country. This is a metadata path that complements TradeRequest-based analysis.",
        "tables": ["Employee", "Account", "BrokerDealer"],
        "relevant_columns": ["EmployeeID", "EmployeeName", "Department", "AccountID", "AccountType", "AccountNumber", "BrokerDealerName", "Country", "RegistrationNumber"],
        "join_columns": ["EmployeeID", "BrokerDealerID"],
        "sql_template": (
            "SELECT bd.BrokerDealerName AS broker_dealer_name, bd.Country AS country, COUNT(a.AccountID) AS account_count "
            "FROM Account a "
            "JOIN BrokerDealer bd ON a.BrokerDealerID = bd.BrokerDealerID "
            "GROUP BY bd.BrokerDealerName, bd.Country "
            "ORDER BY account_count DESC"
        ),
        "tags": ["schema", "account", "broker_registration", "country", "employee_path"]
    },
    {
        "capsule_id": "schema_context_trend_patterns",
        "summary": "For trend questions, prefer TradeRequest.RequestDate, ComplianceAlert.AlertDate, and ApprovalWorkflow.ReviewDate. Group using FORMAT(date_col, 'yyyy-MM') for monthly trends or DATEPART(ISO_WEEK, date_col) for weekly views.",
        "tables": ["TradeRequest", "ComplianceAlert", "ApprovalWorkflow", "BrokerDealer", "Employee"],
        "relevant_columns": ["RequestDate", "AlertDate", "ReviewDate", "Status", "Severity", "Department", "BrokerDealerName"],
        "join_columns": ["BrokerDealerID", "EmployeeID", "TradeRequestID", "ReviewerID"],
        "sql_template": (
            "SELECT FORMAT(tr.RequestDate, 'yyyy-MM') AS month_key, bd.BrokerDealerName AS broker_dealer, COUNT(tr.TradeRequestID) AS request_count "
            "FROM TradeRequest tr "
            "JOIN BrokerDealer bd ON tr.BrokerDealerID = bd.BrokerDealerID "
            "WHERE tr.RequestDate >= DATEADD(MONTH, -6, GETDATE()) "
            "GROUP BY FORMAT(tr.RequestDate, 'yyyy-MM'), bd.BrokerDealerName "
            "ORDER BY month_key ASC, request_count DESC"
        ),
        "tags": ["schema", "trend", "time_series", "broker", "department"]
    }
]

CAPSULE_DEFINITIONS: list[dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1 — VOLUME & ACTIVITY
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "trade_requests_by_broker_dealer",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Trade requests per broker dealer",
        "how": "Count total, approved, rejected, and escalated requests per broker dealer",
        "sql": """
SELECT
    bd.BrokerDealerName          AS broker_dealer,
    bd.Country                   AS country,
    COUNT(tr.TradeRequestID)     AS total_requests,
    SUM(CASE WHEN tr.Status = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    SUM(CASE WHEN tr.Status = 'Pending'   THEN 1 ELSE 0 END) AS pending_count,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS rejection_rate_pct
FROM BrokerDealer bd
LEFT JOIN TradeRequest tr ON tr.BrokerDealerID = bd.BrokerDealerID
GROUP BY bd.BrokerDealerID, bd.BrokerDealerName, bd.Country
ORDER BY total_requests DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trade request volume and approval outcomes broken down by broker dealer partner. "
            "This capsule answers: which broker dealers submit the most trade requests, "
            "and what are their approval, rejection, and escalation rates? "
            "Compliance officers use this to identify broker dealers with unusual rejection patterns "
            "or high escalation rates that may signal systemic risk or policy non-compliance. "
            "Related questions: broker dealer activity levels, which broker has most rejections, "
            "broker dealer compliance performance, rejection rate by counterparty. "
            "Finding: {signal} "
            "Tables: BrokerDealer, TradeRequest. "
            "Key columns: BrokerDealerName, Country, total_requests, rejected_count, escalated_count, rejection_rate_pct."
        ),
        "ttl_hours": 24,
        "tags": ["broker_dealer", "volume", "aggregation", "rejection_rate", "escalation"],
        "tables_used": ["BrokerDealer", "TradeRequest"],
        "key_columns": ["BrokerDealerName", "total_requests", "rejected_count", "escalated_count", "rejection_rate_pct"],
        "staleness_trigger": "new TradeRequest row inserted",
        "related_capsule_ids": ["violations_by_broker_dealer", "broker_dealers_high_rejection_and_alerts"],
        "relationship_types": ["corroborates", "aggregates_up"],
    },

    {
        "capsule_id": "trade_requests_by_department",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Trade requests per employee department",
        "how": "Count total, approved, rejected, and escalated requests grouped by department",
        "sql": """
SELECT
    e.Department                 AS department,
    COUNT(tr.TradeRequestID)     AS total_requests,
    SUM(CASE WHEN tr.Status = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    COUNT(DISTINCT tr.EmployeeID) AS active_employees,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status IN ('Rejected','Escalated') THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS non_approval_rate_pct
FROM Employee e
LEFT JOIN TradeRequest tr ON tr.EmployeeID = e.EmployeeID
GROUP BY e.Department
ORDER BY total_requests DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trade request activity by organizational department, showing which business units "
            "generate the highest trading volumes and which face the most compliance friction. "
            "This capsule answers: which departments submit most trade requests, "
            "what is each department's rejection rate, which team has the highest escalation frequency. "
            "Useful for: department-level compliance risk profiling, identifying teams that need "
            "additional training, understanding trading behaviour by business unit. "
            "Synonyms: business unit activity, division trading volume, team-level compliance outcomes. "
            "Finding: {signal} "
            "Tables: Employee, TradeRequest. "
            "Key columns: Department, total_requests, rejected_count, escalated_count, non_approval_rate_pct."
        ),
        "ttl_hours": 24,
        "tags": ["department", "volume", "aggregation", "rejection_rate"],
        "tables_used": ["Employee", "TradeRequest"],
        "key_columns": ["Department", "total_requests", "rejected_count", "non_approval_rate_pct"],
        "staleness_trigger": "new TradeRequest or Employee row",
        "related_capsule_ids": ["violations_by_department", "escalation_trend_by_department", "department_compliance_scorecard"],
        "relationship_types": ["corroborates", "corroborates", "aggregates_up"],
    },

    {
        "capsule_id": "trade_requests_by_security_symbol",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Trade requests per security symbol (top 10 by volume)",
        "how": "Count and sum trade quantity per symbol, show top 10 most traded",
        "sql": """
SELECT TOP 10
    tr.SecuritySymbol                AS security_symbol,
    COUNT(tr.TradeRequestID)         AS request_count,
    SUM(tr.Quantity)                 AS total_quantity,
    SUM(CASE WHEN tr.TradeType = 'BUY'  THEN tr.Quantity ELSE 0 END) AS buy_quantity,
    SUM(CASE WHEN tr.TradeType = 'SELL' THEN tr.Quantity ELSE 0 END) AS sell_quantity,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    MAX(CASE WHEN rs.RestrictionID IS NOT NULL THEN 1 ELSE 0 END) AS has_restriction_history
FROM TradeRequest tr
LEFT JOIN RestrictedSecurity rs ON rs.SecuritySymbol = tr.SecuritySymbol
GROUP BY tr.SecuritySymbol
ORDER BY request_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Top securities by trade request volume showing which ticker symbols are most actively "
            "traded by employees. This capsule answers: which securities are traded most frequently, "
            "what is the buy vs sell split per symbol, which high-volume securities also appear on "
            "restriction lists creating potential violation risk. "
            "Compliance relevance: securities with high volume plus restriction history are the "
            "highest priority for review — combining popularity with policy breach exposure. "
            "Related terms: most active securities, ticker symbol activity, top traded stocks, "
            "security volume analysis, restricted security trading. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity. "
            "Key columns: SecuritySymbol, request_count, total_quantity, has_restriction_history."
        ),
        "ttl_hours": 24,
        "tags": ["security_symbol", "volume", "aggregation", "restriction"],
        "tables_used": ["TradeRequest", "RestrictedSecurity"],
        "key_columns": ["SecuritySymbol", "request_count", "total_quantity", "rejected_count", "has_restriction_history"],
        "staleness_trigger": "new TradeRequest or RestrictedSecurity row",
        "related_capsule_ids": ["currently_active_restrictions", "securities_in_restrictions_and_alerts"],
        "relationship_types": ["corroborates", "drills_down"],
    },

    {
        "capsule_id": "trade_requests_by_trade_type",
        "capsule_type": "aggregation",
        "priority": "P3",
        "what": "Trade request split between BUY and SELL",
        "how": "Count and percentage breakdown of BUY vs SELL across all requests",
        "sql": """
SELECT
    tr.TradeType                             AS trade_type,
    COUNT(tr.TradeRequestID)                 AS request_count,
    SUM(tr.Quantity)                         AS total_quantity,
    CAST(
        100.0 * COUNT(tr.TradeRequestID)
        / NULLIF((SELECT COUNT(*) FROM TradeRequest), 0)
    AS DECIMAL(5,2))                         AS pct_of_total,
    SUM(CASE WHEN tr.Status = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count
FROM TradeRequest tr
GROUP BY tr.TradeType
ORDER BY request_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Overall split of trade requests between buy orders and sell orders across the compliance database. "
            "This capsule answers: what proportion of all trade requests are buys versus sells, "
            "how do approval and rejection rates differ between buy and sell trade types. "
            "An unusual skew toward sells with high rejection rates may indicate insider trading "
            "concerns or pre-restriction dumping behaviour. "
            "Related terms: buy sell ratio, trade direction distribution, long short activity. "
            "Finding: {signal} "
            "Tables: TradeRequest. "
            "Key columns: TradeType, request_count, total_quantity, pct_of_total, rejected_count."
        ),
        "ttl_hours": 48,
        "tags": ["trade_type", "aggregation", "buy_sell", "distribution"],
        "tables_used": ["TradeRequest"],
        "key_columns": ["TradeType", "request_count", "total_quantity", "pct_of_total"],
        "staleness_trigger": "new TradeRequest row",
        "related_capsule_ids": ["trade_requests_by_security_symbol"],
        "relationship_types": ["corroborates"],
    },

    {
        "capsule_id": "monthly_request_volume",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Monthly trade request volume over the last 6 months",
        "how": "Count requests grouped by month and status",
        "sql": """
SELECT
    FORMAT(tr.RequestDate, 'yyyy-MM')        AS request_month,
    COUNT(tr.TradeRequestID)                 AS total_requests,
    SUM(CASE WHEN tr.Status = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    SUM(CASE WHEN tr.Status = 'Pending'   THEN 1 ELSE 0 END) AS pending_count
FROM TradeRequest tr
WHERE tr.RequestDate >= DATEADD(MONTH, -6, GETDATE())
GROUP BY FORMAT(tr.RequestDate, 'yyyy-MM')
ORDER BY request_month ASC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Month-by-month trade request volume for the last 6 months showing activity trends "
            "and outcome distributions across time. "
            "This capsule answers: is trade request volume growing or declining, "
            "which months had the most rejections or escalations, is there a seasonal compliance pattern. "
            "A rising trend in total requests combined with rising rejections signals increasing "
            "compliance pressure. A spike in escalations in a specific month warrants investigation. "
            "Related terms: monthly volume trend, request activity over time, trading frequency history. "
            "Finding: {signal} "
            "Tables: TradeRequest. "
            "Key columns: request_month, total_requests, rejected_count, escalated_count."
        ),
        "ttl_hours": 12,
        "tags": ["monthly", "trend", "volume", "aggregation"],
        "tables_used": ["TradeRequest"],
        "key_columns": ["request_month", "total_requests", "rejected_count", "escalated_count"],
        "staleness_trigger": "daily",
        "related_capsule_ids": ["monthly_alert_volume_trend", "weekly_trade_request_volume"],
        "relationship_types": ["corroborates", "same_entity"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 2 — VIOLATIONS (P1)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "violations_on_restricted_securities",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Trade requests made while the security was on an active restriction",
        "how": "Date-overlap join between TradeRequest and RestrictedSecurity",
        "sql": """
SELECT
    tr.TradeRequestID                    AS trade_request_id,
    e.EmployeeName                       AS employee_name,
    e.Department                         AS department,
    bd.BrokerDealerName                  AS broker_dealer,
    tr.SecuritySymbol                    AS security_symbol,
    tr.TradeType                         AS trade_type,
    tr.Quantity                          AS quantity,
    tr.RequestDate                       AS request_date,
    tr.Status                            AS request_status,
    rs.RestrictionType                   AS restriction_type,
    rs.StartDate                         AS restriction_start,
    rs.EndDate                           AS restriction_end,
    rs.Reason                            AS restriction_reason
FROM TradeRequest tr
-- date overlap: request falls within restriction window
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
ORDER BY tr.RequestDate DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Policy violations where employees submitted trade requests for securities that were "
            "on an active restriction list at the time of the request. "
            "This is the most critical compliance breach capsule. "
            "Answers: which employees breached trading restrictions, which securities were traded "
            "in violation of policy, which broker dealers facilitated non-compliant trades, "
            "what restriction types were breached most frequently. "
            "Also known as: trading violations, restriction breaches, blackout violations, "
            "insider list breaches, policy non-compliance, prohibited security trading. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity, Employee, BrokerDealer. "
            "Key columns: employee_name, security_symbol, restriction_type, request_date, request_status."
        ),
        "ttl_hours": 6,
        "tags": ["violation", "restriction", "breach", "P1", "critical"],
        "tables_used": ["TradeRequest", "RestrictedSecurity", "Employee", "BrokerDealer"],
        "key_columns": ["employee_name", "security_symbol", "restriction_type", "request_date", "request_status"],
        "staleness_trigger": "any new TradeRequest or RestrictedSecurity change",
        "related_capsule_ids": ["violations_by_restriction_type", "violations_by_broker_dealer", "violations_by_department", "repeat_violators"],
        "relationship_types": ["drills_down", "aggregates_up", "aggregates_up", "corroborates"],
    },

    {
        "capsule_id": "violations_by_restriction_type",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Violation counts grouped by restriction type",
        "how": "Count overlapping requests per restriction category (Blackout, Insider List, Watch List)",
        "sql": """
SELECT
    rs.RestrictionType                   AS restriction_type,
    COUNT(tr.TradeRequestID)             AS violation_count,
    COUNT(DISTINCT tr.EmployeeID)        AS unique_employees,
    COUNT(DISTINCT tr.SecuritySymbol)    AS unique_securities,
    COUNT(DISTINCT tr.BrokerDealerID)    AS unique_broker_dealers
FROM TradeRequest tr
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
GROUP BY rs.RestrictionType
ORDER BY violation_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Compliance violations broken down by restriction category — Blackout, Insider List, "
            "and Watch List. "
            "This capsule answers: which restriction type has the most violations, "
            "how many unique employees and securities are involved in each category, "
            "which type of trading ban is being ignored most frequently. "
            "Blackout violations are the most serious as they relate to earnings period trading bans. "
            "Insider List breaches involve material non-public information risk. "
            "Watch List infractions indicate insufficient monitoring controls. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity. "
            "Key columns: restriction_type, violation_count, unique_employees, unique_securities."
        ),
        "ttl_hours": 12,
        "tags": ["violation", "restriction_type", "blackout", "insider_list", "watch_list"],
        "tables_used": ["TradeRequest", "RestrictedSecurity"],
        "key_columns": ["restriction_type", "violation_count", "unique_employees", "unique_securities"],
        "staleness_trigger": "new violation or restriction change",
        "related_capsule_ids": ["violations_on_restricted_securities", "currently_active_restrictions"],
        "relationship_types": ["aggregates_up", "corroborates"],
    },

    {
        "capsule_id": "violations_by_broker_dealer",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Violations per broker dealer counterparty",
        "how": "Count restriction-overlap trades grouped by broker dealer",
        "sql": """
SELECT
    bd.BrokerDealerName                  AS broker_dealer,
    bd.Country                           AS country,
    COUNT(tr.TradeRequestID)             AS violation_count,
    COUNT(DISTINCT tr.EmployeeID)        AS unique_employees,
    COUNT(DISTINCT tr.SecuritySymbol)    AS unique_securities
FROM TradeRequest tr
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
GROUP BY bd.BrokerDealerID, bd.BrokerDealerName, bd.Country
ORDER BY violation_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trading policy violations attributed to each broker dealer counterparty. "
            "This capsule reveals which broker dealers are most frequently involved in non-compliant "
            "trades where the security was on an active restriction list. "
            "Answers: which brokers facilitated the most violations, which brokers pose the highest "
            "counterparty compliance risk, how many employees used each broker in violation transactions. "
            "A broker dealer with repeated violation involvement may require enhanced due diligence, "
            "restriction of trading permissions, or regulatory reporting. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity, BrokerDealer. "
            "Key columns: broker_dealer, country, violation_count, unique_employees."
        ),
        "ttl_hours": 12,
        "tags": ["violation", "broker_dealer", "counterparty_risk"],
        "tables_used": ["TradeRequest", "RestrictedSecurity", "BrokerDealer"],
        "key_columns": ["broker_dealer", "country", "violation_count", "unique_employees"],
        "staleness_trigger": "new violation detected",
        "related_capsule_ids": ["trade_requests_by_broker_dealer", "broker_dealers_high_rejection_and_alerts"],
        "relationship_types": ["same_entity", "corroborates"],
    },

    {
        "capsule_id": "violations_by_department",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Violations per organizational department",
        "how": "Count restriction-overlap trades grouped by employee department",
        "sql": """
SELECT
    e.Department                         AS department,
    COUNT(tr.TradeRequestID)             AS violation_count,
    COUNT(DISTINCT tr.EmployeeID)        AS unique_violators,
    COUNT(DISTINCT tr.SecuritySymbol)    AS unique_securities,
    COUNT(DISTINCT rs.RestrictionType)   AS restriction_types_hit
FROM TradeRequest tr
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
GROUP BY e.Department
ORDER BY violation_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Compliance violations aggregated by organizational department showing which business units "
            "have the worst trading policy adherence. "
            "This capsule answers: which department has the most restriction breaches, "
            "how many unique employees in each department committed violations, "
            "which departments hit the broadest range of restriction types. "
            "High violation counts in a specific department may indicate inadequate compliance "
            "training, culture issues, or a department with elevated access to sensitive securities. "
            "Related terms: department violations, business unit breaches, team non-compliance. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity, Employee. "
            "Key columns: department, violation_count, unique_violators, restriction_types_hit."
        ),
        "ttl_hours": 12,
        "tags": ["violation", "department", "business_unit"],
        "tables_used": ["TradeRequest", "RestrictedSecurity", "Employee"],
        "key_columns": ["department", "violation_count", "unique_violators", "restriction_types_hit"],
        "staleness_trigger": "new violation detected",
        "related_capsule_ids": ["trade_requests_by_department", "department_compliance_scorecard"],
        "relationship_types": ["same_entity", "drills_down"],
    },

    {
        "capsule_id": "repeat_violators",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Employees with two or more compliance alerts (repeat offenders)",
        "how": "Count alerts per employee, filter to those with 2 or more",
        "sql": """
SELECT
    e.EmployeeID                         AS employee_id,
    e.EmployeeName                       AS employee_name,
    e.Department                         AS department,
    e.JobTitle                           AS job_title,
    COUNT(ca.AlertID)                    AS alert_count,
    COUNT(DISTINCT ca.AlertType)         AS distinct_alert_types,
    MAX(ca.Severity)                     AS max_severity,
    SUM(CASE WHEN ca.Status IN ('Open','Investigating') THEN 1 ELSE 0 END) AS open_alerts
FROM Employee e
JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle
HAVING COUNT(ca.AlertID) >= 2
ORDER BY alert_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Employees with a history of multiple compliance alerts, flagged as repeat violators "
            "or habitual non-compliant traders. "
            "This capsule answers: which employees have been alerted more than once, "
            "how many distinct types of alerts do repeat offenders accumulate, "
            "which repeat violators still have open or investigating alerts requiring action. "
            "Repeat violators represent the highest individual employee risk — they have demonstrated "
            "inability or unwillingness to follow trading compliance rules after initial alerts. "
            "Also known as: habitual offenders, multi-alert employees, high-risk employees. "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert. "
            "Key columns: employee_name, department, alert_count, distinct_alert_types, open_alerts."
        ),
        "ttl_hours": 8,
        "tags": ["violation", "repeat_violator", "employee_risk", "P1"],
        "tables_used": ["Employee", "ComplianceAlert"],
        "key_columns": ["employee_name", "department", "alert_count", "open_alerts"],
        "staleness_trigger": "new ComplianceAlert row",
        "related_capsule_ids": ["violations_on_restricted_securities", "employees_multiple_alert_types"],
        "relationship_types": ["corroborates", "same_entity"],
    },

    {
        "capsule_id": "active_restrictions_recent_trade_attempts",
        "capsule_type": "violation",
        "priority": "P1",
        "what": "Active restrictions with trade attempts in the last 30 days",
        "how": "Join active restrictions to recent trade requests on same symbol",
        "sql": """
SELECT
    rs.SecuritySymbol                    AS security_symbol,
    rs.RestrictionType                   AS restriction_type,
    rs.StartDate                         AS restriction_start,
    rs.Reason                            AS restriction_reason,
    rs.AddedBy                           AS added_by,
    COUNT(tr.TradeRequestID)             AS recent_trade_attempts,
    COUNT(DISTINCT tr.EmployeeID)        AS unique_employees,
    MAX(tr.RequestDate)                  AS latest_attempt_date
FROM RestrictedSecurity rs
-- active restriction: EndDate is NULL (no end set) or in the future
JOIN TradeRequest tr
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate >= DATEADD(DAY, -30, GETDATE())
WHERE rs.EndDate IS NULL OR rs.EndDate >= GETDATE()
GROUP BY rs.RestrictionID, rs.SecuritySymbol, rs.RestrictionType,
         rs.StartDate, rs.Reason, rs.AddedBy
ORDER BY recent_trade_attempts DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Currently active trading restrictions that have seen actual trade attempts within the "
            "last 30 days — the most time-critical violation signal in the compliance system. "
            "This capsule answers: which actively restricted securities are employees still attempting "
            "to trade, how many employees are attempting restricted trades right now, "
            "which active restrictions are being ignored most frequently. "
            "This is an immediate action capsule — any security on an active restriction with recent "
            "trade attempts requires same-day compliance officer review. "
            "Also known as: live violations, real-time restriction breaches, current policy breaches. "
            "Finding: {signal} "
            "Tables: RestrictedSecurity, TradeRequest. "
            "Key columns: security_symbol, restriction_type, recent_trade_attempts, unique_employees, latest_attempt_date."
        ),
        "ttl_hours": 6,
        "tags": ["violation", "active_restriction", "real_time", "P1", "urgent"],
        "tables_used": ["RestrictedSecurity", "TradeRequest"],
        "key_columns": ["security_symbol", "restriction_type", "recent_trade_attempts", "unique_employees", "latest_attempt_date"],
        "staleness_trigger": "any TradeRequest or RestrictedSecurity change",
        "related_capsule_ids": ["currently_active_restrictions", "violations_on_restricted_securities"],
        "relationship_types": ["drills_down", "corroborates"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 3 — TRENDS
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "monthly_alert_volume_trend",
        "capsule_type": "trend",
        "priority": "P2",
        "what": "Monthly compliance alert volume over the last 6 months",
        "how": "Count alerts per month, break down by severity and status",
        "sql": """
SELECT
    FORMAT(ca.AlertDate, 'yyyy-MM')          AS alert_month,
    COUNT(ca.AlertID)                        AS total_alerts,
    SUM(CASE WHEN ca.Severity = 'Critical' THEN 1 ELSE 0 END) AS critical_count,
    SUM(CASE WHEN ca.Severity = 'High'     THEN 1 ELSE 0 END) AS high_count,
    SUM(CASE WHEN ca.Severity = 'Medium'   THEN 1 ELSE 0 END) AS medium_count,
    SUM(CASE WHEN ca.Severity = 'Low'      THEN 1 ELSE 0 END) AS low_count,
    SUM(CASE WHEN ca.Status IN ('Open','Investigating') THEN 1 ELSE 0 END) AS open_count
FROM ComplianceAlert ca
WHERE ca.AlertDate >= DATEADD(MONTH, -6, GETDATE())
GROUP BY FORMAT(ca.AlertDate, 'yyyy-MM')
ORDER BY alert_month ASC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Month-by-month trend of compliance alerts showing whether the firm's risk exposure "
            "is improving or deteriorating over the last 6 months. "
            "This capsule answers: is the number of compliance alerts increasing or decreasing, "
            "which months had the most critical or high severity alerts, "
            "is there a seasonal or event-driven spike in alerts. "
            "A rising alert trend combined with growing critical counts is the most concerning signal — "
            "it indicates deteriorating compliance culture or a specific risk event driving alerts. "
            "Related terms: alert volume trend, compliance incident trend, risk escalation over time. "
            "Finding: {signal} "
            "Tables: ComplianceAlert. "
            "Key columns: alert_month, total_alerts, critical_count, high_count, open_count."
        ),
        "ttl_hours": 12,
        "tags": ["trend", "alert", "monthly", "severity"],
        "tables_used": ["ComplianceAlert"],
        "key_columns": ["alert_month", "total_alerts", "critical_count", "high_count", "open_count"],
        "staleness_trigger": "daily",
        "related_capsule_ids": ["monthly_request_volume", "alert_severity_trend"],
        "relationship_types": ["corroborates", "same_entity"],
    },

    {
        "capsule_id": "weekly_trade_request_volume",
        "capsule_type": "trend",
        "priority": "P2",
        "what": "Weekly trade request volume over the last 8 weeks",
        "how": "Count requests per ISO week with status breakdown",
        "sql": """
SELECT
    FORMAT(tr.RequestDate, 'yyyy-') + 'W' + RIGHT('0' + CAST(DATEPART(ISO_WEEK, tr.RequestDate) AS VARCHAR), 2) AS request_week,
    COUNT(tr.TradeRequestID)                 AS total_requests,
    SUM(CASE WHEN tr.Status = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count
FROM TradeRequest tr
WHERE tr.RequestDate >= DATEADD(WEEK, -8, GETDATE())
GROUP BY FORMAT(tr.RequestDate, 'yyyy-') + 'W' + RIGHT('0' + CAST(DATEPART(ISO_WEEK, tr.RequestDate) AS VARCHAR), 2)
ORDER BY request_week ASC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Week-by-week trade request activity for the last 8 weeks showing short-term trading "
            "volume patterns and compliance outcomes. "
            "This capsule answers: is weekly trading volume increasing or decreasing recently, "
            "which specific weeks had unusual spikes in rejections or escalations, "
            "is there a day-of-week or week-of-month pattern to trading compliance breaches. "
            "Weekly granularity catches intra-quarter events that monthly data misses, "
            "such as pre-earnings blackout period violations or sudden volume spikes. "
            "Finding: {signal} "
            "Tables: TradeRequest. "
            "Key columns: request_week, total_requests, rejected_count, escalated_count."
        ),
        "ttl_hours": 8,
        "tags": ["trend", "weekly", "volume"],
        "tables_used": ["TradeRequest"],
        "key_columns": ["request_week", "total_requests", "rejected_count", "escalated_count"],
        "staleness_trigger": "daily",
        "related_capsule_ids": ["monthly_request_volume"],
        "relationship_types": ["drills_down"],
    },

    {
        "capsule_id": "monthly_rejection_rate_by_broker_dealer",
        "capsule_type": "trend",
        "priority": "P2",
        "what": "Monthly rejection rate per broker dealer over last 6 months",
        "how": "Compute rejection rate per broker per month",
        "sql": """
SELECT
    FORMAT(tr.RequestDate, 'yyyy-MM')        AS request_month,
    bd.BrokerDealerName                      AS broker_dealer,
    COUNT(tr.TradeRequestID)                 AS total_requests,
    SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) AS rejected_count,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS rejection_rate_pct
FROM TradeRequest tr
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
WHERE tr.RequestDate >= DATEADD(MONTH, -6, GETDATE())
GROUP BY FORMAT(tr.RequestDate, 'yyyy-MM'), bd.BrokerDealerID, bd.BrokerDealerName
ORDER BY request_month ASC, rejection_rate_pct DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Trend of rejection rates per broker dealer over the last 6 months, revealing whether "
            "specific brokers are experiencing a worsening or improving compliance profile. "
            "This capsule answers: is any broker dealer's rejection rate trending upward, "
            "which broker dealer went from low to high rejection rate suddenly, "
            "do rejection rate spikes correlate with specific months or events. "
            "A broker dealer whose rejection rate increases month-over-month for 3+ consecutive months "
            "is a systemic risk indicator requiring relationship review. "
            "Finding: {signal} "
            "Tables: TradeRequest, BrokerDealer. "
            "Key columns: request_month, broker_dealer, rejection_rate_pct."
        ),
        "ttl_hours": 12,
        "tags": ["trend", "broker_dealer", "rejection_rate", "monthly"],
        "tables_used": ["TradeRequest", "BrokerDealer"],
        "key_columns": ["request_month", "broker_dealer", "rejection_rate_pct"],
        "staleness_trigger": "monthly",
        "related_capsule_ids": ["trade_requests_by_broker_dealer"],
        "relationship_types": ["drills_down"],
    },

    {
        "capsule_id": "escalation_trend_by_department",
        "capsule_type": "trend",
        "priority": "P2",
        "what": "Monthly escalation trend per department",
        "how": "Count escalated requests per department per month over 6 months",
        "sql": """
SELECT
    FORMAT(tr.RequestDate, 'yyyy-MM')        AS request_month,
    e.Department                             AS department,
    COUNT(tr.TradeRequestID)                 AS total_requests,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS escalation_rate_pct
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
WHERE tr.RequestDate >= DATEADD(MONTH, -6, GETDATE())
GROUP BY FORMAT(tr.RequestDate, 'yyyy-MM'), e.Department
ORDER BY request_month ASC, escalated_count DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Month-by-month escalation rate per organizational department over the last 6 months. "
            "This capsule answers: which departments show a worsening escalation trend, "
            "is any department's escalation rate accelerating, "
            "which month saw the highest escalation rate for a specific department. "
            "Rising escalation rates in a department indicate either systemic compliance failures "
            "or a category of trade types that require senior compliance officer involvement. "
            "Related terms: department escalation history, team escalation trend, business unit risk trend. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee. "
            "Key columns: request_month, department, escalated_count, escalation_rate_pct."
        ),
        "ttl_hours": 12,
        "tags": ["trend", "department", "escalation", "monthly"],
        "tables_used": ["TradeRequest", "Employee"],
        "key_columns": ["request_month", "department", "escalated_count", "escalation_rate_pct"],
        "staleness_trigger": "monthly",
        "related_capsule_ids": ["trade_requests_by_department"],
        "relationship_types": ["drills_down"],
    },

    {
        "capsule_id": "alert_severity_trend",
        "capsule_type": "trend",
        "priority": "P2",
        "what": "Alert severity distribution trend over time",
        "how": "Count alerts per severity per month over last 6 months",
        "sql": """
SELECT
    FORMAT(ca.AlertDate, 'yyyy-MM')          AS alert_month,
    ca.Severity                              AS severity,
    COUNT(ca.AlertID)                        AS alert_count,
    SUM(CASE WHEN ca.Status = 'Open' THEN 1 ELSE 0 END)         AS open_count,
    SUM(CASE WHEN ca.Status = 'Closed' THEN 1 ELSE 0 END)       AS closed_count,
    SUM(CASE WHEN ca.Status = 'Escalated' THEN 1 ELSE 0 END)    AS escalated_count
FROM ComplianceAlert ca
WHERE ca.AlertDate >= DATEADD(MONTH, -6, GETDATE())
GROUP BY FORMAT(ca.AlertDate, 'yyyy-MM'), ca.Severity
ORDER BY alert_month ASC, ca.Severity ASC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trend of compliance alert severity over time — showing whether critical and high severity "
            "alerts are growing, shrinking, or stable. "
            "This capsule answers: is the proportion of critical alerts increasing, "
            "are high severity alerts resolving (moving to closed) or accumulating as open, "
            "which months show the worst severity profile. "
            "An increasing share of Critical severity alerts in recent months indicates worsening "
            "compliance risk that may require escalation to the board or regulators. "
            "Related terms: alert severity trend, risk severity distribution, compliance risk profile. "
            "Finding: {signal} "
            "Tables: ComplianceAlert. "
            "Key columns: alert_month, severity, alert_count, open_count, escalated_count."
        ),
        "ttl_hours": 12,
        "tags": ["trend", "alert", "severity", "monthly"],
        "tables_used": ["ComplianceAlert"],
        "key_columns": ["alert_month", "severity", "alert_count", "open_count"],
        "staleness_trigger": "daily",
        "related_capsule_ids": ["monthly_alert_volume_trend", "high_severity_open_alerts"],
        "relationship_types": ["drills_down", "corroborates"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 4 — RISK PATTERNS
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "employees_multiple_alert_types",
        "capsule_type": "pattern",
        "priority": "P1",
        "what": "Employees who have been flagged with multiple distinct alert types",
        "how": "Count distinct AlertType values per employee, filter to employees with 2 or more",
        "sql": """
SELECT
    e.EmployeeID                             AS employee_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    e.JobTitle                               AS job_title,
    COUNT(ca.AlertID)                        AS total_alerts,
    COUNT(DISTINCT ca.AlertType)             AS distinct_alert_types,
    STUFF((SELECT DISTINCT ', ' + ca2.AlertType FROM ComplianceAlert ca2 WHERE ca2.EmployeeID = e.EmployeeID FOR XML PATH(''), TYPE).value('.','NVARCHAR(MAX)'), 1, 2, '') AS alert_types_list,
    MAX(ca.Severity)                         AS max_severity,
    SUM(CASE WHEN ca.Status IN ('Open','Investigating') THEN 1 ELSE 0 END) AS unresolved_alerts
FROM Employee e
JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle
HAVING COUNT(DISTINCT ca.AlertType) >= 2
ORDER BY distinct_alert_types DESC, total_alerts DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "High-risk employees who have been flagged for multiple distinct types of compliance alerts, "
            "indicating a broad pattern of non-compliance rather than a single incident. "
            "This capsule answers: which employees have both restriction violations AND other alert types, "
            "which employees show the most diverse compliance failure profile, "
            "who are the broadest risk individuals requiring comprehensive investigation. "
            "An employee with 3+ distinct alert types across multiple categories represents "
            "a systemic compliance risk requiring HR and legal involvement alongside remediation. "
            "Related terms: multi-type violations, broad compliance failures, high-risk employee profile. "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert. "
            "Key columns: employee_name, department, distinct_alert_types, alert_types_list, unresolved_alerts."
        ),
        "ttl_hours": 8,
        "tags": ["pattern", "employee_risk", "multi_alert", "P1"],
        "tables_used": ["Employee", "ComplianceAlert"],
        "key_columns": ["employee_name", "department", "distinct_alert_types", "alert_types_list", "unresolved_alerts"],
        "staleness_trigger": "new ComplianceAlert row",
        "related_capsule_ids": ["repeat_violators", "high_severity_open_alerts"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    {
        "capsule_id": "high_severity_open_alerts",
        "capsule_type": "risk",
        "priority": "P1",
        "what": "Open or investigating alerts with Critical or High severity",
        "how": "Filter ComplianceAlert for unresolved high-impact alerts",
        "sql": """
SELECT
    ca.AlertID                               AS alert_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    ca.AlertType                             AS alert_type,
    ca.Severity                              AS severity,
    ca.AlertDate                             AS alert_date,
    ca.Status                                AS status,
    ca.Description                           AS description,
    DATEDIFF(DAY, ca.AlertDate, GETDATE())   AS days_open,
    tr.SecuritySymbol                        AS security_symbol,
    bd.BrokerDealerName                      AS broker_dealer
FROM ComplianceAlert ca
JOIN Employee e ON e.EmployeeID = ca.EmployeeID
LEFT JOIN TradeRequest tr ON tr.TradeRequestID = ca.TradeRequestID
LEFT JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
WHERE ca.Severity IN ('Critical', 'High')
  AND ca.Status IN ('Open', 'Investigating')
ORDER BY ca.Severity DESC, days_open DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "All unresolved Critical and High severity compliance alerts — the immediate action queue "
            "for compliance officers. "
            "This capsule answers: which high-severity alerts are still open and awaiting resolution, "
            "how many days have these critical alerts been unresolved, "
            "which employees and departments have active high-severity compliance issues. "
            "Alerts in Investigating status for more than 5 days warrant escalation to senior management. "
            "Critical alerts with no ResolvedDate after 48 hours may require regulatory notification. "
            "Related terms: open critical alerts, unresolved high risk, compliance backlog, urgent alerts. "
            "Finding: {signal} "
            "Tables: ComplianceAlert, Employee, TradeRequest, BrokerDealer. "
            "Key columns: employee_name, severity, alert_date, status, days_open, description."
        ),
        "ttl_hours": 4,
        "tags": ["risk", "open_alert", "critical", "high_severity", "urgent", "P1"],
        "tables_used": ["ComplianceAlert", "Employee", "TradeRequest", "BrokerDealer"],
        "key_columns": ["employee_name", "severity", "alert_date", "status", "days_open", "description"],
        "staleness_trigger": "any ComplianceAlert status change",
        "related_capsule_ids": ["repeat_violators", "alert_severity_trend"],
        "relationship_types": ["corroborates", "drills_down"],
    },

    {
        "capsule_id": "broker_dealers_high_rejection_and_alerts",
        "capsule_type": "risk",
        "priority": "P1",
        "what": "Broker dealers with both high rejection rates and high alert counts",
        "how": "Join trade stats and alert counts per broker dealer, filter to high-risk profiles",
        "sql": """
SELECT
    bd.BrokerDealerName                      AS broker_dealer,
    bd.Country                               AS country,
    COUNT(DISTINCT tr.TradeRequestID)        AS total_requests,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS rejection_rate_pct,
    COUNT(DISTINCT ca.AlertID)               AS total_alerts,
    SUM(CASE WHEN ca.Severity IN ('Critical','High') THEN 1 ELSE 0 END) AS high_severity_alerts
FROM BrokerDealer bd
LEFT JOIN TradeRequest tr ON tr.BrokerDealerID = bd.BrokerDealerID
LEFT JOIN ComplianceAlert ca
    ON ca.TradeRequestID = tr.TradeRequestID
GROUP BY bd.BrokerDealerID, bd.BrokerDealerName, bd.Country
HAVING COUNT(DISTINCT tr.TradeRequestID) > 0
ORDER BY rejection_rate_pct DESC, total_alerts DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Broker dealers showing both elevated rejection rates and high compliance alert counts — "
            "the combined risk profile that signals systemic broker-level problems. "
            "This capsule answers: which broker dealers have both poor approval rates and a high "
            "number of compliance alerts, identifying those that represent dual-dimension risk. "
            "A broker dealer with >30% rejection rate AND multiple high-severity alerts is a "
            "candidate for enhanced monitoring, reduced trading permissions, or relationship termination. "
            "Related terms: systemic broker risk, dual-metric high-risk broker, counterparty risk profile. "
            "Finding: {signal} "
            "Tables: BrokerDealer, TradeRequest, ComplianceAlert. "
            "Key columns: broker_dealer, rejection_rate_pct, total_alerts, high_severity_alerts."
        ),
        "ttl_hours": 12,
        "tags": ["risk", "broker_dealer", "systemic_risk", "dual_metric", "P1"],
        "tables_used": ["BrokerDealer", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["broker_dealer", "rejection_rate_pct", "total_alerts", "high_severity_alerts"],
        "staleness_trigger": "new TradeRequest or ComplianceAlert row",
        "related_capsule_ids": ["trade_requests_by_broker_dealer", "violations_by_broker_dealer"],
        "relationship_types": ["same_entity", "corroborates"],
    },

    {
        "capsule_id": "escalation_patterns",
        "capsule_type": "pattern",
        "priority": "P2",
        "what": "Characteristics that lead to escalation vs rejection in approval workflow",
        "how": "Compare approved vs escalated requests across dimensions",
        "sql": """
SELECT
    tr.Status                                AS request_status,
    e.Department                             AS department,
    tr.TradeType                             AS trade_type,
    bd.Country                               AS broker_country,
    COUNT(tr.TradeRequestID)                 AS request_count,
    AVG(CAST(tr.Quantity AS FLOAT))          AS avg_quantity,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))    AS avg_turnaround_days,
    SUM(CASE WHEN ca.AlertID IS NOT NULL THEN 1 ELSE 0 END) AS had_compliance_alert
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
LEFT JOIN ApprovalWorkflow aw ON aw.TradeRequestID = tr.TradeRequestID
LEFT JOIN ComplianceAlert ca ON ca.TradeRequestID = tr.TradeRequestID
WHERE tr.Status IN ('Escalated', 'Rejected', 'Approved')
GROUP BY tr.Status, e.Department, tr.TradeType, bd.Country
ORDER BY tr.Status, request_count DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Pattern analysis of what differentiates escalated trade requests from rejections and approvals. "
            "This capsule answers: which departments, trade types, and broker countries are most likely "
            "to lead to escalation versus outright rejection, what is the average quantity profile for "
            "escalated trades compared to rejected ones, do escalated trades consistently have compliance alerts. "
            "Understanding escalation patterns helps predict which pending trades will require "
            "senior compliance review before they are submitted. "
            "Related terms: escalation drivers, what causes escalations, escalation risk factors. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee, BrokerDealer, ApprovalWorkflow, ComplianceAlert. "
            "Key columns: request_status, department, trade_type, request_count, avg_turnaround_days."
        ),
        "ttl_hours": 24,
        "tags": ["pattern", "escalation", "workflow", "risk_factor"],
        "tables_used": ["TradeRequest", "Employee", "BrokerDealer", "ApprovalWorkflow", "ComplianceAlert"],
        "key_columns": ["request_status", "department", "trade_type", "request_count", "avg_turnaround_days"],
        "staleness_trigger": "weekly",
        "related_capsule_ids": ["escalation_trend_by_department", "reviewer_decision_distribution"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 5 — APPROVAL WORKFLOW
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "reviewer_decision_distribution",
        "capsule_type": "operational",
        "priority": "P2",
        "what": "Decision breakdown per reviewer (who approves vs rejects vs escalates)",
        "how": "Count decisions per reviewer employee, compute percentages",
        "sql": """
SELECT
    reviewer.EmployeeName                    AS reviewer_name,
    reviewer.Department                      AS reviewer_department,
    reviewer.JobTitle                        AS reviewer_job_title,
    COUNT(aw.WorkflowID)                     AS total_reviews,
    SUM(CASE WHEN aw.Decision = 'Approved'  THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN aw.Decision = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN aw.Decision = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    SUM(CASE WHEN aw.Decision = 'Pending'   THEN 1 ELSE 0 END) AS pending_count,
    CAST(
        100.0 * SUM(CASE WHEN aw.Decision = 'Approved' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(aw.WorkflowID), 0)
    AS DECIMAL(5,2)) AS approval_rate_pct,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))    AS avg_turnaround_days
FROM ApprovalWorkflow aw
-- ReviewerID references Employee (reviewer is a compliance/risk employee)
JOIN Employee reviewer ON reviewer.EmployeeID = aw.ReviewerID
GROUP BY aw.ReviewerID, reviewer.EmployeeName, reviewer.Department, reviewer.JobTitle
ORDER BY total_reviews DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Approval workflow decision distribution per compliance reviewer — who approves, "
            "rejects, and escalates trade requests and at what rate. "
            "This capsule answers: which reviewers approve the highest proportion of requests, "
            "which reviewers are most likely to escalate rather than decide directly, "
            "what is each reviewer's average turnaround time. "
            "Reviewers with abnormally high approval rates may be rubber-stamping requests. "
            "Reviewers with very high escalation rates may lack decision authority or face complex cases. "
            "Related terms: reviewer performance, approval bias, who approves most trades. "
            "Finding: {signal} "
            "Tables: ApprovalWorkflow, Employee. "
            "Key columns: reviewer_name, total_reviews, approved_count, rejected_count, approval_rate_pct, avg_turnaround_days."
        ),
        "ttl_hours": 24,
        "tags": ["operational", "workflow", "reviewer", "approval_rate"],
        "tables_used": ["ApprovalWorkflow", "Employee"],
        "key_columns": ["reviewer_name", "total_reviews", "approval_rate_pct", "avg_turnaround_days"],
        "staleness_trigger": "new ApprovalWorkflow row",
        "related_capsule_ids": ["avg_turnaround_by_reviewer", "reviewer_coverage_gaps"],
        "relationship_types": ["same_entity", "corroborates"],
    },

    {
        "capsule_id": "avg_turnaround_by_department",
        "capsule_type": "operational",
        "priority": "P3",
        "what": "Average approval turnaround time per requesting department",
        "how": "Average TurnaroundDays from ApprovalWorkflow grouped by employee department",
        "sql": """
SELECT
    e.Department                             AS department,
    COUNT(aw.WorkflowID)                     AS reviewed_requests,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))    AS avg_turnaround_days,
    MIN(aw.TurnaroundDays)                   AS min_turnaround_days,
    MAX(aw.TurnaroundDays)                   AS max_turnaround_days,
    SUM(CASE WHEN aw.TurnaroundDays > 3 THEN 1 ELSE 0 END) AS slow_reviews_over_3_days
FROM ApprovalWorkflow aw
JOIN TradeRequest tr ON tr.TradeRequestID = aw.TradeRequestID
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
GROUP BY e.Department
ORDER BY avg_turnaround_days DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Average trade request approval turnaround time broken down by requesting department. "
            "This capsule answers: which departments wait the longest for trade approvals, "
            "which departments have reviewers who process requests most quickly, "
            "how many department requests take more than 3 days to review. "
            "Long average turnaround for a department can indicate insufficient reviewer capacity, "
            "overly complex trade requests, or systematic compliance issues causing delays. "
            "SLA violation: requests older than 3 days without a decision are SLA breaches. "
            "Related terms: approval speed by department, review latency, workflow efficiency. "
            "Finding: {signal} "
            "Tables: ApprovalWorkflow, TradeRequest, Employee. "
            "Key columns: department, avg_turnaround_days, slow_reviews_over_3_days."
        ),
        "ttl_hours": 24,
        "tags": ["operational", "workflow", "turnaround", "department", "sla"],
        "tables_used": ["ApprovalWorkflow", "TradeRequest", "Employee"],
        "key_columns": ["department", "avg_turnaround_days", "slow_reviews_over_3_days"],
        "staleness_trigger": "new ApprovalWorkflow row",
        "related_capsule_ids": ["avg_turnaround_by_reviewer", "pending_requests_no_review"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    {
        "capsule_id": "avg_turnaround_by_reviewer",
        "capsule_type": "operational",
        "priority": "P3",
        "what": "Average turnaround time per reviewer",
        "how": "Average TurnaroundDays from ApprovalWorkflow per reviewer",
        "sql": """
SELECT
    reviewer.EmployeeName                    AS reviewer_name,
    reviewer.Department                      AS reviewer_department,
    reviewer.JobTitle                        AS reviewer_title,
    COUNT(aw.WorkflowID)                     AS total_reviews,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))    AS avg_turnaround_days,
    MAX(aw.TurnaroundDays)                   AS max_turnaround_days,
    SUM(CASE WHEN aw.TurnaroundDays > 3 THEN 1 ELSE 0 END) AS sla_breaches
FROM ApprovalWorkflow aw
JOIN Employee reviewer ON reviewer.EmployeeID = aw.ReviewerID
GROUP BY aw.ReviewerID, reviewer.EmployeeName, reviewer.Department, reviewer.JobTitle
ORDER BY avg_turnaround_days DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Individual reviewer turnaround time analysis for the compliance approval workflow. "
            "This capsule answers: which reviewers process requests the slowest on average, "
            "which reviewers have the most SLA breaches (decisions taking longer than 3 days), "
            "which reviewer has the highest single review delay (max turnaround). "
            "Reviewers with consistently high turnaround times may need capacity support, "
            "training, or workload rebalancing. SLA breaches above a threshold warrant management attention. "
            "Related terms: reviewer speed, reviewer efficiency, compliance review SLA, individual reviewer metrics. "
            "Finding: {signal} "
            "Tables: ApprovalWorkflow, Employee. "
            "Key columns: reviewer_name, avg_turnaround_days, sla_breaches."
        ),
        "ttl_hours": 24,
        "tags": ["operational", "workflow", "reviewer", "turnaround", "sla"],
        "tables_used": ["ApprovalWorkflow", "Employee"],
        "key_columns": ["reviewer_name", "avg_turnaround_days", "sla_breaches"],
        "staleness_trigger": "new ApprovalWorkflow row",
        "related_capsule_ids": ["reviewer_decision_distribution"],
        "relationship_types": ["same_entity"],
    },

    {
        "capsule_id": "pending_requests_no_review",
        "capsule_type": "operational",
        "priority": "P1",
        "what": "Pending trade requests with no ApprovalWorkflow entry",
        "how": "LEFT JOIN TradeRequest to ApprovalWorkflow, filter where no workflow and status=Pending",
        "sql": """
SELECT
    tr.TradeRequestID                        AS trade_request_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    bd.BrokerDealerName                      AS broker_dealer,
    tr.SecuritySymbol                        AS security_symbol,
    tr.TradeType                             AS trade_type,
    tr.Quantity                              AS quantity,
    tr.RequestDate                           AS request_date,
    DATEDIFF(DAY, tr.RequestDate, GETDATE()) AS days_waiting
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
-- no workflow entry exists for this request
LEFT JOIN ApprovalWorkflow aw ON aw.TradeRequestID = tr.TradeRequestID
WHERE tr.Status = 'Pending'
  AND aw.WorkflowID IS NULL
ORDER BY days_waiting DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trade requests in Pending status that have no ApprovalWorkflow entry — meaning no "
            "reviewer has been assigned or started the review process. "
            "This is the most operationally urgent capsule — these requests are in a dead queue. "
            "This capsule answers: which pending requests have been completely ignored by the review system, "
            "how many days have unreviewed pending requests been waiting, "
            "which departments and broker dealers have the most orphaned pending requests. "
            "Any pending request older than 1 day with no workflow is an SLA breach requiring immediate triage. "
            "Related terms: unreviewed requests, orphan queue, pending without review, stuck requests. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee, BrokerDealer, ApprovalWorkflow. "
            "Key columns: employee_name, department, security_symbol, request_date, days_waiting."
        ),
        "ttl_hours": 2,
        "tags": ["operational", "pending", "no_review", "urgent", "P1"],
        "tables_used": ["TradeRequest", "Employee", "BrokerDealer", "ApprovalWorkflow"],
        "key_columns": ["employee_name", "department", "security_symbol", "request_date", "days_waiting"],
        "staleness_trigger": "any TradeRequest or ApprovalWorkflow change",
        "related_capsule_ids": ["requests_pending_over_3_days", "reviewer_coverage_gaps"],
        "relationship_types": ["same_entity", "corroborates"],
    },

    {
        "capsule_id": "requests_pending_over_3_days",
        "capsule_type": "operational",
        "priority": "P2",
        "what": "Trade requests pending for more than 3 days (SLA breach)",
        "how": "Filter TradeRequest for Pending/Pending-in-workflow requests older than 3 days",
        "sql": """
SELECT
    tr.TradeRequestID                        AS trade_request_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    bd.BrokerDealerName                      AS broker_dealer,
    tr.SecuritySymbol                        AS security_symbol,
    tr.TradeType                             AS trade_type,
    tr.RequestDate                           AS request_date,
    DATEDIFF(DAY, tr.RequestDate, GETDATE()) AS days_waiting,
    aw.Decision                              AS current_decision,
    reviewer.EmployeeName                    AS reviewer_name
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
LEFT JOIN ApprovalWorkflow aw ON aw.TradeRequestID = tr.TradeRequestID
LEFT JOIN Employee reviewer ON reviewer.EmployeeID = aw.ReviewerID
WHERE tr.Status = 'Pending'
  AND tr.RequestDate < DATEADD(DAY, -3, GETDATE())
ORDER BY days_waiting DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trade requests that have been pending for more than 3 days, representing SLA violations "
            "in the compliance approval workflow. "
            "This capsule answers: which specific requests have breached the 3-day review SLA, "
            "which reviewers are assigned to these delayed cases, "
            "which departments have the most SLA-breaching pending requests. "
            "SLA breaches represent compliance workflow failures — delays expose the firm to risk "
            "because employees may act on unreviewed requests or restrictions may change during delays. "
            "Related terms: overdue reviews, SLA breach, late approvals, delayed compliance review. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee, BrokerDealer, ApprovalWorkflow. "
            "Key columns: employee_name, department, request_date, days_waiting, reviewer_name."
        ),
        "ttl_hours": 4,
        "tags": ["operational", "pending", "sla_breach", "overdue"],
        "tables_used": ["TradeRequest", "Employee", "BrokerDealer", "ApprovalWorkflow"],
        "key_columns": ["employee_name", "department", "request_date", "days_waiting", "reviewer_name"],
        "staleness_trigger": "daily",
        "related_capsule_ids": ["pending_requests_no_review"],
        "relationship_types": ["same_entity"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 6 — SECURITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "most_traded_securities_by_volume",
        "capsule_type": "distribution",
        "priority": "P2",
        "what": "Securities ranked by total trade quantity",
        "how": "Sum Quantity per SecuritySymbol across all TradeRequests",
        "sql": """
SELECT TOP 20
    tr.SecuritySymbol                        AS security_symbol,
    COUNT(tr.TradeRequestID)                 AS request_count,
    SUM(tr.Quantity)                         AS total_quantity,
    SUM(CASE WHEN tr.TradeType = 'BUY'  THEN tr.Quantity ELSE 0 END) AS buy_quantity,
    SUM(CASE WHEN tr.TradeType = 'SELL' THEN tr.Quantity ELSE 0 END) AS sell_quantity,
    COUNT(DISTINCT tr.EmployeeID)            AS unique_employees,
    COUNT(DISTINCT tr.BrokerDealerID)        AS unique_brokers
FROM TradeRequest tr
GROUP BY tr.SecuritySymbol
ORDER BY total_quantity DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Top 20 securities by total trade quantity — showing which ticker symbols represent "
            "the largest volume of trading activity in the compliance system. "
            "This capsule answers: which securities are traded in the highest quantities, "
            "how many unique employees trade each high-volume security, "
            "what is the buy vs sell quantity split for the most active securities. "
            "High-volume securities warrant enhanced monitoring especially if they also appear "
            "on any restriction list or have a history of compliance alerts. "
            "Related terms: most traded securities, highest volume stocks, top positions by quantity. "
            "Finding: {signal} "
            "Tables: TradeRequest. "
            "Key columns: security_symbol, total_quantity, buy_quantity, sell_quantity, unique_employees."
        ),
        "ttl_hours": 24,
        "tags": ["distribution", "security_symbol", "volume", "quantity"],
        "tables_used": ["TradeRequest"],
        "key_columns": ["security_symbol", "total_quantity", "request_count", "unique_employees"],
        "staleness_trigger": "new TradeRequest row",
        "related_capsule_ids": ["trade_requests_by_security_symbol", "securities_with_restriction_history"],
        "relationship_types": ["same_entity", "corroborates"],
    },

    {
        "capsule_id": "securities_with_restriction_history",
        "capsule_type": "distribution",
        "priority": "P2",
        "what": "Securities that have ever appeared on a restriction list",
        "how": "Group RestrictedSecurity by symbol, count restrictions and type coverage",
        "sql": """
SELECT
    rs.SecuritySymbol                        AS security_symbol,
    COUNT(rs.RestrictionID)                  AS restriction_count,
    COUNT(DISTINCT rs.RestrictionType)       AS distinct_restriction_types,
    MIN(rs.StartDate)                        AS earliest_restriction,
    MAX(rs.StartDate)                        AS latest_restriction,
    SUM(CASE WHEN rs.EndDate IS NULL THEN 1 ELSE 0 END) AS currently_active_restrictions
FROM RestrictedSecurity rs
GROUP BY rs.SecuritySymbol
ORDER BY restriction_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Securities that have historically appeared on restriction lists — showing their restriction "
            "frequency, types of restrictions applied, and current active status. "
            "This capsule answers: which securities have the most restriction history, "
            "which securities have been placed under multiple different restriction types, "
            "which historically restricted securities are still under active restriction today. "
            "Securities with multiple restriction types across their history may represent "
            "chronically sensitive instruments requiring permanent enhanced monitoring. "
            "Related terms: restriction history, securities with compliance records, repeat restricted tickers. "
            "Finding: {signal} "
            "Tables: RestrictedSecurity. "
            "Key columns: security_symbol, restriction_count, distinct_restriction_types, currently_active_restrictions."
        ),
        "ttl_hours": 24,
        "tags": ["distribution", "security_symbol", "restriction_history"],
        "tables_used": ["RestrictedSecurity"],
        "key_columns": ["security_symbol", "restriction_count", "distinct_restriction_types", "currently_active_restrictions"],
        "staleness_trigger": "new RestrictedSecurity row",
        "related_capsule_ids": ["currently_active_restrictions"],
        "relationship_types": ["aggregates_up"],
    },

    {
        "capsule_id": "currently_active_restrictions",
        "capsule_type": "distribution",
        "priority": "P1",
        "what": "Securities currently on an active restriction (EndDate IS NULL or future)",
        "how": "Filter RestrictedSecurity where restriction is live today",
        "sql": """
SELECT
    rs.RestrictionID                         AS restriction_id,
    rs.SecuritySymbol                        AS security_symbol,
    rs.RestrictionType                       AS restriction_type,
    rs.StartDate                             AS start_date,
    rs.EndDate                               AS end_date,
    rs.Reason                                AS reason,
    rs.AddedBy                               AS added_by,
    DATEDIFF(DAY, rs.StartDate, GETDATE())   AS days_active
FROM RestrictedSecurity rs
WHERE rs.EndDate IS NULL OR rs.EndDate >= GETDATE()
ORDER BY rs.RestrictionType ASC, rs.StartDate DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Complete list of trading restrictions currently in force — securities where "
            "EndDate is NULL (permanent/indefinite) or set to a future date. "
            "This capsule answers: how many securities are on active trading restrictions right now, "
            "which restriction types are most common among current restrictions, "
            "which restrictions have been active the longest without an end date. "
            "This is the definitive reference for pre-trade compliance checking. "
            "Any trade request for a symbol in this capsule requires immediate compliance review. "
            "Related terms: active restrictions, current blacklist, live trading ban, active blackout, "
            "current insider list, current watch list. "
            "Finding: {signal} "
            "Tables: RestrictedSecurity. "
            "Key columns: security_symbol, restriction_type, start_date, end_date, days_active."
        ),
        "ttl_hours": 4,
        "tags": ["distribution", "active_restriction", "current", "urgent", "P1"],
        "tables_used": ["RestrictedSecurity"],
        "key_columns": ["security_symbol", "restriction_type", "start_date", "end_date", "days_active"],
        "staleness_trigger": "any RestrictedSecurity change",
        "related_capsule_ids": ["active_restrictions_recent_trade_attempts", "securities_with_restriction_history"],
        "relationship_types": ["drills_down", "drills_down"],
    },

    {
        "capsule_id": "securities_in_restrictions_and_alerts",
        "capsule_type": "distribution",
        "priority": "P1",
        "what": "Securities appearing in both restriction records and compliance alerts",
        "how": "Join RestrictedSecurity symbols to TradeRequest symbols that triggered alerts",
        "sql": """
SELECT
    rs.SecuritySymbol                        AS security_symbol,
    COUNT(DISTINCT rs.RestrictionID)         AS restriction_entries,
    COUNT(DISTINCT ca.AlertID)               AS alert_count,
    MAX(ca.Severity)                         AS max_alert_severity,
    COUNT(DISTINCT ca.EmployeeID)            AS unique_alerted_employees,
    SUM(CASE WHEN rs.EndDate IS NULL THEN 1 ELSE 0 END) AS active_restrictions
FROM RestrictedSecurity rs
JOIN TradeRequest tr ON tr.SecuritySymbol = rs.SecuritySymbol
JOIN ComplianceAlert ca ON ca.TradeRequestID = tr.TradeRequestID
GROUP BY rs.SecuritySymbol
ORDER BY alert_count DESC, restriction_entries DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Securities at the intersection of trading restrictions and compliance alerts — "
            "those that have both been officially restricted and have triggered compliance alerts. "
            "This capsule answers: which securities appear in both the restriction registry and the alert system, "
            "how many alerts were raised for trades in restricted securities, "
            "which securities represent the combined worst-case scenario of active restriction plus alert history. "
            "These dual-flagged securities require the most intensive ongoing monitoring and "
            "any new trade attempt should trigger immediate escalation. "
            "Finding: {signal} "
            "Tables: RestrictedSecurity, TradeRequest, ComplianceAlert. "
            "Key columns: security_symbol, restriction_entries, alert_count, max_alert_severity."
        ),
        "ttl_hours": 8,
        "tags": ["distribution", "security_symbol", "dual_flag", "restriction", "alert"],
        "tables_used": ["RestrictedSecurity", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["security_symbol", "restriction_entries", "alert_count", "max_alert_severity"],
        "staleness_trigger": "new alert or restriction",
        "related_capsule_ids": ["currently_active_restrictions", "violations_on_restricted_securities"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 7 — DEPARTMENT & EMPLOYEE HEALTH
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "department_compliance_scorecard",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Combined compliance scorecard per department",
        "how": "Join requests, rejections, escalations, and alerts per department into one scorecard",
        "sql": """
SELECT
    e.Department                             AS department,
    COUNT(DISTINCT e.EmployeeID)             AS headcount,
    COUNT(DISTINCT tr.TradeRequestID)        AS total_requests,
    SUM(CASE WHEN tr.Status = 'Rejected'  THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    COUNT(DISTINCT ca.AlertID)               AS total_alerts,
    SUM(CASE WHEN ca.Severity IN ('Critical','High') THEN 1 ELSE 0 END) AS high_severity_alerts,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status IN ('Rejected','Escalated') THEN 1 ELSE 0 END)
        / NULLIF(COUNT(DISTINCT tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS non_approval_rate_pct
FROM Employee e
LEFT JOIN TradeRequest tr ON tr.EmployeeID = e.EmployeeID
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
GROUP BY e.Department
ORDER BY high_severity_alerts DESC, non_approval_rate_pct DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Comprehensive compliance scorecard for each department combining all key metrics: "
            "headcount, trade request volume, rejection rate, escalation rate, total alerts, "
            "and high severity alert count. "
            "This capsule answers: which department has the worst overall compliance profile, "
            "how does each department compare across all compliance dimensions simultaneously, "
            "which departments are clean vs problematic across the full scorecard. "
            "The scorecard is the primary tool for department-level compliance review presentations "
            "and regulatory reporting on organizational risk distribution. "
            "Related terms: department health, business unit scorecard, division compliance rating. "
            "Finding: {signal} "
            "Tables: Employee, TradeRequest, ComplianceAlert. "
            "Key columns: department, headcount, non_approval_rate_pct, total_alerts, high_severity_alerts."
        ),
        "ttl_hours": 24,
        "tags": ["aggregation", "department", "scorecard", "health_metric"],
        "tables_used": ["Employee", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["department", "headcount", "non_approval_rate_pct", "total_alerts", "high_severity_alerts"],
        "staleness_trigger": "new TradeRequest or ComplianceAlert",
        "related_capsule_ids": ["trade_requests_by_department", "violations_by_department"],
        "relationship_types": ["aggregates_up", "aggregates_up"],
    },

    {
        "capsule_id": "employees_zero_alerts",
        "capsule_type": "aggregation",
        "priority": "P4",
        "what": "Employees with no compliance alerts on record",
        "how": "LEFT JOIN Employee to ComplianceAlert, filter to those with no alerts",
        "sql": """
SELECT
    e.EmployeeID                             AS employee_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    e.JobTitle                               AS job_title,
    e.HireDate                               AS hire_date,
    COUNT(tr.TradeRequestID)                 AS total_requests,
    DATEDIFF(YEAR, e.HireDate, GETDATE())    AS years_of_service
FROM Employee e
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
LEFT JOIN TradeRequest tr ON tr.EmployeeID = e.EmployeeID
WHERE ca.AlertID IS NULL
  AND e.Status = 'Active'
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle, e.HireDate
ORDER BY years_of_service DESC, total_requests DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Active employees with a completely clean compliance record — no alerts of any kind. "
            "This capsule answers: how many active employees have never received a compliance alert, "
            "which departments have the highest proportion of clean-record employees, "
            "how many years of service do compliant employees average. "
            "This positive signal helps assess overall compliance culture and identifies departments "
            "with strong trading discipline that could serve as models for others. "
            "Related terms: clean record employees, no violations, compliant employees, zero alerts. "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert, TradeRequest. "
            "Key columns: employee_name, department, total_requests, years_of_service."
        ),
        "ttl_hours": 48,
        "tags": ["aggregation", "employee", "clean_record", "positive_indicator"],
        "tables_used": ["Employee", "ComplianceAlert", "TradeRequest"],
        "key_columns": ["employee_name", "department", "total_requests", "years_of_service"],
        "staleness_trigger": "new ComplianceAlert row",
        "related_capsule_ids": ["department_compliance_scorecard"],
        "relationship_types": ["corroborates"],
    },

    {
        "capsule_id": "alert_rate_by_job_title",
        "capsule_type": "aggregation",
        "priority": "P3",
        "what": "Compliance alert rate per job title",
        "how": "Count employees and alerts per job title, compute alerts-per-employee",
        "sql": """
SELECT
    e.JobTitle                               AS job_title,
    COUNT(DISTINCT e.EmployeeID)             AS employee_count,
    COUNT(DISTINCT ca.AlertID)               AS total_alerts,
    CAST(
        1.0 * COUNT(DISTINCT ca.AlertID)
        / NULLIF(COUNT(DISTINCT e.EmployeeID), 0)
    AS DECIMAL(5,2)) AS alerts_per_employee,
    SUM(CASE WHEN ca.Severity IN ('Critical','High') THEN 1 ELSE 0 END) AS high_severity_alerts
FROM Employee e
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
GROUP BY e.JobTitle
ORDER BY alerts_per_employee DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Compliance alert rate broken down by employee job title — identifying which roles "
            "produce the most compliance risk per employee. "
            "This capsule answers: which job titles have the highest compliance alert rates, "
            "are senior roles (directors, VPs) or junior roles (analysts, associates) "
            "generating more alerts per person, which specific job title has the worst compliance record. "
            "High-alert roles may need role-specific compliance training or trading restrictions. "
            "Related terms: role-based risk, job title compliance profile, position risk analysis. "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert. "
            "Key columns: job_title, employee_count, alerts_per_employee, high_severity_alerts."
        ),
        "ttl_hours": 48,
        "tags": ["aggregation", "job_title", "alert_rate", "role_risk"],
        "tables_used": ["Employee", "ComplianceAlert"],
        "key_columns": ["job_title", "employee_count", "alerts_per_employee", "high_severity_alerts"],
        "staleness_trigger": "new ComplianceAlert row",
        "related_capsule_ids": ["department_compliance_scorecard"],
        "relationship_types": ["drills_down"],
    },

    {
        "capsule_id": "new_employee_compliance",
        "capsule_type": "aggregation",
        "priority": "P3",
        "what": "Compliance profile of employees hired within the last 2 years",
        "how": "Filter Employee for HireDate < 2 years ago, count their alerts",
        "sql": """
SELECT
    e.EmployeeID                             AS employee_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    e.JobTitle                               AS job_title,
    e.HireDate                               AS hire_date,
    DATEDIFF(MONTH, e.HireDate, GETDATE())   AS months_of_service,
    COUNT(DISTINCT tr.TradeRequestID)        AS total_requests,
    COUNT(DISTINCT ca.AlertID)               AS total_alerts,
    MAX(ca.Severity)                         AS max_severity
FROM Employee e
LEFT JOIN TradeRequest tr ON tr.EmployeeID = e.EmployeeID
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
WHERE e.HireDate >= DATEADD(YEAR, -2, GETDATE())
  AND e.Status = 'Active'
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle, e.HireDate
ORDER BY total_alerts DESC, months_of_service ASC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Compliance alert profile for employees who joined in the last 2 years — "
            "a key metric for assessing onboarding compliance effectiveness. "
            "This capsule answers: how many new employees already have compliance alerts, "
            "which departments have the most non-compliant new hires, "
            "are recently hired employees generating more alerts per month than their tenure suggests. "
            "New employees with alerts early in their tenure indicate onboarding compliance training gaps. "
            "Related terms: new hire compliance, recent employee alerts, onboarding risk, junior employee violations. "
            "Finding: {signal} "
            "Tables: Employee, TradeRequest, ComplianceAlert. "
            "Key columns: employee_name, department, months_of_service, total_alerts, max_severity."
        ),
        "ttl_hours": 48,
        "tags": ["aggregation", "new_employee", "onboarding", "compliance_profile"],
        "tables_used": ["Employee", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["employee_name", "department", "months_of_service", "total_alerts"],
        "staleness_trigger": "new ComplianceAlert or Employee row",
        "related_capsule_ids": ["department_compliance_scorecard"],
        "relationship_types": ["drills_down"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 8 — CROSS-ENTITY RISK
    # ══════════════════════════════════════════════════════════════════════════

    {
        "capsule_id": "employee_broker_dealer_combination_risk",
        "capsule_type": "risk",
        "priority": "P1",
        "what": "Employees with compliance alerts across multiple broker dealers",
        "how": "Find employees whose alerted trades span 2+ different broker dealers",
        "sql": """
SELECT
    e.EmployeeID                             AS employee_id,
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    e.JobTitle                               AS job_title,
    COUNT(DISTINCT tr.BrokerDealerID)        AS broker_dealers_involved,
    COUNT(DISTINCT ca.AlertID)               AS total_alerts,
    COUNT(DISTINCT ca.AlertType)             AS distinct_alert_types,
    STUFF((SELECT DISTINCT ' | ' + bd2.BrokerDealerName FROM Account a2 JOIN BrokerDealer bd2 ON a2.BrokerDealerID = bd2.BrokerDealerID WHERE a2.EmployeeID = e.EmployeeID FOR XML PATH(''), TYPE).value('.','NVARCHAR(MAX)'), 1, 3, '') AS broker_dealer_names,
    MAX(ca.Severity)                         AS max_severity
FROM Employee e
JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
JOIN TradeRequest tr ON tr.TradeRequestID = ca.TradeRequestID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle
HAVING COUNT(DISTINCT tr.BrokerDealerID) >= 2
ORDER BY broker_dealers_involved DESC, total_alerts DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Employees whose compliance alerts span multiple broker dealer counterparties — "
            "the highest individual risk profile combining personal violation history with "
            "cross-counterparty exposure. "
            "This capsule answers: which employees have compliance issues with more than one broker dealer, "
            "are any employees using multiple brokers to circumvent monitoring, "
            "which employees have the broadest broker dealer violation footprint. "
            "An employee with alerts at 3+ broker dealers represents a potential deliberate circumvention "
            "pattern requiring immediate investigation and possible trading suspension. "
            "Related terms: multi-broker violations, cross-counterparty risk, employee broker risk. "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert, TradeRequest, BrokerDealer. "
            "Key columns: employee_name, department, broker_dealers_involved, total_alerts, broker_dealer_names."
        ),
        "ttl_hours": 8,
        "tags": ["risk", "employee_risk", "broker_dealer", "cross_entity", "P1"],
        "tables_used": ["Employee", "ComplianceAlert", "TradeRequest", "BrokerDealer"],
        "key_columns": ["employee_name", "department", "broker_dealers_involved", "total_alerts", "broker_dealer_names"],
        "staleness_trigger": "new ComplianceAlert row",
        "related_capsule_ids": ["repeat_violators", "broker_dealers_high_rejection_and_alerts"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    {
        "capsule_id": "department_restriction_type_concentration",
        "capsule_type": "risk",
        "priority": "P2",
        "what": "Which departments repeatedly trigger the same restriction type",
        "how": "Count violations per department-restriction_type pair",
        "sql": """
SELECT
    e.Department                             AS department,
    rs.RestrictionType                       AS restriction_type,
    COUNT(tr.TradeRequestID)                 AS violation_count,
    COUNT(DISTINCT tr.EmployeeID)            AS unique_employees,
    COUNT(DISTINCT tr.SecuritySymbol)        AS unique_securities,
    MAX(tr.RequestDate)                      AS latest_violation_date
FROM TradeRequest tr
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
GROUP BY e.Department, rs.RestrictionType
ORDER BY violation_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Concentration analysis of which departments repeatedly violate the same restriction type — "
            "revealing structural compliance gaps specific to certain business units and rule categories. "
            "This capsule answers: which department-restriction type combination is most frequently violated, "
            "does a particular department consistently ignore Blackout restrictions while another hits Insider List, "
            "which departments have a single dominant restriction type that explains most of their violations. "
            "Concentrated violations of a single restriction type in one department suggest a specific "
            "knowledge gap or cultural problem with that particular compliance rule. "
            "Finding: {signal} "
            "Tables: TradeRequest, RestrictedSecurity, Employee. "
            "Key columns: department, restriction_type, violation_count, unique_employees."
        ),
        "ttl_hours": 24,
        "tags": ["risk", "department", "restriction_type", "concentration", "cross_entity"],
        "tables_used": ["TradeRequest", "RestrictedSecurity", "Employee"],
        "key_columns": ["department", "restriction_type", "violation_count", "unique_employees"],
        "staleness_trigger": "new violation",
        "related_capsule_ids": ["violations_by_department", "violations_by_restriction_type"],
        "relationship_types": ["drills_down", "drills_down"],
    },

    {
        "capsule_id": "reviewer_coverage_gaps",
        "capsule_type": "risk",
        "priority": "P2",
        "what": "Departments whose trade requests are not being reviewed",
        "how": "Identify departments where requests lack matching ApprovalWorkflow entries",
        "sql": """
SELECT
    e.Department                             AS department,
    COUNT(DISTINCT tr.TradeRequestID)        AS total_requests,
    SUM(CASE WHEN aw.WorkflowID IS NULL THEN 1 ELSE 0 END) AS unreviewed_count,
    CAST(
        100.0 * SUM(CASE WHEN aw.WorkflowID IS NULL THEN 1 ELSE 0 END)
        / NULLIF(COUNT(DISTINCT tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS unreviewed_rate_pct,
    COUNT(DISTINCT aw.ReviewerID)            AS distinct_reviewers
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
LEFT JOIN ApprovalWorkflow aw ON aw.TradeRequestID = tr.TradeRequestID
GROUP BY e.Department
ORDER BY unreviewed_rate_pct DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Reviewer coverage analysis showing which departments have trade requests that are not "
            "receiving timely compliance review — exposing systemic workflow gaps. "
            "This capsule answers: which departments have the highest proportion of unreviewed requests, "
            "are any departments entirely without reviewer coverage, "
            "how many distinct reviewers service each department's compliance queue. "
            "A department with >20% unreviewed requests represents a critical workflow gap — "
            "employees may be trading without proper compliance oversight. "
            "Related terms: review coverage, unreviewed departments, compliance oversight gaps, reviewer assignment. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee, ApprovalWorkflow. "
            "Key columns: department, unreviewed_count, unreviewed_rate_pct, distinct_reviewers."
        ),
        "ttl_hours": 8,
        "tags": ["risk", "workflow", "coverage_gap", "reviewer", "department"],
        "tables_used": ["TradeRequest", "Employee", "ApprovalWorkflow"],
        "key_columns": ["department", "unreviewed_count", "unreviewed_rate_pct", "distinct_reviewers"],
        "staleness_trigger": "new TradeRequest or ApprovalWorkflow change",
        "related_capsule_ids": ["pending_requests_no_review", "reviewer_decision_distribution"],
        "relationship_types": ["corroborates", "corroborates"],
    },

    {
        "capsule_id": "full_risk_profile_join",
        "capsule_type": "risk",
        "priority": "P1",
        "what": "Full cross-table risk profile joining Employee, TradeRequest, ComplianceAlert, ApprovalWorkflow, RestrictedSecurity",
        "how": "Five-table join to identify the most high-risk trade instances with all context",
        "sql": """
SELECT TOP 50
    e.EmployeeName                           AS employee_name,
    e.Department                             AS department,
    e.JobTitle                               AS job_title,
    tr.TradeRequestID                        AS trade_request_id,
    tr.SecuritySymbol                        AS security_symbol,
    tr.TradeType                             AS trade_type,
    tr.Quantity                              AS quantity,
    tr.RequestDate                           AS request_date,
    tr.Status                               AS request_status,
    bd.BrokerDealerName                      AS broker_dealer,
    ca.AlertType                             AS alert_type,
    ca.Severity                              AS alert_severity,
    ca.Status                                AS alert_status,
    aw.Decision                              AS reviewer_decision,
    aw.TurnaroundDays                        AS turnaround_days,
    rs.RestrictionType                       AS restriction_type,
    rs.Reason                                AS restriction_reason
FROM TradeRequest tr
JOIN Employee e ON e.EmployeeID = tr.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = tr.BrokerDealerID
JOIN ComplianceAlert ca ON ca.TradeRequestID = tr.TradeRequestID
-- Only include trades that also hit a restriction
JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
    AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
LEFT JOIN ApprovalWorkflow aw ON aw.TradeRequestID = tr.TradeRequestID
ORDER BY ca.Severity DESC, tr.RequestDate DESC
""".strip(),
        "signal_method": "llm_summary",
        "embed_text_template": (
            "Full five-table risk profile joining every key compliance entity — Employee, TradeRequest, "
            "ComplianceAlert, ApprovalWorkflow, and RestrictedSecurity — for the highest risk trade instances. "
            "This capsule answers: which specific trades have the worst combined risk profile, "
            "which employees have trades with BOTH a restriction violation AND a compliance alert, "
            "what was the reviewer decision and turnaround on the most critical violation trades. "
            "This is the most comprehensive cross-entity risk view in the compliance system. "
            "Use this capsule for detailed regulatory investigations, breach incident reports, "
            "or when the full audit trail for a specific risk event is needed. "
            "Related terms: full audit, cross-entity risk, compliance incident details, violation audit trail. "
            "Finding: {signal} "
            "Tables: TradeRequest, Employee, BrokerDealer, ComplianceAlert, ApprovalWorkflow, RestrictedSecurity. "
            "Key columns: employee_name, security_symbol, alert_severity, restriction_type, reviewer_decision."
        ),
        "ttl_hours": 8,
        "tags": ["risk", "cross_entity", "full_join", "audit", "P1"],
        "tables_used": ["TradeRequest", "Employee", "BrokerDealer", "ComplianceAlert", "ApprovalWorkflow", "RestrictedSecurity"],
        "key_columns": ["employee_name", "security_symbol", "alert_severity", "restriction_type", "reviewer_decision"],
        "staleness_trigger": "any data change",
        "related_capsule_ids": ["violations_on_restricted_securities", "high_severity_open_alerts"],
        "relationship_types": ["aggregates_up", "corroborates"],
    },

    {
        "capsule_id": "trade_requests_by_employee",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Trade request volume per employee",
        "how": "Count total trade requests grouped by employee",
        "sql": """
SELECT
    e.EmployeeName                       AS employee_name,
    e.Department                         AS department,
    e.JobTitle                           AS job_title,
    COUNT(tr.TradeRequestID)             AS total_requests,
    SUM(CASE WHEN tr.Status = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) AS rejected_count,
    SUM(CASE WHEN tr.Status = 'Escalated' THEN 1 ELSE 0 END) AS escalated_count,
    CAST(
        100.0 * SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(tr.TradeRequestID), 0)
    AS DECIMAL(5,2)) AS rejection_rate_pct
FROM Employee e
JOIN TradeRequest tr ON tr.EmployeeID = e.EmployeeID
GROUP BY e.EmployeeID, e.EmployeeName, e.Department, e.JobTitle
ORDER BY total_requests DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Trade request activity by individual employee, showing staff with the highest trading volume. "
            "This capsule answers: which individuals appear repeatedly in trade request activity across the dataset, "
            "which staff members seem unusually active in personal trading, and what their approval/rejection rates are. "
            "Finding: {signal} "
            "Tables: Employee, TradeRequest."
        ),
        "ttl_hours": 24,
        "tags": ["employee", "volume", "aggregation"],
        "tables_used": ["Employee", "TradeRequest"],
        "key_columns": ["employee_name", "total_requests", "rejected_count"],
        "staleness_trigger": "new TradeRequest",
        "related_capsule_ids": ["trade_requests_by_department"],
        "relationship_types": ["aggregates_up"],
    },

    {
        "capsule_id": "daily_trading_concentration_by_security",
        "capsule_type": "pattern",
        "priority": "P2",
        "what": "Security showing unusually high trading concentration on a given day",
        "how": "Group trades by RequestDate and SecuritySymbol and find peaks",
        "sql": """
SELECT top 20
    tr.RequestDate                       AS request_date,
    tr.SecuritySymbol                    AS security_symbol,
    COUNT(tr.TradeRequestID)             AS distinct_requests,
    COUNT(DISTINCT tr.EmployeeID)        AS unique_employees,
    SUM(tr.Quantity)                     AS total_quantity
FROM TradeRequest tr
GROUP BY tr.RequestDate, tr.SecuritySymbol
HAVING COUNT(DISTINCT tr.EmployeeID) >= 3 OR COUNT(tr.TradeRequestID) >= 5
ORDER BY distinct_requests DESC, total_quantity DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Detects when one security dominates trading on a given day revealing unusual herd behavior. "
            "Answers: Which security shows unusually high trading concentration on a given day? "
            "Finding: {signal} "
            "Tables: TradeRequest."
        ),
        "ttl_hours": 24,
        "tags": ["security", "concentration", "pattern", "daily"],
        "tables_used": ["TradeRequest"],
        "key_columns": ["request_date", "security_symbol", "distinct_requests"],
        "staleness_trigger": "daily",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "broker_dealers_high_rejection_and_alerts",
        "capsule_type": "risk",
        "priority": "P2",
        "what": "Broker dealers combining high rejection rates and multiple compliance alerts",
        "how": "Join TradeRequest rejections and ComplianceAlert counts by broker dealer",
        "sql": """
SELECT
    bd.BrokerDealerName                  AS broker_dealer,
    COUNT(DISTINCT tr.TradeRequestID)    AS total_requests,
    SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) AS rejected_requests,
    COUNT(DISTINCT ca.AlertID)           AS total_alerts,
    CAST(100.0 * SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT tr.TradeRequestID), 0) AS DECIMAL(5,2)) as rejection_rate_pct
FROM BrokerDealer bd
JOIN Account a ON a.BrokerDealerID = bd.BrokerDealerID
LEFT JOIN TradeRequest tr ON tr.EmployeeID = a.EmployeeID
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = a.EmployeeID
GROUP BY bd.BrokerDealerID, bd.BrokerDealerName
HAVING COUNT(DISTINCT ca.AlertID) > 0 AND SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) > 0
ORDER BY rejection_rate_pct DESC, total_alerts DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Identifies broker dealers that look riskiest because they combine high rejection rates and high alert counts. "
            "Finding: {signal} "
            "Tables: BrokerDealer, Account, TradeRequest, ComplianceAlert."
        ),
        "ttl_hours": 24,
        "tags": ["broker_dealer", "risk", "rejection", "alert"],
        "tables_used": ["BrokerDealer", "TradeRequest", "ComplianceAlert", "Account"],
        "key_columns": ["broker_dealer", "rejection_rate_pct", "total_alerts"],
        "staleness_trigger": "weekly",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "alert_rate_by_job_title",
        "capsule_type": "aggregation",
        "priority": "P3",
        "what": "Compliance alert distribution by job title",
        "how": "Count alerts grouped by the employee's JobTitle",
        "sql": """
SELECT
    e.JobTitle                           AS job_title,
    COUNT(DISTINCT e.EmployeeID)         AS total_employees,
    COUNT(ca.AlertID)                    AS total_alerts,
    CAST(COUNT(ca.AlertID) * 1.0 / NULLIF(COUNT(DISTINCT e.EmployeeID), 0) AS DECIMAL(5,2)) AS alerts_per_employee
FROM Employee e
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = e.EmployeeID
GROUP BY e.JobTitle
ORDER BY alerts_per_employee DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Compliance issues distributed across different job titles. "
            "Answers: Which job titles appear to have the highest alert rate? "
            "Finding: {signal} "
            "Tables: Employee, ComplianceAlert."
        ),
        "ttl_hours": 48,
        "tags": ["job_title", "alert", "aggregation"],
        "tables_used": ["Employee", "ComplianceAlert"],
        "key_columns": ["job_title", "total_alerts", "alerts_per_employee"],
        "staleness_trigger": "weekly",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "high_risk_employee_broker_combinations",
        "capsule_type": "risk",
        "priority": "P2",
        "what": "Employee and Broker Dealer combinations with highest risk signals",
        "how": "Group rejections and alerts by Employee and Broker Dealer",
        "sql": """
SELECT top 50
    e.EmployeeName                       AS employee_name,
    bd.BrokerDealerName                  AS broker_dealer,
    COUNT(DISTINCT tr.TradeRequestID)    AS total_requests,
    SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) AS rejected_requests,
    COUNT(DISTINCT ca.AlertID)           AS total_alerts
FROM Account a
JOIN Employee e ON e.EmployeeID = a.EmployeeID
JOIN BrokerDealer bd ON bd.BrokerDealerID = a.BrokerDealerID
LEFT JOIN TradeRequest tr ON tr.EmployeeID = a.EmployeeID AND tr.Status = 'Rejected'
LEFT JOIN ComplianceAlert ca ON ca.EmployeeID = a.EmployeeID
GROUP BY e.EmployeeID, e.EmployeeName, bd.BrokerDealerID, bd.BrokerDealerName
HAVING SUM(CASE WHEN tr.Status = 'Rejected' THEN 1 ELSE 0 END) > 0 OR COUNT(DISTINCT ca.AlertID) > 0
ORDER BY rejected_requests DESC, total_alerts DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Highlights which employee and broker dealer combinations look riskiest based on rejected trades and alerts. "
            "Finding: {signal} "
            "Tables: Employee, BrokerDealer, Account, TradeRequest, ComplianceAlert."
        ),
        "ttl_hours": 24,
        "tags": ["employee", "broker_dealer", "risk"],
        "tables_used": ["Employee", "BrokerDealer", "Account", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["employee_name", "broker_dealer", "rejected_requests", "total_alerts"],
        "staleness_trigger": "weekly",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "securities_in_restrictions_and_alerts",
        "capsule_type": "pattern",
        "priority": "P2",
        "what": "Securities appearing in both active restrictions and compliance alerts",
        "how": "Inner join between RestrictedSecurity and TradeRequest linked to ComplianceAlert",
        "sql": """
SELECT
    rs.SecuritySymbol                    AS security_symbol,
    COUNT(DISTINCT rs.RestrictionID)     AS restriction_count,
    COUNT(DISTINCT ca.AlertID)           AS alert_count,
    STUFF((SELECT DISTINCT ', ' + rs2.RestrictionType FROM RestrictedSecurity rs2 WHERE rs2.SecuritySymbol = rs.SecuritySymbol FOR XML PATH(''), TYPE).value('.','NVARCHAR(MAX)'), 1, 2, '') AS restriction_types
FROM RestrictedSecurity rs
JOIN TradeRequest tr ON tr.SecuritySymbol = rs.SecuritySymbol
JOIN ComplianceAlert ca ON ca.TradeRequestID = tr.TradeRequestID
GROUP BY rs.SecuritySymbol
HAVING COUNT(DISTINCT ca.AlertID) > 0
ORDER BY alert_count DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Identifies which securities appear in both restrictions and alerts, signaling high targeted risk. "
            "Finding: {signal} "
            "Tables: RestrictedSecurity, TradeRequest, ComplianceAlert."
        ),
        "ttl_hours": 24,
        "tags": ["security", "restriction", "alert", "risk"],
        "tables_used": ["RestrictedSecurity", "TradeRequest", "ComplianceAlert"],
        "key_columns": ["security_symbol", "restriction_count", "alert_count", "restriction_types"],
        "staleness_trigger": "daily",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "average_approval_time_by_department",
        "capsule_type": "aggregation",
        "priority": "P2",
        "what": "Average approval turnaround time per department",
        "how": "Average the TurnaroundDays in ApprovalWorkflow grouped by employee's department",
        "sql": """
SELECT
    e.Department                           AS department,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))  AS avg_turnaround_days,
    COUNT(aw.WorkflowID)                   AS total_reviews,
    SUM(CASE WHEN aw.Decision = 'Rejected'  THEN 1 ELSE 0 END) AS rejections,
    SUM(CASE WHEN aw.Decision = 'Escalated' THEN 1 ELSE 0 END) AS escalations
FROM ApprovalWorkflow aw
JOIN TradeRequest tr ON aw.TradeRequestID = tr.TradeRequestID
JOIN Employee e ON tr.EmployeeID = e.EmployeeID
WHERE aw.TurnaroundDays IS NOT NULL
GROUP BY e.Department
ORDER BY avg_turnaround_days DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Measures how long compliance reviews take on average for different departments. "
            "Answers: Which departments have the slowest approval turnaround? "
            "Finding: {signal} "
            "Tables: ApprovalWorkflow, TradeRequest, Employee."
        ),
        "ttl_hours": 24,
        "tags": ["department", "approval", "turnaround", "aggregation"],
        "tables_used": ["ApprovalWorkflow", "TradeRequest", "Employee"],
        "key_columns": ["department", "avg_turnaround_days", "total_reviews"],
        "staleness_trigger": "daily",
        "related_capsule_ids": [],
        "relationship_types": [],
    },

    {
        "capsule_id": "average_review_time_by_reviewer",
        "capsule_type": "aggregation",
        "priority": "P3",
        "what": "Average review time taken by each compliance reviewer",
        "how": "Average the TurnaroundDays in ApprovalWorkflow grouped by ReviewerID",
        "sql": """
SELECT
    r.EmployeeName                         AS reviewer_name,
    COUNT(aw.WorkflowID)                   AS total_reviews,
    AVG(CAST(aw.TurnaroundDays AS FLOAT))  AS avg_turnaround_days,
    SUM(CASE WHEN aw.Decision = 'Rejected' THEN 1 ELSE 0 END) AS total_rejections
FROM ApprovalWorkflow aw
JOIN Employee r ON aw.ReviewerID = r.EmployeeID
WHERE aw.TurnaroundDays IS NOT NULL
GROUP BY r.EmployeeID, r.EmployeeName
ORDER BY avg_turnaround_days DESC
""".strip(),
        "signal_method": "rule_based",
        "embed_text_template": (
            "Identifies which reviewer takes the longest on average to complete reviews. "
            "Answers: Which reviewer takes the longest on average to complete reviews? "
            "Finding: {signal} "
            "Tables: ApprovalWorkflow, Employee."
        ),
        "ttl_hours": 24,
        "tags": ["reviewer", "approval", "turnaround", "aggregation"],
        "tables_used": ["ApprovalWorkflow", "Employee"],
        "key_columns": ["reviewer_name", "avg_turnaround_days", "total_reviews"],
        "staleness_trigger": "daily",
        "related_capsule_ids": [],
        "relationship_types": [],
    }

]
