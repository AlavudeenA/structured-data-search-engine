"""Generate schema-context capsules from live schema metadata, FKs, and domain situations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..database_connection import get_fk_relationships, get_join_paths, get_schema_metadata
from ..embedding import embed_single
from ..models import SchemaContextCapsule

logger = logging.getLogger(__name__)


def _build_capsule(
    capsule_id: str,
    summary: str,
    tables: list[str],
    relevant_columns: list[str],
    recommended_joins: list[str],
    join_columns: list[str],
    recommended_filters: list[str],
    example_questions: list[str],
    sql_template: str,
    tags: list[str],
) -> SchemaContextCapsule:
    generated_at = datetime.now(timezone.utc).isoformat()
    embed_text = (
        f"{summary} Tables: {', '.join(tables)}. Relevant columns: {', '.join(relevant_columns)}. "
        f"Recommended joins: {' | '.join(recommended_joins[:3])}. Example questions: {' | '.join(example_questions[:4])}."
    )
    return SchemaContextCapsule(
        capsule_id=capsule_id,
        summary=summary,
        tables=tables,
        relevant_columns=relevant_columns,
        recommended_joins=recommended_joins,
        join_columns=join_columns,
        recommended_filters=recommended_filters,
        example_questions=example_questions,
        sql_template=sql_template,
        tags=tags,
        generated_at=generated_at,
        vector=embed_single(embed_text),
    )


def generate_schema_capsules() -> list[SchemaContextCapsule]:
    """Build metadata-focused schema capsules for SQL planning."""
    schema = get_schema_metadata()
    fk_relationships = get_fk_relationships()
    join_paths = get_join_paths()

    join_texts = [f"{path['left_table']}.{path['left_column']} = {path['right_table']}.{path['right_column']}" for path in join_paths]
    table_columns = {table: [column["name"] for column in columns] for table, columns in schema.items()}

    capsules: list[SchemaContextCapsule] = []
    capsules.append(
        _build_capsule(
            capsule_id="schema_context_trade_request_backbone",
            summary="TradeRequest is the operational backbone for ranking, volume, broker, employee, and security analysis. Use it as the fact table and join to Employee and BrokerDealer first.",
            tables=["TradeRequest", "Employee", "BrokerDealer"],
            relevant_columns=table_columns.get("TradeRequest", []) + ["EmployeeName", "Department", "BrokerDealerName", "Country"],
            recommended_joins=[
                "TradeRequest.EmployeeID = Employee.EmployeeID",
                "TradeRequest.BrokerDealerID = BrokerDealer.BrokerDealerID",
            ],
            join_columns=["EmployeeID", "BrokerDealerID"],
            recommended_filters=[
                "tr.RequestDate >= DATEADD(MONTH, -6, GETDATE())",
                "tr.Status IN ('Pending', 'Approved', 'Rejected', 'Escalated')",
            ],
            example_questions=[
                "Which broker dealer has the most trade requests?",
                "Which department submits the most requests?",
                "Is buy more or sell more?",
                "Which securities are traded most often?",
            ],
            sql_template=(
                "SELECT bd.BrokerDealerName AS broker_dealer, COUNT(tr.TradeRequestID) AS request_count "
                "FROM TradeRequest tr "
                "JOIN BrokerDealer bd ON tr.BrokerDealerID = bd.BrokerDealerID "
                "GROUP BY bd.BrokerDealerName "
                "ORDER BY request_count DESC"
            ),
            tags=["schema", "trade_request", "broker", "employee", "aggregation"],
        )
    )

    capsules.append(
        _build_capsule(
            capsule_id="schema_context_restriction_overlap",
            summary="Use RestrictedSecurity with TradeRequest on SecuritySymbol plus date overlap when asking about violations, active restrictions, or restricted trading attempts. EndDate NULL means the restriction is still active.",
            tables=["TradeRequest", "RestrictedSecurity", "Employee", "BrokerDealer"],
            relevant_columns=["SecuritySymbol", "RestrictionType", "StartDate", "EndDate", "RequestDate", "EmployeeID", "BrokerDealerID"],
            recommended_joins=[
                "TradeRequest.SecuritySymbol = RestrictedSecurity.SecuritySymbol",
                "TradeRequest.RequestDate BETWEEN RestrictedSecurity.StartDate AND ISNULL(RestrictedSecurity.EndDate, '9999-12-31')",
                "TradeRequest.EmployeeID = Employee.EmployeeID",
                "TradeRequest.BrokerDealerID = BrokerDealer.BrokerDealerID",
            ],
            join_columns=["SecuritySymbol", "EmployeeID", "BrokerDealerID"],
            recommended_filters=[
                "ISNULL(rs.EndDate, '9999-12-31') >= GETDATE()",
                "rs.RestrictionType IN ('Blackout', 'Insider List', 'Watch List')",
            ],
            example_questions=[
                "Which requests violated active restrictions?",
                "Which department hits the watch list most often?",
                "Show active restrictions with recent trade attempts.",
            ],
            sql_template=(
                "SELECT tr.TradeRequestID AS trade_request_id, e.EmployeeName AS employee_name, tr.SecuritySymbol AS security_symbol, "
                "rs.RestrictionType AS restriction_type, tr.RequestDate AS request_date "
                "FROM TradeRequest tr "
                "JOIN RestrictedSecurity rs ON tr.SecuritySymbol = rs.SecuritySymbol "
                "AND tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31') "
                "JOIN Employee e ON tr.EmployeeID = e.EmployeeID "
                "ORDER BY tr.RequestDate DESC"
            ),
            tags=["schema", "restriction", "violation", "date_overlap", "active_restriction"],
        )
    )

    capsules.append(
        _build_capsule(
            capsule_id="schema_context_alerts_and_reviews",
            summary="ComplianceAlert plus ApprovalWorkflow answers escalation, alert severity, reviewer performance, and unresolved issue questions. ReviewerID points back to Employee as the reviewer, not the requester.",
            tables=["ComplianceAlert", "TradeRequest", "ApprovalWorkflow", "Employee"],
            relevant_columns=["AlertType", "AlertDate", "Severity", "Status", "Description", "ResolvedDate", "Decision", "TurnaroundDays", "ReviewerID"],
            recommended_joins=[
                "ComplianceAlert.TradeRequestID = TradeRequest.TradeRequestID",
                "ComplianceAlert.EmployeeID = Employee.EmployeeID",
                "ApprovalWorkflow.TradeRequestID = TradeRequest.TradeRequestID",
                "ApprovalWorkflow.ReviewerID = Employee.EmployeeID",
            ],
            join_columns=["TradeRequestID", "EmployeeID", "ReviewerID"],
            recommended_filters=[
                "ca.Status IN ('Open', 'Investigating')",
                "ca.Severity IN ('High', 'Critical')",
                "aw.Decision IN ('Rejected', 'Escalated', 'Pending')",
            ],
            example_questions=[
                "Which employees have multiple alert types?",
                "Which reviewer has the slowest turnaround?",
                "What characteristics lead to escalation versus rejection?",
            ],
            sql_template=(
                "SELECT ca.AlertType AS alert_type, ca.Severity AS severity, aw.Decision AS decision, aw.TurnaroundDays AS turnaround_days "
                "FROM ComplianceAlert ca "
                "LEFT JOIN ApprovalWorkflow aw ON ca.TradeRequestID = aw.TradeRequestID "
                "ORDER BY ca.AlertDate DESC"
            ),
            tags=["schema", "alerts", "workflow", "reviewer", "operational", "risk"],
        )
    )

    capsules.append(
        _build_capsule(
            capsule_id="schema_context_employee_account_broker_path",
            summary="Use Account when the question is about employee brokerage registration, account coverage, or broker registration by country. This is a metadata path that complements TradeRequest-based analysis.",
            tables=["Employee", "Account", "BrokerDealer"],
            relevant_columns=["EmployeeID", "EmployeeName", "Department", "AccountID", "AccountType", "AccountNumber", "BrokerDealerName", "Country", "RegistrationNumber"],
            recommended_joins=[
                "Account.EmployeeID = Employee.EmployeeID",
                "Account.BrokerDealerID = BrokerDealer.BrokerDealerID",
            ],
            join_columns=["EmployeeID", "BrokerDealerID"],
            recommended_filters=[
                "bd.Country = 'USA'",
                "e.Status = 'Active'",
            ],
            example_questions=[
                "Which broker dealers are registered in the USA?",
                "Which employees have accounts with multiple broker dealers?",
                "Which departments are concentrated with one broker dealer?",
            ],
            sql_template=(
                "SELECT bd.BrokerDealerName AS broker_dealer_name, bd.Country AS country, COUNT(a.AccountID) AS account_count "
                "FROM Account a "
                "JOIN BrokerDealer bd ON a.BrokerDealerID = bd.BrokerDealerID "
                "GROUP BY bd.BrokerDealerName, bd.Country "
                "ORDER BY account_count DESC"
            ),
            tags=["schema", "account", "broker_registration", "country", "employee_path"],
        )
    )

    capsules.append(
        _build_capsule(
            capsule_id="schema_context_trend_patterns",
            summary="For trend questions, prefer TradeRequest.RequestDate, ComplianceAlert.AlertDate, and ApprovalWorkflow.ReviewDate. Group using FORMAT(date_col, 'yyyy-MM') for monthly trends or DATEPART(ISO_WEEK, date_col) for weekly views.",
            tables=["TradeRequest", "ComplianceAlert", "ApprovalWorkflow", "BrokerDealer", "Employee"],
            relevant_columns=["RequestDate", "AlertDate", "ReviewDate", "Status", "Severity", "Department", "BrokerDealerName"],
            recommended_joins=join_texts[:8] if join_texts else [
                "TradeRequest.BrokerDealerID = BrokerDealer.BrokerDealerID",
                "TradeRequest.EmployeeID = Employee.EmployeeID",
            ],
            join_columns=["BrokerDealerID", "EmployeeID", "TradeRequestID", "ReviewerID"],
            recommended_filters=[
                "tr.RequestDate >= DATEADD(MONTH, -6, GETDATE())",
                "ca.AlertDate >= DATEADD(MONTH, -6, GETDATE())",
            ],
            example_questions=[
                "Which broker dealer's trading activity is increasing over time?",
                "Show monthly alert volume trend.",
                "Which department has rising escalation activity?",
            ],
            sql_template=(
                "SELECT FORMAT(tr.RequestDate, 'yyyy-MM') AS month_key, bd.BrokerDealerName AS broker_dealer, COUNT(tr.TradeRequestID) AS request_count "
                "FROM TradeRequest tr "
                "JOIN BrokerDealer bd ON tr.BrokerDealerID = bd.BrokerDealerID "
                "WHERE tr.RequestDate >= DATEADD(MONTH, -6, GETDATE()) "
                "GROUP BY FORMAT(tr.RequestDate, 'yyyy-MM'), bd.BrokerDealerName "
                "ORDER BY month_key ASC, request_count DESC"
            ),
            tags=["schema", "trend", "time_series", "broker", "department"],
        )
    )

    logger.info("Generated %s schema context capsules", len(capsules))
    logger.debug("Schema tables discovered: %s | FK relationships: %s", list(schema), len(fk_relationships))
    return capsules
