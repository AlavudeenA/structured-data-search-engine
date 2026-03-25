# File: src/schema_capsule_generator.py
"""Generate schema-context capsules for SQL planning and fallback guidance."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .app_constants import CAPSULE_TYPE_SCHEMA_CONTEXT
from .database_connection import get_foreign_key_metadata, get_schema_metadata


def generate_schema_context_capsules() -> list[dict[str, Any]]:
    """Build metadata-focused capsules from schema and foreign keys."""
    schema = get_schema_metadata()
    if not schema:
        return []

    foreign_keys = get_foreign_key_metadata()
    capsules: list[dict[str, Any]] = []

    for table_name, columns in schema.items():
        capsules.append(_build_table_overview_capsule(table_name, columns))

    capsules.extend(_build_join_capsules(schema, foreign_keys))
    capsules.extend(_build_situation_capsules(schema, foreign_keys))
    return capsules


def _build_table_overview_capsule(table_name: str, columns: list[dict[str, str]]) -> dict[str, Any]:
    column_names = [str(col.get("name", "")).strip() for col in columns if str(col.get("name", "")).strip()]
    date_columns = [name for name, dtype in _column_pairs(columns) if _is_date_type(dtype)]
    numeric_columns = [name for name, dtype in _column_pairs(columns) if _is_numeric_type(dtype)]
    entity_columns = [name for name, dtype in _column_pairs(columns) if _is_entity_column(name, dtype)]

    summary = (
        f"Schema context for table {table_name}. "
        f"Important columns include {', '.join(column_names[:8]) or 'N/A'}. "
        f"Useful entity columns: {', '.join(entity_columns[:5]) or 'N/A'}. "
        f"Date columns: {', '.join(date_columns[:4]) or 'N/A'}."
    )
    return _build_schema_capsule(
        capsule_name=f"schema_table_{table_name}",
        summary_text=summary,
        tables_used=[table_name],
        key_columns=column_names[:8],
        tags=[CAPSULE_TYPE_SCHEMA_CONTEXT, _slug(table_name), "table_overview"],
        relevant_tables=[table_name],
        relevant_columns=column_names,
        recommended_joins=[],
        recommended_filters=_recommend_filters(date_columns=date_columns, entity_columns=entity_columns),
        sql_template=f"SELECT TOP 100 * FROM [{table_name}]",
        example_questions=[
            f"Show records from {table_name}",
            f"What are the main columns in {table_name}?",
        ],
        metric_columns=numeric_columns,
        time_columns=date_columns,
        entity_columns=entity_columns,
        business_intents=["lookup", "listing", "table exploration"],
    )


def _build_join_capsules(
    schema: dict[str, list[dict[str, str]]],
    foreign_keys: list[dict[str, str]],
) -> list[dict[str, Any]]:
    capsules: list[dict[str, Any]] = []
    for fk in foreign_keys:
        parent_table = fk["parent_table"]
        ref_table = fk["ref_table"]
        parent_column = fk["parent_column"]
        ref_column = fk["ref_column"]
        join_sql = (
            f"SELECT TOP 100 * FROM [{parent_table}] p "
            f"JOIN [{ref_table}] r ON p.[{parent_column}] = r.[{ref_column}]"
        )
        join_desc = f"{parent_table}.{parent_column} -> {ref_table}.{ref_column}"
        parent_entity_cols = _entity_columns_from_schema(schema.get(parent_table, []))
        ref_entity_cols = _entity_columns_from_schema(schema.get(ref_table, []))
        date_columns = _date_columns_from_schema(schema.get(parent_table, [])) + _date_columns_from_schema(
            schema.get(ref_table, [])
        )
        capsules.append(
            _build_schema_capsule(
                capsule_name=f"situation_join_{parent_table}_{ref_table}_{parent_column}",
                summary_text=(
                    f"Join path between {parent_table} and {ref_table}. "
                    f"Use {join_desc} when the question spans both tables."
                ),
                tables_used=[parent_table, ref_table],
                key_columns=[parent_column, ref_column],
                tags=[CAPSULE_TYPE_SCHEMA_CONTEXT, "join_path", _slug(parent_table), _slug(ref_table)],
                relevant_tables=[parent_table, ref_table],
                relevant_columns=list(
                    dict.fromkeys(
                        [parent_column, ref_column, *parent_entity_cols[:3], *ref_entity_cols[:3], *date_columns[:2]]
                    )
                ),
                recommended_joins=[join_desc],
                recommended_filters=_recommend_filters(
                    date_columns=date_columns,
                    entity_columns=list(dict.fromkeys(parent_entity_cols + ref_entity_cols)),
                ),
                sql_template=join_sql,
                example_questions=[
                    f"Show {ref_table} activity connected to {parent_table}",
                    f"Which {ref_table} records are associated with each {parent_table}?",
                ],
                metric_columns=[],
                time_columns=date_columns[:4],
                entity_columns=list(dict.fromkeys(parent_entity_cols + ref_entity_cols))[:8],
                business_intents=["join lookup", "cross-table analysis"],
            )
        )
    return capsules


def _build_situation_capsules(
    schema: dict[str, list[dict[str, str]]],
    foreign_keys: list[dict[str, str]],
) -> list[dict[str, Any]]:
    tables = {name.lower(): name for name in schema}
    if "traderequest" not in tables:
        return []

    trade_request = tables["traderequest"]
    broker_dealer = tables.get("brokerdealer")
    employee = tables.get("employee")
    account = tables.get("account")
    trade_date = _pick_time_column(schema.get(trade_request, []))

    capsules: list[dict[str, Any]] = []
    if broker_dealer and account:
        join_lines = _resolve_join_path(
            foreign_keys,
            [trade_request, account, broker_dealer],
        )
        if join_lines:
            capsules.append(
                _build_schema_capsule(
                    capsule_name="situation_broker_trade_volume",
                    summary_text=(
                        "Situation capsule for broker-dealer trade volume. "
                        "Use TradeRequest with the Account and BrokerDealer join path to rank broker activity."
                    ),
                    tables_used=[trade_request, account, broker_dealer],
                    key_columns=[trade_date] if trade_date else [],
                    tags=[CAPSULE_TYPE_SCHEMA_CONTEXT, "broker_activity", "ranking", "trade_volume"],
                    relevant_tables=[trade_request, account, broker_dealer],
                    relevant_columns=_collect_columns(
                        schema,
                        [trade_request, account, broker_dealer],
                        prefer_tokens=("broker", "name", "trade", "request", "date"),
                    ),
                    recommended_joins=join_lines,
                    recommended_filters=_recommend_filters(
                        date_columns=[trade_date] if trade_date else [],
                        entity_columns=_collect_columns(
                            schema,
                            [broker_dealer, account],
                            prefer_tokens=("name", "broker"),
                        ),
                    ),
                    sql_template=_trade_volume_sql_template(
                        trade_request=trade_request,
                        account=account,
                        broker_dealer=broker_dealer,
                        join_lines=join_lines,
                        trade_date=trade_date,
                    ),
                    example_questions=[
                        "Which broker-dealer has the most trade requests?",
                        "Which broker-dealer's trading activity is increasing over time?",
                        "Show top broker dealers by trade request count",
                    ],
                    metric_columns=["COUNT(*)"],
                    time_columns=[trade_date] if trade_date else [],
                    entity_columns=_collect_columns(
                        schema,
                        [broker_dealer, account],
                        prefer_tokens=("name", "broker"),
                    ),
                    business_intents=["ranking", "trend", "broker activity"],
                )
            )

    if employee:
        employee_join_tables = [trade_request]
        if account and _resolve_join_path(foreign_keys, [trade_request, account, employee]):
            employee_join_tables = [trade_request, account, employee]
        else:
            employee_join_tables = [trade_request, employee]
        join_lines = _resolve_join_path(foreign_keys, employee_join_tables)
        if join_lines:
            capsules.append(
                _build_schema_capsule(
                    capsule_name="situation_employee_trade_activity",
                    summary_text=(
                        "Situation capsule for employee trade request activity. "
                        "Use this path for top employees, departments, and frequent requester analysis."
                    ),
                    tables_used=employee_join_tables,
                    key_columns=[trade_date] if trade_date else [],
                    tags=[CAPSULE_TYPE_SCHEMA_CONTEXT, "employee_activity", "ranking", "department"],
                    relevant_tables=employee_join_tables,
                    relevant_columns=_collect_columns(
                        schema,
                        employee_join_tables,
                        prefer_tokens=("employee", "name", "department", "trade", "request"),
                    ),
                    recommended_joins=join_lines,
                    recommended_filters=_recommend_filters(
                        date_columns=[trade_date] if trade_date else [],
                        entity_columns=_collect_columns(schema, [employee], prefer_tokens=("name", "department")),
                    ),
                    sql_template=_employee_activity_sql_template(
                        join_tables=employee_join_tables,
                        join_lines=join_lines,
                    ),
                    example_questions=[
                        "Which employees are most active in trading requests?",
                        "Which department appears most active in trade requests?",
                    ],
                    metric_columns=["COUNT(*)"],
                    time_columns=[trade_date] if trade_date else [],
                    entity_columns=_collect_columns(schema, [employee], prefer_tokens=("name", "department")),
                    business_intents=["ranking", "employee activity", "department analysis"],
                )
            )

    return capsules


def _build_schema_capsule(
    capsule_name: str,
    summary_text: str,
    tables_used: list[str],
    key_columns: list[str],
    tags: list[str],
    relevant_tables: list[str],
    relevant_columns: list[str],
    recommended_joins: list[str],
    recommended_filters: list[str],
    sql_template: str,
    example_questions: list[str],
    metric_columns: list[str],
    time_columns: list[str],
    entity_columns: list[str],
    business_intents: list[str],
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    structured = {
        "relevant_tables": relevant_tables,
        "relevant_columns": relevant_columns,
        "recommended_joins": recommended_joins,
        "recommended_filters": recommended_filters,
        "sql_template": sql_template,
        "example_questions": example_questions,
        "metric_columns": metric_columns,
        "time_columns": time_columns,
        "entity_columns": entity_columns,
        "business_intents": business_intents,
    }
    return {
        "capsule_id": str(uuid.uuid4()),
        "capsule_name": capsule_name,
        "capsule_type": CAPSULE_TYPE_SCHEMA_CONTEXT,
        "capsule_version": "v1",
        "tables_used": relevant_tables,
        "key_columns": key_columns[:12],
        "tags": tags,
        "summary_text": summary_text,
        "rows_json": "[]",
        "row_count": 0,
        "created_at": created_at,
        "metrics": {"schema_context": True},
        "entity": ",".join(entity_columns[:4]),
        "capsule_topic": ",".join(business_intents[:3]),
        "capsule_priority": "high",
        "schema_context_json": json.dumps(structured, default=str),
        "recommended_joins": recommended_joins,
        "recommended_filters": recommended_filters,
        "relevant_columns": relevant_columns,
        "sql_template": sql_template,
        "example_questions": example_questions,
        "time_columns": time_columns,
        "entity_columns": entity_columns,
        "metric_columns": metric_columns,
        "business_intents": business_intents,
        "source_sql": "",
    }


def _resolve_join_path(foreign_keys: list[dict[str, str]], tables: list[str]) -> list[str]:
    joins: list[str] = []
    for left, right in zip(tables, tables[1:]):
        relation = _find_fk_between(foreign_keys, left, right)
        if relation:
            joins.append(
                f"{relation['parent_table']}.{relation['parent_column']} -> "
                f"{relation['ref_table']}.{relation['ref_column']}"
            )
    if len(joins) != max(0, len(tables) - 1):
        return []
    return joins


def _find_fk_between(foreign_keys: list[dict[str, str]], left: str, right: str) -> dict[str, str] | None:
    for fk in foreign_keys:
        if fk["parent_table"] == left and fk["ref_table"] == right:
            return fk
        if fk["parent_table"] == right and fk["ref_table"] == left:
            return fk
    return None


def _trade_volume_sql_template(
    trade_request: str,
    account: str,
    broker_dealer: str,
    join_lines: list[str],
    trade_date: str | None,
) -> str:
    aliases = {trade_request: "tr", account: "a", broker_dealer: "bd"}
    join_sql = _join_lines_to_sql(join_lines, aliases)
    date_select = f", CAST(tr.[{trade_date}] AS DATE) AS period_date" if trade_date else ""
    date_group = f", CAST(tr.[{trade_date}] AS DATE)" if trade_date else ""
    date_where = f"WHERE tr.[{trade_date}] IS NOT NULL " if trade_date else ""
    return (
        "Template: replace <broker_dimension_column> with a human-readable broker column.\n"
        f"SELECT TOP 20 bd.[<broker_dimension_column>]{date_select}, COUNT(*) AS trade_request_count "
        f"FROM [{trade_request}] tr {join_sql} "
        f"{date_where}GROUP BY bd.[<broker_dimension_column>]{date_group} "
        "ORDER BY trade_request_count DESC"
    )


def _employee_activity_sql_template(join_tables: list[str], join_lines: list[str]) -> str:
    aliases = {join_tables[0]: "tr"}
    if len(join_tables) > 1:
        aliases[join_tables[1]] = "a"
    if len(join_tables) > 2:
        aliases[join_tables[2]] = "e"
    else:
        aliases[join_tables[-1]] = "e"
    fact_table = join_tables[0]
    join_sql = _join_lines_to_sql(join_lines, aliases)
    employee_alias = aliases[join_tables[-1]]
    return (
        "Template: replace <employee_dimension_column> with employee name or department.\n"
        f"SELECT TOP 20 {employee_alias}.[<employee_dimension_column>], COUNT(*) AS trade_request_count "
        f"FROM [{fact_table}] tr {join_sql} "
        f"GROUP BY {employee_alias}.[<employee_dimension_column>] ORDER BY trade_request_count DESC"
    )


def _join_lines_to_sql(join_lines: list[str], aliases: dict[str, str]) -> str:
    join_sql_parts: list[str] = []
    joined_tables: set[str] = {next(iter(aliases))}
    for line in join_lines:
        left, right = [part.strip() for part in line.split("->", 1)]
        left_table, left_col = left.split(".", 1)
        right_table, right_col = right.split(".", 1)
        if left_table in joined_tables and right_table not in joined_tables:
            join_sql_parts.append(
                f"JOIN [{right_table}] {aliases.get(right_table, right_table.lower()[:2])} "
                f"ON {aliases.get(left_table, left_table.lower()[:2])}.[{left_col}] = "
                f"{aliases.get(right_table, right_table.lower()[:2])}.[{right_col}]"
            )
            joined_tables.add(right_table)
        elif right_table in joined_tables and left_table not in joined_tables:
            join_sql_parts.append(
                f"JOIN [{left_table}] {aliases.get(left_table, left_table.lower()[:2])} "
                f"ON {aliases.get(left_table, left_table.lower()[:2])}.[{left_col}] = "
                f"{aliases.get(right_table, right_table.lower()[:2])}.[{right_col}]"
            )
            joined_tables.add(left_table)
    return " ".join(join_sql_parts)


def _recommend_filters(date_columns: list[str], entity_columns: list[str]) -> list[str]:
    filters: list[str] = []
    for column in date_columns[:2]:
        filters.append(f"Filter recent periods with [{column}]")
    for column in entity_columns[:2]:
        filters.append(f"Filter exact entity names on [{column}]")
    return filters


def _collect_columns(
    schema: dict[str, list[dict[str, str]]],
    tables: list[str],
    prefer_tokens: tuple[str, ...],
) -> list[str]:
    picked: list[str] = []
    for table in tables:
        for column in schema.get(table, []):
            name = str(column.get("name", "")).strip()
            lname = name.lower()
            if any(token in lname for token in prefer_tokens):
                picked.append(f"{table}.{name}")
    return list(dict.fromkeys(picked))[:12]


def _entity_columns_from_schema(columns: list[dict[str, str]]) -> list[str]:
    return [name for name, dtype in _column_pairs(columns) if _is_entity_column(name, dtype)]


def _date_columns_from_schema(columns: list[dict[str, str]]) -> list[str]:
    return [name for name, dtype in _column_pairs(columns) if _is_date_type(dtype)]


def _pick_time_column(columns: list[dict[str, str]]) -> str | None:
    date_columns = _date_columns_from_schema(columns)
    return date_columns[0] if date_columns else None


def _column_pairs(columns: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [
        (str(col.get("name", "")).strip(), str(col.get("type", "")).strip())
        for col in columns
        if str(col.get("name", "")).strip()
    ]


def _is_date_type(data_type: str) -> bool:
    dtype = data_type.lower()
    return "date" in dtype or "time" in dtype


def _is_numeric_type(data_type: str) -> bool:
    dtype = data_type.lower()
    return any(token in dtype for token in ("int", "decimal", "numeric", "float", "real", "money", "smallmoney"))


def _is_entity_column(name: str, data_type: str) -> bool:
    lname = name.lower()
    if lname.endswith("id") or lname.endswith("_id"):
        return False
    if _is_date_type(data_type) or _is_numeric_type(data_type):
        return False
    return any(token in lname for token in ("name", "department", "security", "broker", "employee", "account"))


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
