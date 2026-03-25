# File: src/sql_executor.py
"""SQL execution service for structured query flow."""

from __future__ import annotations

from typing import Any

from .database_connection import execute_select
from .sql_autofix import fix_sql_with_groq


def execute_sql(sql: str) -> dict[str, Any]:
    """Execute SQL and return normalized result payload."""
    result = execute_select(sql)
    return {
        "sql": sql,
        "columns": result["columns"],
        "rows": result["rows"],
        "row_count": result["row_count"],
        "truncated": result["truncated"],
    }


def execute_sql_with_autofix(
    sql: str,
    schema_capsules: list[dict[str, Any]] | None = None,
    max_fix_attempts: int = 1,
) -> dict[str, Any]:
    """Execute SQL and retry once with LLM-guided autofix on invalid-column failures."""
    current_sql = sql
    current_reason = ""
    last_error: Exception | None = None

    for attempt in range(max(0, int(max_fix_attempts)) + 1):
        try:
            result = execute_sql(current_sql)
            if current_reason:
                result["autofix_reason"] = current_reason
                result["autofix_applied"] = True
                result["original_sql"] = sql
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= max_fix_attempts or not _is_retryable_sql_error(exc):
                raise
            fix = fix_sql_with_groq(
                bad_sql=current_sql,
                error_text=str(exc),
                schema_capsules=schema_capsules or [],
            )
            if not fix:
                raise
            current_sql = fix["sql"]
            current_reason = fix.get("reason", "")

    if last_error is not None:
        raise last_error
    raise RuntimeError("SQL execution failed without a captured error.")


def _is_retryable_sql_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "invalid column name" in text
