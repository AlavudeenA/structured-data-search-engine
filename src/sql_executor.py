# File: src/sql_executor.py
"""SQL execution service for structured query flow."""

from __future__ import annotations

from typing import Any

from .database_connection import execute_select


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
