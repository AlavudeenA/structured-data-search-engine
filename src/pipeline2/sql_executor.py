"""SQL execution with one autofix retry and normalized execution response."""

from __future__ import annotations

from typing import Any

from ..database_connection import execute_select_with_meta
from .sql_autofix import fix_sql


def execute_with_autofix(sql: str) -> dict[str, Any]:
    """Execute generated SQL and retry once with Groq-based autofix on failure."""
    first_attempt = execute_select_with_meta(sql)
    if first_attempt["error"] is None:
        return {**first_attempt, "autofix_used": False, "autofix_sql": None, "original_sql": sql}

    fixed_sql = fix_sql(sql, first_attempt["error"])
    if not fixed_sql:
        return {**first_attempt, "autofix_used": False, "autofix_sql": None, "original_sql": sql}

    second_attempt = execute_select_with_meta(fixed_sql)
    return {**second_attempt, "autofix_used": True, "autofix_sql": fixed_sql, "original_sql": sql}
