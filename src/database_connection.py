# File: src/database_connection.py
"""SQL Server database connection and query helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import pyodbc

from .app_constants import (
    ALLOWED_TABLES,
    DB_CONNECTION_TIMEOUT_SECONDS,
    DB_DEFAULT_CONN_STR,
    DB_EXECUTE_SELECT_MAX_ROWS,
)


def get_connection_string() -> str:
    """Return SQL Server connection string from env or defaults."""
    return os.getenv("SQLSERVER_CONN_STR", DB_DEFAULT_CONN_STR)


@contextmanager
def get_connection() -> Iterator[pyodbc.Connection]:
    """Yield a SQL Server connection using Windows authentication."""
    conn = pyodbc.connect(get_connection_string(), timeout=DB_CONNECTION_TIMEOUT_SECONDS)
    try:
        yield conn
    finally:
        conn.close()


def execute_select(
    sql: str, params: tuple[Any, ...] | None = None, max_rows: int = DB_EXECUTE_SELECT_MAX_ROWS
) -> dict[str, Any]:
    """Execute a read-only SELECT query and return a structured result."""
    normalized = sql.strip().lower()
    allowed_prefixes = ("select", "with")
    if not normalized.startswith(allowed_prefixes):
        raise ValueError("Only SELECT queries are allowed.")
    if normalized.count(";") > 1:
        raise ValueError("Multiple SQL statements are not allowed.")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        rows = cursor.fetchmany(max_rows)
        columns = [col[0] for col in (cursor.description or [])]
        result_rows = [dict(zip(columns, row)) for row in rows]

    return {
        "columns": columns,
        "rows": result_rows,
        "row_count": len(result_rows),
        "truncated": len(result_rows) == max_rows,
    }


def get_schema_metadata() -> dict[str, list[dict[str, str]]]:
    """Load column metadata for allowed compliance tables."""
    schema: dict[str, list[dict[str, str]]] = {}
    tables = _effective_allowed_tables()
    if not tables:
        return schema

    placeholders = ",".join("?" for _ in tables)
    sql = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME IN ({placeholders})
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(tables))
        for table_name, column_name, data_type in cursor.fetchall():
            schema.setdefault(table_name, []).append(
                {"name": str(column_name), "type": str(data_type)}
            )
    return schema


def get_foreign_keys() -> list[str]:
    """Load foreign key relationships for allowed tables."""
    fk_rows = get_foreign_key_metadata()
    return [
        f"{row['parent_table']}.{row['parent_column']} -> {row['ref_table']}.{row['ref_column']}"
        for row in fk_rows
    ]


def get_foreign_key_metadata() -> list[dict[str, str]]:
    """Load structured foreign key metadata for allowed tables."""
    tables = _effective_allowed_tables()
    if not tables:
        return []

    sql = """
    SELECT
        OBJECT_NAME(f.parent_object_id) AS parent_table,
        COL_NAME(fc.parent_object_id, fc.parent_column_id) AS parent_column,
        OBJECT_NAME(f.referenced_object_id) AS ref_table,
        COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS ref_column
    FROM sys.foreign_keys f
    JOIN sys.foreign_key_columns fc
        ON f.object_id = fc.constraint_object_id
    ORDER BY parent_table, parent_column
    """
    relationships: list[dict[str, str]] = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        for parent_table, parent_column, ref_table, ref_column in cursor.fetchall():
            p_table = str(parent_table)
            r_table = str(ref_table)
            if p_table not in tables or r_table not in tables:
                continue
            relationships.append(
                {
                    "parent_table": p_table,
                    "parent_column": str(parent_column),
                    "ref_table": r_table,
                    "ref_column": str(ref_column),
                }
            )
    return relationships


def discover_tables() -> set[str]:
    """Discover base table names from the current SQL Server database."""
    sql = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    """
    tables: set[str] = set()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        for (table_name,) in cursor.fetchall():
            tables.add(str(table_name))
    return tables


def _effective_allowed_tables() -> set[str]:
    """Use explicit allowed tables if provided; otherwise auto-discover."""
    if ALLOWED_TABLES:
        return set(ALLOWED_TABLES)
    return discover_tables()
