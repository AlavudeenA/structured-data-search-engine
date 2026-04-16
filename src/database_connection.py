"""Database access helpers for SQL Server, schema metadata, and FK discovery."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

import pyodbc

from .app_constants import CORE_TABLES, DB_EXECUTE_MAX_ROWS, DB_MAX_POOL, DB_MAX_RETRIES
from .config import get_settings

logger = logging.getLogger(__name__)

pyodbc.pooling = True
_CONNECTION_POOL: list[pyodbc.Connection] = []


def _make_connection() -> pyodbc.Connection:
    cfg = get_settings()
    last_error: Exception | None = None
    for attempt in range(DB_MAX_RETRIES):
        try:
            return pyodbc.connect(cfg.sqlserver_conn_str, timeout=10, autocommit=False)
        except Exception as exc:
            last_error = exc
            logger.warning("DB connect attempt %s failed: %s", attempt + 1, exc)
            if attempt < DB_MAX_RETRIES - 1:
                time.sleep(1)
    raise RuntimeError(f"Could not connect to SQL Server after {DB_MAX_RETRIES} attempts: {last_error}")


@contextmanager
def get_connection() -> Iterator[pyodbc.Connection]:
    """Yield a pooled SQL Server connection."""
    if _CONNECTION_POOL:
        conn = _CONNECTION_POOL.pop()
        try:
            conn.execute("SELECT 1")
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            conn = _make_connection()
    else:
        conn = _make_connection()

    try:
        yield conn
    finally:
        if len(_CONNECTION_POOL) < DB_MAX_POOL:
            _CONNECTION_POOL.append(conn)
        else:
            conn.close()


def execute_select(sql: str, max_rows: int = DB_EXECUTE_MAX_ROWS) -> list[dict[str, Any]]:
    """Execute a SELECT statement and return rows as dictionaries."""
    result = execute_select_with_meta(sql, max_rows=max_rows)
    return result["rows"]


def execute_select_with_meta(sql: str, max_rows: int = DB_EXECUTE_MAX_ROWS) -> dict[str, Any]:
    """Execute a SELECT query and return columns, rows, row_count, and error."""
    normalized = sql.strip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return {"columns": [], "rows": [], "row_count": 0, "error": "Only SELECT or WITH queries are allowed."}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [column[0] for column in (cursor.description or [])]
            raw_rows = cursor.fetchmany(max_rows)
            rows = [dict(zip(columns, row)) for row in raw_rows]
            return {"columns": columns, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as exc:
        logger.error("SQL execution failed: %s | SQL: %s", exc, sql[:400])
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def get_schema_metadata() -> dict[str, list[dict[str, str]]]:
    """Return schema metadata for the compliance tables."""
    placeholders = ",".join("?" for _ in CORE_TABLES)
    sql = f"""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME IN ({placeholders})
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    schema: dict[str, list[dict[str, str]]] = {}
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(CORE_TABLES))
            for table_name, column_name, data_type in cursor.fetchall():
                schema.setdefault(str(table_name), []).append(
                    {"name": str(column_name), "type": str(data_type)}
                )
    except Exception as exc:
        logger.error("Failed to read schema metadata: %s", exc)
    return schema


def get_fk_relationships() -> list[dict[str, str]]:
    """Return foreign-key relationships between compliance tables."""
    sql = """
        SELECT
            OBJECT_NAME(f.parent_object_id) AS parent_table,
            COL_NAME(fc.parent_object_id, fc.parent_column_id) AS parent_column,
            OBJECT_NAME(f.referenced_object_id) AS ref_table,
            COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS ref_column
        FROM sys.foreign_keys AS f
        INNER JOIN sys.foreign_key_columns AS fc
            ON f.object_id = fc.constraint_object_id
        ORDER BY parent_table, parent_column
    """
    relationships: list[dict[str, str]] = []
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            for parent_table, parent_column, ref_table, ref_column in cursor.fetchall():
                relationship = {
                    "parent_table": str(parent_table),
                    "parent_column": str(parent_column),
                    "ref_table": str(ref_table),
                    "ref_column": str(ref_column),
                }
                if relationship["parent_table"] in CORE_TABLES and relationship["ref_table"] in CORE_TABLES:
                    relationships.append(relationship)
    except Exception as exc:
        logger.error("Failed to read foreign keys: %s", exc)
    return relationships


def get_join_paths() -> list[dict[str, str]]:
    """Return normalized join paths related from FK relationships."""
    join_paths: list[dict[str, str]] = []
    for rel in get_fk_relationships():
        join_paths.append(
            {
                "left_table": rel["parent_table"],
                "left_column": rel["parent_column"],
                "right_table": rel["ref_table"],
                "right_column": rel["ref_column"],
                "join_sql": (
                    f"{rel['parent_table']}.{rel['parent_column']} = "
                    f"{rel['ref_table']}.{rel['ref_column']}"
                ),
            }
        )
    return join_paths


def schema_to_text(schema: dict[str, list[dict[str, str]]]) -> str:
    """Render schema metadata into a readable prompt block."""
    lines: list[str] = []
    for table_name, columns in sorted(schema.items()):
        rendered_columns = ", ".join(f"{column['name']} ({column['type']})" for column in columns)
        lines.append(f"{table_name}: {rendered_columns}")
    return "\n".join(lines)


def fk_to_text(relationships: list[dict[str, str]]) -> str:
    """Render FK relationships into text for prompts."""
    return "\n".join(
        f"{rel['parent_table']}.{rel['parent_column']} -> {rel['ref_table']}.{rel['ref_column']}"
        for rel in relationships
    )
