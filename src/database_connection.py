"""Database access helpers using SQLite."""

from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .app_constants import DB_EXECUTE_MAX_ROWS
from .config import get_settings

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).parent / "business_schema" / "dbscript.sql"
_initialized = False


def _db_path() -> str:
    return str(Path(get_settings().db_path).resolve())


def _initialize_db(conn: sqlite3.Connection) -> None:
    script = _SCRIPT_PATH.read_text(encoding="utf-8")
    conn.executescript(script)


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    db = _db_path()
    conn = sqlite3.connect(db, check_same_thread=False)
    try:
        count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        if count == 0:
            logger.info("Initializing SQLite database at %s", db)
            _initialize_db(conn)
    finally:
        conn.close()
    _initialized = True


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection."""
    _ensure_initialized()
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def execute_select(sql: str, max_rows: int = DB_EXECUTE_MAX_ROWS) -> list[dict[str, Any]]:
    return execute_select_with_meta(sql, max_rows=max_rows)["rows"]


def execute_select_with_meta(sql: str, max_rows: int = DB_EXECUTE_MAX_ROWS) -> dict[str, Any]:
    """Execute a SELECT query and return columns, rows, row_count, and error."""
    normalized = sql.strip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return {"columns": [], "rows": [], "row_count": 0, "error": "Only SELECT or WITH queries are allowed."}
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            raw_rows = cursor.fetchmany(max_rows)
            rows = [dict(zip(columns, row)) for row in raw_rows]
            return {"columns": columns, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as exc:
        logger.error("SQL execution failed: %s | SQL: %s", exc, sql[:400])
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def get_schema_metadata() -> dict[str, list[dict[str, str]]]:
    """Return schema metadata for all tables in the database."""
    schema: dict[str, list[dict[str, str]]] = {}
    try:
        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table_name,) in tables:
                cols = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                schema[table_name] = [
                    {"name": col[1], "type": col[2] or "TEXT"} for col in cols
                ]
    except Exception as exc:
        logger.error("Failed to read schema metadata: %s", exc)
    return schema


def get_fk_relationships() -> list[dict[str, str]]:
    """Return foreign-key relationships between tables."""
    relationships: list[dict[str, str]] = []
    try:
        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (table_name,) in tables:
                fks = conn.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
                for fk in fks:
                    # fk: (id, seq, ref_table, from_col, to_col, ...)
                    relationships.append({
                        "parent_table": table_name,
                        "parent_column": fk[3],
                        "ref_table": fk[2],
                        "ref_column": fk[4],
                    })
    except Exception as exc:
        logger.error("Failed to read foreign keys: %s", exc)
    return relationships


def get_join_paths() -> list[dict[str, str]]:
    """Return normalized join paths derived from FK relationships."""
    return [
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
        for rel in get_fk_relationships()
    ]


def schema_to_text(schema: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []
    for table_name, columns in sorted(schema.items()):
        rendered = ", ".join(f"{col['name']} ({col['type']})" for col in columns)
        lines.append(f"{table_name}: {rendered}")
    return "\n".join(lines)


def fk_to_text(relationships: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{rel['parent_table']}.{rel['parent_column']} -> {rel['ref_table']}.{rel['ref_column']}"
        for rel in relationships
    )
