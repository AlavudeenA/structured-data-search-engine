# File: src/sql_autofix.py
"""Auto-fix SQL typos with Groq and retry ingestion."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib import error, request

from .database_connection import ALLOWED_TABLES, get_schema_metadata
from .embedding import ingest_from_sql


def ingest_sql_with_autofix(
    sql: str,
    source_query: str = "manual_sql_ingestion",
    batch_size: int = 25,
    ingestion_mode: str = "append_unique",
    capsule_metadata: dict[str, Any] | None = None,
    max_fix_attempts: int = 2,
) -> dict[str, Any]:
    """Try ingestion; on SQL error, use Groq to fix query and retry."""
    attempts: list[dict[str, str]] = []
    current_sql = sql.strip()
    last_error = ""

    for _ in range(max_fix_attempts + 1):
        try:
            ingest_result = ingest_from_sql(
                sql=current_sql,
                source_query=source_query,
                batch_size=batch_size,
                ingestion_mode=ingestion_mode,
                capsule_metadata=capsule_metadata,
            )
            return {
                "ingest_result": ingest_result,
                "original_sql": sql,
                "final_sql": current_sql,
                "corrected": current_sql.strip() != sql.strip(),
                "ingestion_mode": ingestion_mode,
                "attempts": attempts,
            }
        except Exception as exc:
            last_error = str(exc)
            fixed_sql = _fix_sql_with_groq(
                bad_sql=current_sql,
                error_text=last_error,
            )
            if not fixed_sql or fixed_sql.strip().lower() == current_sql.strip().lower():
                break
            attempts.append(
                {
                    "error": last_error,
                    "before_sql": current_sql,
                    "after_sql": fixed_sql,
                }
            )
            current_sql = fixed_sql

    raise RuntimeError(f"SQL ingestion failed after auto-fix attempts: {last_error}")


def _fix_sql_with_groq(bad_sql: str, error_text: str) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GROQ_SQL_FIX_MODEL", "llama-3.3-70b-versatile")
    schema = get_schema_metadata()
    schema_text = _schema_to_text(schema)
    prompt = (
        "You are fixing a SQL Server SELECT query with typo/syntax issues.\n"
        "Return strict JSON: {\"sql\": \"...\", \"reason\": \"...\"}\n"
        "Rules:\n"
        "- Return exactly one SELECT statement\n"
        "- No semicolon\n"
        f"Schema:\n{schema_text}\n"
        f"Broken SQL:\n{bad_sql}\n"
        f"Execution Error:\n{error_text}\n"
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "input": [
            {"role": "system", "content": "Fix SQL Server SELECT typos safely."},
            {"role": "user", "content": prompt},
        ],
    }

    req = request.Request(
        url="https://api.groq.com/openai/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Compliance-Data-Assistant/1.0",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = _extract_response_text(body)
            parsed = json.loads(content)
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, ValueError):
        return None

    candidate = str(parsed.get("sql", "")).strip()
    if _is_safe_sql(candidate):
        return candidate
    return None


def _schema_to_text(schema: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []
    for table_name, columns in schema.items():
        col_text = ", ".join(f"{col['name']} ({col['type']})" for col in columns)
        lines.append(f"- {table_name}: {col_text}")
    return "\n".join(lines)


def _is_safe_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    if not normalized.startswith("select "):
        return False
    if ";" in normalized:
        return False
    banned = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "merge ")
    if any(token in normalized for token in banned):
        return False
    referenced_tables = {
        t.lower()
        for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", normalized)
    }
    allowed = {t.lower() for t in ALLOWED_TABLES}
    return referenced_tables.issubset(allowed) if referenced_tables else True


def _extract_response_text(body: dict[str, Any]) -> str:
    """Extract assistant text from Groq/OpenAI response formats."""
    if "output" in body:
        for item in body["output"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text"):
                        return str(part.get("text", ""))

    if "output_text" in body:
        return str(body["output_text"])

    if "choices" in body:
        return str(body["choices"][0]["message"]["content"])

    raise ValueError(f"Unsupported response structure: {body}")
