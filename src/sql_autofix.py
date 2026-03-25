# File: src/sql_autofix.py
"""SQL auto-fix helpers for retrying failed generated SELECT queries."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib import error, request

from .app_constants import ALLOWED_TABLES, DEFAULT_GROQ_MODEL, DEFAULT_USER_AGENT, GROQ_RESPONSES_API_URL, HTTP_DEFAULT_TIMEOUT_SECONDS, INGESTION_MODE_APPEND_UNIQUE, TEMPERATURE_ZERO
from .database_connection import get_foreign_keys, get_schema_metadata


def ingest_sql_with_autofix(
    sql: str,
    source_query: str = "manual_sql_ingestion",
    batch_size: int = 25,
    ingestion_mode: str = INGESTION_MODE_APPEND_UNIQUE,
    capsule_metadata: dict[str, Any] | None = None,
    max_fix_attempts: int = 2,
) -> dict[str, Any]:
    """Deprecated: Capsule  no longer uses manual SQL ingestion path."""
    raise NotImplementedError(
        "ingest_sql_with_autofix is deprecated. "
        "Use generate_and_ingest_capsules from src.embedding."
    )


def fix_sql_with_groq(
    bad_sql: str,
    error_text: str,
    schema_capsules: list[dict[str, Any]] | None = None,
) -> dict[str, str] | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GROQ_SQL_FIX_MODEL", DEFAULT_GROQ_MODEL)
    schema = get_schema_metadata()
    relationships = get_foreign_keys()
    schema_text = _schema_to_text(schema)
    relationships_text = _relationships_to_text(relationships)
    schema_capsules_text = _schema_capsules_to_text(schema_capsules or [])
    prompt = (
        "You are fixing a SQL Server SELECT query with typo/syntax issues.\n"
        "Return strict JSON: {\"sql\": \"...\", \"reason\": \"...\"}\n"
        "Rules:\n"
        "- Return exactly one SELECT statement\n"
        "- No semicolon\n"
        f"Schema:\n{schema_text}\n"
        f"Relationships:\n{relationships_text}\n"
        f"Schema context capsules:\n{schema_capsules_text}\n"
        f"Broken SQL:\n{bad_sql}\n"
        f"Execution Error:\n{error_text}\n"
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": TEMPERATURE_ZERO,
        "input": [
            {"role": "system", "content": "Fix SQL Server SELECT typos safely."},
            {"role": "user", "content": prompt},
        ],
    }

    req = request.Request(
        url=GROQ_RESPONSES_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=HTTP_DEFAULT_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = _extract_response_text(body)
            parsed = json.loads(content)
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, ValueError):
        return None

    candidate = str(parsed.get("sql", "")).strip()
    if _is_safe_sql(candidate):
        return {
            "sql": candidate,
            "reason": str(parsed.get("reason", "Auto-fixed after SQL execution error.")).strip()
            or "Auto-fixed after SQL execution error.",
        }
    return None


def _schema_to_text(schema: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []
    for table_name, columns in schema.items():
        col_text = ", ".join(f"{col['name']} ({col['type']})" for col in columns)
        lines.append(f"- {table_name}: {col_text}")
    return "\n".join(lines)


def _relationships_to_text(relationships: list[str]) -> str:
    if not relationships:
        return "None"
    return "\n".join(relationships)


def _schema_capsules_to_text(schema_capsules: list[dict[str, Any]]) -> str:
    if not schema_capsules:
        return "None"

    lines: list[str] = []
    for idx, hit in enumerate(schema_capsules[:6], start=1):
        payload = hit.get("payload", hit) if isinstance(hit, dict) else {}
        if not isinstance(payload, dict):
            continue
        capsule_name = str(payload.get("capsule_name", f"schema_capsule_{idx}")).strip()
        relevant_columns = ", ".join(_normalize_list(payload.get("relevant_columns", []))[:10]) or "N/A"
        recommended_joins = " | ".join(_normalize_list(payload.get("recommended_joins", []))[:6]) or "N/A"
        join_columns = " | ".join(_normalize_list(payload.get("join_columns", []))[:8]) or "N/A"
        sql_template = str(payload.get("sql_template", "")).strip() or "N/A"
        lines.append(
            "\n".join(
                [
                    f"{idx}. {capsule_name}",
                    f"   Relevant columns: {relevant_columns}",
                    f"   Recommended joins: {recommended_joins}",
                    f"   Exact join columns: {join_columns}",
                    f"   SQL template: {sql_template}",
                ]
            )
        )
    return "\n".join(lines) if lines else "None"


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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
