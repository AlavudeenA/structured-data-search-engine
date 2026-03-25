# File: src/capsule_generator_.py
"""Schema-agnostic capsule  generator for random and aggregated SQL context."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .app_constants import (
    CAPSULE_HEURISTIC_TAG_PREVIEW,
    CAPSULE_NUMERIC_COLUMNS_SUMMARY_LIMIT,
    CAPSULE_SUMMARY_MAX_OUTPUT_TOKENS,
    CAPSULE_SUMMARY_MAX_SENTENCES,
    CAPSULE_SUMMARY_SAMPLE_ROWS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_ANALYTICAL_TEMPERATURE,
    DEFAULT_MAX_GROUP_COLS_PER_TABLE,
    DEFAULT_ROWS_PER_CAPSULE,
    GEN_ROWS_MAX_ALLOWED,
    GEN_ROWS_MAX_EXCLUSIVE,
    GEN_ROWS_MIN,
)
from .capsule_query_planner import build_capsule_sql_plans
from .database_connection import execute_select, get_schema_metadata
from .llm_service import call_llm


def generate_capsules_(
    target_capsules: int | None = None,
    target_rows: int = DEFAULT_ROWS_PER_CAPSULE,
    max_rows_per_capsule: int = GEN_ROWS_MAX_EXCLUSIVE,
    include_temporal_aggregations: bool = True,
    max_group_cols_per_table: int = DEFAULT_MAX_GROUP_COLS_PER_TABLE,
    use_llm_summaries: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate mixed random/aggregation/distribution capsules from discovered schema.

    Notes:
    - Capsules are generated with less than 100 rows.
    - Target row range is tuned around 30-80 by default via `target_rows`.
    - This is non-breaking: it does not alter existing ingestion/retrieval flow.
    """
    schema = get_schema_metadata()
    if not schema:
        return []

    if int(max_rows_per_capsule) < GEN_ROWS_MIN or int(max_rows_per_capsule) >= GEN_ROWS_MAX_EXCLUSIVE:
        raise ValueError(
            f"Rows per capsule must be between {GEN_ROWS_MIN} and {GEN_ROWS_MAX_ALLOWED}."
        )
    if int(target_rows) < GEN_ROWS_MIN or int(target_rows) > int(max_rows_per_capsule):
        raise ValueError("Target rows must be between 1 and the strict row limit.")

    row_cap = max(GEN_ROWS_MIN, min(max_rows_per_capsule, GEN_ROWS_MAX_ALLOWED))
    preferred_rows = max(GEN_ROWS_MIN, min(target_rows, row_cap))

    capsules: list[dict[str, Any]] = []
    sql_plans: list[dict[str, Any]] = build_capsule_sql_plans(
        schema=schema,
        preferred_rows=preferred_rows,
        row_cap=row_cap,
        include_temporal_aggregations=include_temporal_aggregations,
        max_group_cols_per_table=max_group_cols_per_table,
    )

    for plan in sql_plans:
        if target_capsules is not None and target_capsules > 0 and len(capsules) >= target_capsules:
            break
        capsule = _run_plan_to_capsule(
            plan=plan,
            row_cap=row_cap,
            use_llm_summaries=use_llm_summaries,
        )
        if capsule is None:
            continue
        capsules.append(capsule)

    return capsules


def preview_capsule_sql_plans(
    target_rows: int = DEFAULT_ROWS_PER_CAPSULE,
    max_rows_per_capsule: int = GEN_ROWS_MAX_EXCLUSIVE,
    include_temporal_aggregations: bool = True,
    max_group_cols_per_table: int = DEFAULT_MAX_GROUP_COLS_PER_TABLE,
) -> list[str]:
    """Return generated SQL plan texts for UI preview/debug."""
    schema = get_schema_metadata()
    if not schema:
        return []
    row_cap = max(GEN_ROWS_MIN, min(max_rows_per_capsule, GEN_ROWS_MAX_ALLOWED))
    preferred_rows = max(GEN_ROWS_MIN, min(target_rows, row_cap))
    plans = build_capsule_sql_plans(
        schema=schema,
        preferred_rows=preferred_rows,
        row_cap=row_cap,
        include_temporal_aggregations=include_temporal_aggregations,
        max_group_cols_per_table=max_group_cols_per_table,
    )
    return [str(p.get("sql", "")).strip() for p in plans if str(p.get("sql", "")).strip()]


def _run_plan_to_capsule(
    plan: dict[str, Any],
    row_cap: int,
    use_llm_summaries: bool,
) -> dict[str, Any] | None:
    sql = str(plan["sql"])
    try:
        # Fetch one extra row to detect overflow; do not silently truncate.
        result = execute_select(sql, max_rows=row_cap + 1)
    except Exception:
        return None

    row_count = int(result.get("row_count", 0))
    if row_count > row_cap:
        raise ValueError(
            f"Capsule query exceeded row limit ({row_cap}). "
            f"Query returned more than {row_cap} rows: {sql}"
        )

    rows = result.get("rows", [])
    if not rows or row_count <= 0:
        return None

    summary = _summarize_capsule(
        capsule_type=str(plan["capsule_type"]),
        tables_used=plan.get("tables_used", []),
        key_columns=plan.get("key_columns", []),
        tags=plan.get("tags", []),
        rows=rows,
        use_llm=use_llm_summaries,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    metrics = _build_metrics(rows)
    return {
        "capsule_id": str(uuid.uuid4()),
        "capsule_type": str(plan["capsule_type"]),
        "capsule_version": "",
        "tables_used": list(plan.get("tables_used", [])),
        "key_columns": list(plan.get("key_columns", [])),
        "tags": list(plan.get("tags", [])),
        "summary_text": summary,
        "rows_json": json.dumps(rows, default=str),
        "row_count": row_count,
        "created_at": created_at,
        "metrics": metrics,
        # optional lineage/debug
        "source_sql": sql,
    }


def _summarize_capsule(
    capsule_type: str,
    tables_used: list[str],
    key_columns: list[str],
    tags: list[str],
    rows: list[dict[str, Any]],
    use_llm: bool,
) -> str:
    sample_rows = rows[:CAPSULE_SUMMARY_SAMPLE_ROWS]
    heuristic = _heuristic_summary(
        capsule_type=capsule_type,
        tables_used=tables_used,
        key_columns=key_columns,
        tags=tags,
        rows=rows,
    )
    if not use_llm:
        return heuristic

    prompt = (
        "Write a concise plain-English summary for this structured-data capsule.\n"
        f"Capsule type: {capsule_type}\n"
        f"Tables used: {tables_used}\n"
        f"Key columns: {key_columns}\n"
        f"Tags: {tags}\n"
        f"Sample rows JSON: {json.dumps(sample_rows, default=str)}\n\n"
        "Rules:\n"
        f"- Max {CAPSULE_SUMMARY_MAX_SENTENCES} sentences.\n"
        "- Mention important identifiers/fields if present.\n"
        "- Do not invent values not present in rows.\n"
    )
    llm_summary = call_llm(
        prompt=prompt,
        system_prompt="You summarize structured SQL result samples into context-capsule text.",
        model_env_var="GROQ_SUMMARY_MODEL",
        default_model=DEFAULT_GROQ_MODEL,
        temperature=DEFAULT_ANALYTICAL_TEMPERATURE,
        max_output_tokens=CAPSULE_SUMMARY_MAX_OUTPUT_TOKENS,
    )
    if llm_summary:
        return llm_summary.strip()
    raise RuntimeError(
        "LLM summary generation failed while use_llm_summaries is enabled."
    )


def _heuristic_summary(
    capsule_type: str,
    tables_used: list[str],
    key_columns: list[str],
    tags: list[str],
    rows: list[dict[str, Any]],
) -> str:
    sample_rows = rows[:CAPSULE_SUMMARY_SAMPLE_ROWS]
    keys = list(sample_rows[0].keys()) if sample_rows else []
    return (
        f"{capsule_type.replace('_', ' ').title()} capsule over {', '.join(tables_used)}. "
        f"Contains {len(rows)} rows with key columns {', '.join(key_columns) or 'N/A'}. "
        f"Tags: {', '.join(tags[:CAPSULE_HEURISTIC_TAG_PREVIEW]) or 'N/A'}. "
        f"Sample fields: {', '.join(keys[:6])}."
    )


def _build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    first = rows[0]
    numeric_cols = [k for k, v in first.items() if _is_number(v)]
    metrics: dict[str, Any] = {"row_count": len(rows), "numeric_summaries": {}}
    for col in numeric_cols[:CAPSULE_NUMERIC_COLUMNS_SUMMARY_LIMIT]:
        values = [float(r[col]) for r in rows if col in r and _is_number(r[col])]
        if not values:
            continue
        metrics["numeric_summaries"][col] = {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 2),
        }
    return metrics


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
