"""
Pipeline 1 core: executes each capsule definition's SQL, extracts signal,
fills embed_text_template, and returns GeneratedCapsule objects for storage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from ..app_constants import CAPSULE_RESULT_MAX_ROWS
from ..database_connection import execute_select
from ..embedding import embed_single
from ..llm_service import call_llm
from ..models import CapsuleDefinition, GeneratedCapsule
from ..prompts import SIGNAL_GENERATION_SYSTEM, SIGNAL_GENERATION_USER
from .ml_enricher import enrich_capsule

logger = logging.getLogger(__name__)


def _dominant_signal(rows: list[dict[str, Any]], capsule_def: CapsuleDefinition) -> str:
    """
    Rule-based signal extraction:
    - find dominant value (row with max count/sum column)
    - compute concentration % (top / total * 100)
    - detect trend direction if date column present
    """
    if not rows:
        return "No data available for this capsule."

    # Find the first purely numeric column (not an id column)
    numeric_col: str | None = None
    for col in rows[0]:
        val = rows[0][col]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if "id" not in col.lower():
                numeric_col = col
                break

    if numeric_col is None:
        # Fallback: just describe row count and top value
        top = rows[0]
        label_col = list(top.keys())[0]
        return (
            f"Top result: {top.get(label_col, 'N/A')} across {len(rows)} records."
        )

    total = sum(
        row[numeric_col] for row in rows if isinstance(row.get(numeric_col), (int, float))
    )
    if total == 0:
        return f"No activity recorded across {len(rows)} records."

    top_row = max(rows, key=lambda r: r.get(numeric_col, 0) or 0)
    top_val = top_row.get(numeric_col, 0)
    concentration_pct = round(top_val / total * 100, 1) if total > 0 else 0

    # First non-numeric, non-id column = label
    label_col = next(
        (
            c
            for c in top_row
            if not isinstance(top_row[c], (int, float)) and "id" not in c.lower()
        ),
        list(top_row.keys())[0],
    )
    top_label = top_row.get(label_col, "Unknown")

    signal = (
        f"{len(rows)} records analyzed. "
        f"Top: {top_label} with {top_val:,} ({concentration_pct}% of total {total:,}). "
    )

    # Trend detection across date-grouped rows
    date_col = next(
        (
            c
            for c in rows[0]
            if isinstance(rows[0][c], str) and "-" in str(rows[0][c]) and len(str(rows[0][c])) >= 7
        ),
        None,
    )
    if date_col and len(rows) >= 4:
        sorted_rows = sorted(rows, key=lambda r: str(r.get(date_col, "")))
        mid = len(sorted_rows) // 2
        first_avg = _safe_mean(sorted_rows[:mid], numeric_col)
        second_avg = _safe_mean(sorted_rows[mid:], numeric_col)
        if first_avg and second_avg and first_avg != 0:
            pct = (second_avg - first_avg) / abs(first_avg) * 100
            if pct > 10:
                signal += f"Trend: increasing ({pct:+.1f}% in second half vs first half). "
            elif pct < -10:
                signal += f"Trend: decreasing ({pct:+.1f}% in second half vs first half). "

    return signal.strip()


def _safe_mean(rows: list[dict], col: str) -> float | None:
    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
    return sum(vals) / len(vals) if vals else None


def _llm_signal(rows: list[dict[str, Any]], capsule_def: CapsuleDefinition) -> str:
    """LLM-based signal extraction using Groq."""
    rows_sample = rows[:20]
    rows_json = json.dumps(rows_sample, default=str, indent=2)
    prompt = SIGNAL_GENERATION_USER.format(
        capsule_what=capsule_def.what,
        capsule_how=capsule_def.how,
        rows_json=rows_json,
    )
    result = call_llm(
        system_prompt=SIGNAL_GENERATION_SYSTEM,
        user_prompt=prompt,
        model_slot="groq_signal_model",
        max_tokens=200,
    )
    if not result or result.startswith("[LLM"):
        return _dominant_signal(rows, capsule_def)
    return result


def generate_capsule(capsule_def: CapsuleDefinition) -> GeneratedCapsule | None:
    """
    Execute SQL, extract signal, embed, and return a GeneratedCapsule.
    Returns None on SQL failure.
    """
    logger.info("Generating capsule: %s", capsule_def.capsule_id)

    rows = execute_select(capsule_def.sql, max_rows=CAPSULE_RESULT_MAX_ROWS)
    if rows is None:
        rows = []

    # Extract signal
    if capsule_def.signal_method == "llm_summary" and rows:
        signal = _llm_signal(rows, capsule_def)
    else:
        signal = _dominant_signal(rows, capsule_def)

    # Fill embed template
    tables_str = ", ".join(capsule_def.tables_used)
    key_cols_str = ", ".join(capsule_def.key_columns)
    embed_text = capsule_def.embed_text_template.replace("{signal}", signal)
    embed_text = embed_text.replace("{tables}", tables_str).replace("{key_columns}", key_cols_str)

    # Enrich
    enrichment = enrich_capsule(capsule_def.capsule_type, capsule_def.tags, rows)

    # Embedding
    vector = embed_single(embed_text)

    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=capsule_def.ttl_hours)

    capsule = GeneratedCapsule(
        capsule_id=capsule_def.capsule_id,
        capsule_type=capsule_def.capsule_type,
        priority=capsule_def.priority,
        what=capsule_def.what,
        how=capsule_def.how,
        sql=capsule_def.sql,
        signal_method=capsule_def.signal_method,
        signal=signal,
        embed_text=embed_text,
        tables_used=capsule_def.tables_used,
        key_columns=capsule_def.key_columns,
        tags=enrichment["tags"],
        ttl_hours=capsule_def.ttl_hours,
        generated_at=now.isoformat(),
        expires_at=expires.isoformat(),
        is_stale=False,
        staleness_trigger=capsule_def.staleness_trigger,
        result_rows=rows,
        anomaly_score=enrichment["anomaly_score"],
        trend_direction=enrichment["trend_direction"],
        related_capsule_ids=capsule_def.related_capsule_ids,
        relationship_types=capsule_def.relationship_types,
        vector=vector,
    )
    logger.info(
        "Capsule generated: %s | signal_len=%d | rows=%d | anomaly=%.2f | trend=%s",
        capsule_def.capsule_id,
        len(signal),
        len(rows),
        enrichment["anomaly_score"],
        enrichment["trend_direction"],
    )
    return capsule


def generate_all_capsules(
    definitions: list[CapsuleDefinition],
    progress_callback=None,
) -> list[GeneratedCapsule]:
    """
    Run generate_capsule for each definition.
    progress_callback(capsule_id, status, signal_preview) called after each.
    """
    results: list[GeneratedCapsule] = []
    for i, defn in enumerate(definitions):
        try:
            cap = generate_capsule(defn)
            if cap:
                results.append(cap)
                if progress_callback:
                    progress_callback(defn.capsule_id, "ok", cap.signal[:80])
        except Exception as exc:
            logger.error("Failed capsule %s: %s", defn.capsule_id, exc)
            if progress_callback:
                progress_callback(defn.capsule_id, "fail", str(exc)[:80])
    return results
