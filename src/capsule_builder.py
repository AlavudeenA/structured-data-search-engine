# File: src/capsule_builder.py
"""Build compact context capsules from SQL execution results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def build_capsules(
    columns: list[str],
    rows: list[dict[str, Any]],
    batch_size: int = 25,
    capsule_type: str = "aggregation",
    capsule_name: str | None = None,
    entity: str = "",
    capsule_topic: str = "",
    metric_tags: list[str] | None = None,
    capsule_version: str = "v1",
    schema_version: str = "v1",
    source_query_hash: str = "",
    capsule_priority: str = "",
    max_capsules: int = 200,
) -> list[dict[str, Any]]:
    """Split rows into batches and convert each batch to a capsule."""
    capsules: list[dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc).isoformat()
    size = max(1, batch_size)

    for idx in range(0, len(rows), size):
        batch = rows[idx : idx + size]
        capsule_idx = (idx // size) + 1
        if capsule_idx > max(1, max_capsules):
            break
        metrics = derive_metrics(batch, columns)
        resolved_priority = capsule_priority or _default_priority_for_type(capsule_type)
        summary_text = build_summary_text(
            capsule_idx=capsule_idx,
            columns=columns,
            rows=batch,
            metrics=metrics,
            capsule_type=capsule_type,
            entity=entity,
            capsule_topic=capsule_topic,
        )
        capsules.append(
            {
                "capsule_index": capsule_idx,
                "capsule_name": _build_capsule_name(capsule_name, capsule_type, capsule_idx),
                "capsule_type": capsule_type or "aggregation",
                "entity": entity or "",
                "capsule_topic": capsule_topic or "",
                "metric_tags": metric_tags or [],
                "capsule_priority": resolved_priority,
                "capsule_version": capsule_version,
                "schema_version": schema_version,
                "source_query_hash": source_query_hash or "",
                "summary_text": summary_text,
                "metrics": metrics,
                "rows_json": json.dumps(batch[:5], default=str),
                "refreshed_at_utc": now_utc,
                "row_count": len(batch),
            }
        )
    return capsules


def derive_metrics(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    """Create lightweight numeric summaries per capsule."""
    numeric_cols: list[str] = []
    for col in columns:
        values = [r.get(col) for r in rows if r.get(col) is not None]
        if values and all(is_number(v) for v in values):
            numeric_cols.append(col)

    metrics: dict[str, Any] = {
        "row_count": len(rows),
        "numeric_summaries": {},
        "entities": detect_entities(rows, columns),
    }
    for col in numeric_cols:
        values = [float(r[col]) for r in rows if r.get(col) is not None]
        if not values:
            continue
        metrics["numeric_summaries"][col] = {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }
    return metrics


def build_summary_text(
    capsule_idx: int,
    columns: list[str],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    capsule_type: str,
    entity: str,
    capsule_topic: str,
) -> str:
    """Build insight-oriented summary text for embedding."""
    ctype = (capsule_type or "aggregation").lower()
    if ctype == "trend":
        return _build_trend_summary(capsule_idx, columns, rows, metrics, entity, capsule_topic)
    if ctype == "entity_profile":
        return _build_entity_profile_summary(
            capsule_idx, columns, rows, metrics, entity, capsule_topic
        )
    return _build_aggregation_summary(capsule_idx, columns, rows, metrics, entity, capsule_topic)


def _build_aggregation_summary(
    capsule_idx: int,
    columns: list[str],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    entity: str,
    capsule_topic: str,
) -> str:
    sample_rows = rows[:3]
    metrics_text = _format_key_metrics(metrics)
    sample_text = json.dumps(sample_rows, default=str)
    entity_hint = f"Entity focus: {entity}. " if entity else ""
    topic_hint = f"Topic: {capsule_topic}. " if capsule_topic else ""
    top_entity_hint = _best_entity_signal(metrics)
    return (
        f"Context capsule {capsule_idx}. "
        f"{topic_hint}{entity_hint}"
        f"Dataset columns: {', '.join(columns)}. "
        f"This capsule contains {len(rows)} records. "
        f"Computed metrics: {metrics_text}. "
        f"{top_entity_hint}"
        f"Example records: {sample_text}."
    )


def is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _build_capsule_name(
    base_name: str | None, capsule_type: str, capsule_idx: int
) -> str:
    root = (base_name or f"{capsule_type}_capsule").strip().lower().replace(" ", "_")
    return f"{root}_{capsule_idx}"


def detect_entities(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, list[str]]:
    """Detect entity-like categorical columns without domain keywords."""
    entity_values: dict[str, list[str]] = {}

    for col in columns:
        values = [
            str(r.get(col)).strip()
            for r in rows
            if r.get(col) not in (None, "")
        ]

        if not values:
            continue

        unique_vals = sorted(set(values))
        unique_count = len(unique_vals)

        if 1 < unique_count <= 20:
            entity_values[col] = unique_vals[:5]
        elif col.lower().endswith(("id", "_id")) and unique_count <= 50:
            entity_values[col] = unique_vals[:5]

    return entity_values


def _best_entity_signal(metrics: dict[str, Any]) -> str:
    entities = metrics.get("entities", {})
    if not entities:
        return ""
    first_col = next(iter(entities.keys()))
    values = entities.get(first_col, [])
    if not values:
        return ""
    return f"Representative values for {first_col}: {', '.join(values[:3])}. "


def _build_trend_summary(
    capsule_idx: int,
    columns: list[str],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    entity: str,
    capsule_topic: str,
) -> str:
    metrics_text = _format_key_metrics(metrics)
    sample_text = json.dumps(rows[:3], default=str)
    date_col = _find_date_column(columns)
    numeric_col = _first_numeric_column(metrics)
    trend_sentence = "Trend signal is limited for this capsule."
    if date_col and numeric_col:
        trend_sentence = _compute_simple_trend(rows, date_col, numeric_col)
    return (
        f"Context capsule {capsule_idx}. "
        f"Topic: {capsule_topic or 'trend'}. "
        f"Entity focus: {entity or 'N/A'}. "
        f"Columns: {', '.join(columns)}. "
        f"{trend_sentence} "
        f"Metrics: {metrics_text}. "
        f"Example records: {sample_text}."
    )


def _build_entity_profile_summary(
    capsule_idx: int,
    columns: list[str],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    entity: str,
    capsule_topic: str,
) -> str:
    metrics_text = _format_key_metrics(metrics)
    sample_text = json.dumps(rows[:3], default=str)
    entity_hint = _best_entity_signal(metrics)
    return (
        f"Context capsule {capsule_idx}. "
        f"Topic: {capsule_topic or 'entity_profile'}. "
        f"Entity profile focus: {entity or 'N/A'}. "
        f"Columns: {', '.join(columns)}. "
        f"{entity_hint}"
        f"Computed metrics: {metrics_text}. "
        f"Example records: {sample_text}."
    )


def _find_date_column(columns: list[str]) -> str | None:
    for col in columns:
        c = col.lower()
        if "date" in c or "time" in c:
            return col
    return None


def _first_numeric_column(metrics: dict[str, Any]) -> str | None:
    nums = metrics.get("numeric_summaries", {})
    return next(iter(nums.keys()), None)


def _compute_simple_trend(rows: list[dict[str, Any]], date_col: str, num_col: str) -> str:
    parsed: list[tuple[datetime, float]] = []
    for row in rows:
        dt_raw = row.get(date_col)
        val_raw = row.get(num_col)
        if dt_raw is None or val_raw is None or not is_number(val_raw):
            continue
        dt = _parse_datetime(dt_raw)
        if dt is None:
            continue
        parsed.append((dt, float(val_raw)))
    if len(parsed) < 2:
        return "Not enough temporal points for trend direction."
    parsed.sort(key=lambda x: x[0])
    first_val = parsed[0][1]
    last_val = parsed[-1][1]
    if first_val == 0:
        delta_text = "baseline is zero, directional change observed"
    else:
        pct = ((last_val - first_val) / abs(first_val)) * 100.0
        delta_text = f"{pct:.2f}% change from first to last observation"
    direction = "increased" if last_val > first_val else "decreased" if last_val < first_val else "remained stable"
    return (
        f"{num_col} {direction} over {date_col} with {delta_text}."
    )


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _default_priority_for_type(capsule_type: str) -> str:
    ctype = (capsule_type or "").lower()
    if ctype in {"aggregation", "trend", "anomaly"}:
        return "high"
    if ctype in {"correlation", "entity_profile"}:
        return "medium"
    return "low"


def _format_key_metrics(metrics: dict[str, Any]) -> str:
    numeric = metrics.get("numeric_summaries", {})
    if not numeric:
        return json.dumps(metrics, default=str)
    parts: list[str] = []
    for col, vals in numeric.items():
        avg_val = vals.get("avg")
        if isinstance(avg_val, (float, int)):
            avg_val = round(float(avg_val), 2)
        min_val = vals.get("min")
        if isinstance(min_val, (float, int)):
            min_val = round(float(min_val), 2)
        max_val = vals.get("max")
        if isinstance(max_val, (float, int)):
            max_val = round(float(max_val), 2)
        parts.append(
            f"{col}: min={min_val}, max={max_val}, avg={avg_val}"
        )
    return "; ".join(parts)
