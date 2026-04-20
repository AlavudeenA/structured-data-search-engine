"""
ML enrichment for generated capsules.
Computes anomaly scores and trend direction from SQL result rows.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

_DATE_RE = re.compile(r"^\d{4}-\d{2}")

from ..app_constants import (
    ANOMALY_DETECTED_THRESHOLD,
    ANOMALY_STDDEV_MULTIPLIER,
    CAPSULE_TYPE_SAMPLE,
    TAG_ANOMALY,
    TAG_ESCALATING,
    TREND_CHANGE_PCT,
    VIOLATION_TYPES,
)

logger = logging.getLogger(__name__)


def _numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return column names that contain numeric values."""
    if not rows:
        return []
    cols: list[str] = []
    for key, val in rows[0].items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            cols.append(key)
    return cols


def _date_column(rows: list[dict[str, Any]]) -> str | None:
    """Return the first string column whose value looks like a date (yyyy-MM...)."""
    if not rows:
        return None
    for key, val in rows[0].items():
        if isinstance(val, str) and _DATE_RE.match(val):
            return key
    return None


def compute_anomaly_score(rows: list[dict[str, Any]]) -> float:
    """
    Fraction of numeric values that exceed mean + N*stddev.
    Returns 0.0 to 1.0. 0.0 if insufficient data.
    """
    if len(rows) < 4:
        return 0.0
    num_cols = _numeric_columns(rows)
    if not num_cols:
        return 0.0

    anomaly_cells = 0
    total_cells = 0
    for col in num_cols:
        values = [row[col] for row in rows if row.get(col) is not None]
        if len(values) < 3:
            continue
        arr = np.array(values, dtype=float)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            continue
        threshold = mean + ANOMALY_STDDEV_MULTIPLIER * std
        anomaly_cells += int(np.sum(arr > threshold))
        total_cells += len(values)

    if total_cells == 0:
        return 0.0
    return round(min(anomaly_cells / total_cells, 1.0), 4)


def compute_trend_direction(rows: list[dict[str, Any]]) -> str:
    """
    Detect trend direction using a date column to split rows into halves.
    Returns 'increasing', 'decreasing', or 'flat'.
    """
    if len(rows) < 4:
        return "flat"
    date_col = _date_column(rows)
    if date_col is None:
        return "flat"
    num_cols = _numeric_columns(rows)
    if not num_cols:
        return "flat"

    # Sort by date column
    try:
        sorted_rows = sorted(rows, key=lambda r: str(r.get(date_col, "")))
    except Exception:
        return "flat"

    mid = len(sorted_rows) // 2
    first_half = sorted_rows[:mid]
    second_half = sorted_rows[mid:]

    # Average the primary numeric column
    primary = num_cols[0]
    first_avg = _mean_of_col(first_half, primary)
    second_avg = _mean_of_col(second_half, primary)

    if first_avg is None or second_avg is None or first_avg == 0:
        return "flat"

    pct_change = (second_avg - first_avg) / abs(first_avg) * 100
    if pct_change > TREND_CHANGE_PCT:
        return "increasing"
    if pct_change < -TREND_CHANGE_PCT:
        return "decreasing"
    return "flat"


def _mean_of_col(rows: list[dict[str, Any]], col: str) -> float | None:
    values = [row[col] for row in rows if row.get(col) is not None and isinstance(row[col], (int, float))]
    if not values:
        return None
    return float(np.mean(values))


def enrich_capsule(
    capsule_type: str,
    tags: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute anomaly_score and trend_direction, add tags if thresholds are met.
    Returns dict with anomaly_score, trend_direction, and updated tags.
    Sample capsules are skipped — random rows have no statistical baseline.
    """
    if capsule_type == CAPSULE_TYPE_SAMPLE:
        return {"anomaly_score": 0.0, "trend_direction": "flat", "tags": list(tags)}

    anomaly_score = compute_anomaly_score(rows)
    trend_direction = compute_trend_direction(rows)
    updated_tags = list(tags)

    if anomaly_score > ANOMALY_DETECTED_THRESHOLD:
        if TAG_ANOMALY not in updated_tags:
            updated_tags.append(TAG_ANOMALY)
        logger.debug("Anomaly detected (score=%.3f)", anomaly_score)

    if trend_direction == "increasing" and capsule_type in VIOLATION_TYPES:
        if TAG_ESCALATING not in updated_tags:
            updated_tags.append(TAG_ESCALATING)
        logger.debug("Escalating risk trend detected")

    return {
        "anomaly_score": anomaly_score,
        "trend_direction": trend_direction,
        "tags": updated_tags,
    }
