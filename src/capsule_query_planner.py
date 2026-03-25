# File: src/capsule_query_planner.py
"""Schema-driven SQL plan generation for context capsules."""

from __future__ import annotations

import random
import re
from typing import Any

from .app_constants import (
    CAPSULE_TYPE_AGGREGATION,
    CAPSULE_TYPE_ANOMALY,
    CAPSULE_TYPE_DISTRIBUTION,
    CAPSULE_TYPE_RANDOM_SAMPLE,
    CAPSULE_TYPE_TREND,
    DEFAULT_MAX_COLUMNS_PER_RANDOM_PROJECTION,
    DEFAULT_MAX_DATE_COLUMNS_PER_TABLE,
    DEFAULT_MAX_KEY_COLUMNS_IN_PROJECTION,
    DEFAULT_MAX_NUMERIC_OUTLIER_COLUMNS,
)


def build_capsule_sql_plans(
    schema: dict[str, list[dict[str, str]]],
    preferred_rows: int,
    row_cap: int,
    include_temporal_aggregations: bool,
    max_group_cols_per_table: int,
) -> list[dict[str, Any]]:
    """
    Build diverse SQL plans from schema metadata.

    Plan families:
    - random_sample
    - aggregation
    - distribution
    - trend
    - anomaly
    """
    plans: list[dict[str, Any]] = []
    top_n = min(preferred_rows, row_cap)
    max_group = max(1, int(max_group_cols_per_table))

    for table, cols in schema.items():
        if not cols:
            continue
        col_names = [str(c["name"]) for c in cols]
        key_cols = _detect_key_columns(cols)
        date_cols = [str(c["name"]) for c in cols if _is_date_type(str(c["type"]))]
        numeric_cols = [str(c["name"]) for c in cols if _is_numeric_type(str(c["type"]))]
        dimension_cols = [
            str(c["name"])
            for c in cols
            if _is_groupable_dimension(str(c["name"]), str(c["type"]))
        ]

        # 1) Random sample plans (broad context snapshots).
        plans.append(
            {
                "capsule_type": CAPSULE_TYPE_RANDOM_SAMPLE,
                "tables_used": [table],
                "key_columns": key_cols,
                "tags": _tags(table, "", CAPSULE_TYPE_RANDOM_SAMPLE),
                "sql": (
                    f"SELECT TOP {top_n} {', '.join(_q(c) for c in _select_random_projection(col_names, key_cols))} "
                    f"FROM {_q(table)} ORDER BY NEWID()"
                ),
            }
        )

        # 2) Aggregation + Distribution plans by selected dimensions.
        for dim in dimension_cols[:max_group]:
            plans.append(
                {
                    "capsule_type": CAPSULE_TYPE_AGGREGATION,
                    "tables_used": [table],
                    "key_columns": _merge_unique([dim], key_cols),
                    "tags": _tags(table, dim, CAPSULE_TYPE_AGGREGATION),
                    "sql": (
                        f"SELECT TOP {top_n} {_q(dim)} AS group_key, COUNT(*) AS record_count "
                        f"FROM {_q(table)} GROUP BY {_q(dim)} ORDER BY record_count DESC"
                    ),
                }
            )
            plans.append(
                {
                    "capsule_type": CAPSULE_TYPE_DISTRIBUTION,
                    "tables_used": [table],
                    "key_columns": _merge_unique([dim], key_cols),
                    "tags": _tags(table, dim, CAPSULE_TYPE_DISTRIBUTION),
                    "sql": (
                        f"SELECT TOP {top_n} {_q(dim)} AS group_key, COUNT(*) AS record_count "
                        f"FROM {_q(table)} WHERE {_q(dim)} IS NOT NULL "
                        f"GROUP BY {_q(dim)} ORDER BY record_count DESC"
                    ),
                }
            )

        # 3) Temporal trends from date columns.
        if include_temporal_aggregations and date_cols:
            for dcol in date_cols[:DEFAULT_MAX_DATE_COLUMNS_PER_TABLE]:
                plans.append(
                    {
                        "capsule_type": CAPSULE_TYPE_TREND,
                        "tables_used": [table],
                        "key_columns": _merge_unique([dcol], key_cols),
                        "tags": _tags(table, dcol, CAPSULE_TYPE_TREND),
                        "sql": (
                            f"SELECT TOP {top_n} CAST({_q(dcol)} AS DATE) AS period_date, COUNT(*) AS record_count "
                            f"FROM {_q(table)} WHERE {_q(dcol)} IS NOT NULL "
                            f"GROUP BY CAST({_q(dcol)} AS DATE) ORDER BY period_date DESC"
                        ),
                    }
                )

                # 4) Temporal anomaly spikes (date + one dimension if available).
                if dimension_cols:
                    dim = dimension_cols[0]
                    plans.append(
                        {
                            "capsule_type": CAPSULE_TYPE_ANOMALY,
                            "tables_used": [table],
                            "key_columns": _merge_unique([dcol, dim], key_cols),
                            "tags": _tags(table, dim, CAPSULE_TYPE_ANOMALY),
                            "sql": (
                                f"SELECT TOP {top_n} CAST({_q(dcol)} AS DATE) AS period_date, {_q(dim)} AS group_key, "
                                f"COUNT(*) AS record_count FROM {_q(table)} "
                                f"WHERE {_q(dcol)} IS NOT NULL AND {_q(dim)} IS NOT NULL "
                                f"GROUP BY CAST({_q(dcol)} AS DATE), {_q(dim)} "
                                f"ORDER BY record_count DESC"
                            ),
                        }
                    )

        # 5) Numeric outlier views (high values).
        for ncol in numeric_cols[:DEFAULT_MAX_NUMERIC_OUTLIER_COLUMNS]:
            if _is_identifier_column(ncol):
                continue
            projection = _merge_unique(key_cols, [ncol])[:DEFAULT_MAX_COLUMNS_PER_RANDOM_PROJECTION]
            plans.append(
                {
                    "capsule_type": CAPSULE_TYPE_ANOMALY,
                    "tables_used": [table],
                    "key_columns": _merge_unique([ncol], key_cols),
                    "tags": _tags(table, ncol, CAPSULE_TYPE_ANOMALY),
                    "sql": (
                        f"SELECT TOP {top_n} {', '.join(_q(c) for c in projection)} "
                        f"FROM {_q(table)} WHERE {_q(ncol)} IS NOT NULL ORDER BY {_q(ncol)} DESC"
                    ),
                }
            )

    return plans


def _q(identifier: str) -> str:
    return f"[{identifier}]"


def _detect_key_columns(columns: list[dict[str, str]]) -> list[str]:
    keys: list[str] = []
    for col in columns:
        name = str(col["name"])
        lname = name.lower()
        if lname.endswith("id") or lname.endswith("_id") or "symbol" in lname or "account" in lname:
            keys.append(name)
    return keys


def _is_groupable_dimension(col_name: str, data_type: str) -> bool:
    if _is_identifier_column(col_name):
        return False
    t = data_type.lower()
    blocked = ("text", "ntext", "image", "xml")
    if any(x in t for x in blocked):
        return False
    return True


def _is_identifier_column(col_name: str) -> bool:
    c = col_name.strip().lower()
    return c.endswith("id") or c.endswith("_id")


def _is_date_type(data_type: str) -> bool:
    t = data_type.lower()
    return "date" in t or "time" in t


def _is_numeric_type(data_type: str) -> bool:
    t = data_type.lower()
    numeric_tokens = ("int", "decimal", "numeric", "float", "real", "money", "smallmoney", "bit")
    return any(tok in t for tok in numeric_tokens)


def _select_random_projection(columns: list[str], key_columns: list[str]) -> list[str]:
    if not columns:
        return []
    keys = [c for c in key_columns if c in columns][:DEFAULT_MAX_KEY_COLUMNS_IN_PROJECTION]
    remaining = [c for c in columns if c not in keys]
    random.shuffle(remaining)
    keys = keys[:DEFAULT_MAX_KEY_COLUMNS_IN_PROJECTION]
    take = max(0, DEFAULT_MAX_COLUMNS_PER_RANDOM_PROJECTION - len(keys))
    return (keys + remaining[:take])[:DEFAULT_MAX_COLUMNS_PER_RANDOM_PROJECTION]


def _tags(table: str, group_col: str, capsule_type: str) -> list[str]:
    table_tag = _slug(table)
    col_tag = _slug(group_col) if group_col else ""
    tags = [capsule_type, f"{table_tag}_{capsule_type}"]
    if col_tag:
        tags.append(f"{col_tag}_signal")
    return tags


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")


def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*primary, *secondary]:
        if item not in merged:
            merged.append(item)
    return merged
