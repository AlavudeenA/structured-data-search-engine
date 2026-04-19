"""Data-change detection via per-table row-count + max-timestamp fingerprint."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from .app_constants import DATA_FINGERPRINT_FILE
from .database_connection import execute_select
from .models import DataFingerprint

logger = logging.getLogger(__name__)

# Map each table to its most meaningful "last changed" timestamp column.
# None means no timestamp available — only row count is used.
_TABLE_TIMESTAMP_COLS: dict[str, str | None] = {
    "Employee":           "UpdatedAt",
    "BrokerDealer":       "UpdatedAt",
    "Account":            "UpdatedAt",
    "RestrictedSecurity": "UpdatedAt",
    "TradeRequest":       "UpdatedAt",
    "ComplianceAlert":    "UpdatedAt",
    "ApprovalWorkflow":   "UpdatedAt",
}


def _fetch_table_stats() -> dict[str, dict[str, str]]:
    stats: dict[str, dict[str, str]] = {}
    for table, ts_col in _TABLE_TIMESTAMP_COLS.items():
        try:
            sql = (
                f"SELECT COUNT(*) AS cnt, MAX({ts_col}) AS latest FROM {table}"
                if ts_col
                else f"SELECT COUNT(*) AS cnt FROM {table}"
            )
            rows = execute_select(sql)
            row = rows[0] if rows else {}
            stats[table] = {
                "count": str(row.get("cnt", 0)),
                "latest": str(row.get("latest", "")),
            }
        except Exception as exc:
            logger.warning("Could not fetch stats for %s: %s", table, exc)
            stats[table] = {"count": "error", "latest": "error"}
    return stats


def _hash_stats(stats: dict[str, dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()


def compute_data_fingerprint() -> DataFingerprint:
    stats = _fetch_table_stats()
    return DataFingerprint(
        generated_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=_hash_stats(stats),
        table_stats=stats,
    )


def save_data_fingerprint(fp: DataFingerprint) -> None:
    DATA_FINGERPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FINGERPRINT_FILE.write_text(fp.model_dump_json(indent=2), encoding="utf-8")


def load_data_fingerprint() -> DataFingerprint | None:
    if not DATA_FINGERPRINT_FILE.exists():
        return None
    try:
        return DataFingerprint.model_validate_json(
            DATA_FINGERPRINT_FILE.read_text(encoding="utf-8")
        )
    except Exception as exc:
        logger.error("Failed to load data fingerprint: %s", exc)
        return None


def check_schema_and_refresh_if_needed() -> bool:
    """Check if DDL changed; if so, rebuild all capsules (full rebuild).
    Returns True if a rebuild was triggered.
    """
    from .capsule_builder.store_manager import generate_all_capsule_collections
    from .database_connection import get_fk_relationships, get_schema_metadata
    from .embedding import compute_schema_fingerprint, load_schema_fingerprint, save_schema_fingerprint

    current_schema = get_schema_metadata()
    current_fks = get_fk_relationships()
    current_fp = compute_schema_fingerprint(current_schema, current_fks)
    previous = load_schema_fingerprint()

    if previous is not None and previous.fingerprint == current_fp:
        return False

    logger.info("DDL changed — triggering full capsule rebuild")
    try:
        generate_all_capsule_collections()
        save_schema_fingerprint(current_schema, current_fks)
        save_data_fingerprint(compute_data_fingerprint())
        return True
    except Exception as exc:
        logger.error("Full rebuild after DDL change failed: %s", exc)
        return False


def check_and_refresh_if_needed(
    analytical_ids: list[str],
    linked_ids: list[str],
) -> bool:
    """Check if data changed; if so, refresh only the specified capsules.
    Returns True if a refresh was triggered.
    """
    from .capsule_builder.store_manager import refresh_targeted_capsules

    previous = load_data_fingerprint()
    if previous is None:
        # First run — save baseline; user must do initial full build manually.
        try:
            save_data_fingerprint(compute_data_fingerprint())
        except Exception:
            pass
        return False

    if not analytical_ids:
        return False

    try:
        current = compute_data_fingerprint()
    except Exception as exc:
        logger.warning("Could not compute data fingerprint: %s", exc)
        return False

    if current.fingerprint == previous.fingerprint:
        return False

    logger.info(
        "Data changed — refreshing %d analytical + %d linked capsules",
        len(analytical_ids),
        len(linked_ids),
    )
    try:
        refresh_targeted_capsules(analytical_ids, linked_ids)
        save_data_fingerprint(current)
        return True
    except Exception as exc:
        logger.error("Targeted capsule refresh failed: %s", exc)
        return False
