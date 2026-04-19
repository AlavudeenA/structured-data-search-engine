"""Build a single user-created capsule, persist it, and link it to existing capsules."""

from __future__ import annotations

import json
import logging
import re

from ..llm_instructions import CAPSULE_ENRICH_SYSTEM, CAPSULE_ENRICH_USER, CAPSULE_SQL_GEN_SYSTEM, CAPSULE_SQL_GEN_USER
from ..llm_service import call_llm, call_llm_json
from ..models import CapsuleDefinition
from .capsule_generator import generate_all_capsules
from .store_manager import _persist_analytical

logger = logging.getLogger(__name__)

_VALID_TYPES = {
    "aggregation", "trend", "violation", "pattern",
    "risk", "operational", "distribution", "sample",
}
_VALID_SIGNAL_METHODS = {"rule_based", "llm_summary", "sample"}


def enrich_user_capsule_metadata(
    what: str,
    sql: str,
    rows: list[dict],
) -> dict:
    """Call LLM to derive capsule_type, how, tags, tables_used, key_columns,
    staleness_trigger, ttl_hours, signal_method, and embed_text from the SQL
    and its result rows.

    Returns a dict of enriched fields (all with safe fallbacks if LLM fails).
    """
    columns = list(rows[0].keys()) if rows else []
    rows_json = json.dumps(rows[:10], default=str)

    raw = call_llm_json(
        system_prompt=CAPSULE_ENRICH_SYSTEM,
        user_prompt=CAPSULE_ENRICH_USER.format(
            what=what,
            sql=sql,
            columns=", ".join(columns),
            row_count=len(rows),
            rows_json=rows_json,
        ),
        max_tokens=512,
    )

    # Parse table names directly from SQL as a reliable fallback
    sql_tables = sorted({
        t for t in re.findall(r"(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.IGNORECASE)
    })

    if not raw or not isinstance(raw, dict):
        logger.warning("LLM enrichment failed — using safe defaults")
        return _safe_defaults(what, sql_tables, columns)

    capsule_type = raw.get("capsule_type", "aggregation")
    if capsule_type not in _VALID_TYPES:
        capsule_type = "aggregation"

    signal_method = raw.get("signal_method", "rule_based")
    if signal_method not in _VALID_SIGNAL_METHODS:
        signal_method = "rule_based"

    tables = raw.get("tables_used") or sql_tables  # always prefer LLM-parsed, fall back to regex

    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower().replace(" ", "_") for t in tags if t]

    key_cols = raw.get("key_columns") or columns[:6]
    if not isinstance(key_cols, list):
        key_cols = columns[:6]

    return {
        "capsule_type":        capsule_type,
        "how":                 str(raw.get("how") or f"User-defined SQL query against {', '.join(tables)}"),
        "tags":                tags + ["user_defined"],
        "tables_used":         tables if tables else sql_tables,
        "key_columns":         key_cols,
        "staleness_trigger":   str(raw.get("staleness_trigger") or "data_change"),
        "ttl_hours":           int(raw.get("ttl_hours") or 24),
        "signal_method":       signal_method,
        "embed_text_template": str(raw.get("embed_text") or what),
    }


def _safe_defaults(what: str, sql_tables: list[str], columns: list[str]) -> dict:
    return {
        "capsule_type":        "aggregation",
        "how":                 f"User-defined SQL query against {', '.join(sql_tables)}",
        "tags":                ["user_defined"],
        "tables_used":         sql_tables,
        "key_columns":         columns[:6],
        "staleness_trigger":   "data_change",
        "ttl_hours":           24,
        "signal_method":       "rule_based",
        "embed_text_template": what,
    }


def generate_capsule_sql(intent: str) -> str:
    """Generate an analytical SQLite SELECT query from a plain-English intent description.

    Returns the raw SQL string, or an empty string if generation fails.
    """
    from ..database_connection import get_fk_relationships, get_schema_metadata

    schema_meta = get_schema_metadata()
    schema_text = "\n".join(
        f"{table}: " + ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        for table, cols in schema_meta.items()
    )
    fk_meta = get_fk_relationships()
    fk_text = "\n".join(
        f"{r['parent_table']}.{r['parent_column']} → {r['ref_table']}.{r['ref_column']}"
        for r in fk_meta
    ) or "None"

    sql = call_llm(
        system_prompt=CAPSULE_SQL_GEN_SYSTEM.format(schema=schema_text, fk_relationships=fk_text),
        user_prompt=CAPSULE_SQL_GEN_USER.format(intent=intent),
        max_tokens=512,
    )

    if not sql or sql.startswith("[LLM"):
        logger.warning("SQL generation failed for intent: %s", intent)
        return ""

    # Strip any accidental markdown fences
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)
    return sql.strip()


def build_single_capsule(capsule_def_dict: dict) -> bool:
    """Build and upsert one user capsule, then wire bidirectional links to related existing capsules."""
    try:
        definition = CapsuleDefinition(**capsule_def_dict)
        capsules = generate_all_capsules([definition])
        if not capsules:
            return False

        _persist_analytical(capsules)

        # Link the new capsule to any related existing capsules (O(n) scan, set_payload only)
        from .relationship_builder import link_user_capsule_to_existing
        link_user_capsule_to_existing(capsules[0])

        return True
    except Exception as exc:
        logger.error("Failed to build capsule %s: %s", capsule_def_dict.get("capsule_id"), exc)
        return False
