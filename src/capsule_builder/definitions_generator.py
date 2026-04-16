"""Generates a new CAPSULE_DEFINITIONS list from live DB schema using the LLM."""

from __future__ import annotations

import json
import textwrap

from ..database_connection import get_fk_relationships, get_schema_metadata
from ..llm_instructions import CAPSULE_REGEN_SYSTEM, CAPSULE_REGEN_USER
from ..llm_service import call_llm
from ..business_schema.capsule_definitions import CAPSULE_DEFINITIONS


# One capsule pulled from the existing definitions to show the LLM the exact format
_FORMAT_EXAMPLE = json.dumps([CAPSULE_DEFINITIONS[0]], indent=2)


def _build_schema_text(schema_meta: list[dict]) -> str:
    """Render schema metadata as a readable table → column list."""
    tables: dict[str, list[str]] = {}
    for row in schema_meta:
        tbl = row.get("table_name", "")
        col = f"{row.get('column_name', '')} ({row.get('data_type', '')})"
        tables.setdefault(tbl, []).append(col)
    lines = []
    for tbl, cols in tables.items():
        lines.append(f"Table: {tbl}")
        for c in cols:
            lines.append(f"  - {c}")
    return "\n".join(lines)


def _build_fk_text(fk_rows: list[dict]) -> str:
    """Render FK relationships as readable lines."""
    lines = []
    for row in fk_rows:
        lines.append(
            f"{row.get('table_name')}.{row.get('column_name')} → "
            f"{row.get('referenced_table')}.{row.get('referenced_column')}"
        )
    return "\n".join(lines) if lines else "No explicit foreign keys detected."


def generate_capsule_definitions_via_llm() -> str:
    """
    Query the live DB schema and ask the LLM to produce a new CAPSULE_DEFINITIONS list.
    Returns the raw LLM output string (Python list literal).
    """
    schema_meta = get_schema_metadata()
    fk_rows = get_fk_relationships()

    schema_text = _build_schema_text(schema_meta)
    fk_text = _build_fk_text(fk_rows)

    user_prompt = CAPSULE_REGEN_USER.format(
        schema=schema_text,
        fk_relationships=fk_text,
        format_example=_FORMAT_EXAMPLE,
    )

    return call_llm(
        system_prompt=CAPSULE_REGEN_SYSTEM,
        user_prompt=user_prompt,
        model_slot="groq_sql_model",   # strongest model — same one used for SQL generation
        temperature=0.2,
        max_tokens=8000,
    )


def save_generated_definitions(raw_python: str, target_path: str) -> None:
    """
    Overwrite the CAPSULE_DEFINITIONS block in capsule_definitions.py with
    the LLM-generated Python list. Preserves SCHEMA_DEFINITIONS.
    """
    import re
    from pathlib import Path

    path = Path(target_path)
    existing = path.read_text(encoding="utf-8")

    # Replace everything from CAPSULE_DEFINITIONS = [ ... ] to the closing ]
    new_block = f"CAPSULE_DEFINITIONS = {raw_python.strip()}\n"
    updated = re.sub(
        r"CAPSULE_DEFINITIONS\s*=\s*\[.*?\n\]",
        new_block,
        existing,
        flags=re.DOTALL,
    )
    path.write_text(updated, encoding="utf-8")
