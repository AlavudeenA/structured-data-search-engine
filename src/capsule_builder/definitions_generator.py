"""
LLM-driven capsule definition regenerator.

generate_capsule_definitions_via_llm()
  → reads live schema + FKs from SQLite
  → builds two rich format examples from the existing definitions
  → calls the language model with the full CAPSULE_REGEN prompt
  → returns raw Python list literal string

save_generated_definitions(raw_python, target_path)
  → validates the output is syntactically valid Python
  → rewrites the CAPSULE_DEFINITIONS block in the target .py file
  → always saves to the canonical business_schema/capsule_definitions.py
    regardless of which path is passed in (the UI path arg is accepted for
    forward-compatibility but the canonical path is always used)
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from ..database_connection import get_fk_relationships, get_schema_metadata
from ..llm_instructions import CAPSULE_REGEN_SYSTEM, CAPSULE_REGEN_USER
from ..llm_service import call_llm
from ..business_schema.domain import DOMAIN_NAME, load_db_metadata

logger = logging.getLogger(__name__)

# Canonical file to overwrite — resolved from domain config so swapping business_schema/ works automatically.
from ..business_schema.domain import DB_SCRIPT_PATH as _DOMAIN_DB_SCRIPT_PATH
_CANONICAL_PATH = _DOMAIN_DB_SCRIPT_PATH.parent / "capsule_definitions.py"


# ── Format example builder ────────────────────────────────────────────────────

def _pick_format_examples() -> str:
    """
    Return two representative capsules as a JSON string:
    - a P1 violation capsule (complex date-overlap SQL, llm_summary)
    - a P2 aggregation capsule (simple group-by SQL, rule_based)
    These two examples together cover the full range of structures
    the LLM must produce.
    """
    from ..business_schema.capsule_definitions import CAPSULE_DEFINITIONS

    # Prefer specific IDs; fall back to first and last in list
    p1_id = "violations_on_restricted_securities"
    p2_id = "trade_requests_by_broker_dealer"

    caps = {c["capsule_id"]: c for c in CAPSULE_DEFINITIONS}

    p1 = caps.get(p1_id) or next(
        (c for c in CAPSULE_DEFINITIONS if c.get("capsule_type") == "violation"), None
    )
    p2 = caps.get(p2_id) or next(
        (c for c in CAPSULE_DEFINITIONS if c.get("capsule_type") == "aggregation"), None
    )

    examples = [c for c in [p1, p2] if c is not None]
    if not examples:
        examples = CAPSULE_DEFINITIONS[:2]

    return json.dumps(examples, indent=2)


# ── Schema / FK formatters ────────────────────────────────────────────────────

def _build_schema_text(schema: dict[str, list[dict[str, str]]]) -> str:
    """
    Render {table: [{name, type},...]} as readable text for the LLM prompt.
    get_schema_metadata() returns this format — not a flat list of rows.
    """
    lines: list[str] = []
    for table_name, columns in sorted(schema.items()):
        col_str = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        lines.append(f"Table: {table_name}")
        lines.append(f"  Columns: {col_str}")
    return "\n".join(lines)


def _build_fk_text(fks: list[dict[str, str]]) -> str:
    """
    Render FK list as readable text.
    get_fk_relationships() returns {parent_table, parent_column, ref_table, ref_column}.
    """
    if not fks:
        return "No explicit foreign keys detected."
    lines = [
        f"{r['parent_table']}.{r['parent_column']} → {r['ref_table']}.{r['ref_column']}"
        for r in fks
    ]
    return "\n".join(lines)


# ── Main generation function ──────────────────────────────────────────────────

def generate_capsule_definitions_via_llm() -> str:
    """
    Query the live DB schema and ask Groq to produce a new CAPSULE_DEFINITIONS list.
    Returns the raw LLM output string (Python list literal).
    Uses the strongest model slot (groq_sql_model = llama-3.3-70b-versatile).
    """
    schema = get_schema_metadata()
    fks = get_fk_relationships()

    schema_text = _build_schema_text(schema)
    fk_text = _build_fk_text(fks)
    format_example = _pick_format_examples()

    user_prompt = CAPSULE_REGEN_USER.format(
        schema=schema_text,
        fk_relationships=fk_text,
        db_metadata=load_db_metadata(),
        format_example=format_example,
        DOMAIN_NAME=DOMAIN_NAME,
    )

    logger.info(
        "Requesting capsule definition regeneration | schema_tables=%d | fk_count=%d",
        len(schema),
        len(fks),
    )

    raw = call_llm(
        system_prompt=CAPSULE_REGEN_SYSTEM,
        user_prompt=user_prompt,
        model_slot="groq_sql_model",  # strongest model — 70b versatile
        temperature=0.2,
        max_tokens=8000,
    )

    # Strip any accidental markdown fences the model may emit
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("python"):
            raw = raw[6:]
        raw = raw.rsplit("```", 1)[0].strip()

    logger.info("LLM regeneration complete | output_chars=%d", len(raw))
    return raw


# ── Validation ────────────────────────────────────────────────────────────────

def validate_definitions_output(raw_python: str) -> tuple[bool, str]:
    """
    Validate the LLM output is a syntactically valid Python list literal.
    Returns (is_valid, error_message_or_empty).
    """
    candidate = raw_python.strip()
    # The LLM may return just the list (no assignment) or include assignment
    if not (candidate.startswith("[") or "CAPSULE_DEFINITIONS" in candidate):
        return False, "Output does not start with '[' — not a Python list."

    # Wrap in an assignment so ast.parse can check it as a statement
    test_src = f"_x = {candidate}" if candidate.startswith("[") else candidate
    try:
        ast.parse(test_src)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"


# ── Save function ─────────────────────────────────────────────────────────────

def save_generated_definitions(raw_python: str, target_path: str | None = None) -> None:
    """
    Overwrite the CAPSULE_DEFINITIONS block in business_schema/capsule_definitions.py.

    The file is always written to _CANONICAL_PATH (src/business_schema/capsule_definitions.py)
    regardless of what target_path is passed — the UI sends an approximate path for UI display
    only; the authoritative save location is always the canonical one.

    Raises ValueError if the Python output fails syntax validation.
    """
    is_valid, error = validate_definitions_output(raw_python)
    if not is_valid:
        raise ValueError(f"LLM output failed Python syntax validation: {error}")

    # Normalise: ensure we have just the list, not an assignment
    list_literal = raw_python.strip()
    if list_literal.startswith("CAPSULE_DEFINITIONS"):
        # strip "CAPSULE_DEFINITIONS = " prefix if model included it
        import re
        list_literal = re.sub(r"^CAPSULE_DEFINITIONS\s*=\s*", "", list_literal).strip()

    path = _CANONICAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        import re as _re
        # Replace the entire CAPSULE_DEFINITIONS block (from = [ to closing \n])
        new_block = f"CAPSULE_DEFINITIONS: list[dict] = {list_literal}\n"
        updated = _re.sub(
            r"CAPSULE_DEFINITIONS\s*(?::\s*list\[dict\])?\s*=\s*\[.*?\n\]",
            new_block,
            existing,
            flags=_re.DOTALL,
        )
        if "CAPSULE_DEFINITIONS" not in updated:
            # Pattern didn't match — append at the end
            updated = existing.rstrip() + "\n\n" + new_block
        path.write_text(updated, encoding="utf-8")
    else:
        # First-time write — create the full module
        module_text = (
            '"""\n'
            'Domain-specific capsule definitions regenerated by AI.\n'
            'Edit via the "Generate Capsule Definitions" button in the UI.\n'
            '"""\n\nfrom __future__ import annotations\n\n'
            f"CAPSULE_DEFINITIONS: list[dict] = {list_literal}\n"
        )
        path.write_text(module_text, encoding="utf-8")

    logger.info("Capsule definitions saved to %s", path)

    # Reload the module so the next Generate All Capsules picks up the new definitions
    try:
        import importlib
        from ..business_schema import capsule_definitions as _mod
        importlib.reload(_mod)
        logger.info("Module reloaded: %d definitions now active", len(_mod.CAPSULE_DEFINITIONS))
    except Exception as exc:
        logger.warning("Module reload failed (will take effect on next process start): %s", exc)
