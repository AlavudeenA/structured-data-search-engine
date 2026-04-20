"""SQL generation with live schema, foreign keys, schema-context capsules, and related signals."""

from __future__ import annotations

import logging
import re

from ..database_connection import fk_to_text, get_fk_relationships, get_schema_metadata, schema_to_text
from ..business_schema.domain import load_db_metadata
from ..llm_service import call_llm
from ..models import ContextPackage
from ..llm_instructions import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER, SQL_REASON_SYSTEM, SQL_REASON_USER

logger = logging.getLogger(__name__)


def _clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text.rstrip(";").strip()


def _is_safe_sql(sql: str) -> bool:
    normalized = sql.lower().strip()
    return normalized.startswith("select") or normalized.startswith("with")


def _render_schema_capsules(context_package: ContextPackage) -> str:
    lines: list[str] = []
    for capsule in context_package.schema_capsules:
        lines.append(f"[{capsule.get('capsule_id')}] {capsule.get('summary')}")
        if capsule.get("recommended_joins"):
            lines.append(f"Joins: {' | '.join(capsule.get('recommended_joins', [])[:4])}")
        if capsule.get("recommended_filters"):
            lines.append(f"Filters: {' | '.join(capsule.get('recommended_filters', [])[:4])}")
        if capsule.get("sql_template"):
            lines.append(f"Template: {capsule.get('sql_template')}")
    return "\n".join(lines) if lines else "No schema capsules available."


def _render_linked_context(context_package: ContextPackage) -> str:
    if not context_package.linked_capsules:
        return "No linked capsules available."
    return "\n".join(
        f"[{capsule.get('capsule_id')}] {capsule.get('signal')}"
        for capsule in context_package.linked_capsules[:5]
    )


def generate_sql(question: str, context_package: ContextPackage) -> dict[str, str] | None:
    """Generate SQL and a short reasoning note."""
    schema = get_schema_metadata()
    fk_relationships = get_fk_relationships()
    system_prompt = SQL_GENERATION_SYSTEM.format(
        schema=schema_to_text(schema),
        fk_relationships=fk_to_text(fk_relationships),
        related_context=_render_linked_context(context_package),
        capsule_context=_render_schema_capsules(context_package),
        db_metadata=load_db_metadata(),
    )
    user_prompt = SQL_GENERATION_USER.format(question=question)
    raw_sql = call_llm(system_prompt, user_prompt, model_slot="groq_sql_model", temperature=0.0, max_tokens=700)
    if not raw_sql or raw_sql.startswith("[LLM"):
        return None

    sql = _clean_sql(raw_sql)
    if not _is_safe_sql(sql):
        logger.error("Rejected unsafe SQL: %s", sql[:200])
        return None

    reason = call_llm(
        SQL_REASON_SYSTEM,
        SQL_REASON_USER.format(
            question=question,
            intent_payload={},
            schema_capsules=_render_schema_capsules(context_package),
            sql=sql,
        ),
        model_slot="groq_summary_model",
        temperature=0.1,
        max_tokens=180,
    )
    return {"sql": sql, "reason": reason if reason and not reason.startswith("[LLM") else "SQL was generated using live schema, foreign keys, and schema-context capsule guidance."}
