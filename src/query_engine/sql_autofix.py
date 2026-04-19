"""SQL autofix that retries failed generated SQL once using error-aware prompt repair."""

from __future__ import annotations

import logging
import re

from ..database_connection import fk_to_text, get_fk_relationships, get_schema_metadata, schema_to_text
from ..llm_service import call_llm
from ..llm_instructions import SQL_AUTOFIX_SYSTEM, SQL_AUTOFIX_USER

logger = logging.getLogger(__name__)


def _clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text.rstrip(";").strip()


def fix_sql(original_sql: str, error_message: str) -> str | None:
    """Ask Groq to repair SQL using schema and the actual database error."""
    raw_sql = call_llm(
        SQL_AUTOFIX_SYSTEM.format(
            schema=schema_to_text(get_schema_metadata()),
            fk_relationships=fk_to_text(get_fk_relationships()),
        ),
        SQL_AUTOFIX_USER.format(original_sql=original_sql, error_message=error_message),
        model_slot="groq_sql_fix_model",
        temperature=0.0,
        max_tokens=700,
    )
    if not raw_sql or raw_sql.startswith("[LLM"):
        return None
    fixed_sql = _clean_sql(raw_sql)
    if not fixed_sql.lower().startswith(("select", "with")):
        logger.error("Autofix returned unsafe SQL: %s", fixed_sql[:200])
        return None
    return fixed_sql
