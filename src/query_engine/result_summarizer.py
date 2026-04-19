"""Summarize SQL result rows into concise compliance-facing language."""

from __future__ import annotations

import json
from typing import Any

from ..llm_service import call_llm
from ..llm_instructions import RESULT_SUMMARIZER_SYSTEM, RESULT_SUMMARIZER_USER


def summarize_sql_result(question: str, rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Summarize query results using Groq with deterministic fallback."""
    if not rows:
        return "The query returned no rows."
    answer = call_llm(
        RESULT_SUMMARIZER_SYSTEM,
        RESULT_SUMMARIZER_USER.format(question=question, row_count=len(rows), rows_sample=json.dumps(rows[:25], default=str, indent=2)),
        model_slot="groq_summary_model",
        temperature=0.2,
        max_tokens=300,
    )
    if not answer or answer.startswith("[LLM"):
        first_row = rows[0]
        preview = ", ".join(f"{key}={value}" for key, value in list(first_row.items())[:4])
        return f"The query returned {len(rows)} rows. The top row is {preview}. Columns included: {', '.join(columns[:6])}."
    return answer
