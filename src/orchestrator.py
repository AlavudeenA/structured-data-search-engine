# File: src/orchestrator.py
"""Orchestrate routing, SQL execution, and response summarization."""

from __future__ import annotations

import json
import sys
from typing import Any

from .app_constants import (
    CONFIDENCE_LOW_THRESHOLD,
    DEFAULT_ANALYTICAL_MIN_SCORE,
    DEFAULT_ANALYTICAL_TOP_K,
    DEFAULT_ANALYTICAL_TOP_K_PER_TYPE,
    FORCED_ANALYTICAL_TOP_K,
    FORCED_ANALYTICAL_TOP_K_PER_TYPE,
)
from .analytical_retriever import retrieve_analytical_context
from .intent_router_service import detect_intent
from .result_summarizer import summarize_result
from .sql_executor import execute_sql
from .text_to_sql import generate_sql_from_question


def handle_user_query(
    user_query: str,
    capsule_type_filter: str | None = None,
    entity_filter: str | None = None,
    capsule_topic_filter: str | None = None,
    force_analytical: bool = False,
) -> dict[str, Any]:
    """Run the hybrid flow for a user question."""
    intent_data = detect_intent(user_query)
    intent = intent_data["intent"]
    confidence = float(intent_data.get("confidence", 0.0))

    if force_analytical:
        analytical = retrieve_analytical_context(
            user_query,
            top_k=FORCED_ANALYTICAL_TOP_K,
            min_score=DEFAULT_ANALYTICAL_MIN_SCORE,
            top_k_per_type=FORCED_ANALYTICAL_TOP_K_PER_TYPE,
            capsule_type=capsule_type_filter,
            entity=entity_filter,
            capsule_topic=capsule_topic_filter,
        )
        return {
            "intent": {
                **intent_data,
                "forced_route": "analytical_query",
            },
            "route": analytical["route"],
            "retrieval": {
                "hits": analytical["hits"],
                "supporting": analytical.get("supporting", {}),
            },
            "answer": analytical["answer"],
        }

    if confidence < CONFIDENCE_LOW_THRESHOLD:
        return {
            "intent": intent_data,
            "route": "invalid_query",
            "answer": (
                "The question could not be understood. "
                "Please ask a meaningful data question."
            ),
        }

    if intent == "structured_query":
        sql_data = generate_sql_from_question(user_query)
        execution = execute_sql(sql_data["sql"])
        summary = summarize_result(user_query, execution)
        return {
            "intent": intent_data,
            "route": "text_to_sql",
            "sql_reason": sql_data["reason"],
            "execution": execution,
            "answer": summary,
        }

    analytical = retrieve_analytical_context(
        user_query,
        top_k=DEFAULT_ANALYTICAL_TOP_K,
        min_score=DEFAULT_ANALYTICAL_MIN_SCORE,
        top_k_per_type=DEFAULT_ANALYTICAL_TOP_K_PER_TYPE,
        capsule_type=capsule_type_filter,
        entity=entity_filter,
        capsule_topic=capsule_topic_filter,
    )
    return {
        "intent": intent_data,
        "route": analytical["route"],
        "retrieval": {
            "hits": analytical["hits"],
            "supporting": analytical.get("supporting", {}),
        },
        "answer": analytical["answer"],
    }


def _format_console_output(result: dict[str, Any]) -> str:
    output = {
        "intent": result.get("intent", {}),
        "route": result.get("route"),
        "sql": result.get("execution", {}).get("sql"),
        "row_count": result.get("execution", {}).get("row_count"),
        "answer": result.get("answer"),
    }
    return json.dumps(output, indent=2, default=str)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit("Usage: py -3 -m src.orchestrator \"your question\"")
    result = handle_user_query(query)
    print(_format_console_output(result))
