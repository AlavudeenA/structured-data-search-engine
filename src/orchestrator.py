# File: src/orchestrator.py
"""Orchestrate routing, SQL execution, and response summarization."""

from __future__ import annotations

import json
import sys
from typing import Any

from .app_constants import (
    ANALYTICAL_INSUFFICIENT_ANSWER_MARKERS,
    CAPSULE_TYPE_SCHEMA_CONTEXT,
    CONFIDENCE_LOW_THRESHOLD,
    DEFAULT_ANALYTICAL_MIN_SCORE,
    DEFAULT_SCHEMA_CONTEXT_TOP_K,
    DEFAULT_ANALYTICAL_TOP_K,
    DEFAULT_ANALYTICAL_TOP_K_PER_TYPE,
    FORCED_ANALYTICAL_TOP_K,
    FORCED_ANALYTICAL_TOP_K_PER_TYPE,
    SCHEMA_CONTEXT_MIN_SCORE,
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
        forced_result = _maybe_switch_to_schema_context_sql(
            user_query=user_query,
            intent_data={
                **intent_data,
                "forced_route": "analytical_query",
            },
            analytical=analytical,
        )
        if forced_result is not None:
            return forced_result
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
    fallback_result = _maybe_switch_to_schema_context_sql(
        user_query=user_query,
        intent_data=intent_data,
        analytical=analytical,
    )
    if fallback_result is not None:
        return fallback_result
    return {
        "intent": intent_data,
        "route": analytical["route"],
        "retrieval": {
            "hits": analytical["hits"],
            "supporting": analytical.get("supporting", {}),
        },
        "answer": analytical["answer"],
    }


def _maybe_switch_to_schema_context_sql(
    user_query: str,
    intent_data: dict[str, Any],
    analytical: dict[str, Any],
) -> dict[str, Any] | None:
    if not _should_switch_to_schema_context_sql(analytical):
        return None

    schema_hits = _resolve_schema_context_hits(user_query, analytical)
    sql_data = generate_sql_from_question(user_query, schema_capsules=schema_hits)
    execution = execute_sql(sql_data["sql"])
    summary = summarize_result(user_query, execution)
    return {
        "intent": intent_data,
        "route": "vector_retrieval_schema_context_llm",
        "sql_reason": sql_data["reason"],
        "execution": execution,
        "retrieval": {
            "hits": schema_hits,
            "supporting": _schema_supporting_payload(schema_hits),
        },
        "answer": summary,
    }


def _should_switch_to_schema_context_sql(analytical: dict[str, Any]) -> bool:
    hits = analytical.get("hits", []) or []
    if not hits:
        return True

    top_hit = hits[0]
    top_payload = top_hit.get("payload", {}) or {}
    top_type = str(top_payload.get("capsule_type", "")).strip().lower()
    if top_type == CAPSULE_TYPE_SCHEMA_CONTEXT:
        return True

    top_score = float(top_hit.get("score", 0.0))
    if top_score < DEFAULT_ANALYTICAL_MIN_SCORE:
        return True

    answer = str(analytical.get("answer", "")).strip().lower()
    return any(marker in answer for marker in ANALYTICAL_INSUFFICIENT_ANSWER_MARKERS)


def _resolve_schema_context_hits(user_query: str, analytical: dict[str, Any]) -> list[dict[str, Any]]:
    analytical_hits = analytical.get("hits", []) or []
    schema_hits_from_analytical = [
        hit
        for hit in analytical_hits
        if str(hit.get("payload", {}).get("capsule_type", "")).strip().lower() == CAPSULE_TYPE_SCHEMA_CONTEXT
    ]
    if schema_hits_from_analytical:
        return schema_hits_from_analytical

    schema_retrieval = retrieve_analytical_context(
        user_query,
        top_k=DEFAULT_SCHEMA_CONTEXT_TOP_K,
        min_score=SCHEMA_CONTEXT_MIN_SCORE,
        top_k_per_type=None,
        capsule_type=CAPSULE_TYPE_SCHEMA_CONTEXT,
    )
    return schema_retrieval.get("hits", []) or []


def _schema_supporting_payload(schema_hits: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    for hit in schema_hits:
        payload = hit.get("payload", {}) or {}
        name = str(payload.get("capsule_name", "")).strip()
        if name and name not in names:
            names.append(name)
    return {
        "mode": "schema_context",
        "capsule_names": names,
        "capsule_count": len(schema_hits),
        "source_sql_hash": "",
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
