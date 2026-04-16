"""Main query orchestration for compliance question answering.

CLI usage:
    py -3 -m src.pipeline2.orchestrator "Which broker has the most escalations?"
"""

from __future__ import annotations

import json
import logging
import sys
import time

from ..app_constants import (
    CONFIDENCE_THRESHOLD_CAPSULE_ANSWER,
    INTENT_ANALYTICAL,
    INTENT_HYBRID,
    INTENT_OPERATIONAL,
    INTENT_STRUCTURED,
)
from ..models import ContextPackage, QueryResponse
from .analytical_retriever import answer_from_capsules
from .context_packager import build_context_package
from .context_searcher import search_all_collections
from .query_router import detect_intent
from .result_summarizer import summarize_sql_result
from .sql_executor import execute_with_autofix
from .sql_generator import generate_sql

logger = logging.getLogger(__name__)


def _capsules_used(context_package: ContextPackage) -> list[str]:
    capsule_ids: list[str] = []
    if context_package.primary_capsule:
        capsule_ids.append(context_package.primary_capsule.get("capsule_id", ""))
    capsule_ids.extend(capsule.get("capsule_id", "") for capsule in context_package.linked_capsules)
    capsule_ids.extend(capsule.get("capsule_id", "") for capsule in context_package.derived_capsules)
    capsule_ids.extend(capsule.get("capsule_id", "") for capsule in context_package.schema_capsules)
    return [capsule_id for capsule_id in capsule_ids if capsule_id]


def _top_hit_is_schema_context(context_package: ContextPackage) -> bool:
    return context_package.primary_capsule is None and bool(context_package.schema_capsules)


def _run_sql_path(
    question: str,
    intent_payload: dict,
    context_package: ContextPackage,
    route_name: str,
    started_at: float,
) -> QueryResponse:
    sql_plan = generate_sql(question, context_package)
    if not sql_plan:
        return QueryResponse(
            answer="The system could not generate a valid SQL query for this question.",
            route_taken="sql_generation_failed",
            intent=intent_payload["intent"],
            confidence=float(intent_payload["confidence"]),
            capsules_used=_capsules_used(context_package),
            context_package=context_package.model_dump(),
            answer_ms=int((time.time() - started_at) * 1000),
            sql_reason=None,
            intent_payload=intent_payload,
        )

    execution = execute_with_autofix(sql_plan["sql"])
    sql_rows = execution.get("rows", [])
    sql_generated = execution.get("autofix_sql") or sql_plan["sql"]
    answer = summarize_sql_result(question, sql_rows, execution.get("columns", [])) if execution.get("error") is None else f"SQL execution failed: {execution.get('error')}"

    return QueryResponse(
        answer=answer,
        route_taken=route_name,
        intent=intent_payload["intent"],
        confidence=float(intent_payload["confidence"]),
        capsules_used=_capsules_used(context_package),
        sql_generated=sql_generated,
        sql_rows=sql_rows[:50],
        context_package=context_package.model_dump(),
        autofix_used=bool(execution.get("autofix_used")),
        error=execution.get("error"),
        answer_ms=int((time.time() - started_at) * 1000),
        sql_reason=sql_plan.get("reason"),
        intent_payload=intent_payload,
    )


def handle_query(question: str) -> QueryResponse:
    """Run the full pipeline for one user question."""
    started_at = time.time()
    intent = detect_intent(question)
    intent_payload = intent.model_dump()
    search_hits = search_all_collections(question)
    context_package = build_context_package(search_hits)

    if intent.intent in {INTENT_STRUCTURED, INTENT_OPERATIONAL}:
        return _run_sql_path(question, intent_payload, context_package, "text_to_sql", started_at)

    if intent.intent == INTENT_ANALYTICAL:
        if context_package.overall_confidence >= CONFIDENCE_THRESHOLD_CAPSULE_ANSWER and not _top_hit_is_schema_context(context_package):
            return QueryResponse(
                answer=answer_from_capsules(question, context_package),
                route_taken="vector_retrieval",
                intent=intent.intent,
                confidence=intent.confidence,
                capsules_used=_capsules_used(context_package),
                context_package=context_package.model_dump(),
                answer_ms=int((time.time() - started_at) * 1000),
                intent_payload=intent_payload,
            )
        return _run_sql_path(question, intent_payload, context_package, "vector_retrieval_schema_context_llm", started_at)

    if intent.intent == INTENT_HYBRID:
        capsule_answer = answer_from_capsules(question, context_package)
        sql_response = _run_sql_path(question, intent_payload, context_package, "hybrid_sql", started_at)
        sql_response.route_taken = "hybrid"
        sql_response.answer = f"Capsule view: {capsule_answer}\n\nSQL view: {sql_response.answer}"
        return sql_response

    return _run_sql_path(question, intent_payload, context_package, "text_to_sql", started_at)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    user_question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Which broker has the most escalations?"
    result = handle_query(user_question)
    print(json.dumps(result.model_dump(), indent=2, default=str))
