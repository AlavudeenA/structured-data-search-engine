"""Analytical answer synthesis from precomputed capsule context."""

from __future__ import annotations

from ..llm_service import call_llm
from ..models import ContextPackage
from ..prompts import ANALYTICAL_ANSWER_SYSTEM, ANALYTICAL_ANSWER_USER


def answer_from_capsules(question: str, context_package: ContextPackage) -> str:
    """Answer using capsule-only context when confidence is high enough."""
    if not context_package.combined_context.strip():
        return "No analytical capsule context was strong enough to answer this question directly."
    answer = call_llm(
        ANALYTICAL_ANSWER_SYSTEM,
        ANALYTICAL_ANSWER_USER.format(question=question, combined_context=context_package.combined_context[:4000]),
        model_slot="groq_analytical_model",
        temperature=0.1,
        max_tokens=400,
    )
    if not answer or answer.startswith("[LLM"):
        return context_package.combined_context[:800]
    return answer
