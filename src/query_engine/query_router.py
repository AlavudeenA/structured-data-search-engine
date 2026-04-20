"""Groq-based intent router with deterministic fallback heuristics."""

from __future__ import annotations

import logging

from ..llm_service import call_llm_json
from ..models import IntentResult
from ..llm_instructions import INTENT_DETECTION_SYSTEM, INTENT_DETECTION_USER

logger = logging.getLogger(__name__)

# Keywords that strongly signal analytical intent — trend/pattern/anomaly reasoning.
# Fallback assumes text_to_sql by default; only routes to analytical on a clear signal.
_ANALYTICAL_HINTS = frozenset([
    "trend", "increasing", "decreasing", "pattern", "anomaly",
    "over time", "unusual", "compare", "comparison", "why is",
    "highest risk", "escalating", "spike", "insight",
])


def _fallback(question: str) -> IntentResult:
    """Default to text_to_sql; only choose analytical on an explicit signal."""
    text = question.lower()
    if any(hint in text for hint in _ANALYTICAL_HINTS):
        return IntentResult(intent="analytical", confidence=0.65, reasoning="Analytical language detected in fallback")
    return IntentResult(intent="text_to_sql", confidence=0.65, reasoning="Defaulted to text_to_sql — no analytical signal detected")


def detect_intent(question: str) -> IntentResult:
    """Detect question intent using LLM, with rule-based fallback."""
    payload = call_llm_json(
        INTENT_DETECTION_SYSTEM,
        INTENT_DETECTION_USER.format(question=question),
        model_slot="groq_intent_model",
        max_tokens=150,
    )
    if not payload:
        return _fallback(question)

    intent = str(payload.get("intent", "text_to_sql")).strip().lower()
    if intent not in {"text_to_sql", "analytical"}:
        logger.warning("LLM returned unknown intent '%s' — falling back", intent)
        return _fallback(question)

    return IntentResult(
        intent=intent,
        confidence=float(payload.get("confidence", 0.7)),
        reasoning=str(payload.get("reasoning", "")),
    )
