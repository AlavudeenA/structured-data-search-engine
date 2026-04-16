"""Groq-based intent router with deterministic fallback heuristics."""

from __future__ import annotations

import logging

from ..llm_service import call_llm_json
from ..models import IntentResult
from ..llm_instructions import INTENT_DETECTION_SYSTEM, INTENT_DETECTION_USER

logger = logging.getLogger(__name__)

STRUCTURED_HINTS = ["which", "show", "list", "how many", "count", "registered", "top", "most"]
ANALYTICAL_HINTS = ["trend", "increasing", "decreasing", "pattern", "anomaly", "over time", "unusual"]
OPERATIONAL_HINTS = ["pending", "active", "open", "unresolved", "currently", "right now"]


def _normalize_parts(values: object) -> list[str]:
    """Normalize LLM part lists into plain strings."""
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, dict):
            text = " ".join(str(part).strip() for part in value.values() if str(part).strip())
        else:
            text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _fallback(question: str) -> IntentResult:
    text = question.lower()
    structured = sum(1 for hint in STRUCTURED_HINTS if hint in text)
    analytical = sum(1 for hint in ANALYTICAL_HINTS if hint in text)
    operational = sum(1 for hint in OPERATIONAL_HINTS if hint in text)

    if operational > 0 and analytical == 0:
        return IntentResult(intent="operational", confidence=0.68, reasoning="Operational urgency keywords detected")
    if analytical > 0 and structured > 0:
        return IntentResult(intent="hybrid", confidence=0.66, reasoning="Question contains both retrieval and analysis cues")
    if analytical > 0:
        return IntentResult(intent="analytical", confidence=0.68, reasoning="Analytical trend or anomaly language detected")
    return IntentResult(intent="structured", confidence=0.64, reasoning="Defaulted to structured because the question asks for direct retrieval")


def detect_intent(question: str) -> IntentResult:
    """Detect question intent using Groq, with rule-based fallback."""
    payload = call_llm_json(
        INTENT_DETECTION_SYSTEM,
        INTENT_DETECTION_USER.format(question=question),
        model_slot="groq_intent_model",
        max_tokens=256,
    )
    if not payload:
        return _fallback(question)

    intent = str(payload.get("intent", "structured")).strip().lower()
    if intent not in {"structured", "analytical", "hybrid", "operational"}:
        return _fallback(question)

    return IntentResult(
        intent=intent,
        confidence=float(payload.get("confidence", 0.7)),
        structured_parts=_normalize_parts(payload.get("structured_parts", [])),
        analytical_parts=_normalize_parts(payload.get("analytical_parts", [])),
        reasoning=str(payload.get("reasoning", "")),
    )
