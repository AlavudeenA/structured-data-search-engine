# File: src/intent_router_service.py
"""Service wrapper for intent routing."""

from __future__ import annotations

from typing import Any

from .query_router import route_query


def detect_intent(user_query: str) -> dict[str, Any]:
    """Return normalized routing response for orchestration."""
    decision = route_query(user_query)
    return {
        "intent": decision.intent,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": decision.source,
    }
