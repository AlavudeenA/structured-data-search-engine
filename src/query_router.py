# File: src/query_router.py
"""Route user questions to either structured SQL flow or analytical context flow."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .app_constants import (
    ANALYTICAL_TERMS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_USER_AGENT,
    GROQ_RESPONSES_API_URL,
    HTTP_INTENT_TIMEOUT_SECONDS,
    INTENT_ANALYTICAL_QUERY,
    INTENT_CONFIDENCE_CLAMP_DEFAULT,
    INTENT_FALLBACK_COMPARATIVE_CONFIDENCE,
    INTENT_FALLBACK_DEFAULT_CONFIDENCE,
    INTENT_REASON_MAX_WORDS,
    INTENT_STRUCTURED_QUERY,
    STRUCTURED_TERMS,
    TEMPERATURE_ZERO,
)


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    confidence: float
    reason: str
    source: str


def route_query(user_query: str) -> IntentDecision:
    """Classify query intent as structured or analytical."""
    query = (user_query or "").strip()
    if not query:
        return IntentDecision(
            intent=INTENT_ANALYTICAL_QUERY,
            confidence=0.0,
            reason="Empty query. Defaulting to analytical path.",
            source="fallback_rules",
        )

    llm_decision = _classify_with_groq(query)
    if llm_decision:
        return llm_decision

    return _fallback_rule_classifier(query)


def _classify_with_groq(query: str) -> IntentDecision | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GROQ_INTENT_MODEL", DEFAULT_GROQ_MODEL)
    system_prompt = (
        "Classify query intent for data question routing.\n"
        "Return strict JSON with keys: intent, confidence, reason.\n"
        f"intent must be exactly one of: {INTENT_STRUCTURED_QUERY}, {INTENT_ANALYTICAL_QUERY}.\n"
        "structured_query means direct, concrete table lookup/aggregation expressible in SQL.\n"
        "analytical_query means exploratory, insight-oriented, trend/anomaly/comparative reasoning.\n"
        f"Keep reason under {INTENT_REASON_MAX_WORDS} words."
    )

    payload: dict[str, Any] = {
        "model": model,
        "temperature": TEMPERATURE_ZERO,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    }

    req = request.Request(
        url=GROQ_RESPONSES_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=HTTP_INTENT_TIMEOUT_SECONDS) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            message = _extract_response_text(raw)
            parsed = json.loads(message)
    except (KeyError, ValueError, error.URLError, TimeoutError, error.HTTPError):
        return None

    intent = parsed.get("intent")
    reason = str(parsed.get("reason", "LLM intent classification.")).strip()
    confidence = _clamp_confidence(parsed.get("confidence"))

    if intent not in {INTENT_STRUCTURED_QUERY, INTENT_ANALYTICAL_QUERY}:
        return None

    return IntentDecision(
        intent=intent,
        confidence=confidence,
        reason=reason or "LLM intent classification.",
        source=f"groq:{model}",
    )


def _fallback_rule_classifier(query: str) -> IntentDecision:
    text = re.sub(r"\s+", " ", query.lower())

    if any(k in text for k in [" vs ", " versus ", " compared to ", " compared with "]):
        return IntentDecision(
            intent=INTENT_ANALYTICAL_QUERY,
            confidence=INTENT_FALLBACK_COMPARATIVE_CONFIDENCE,
            reason="Comparative query detected.",
            source="fallback_rules",
        )

    analytical_hits = sum(1 for term in ANALYTICAL_TERMS if term in text)
    structured_hits = sum(1 for term in STRUCTURED_TERMS if term in text)

    if analytical_hits > structured_hits:
        return IntentDecision(
            intent=INTENT_ANALYTICAL_QUERY,
            confidence=INTENT_FALLBACK_DEFAULT_CONFIDENCE,
            reason="Analytical terms dominate query wording.",
            source="fallback_rules",
        )

    return IntentDecision(
        intent=INTENT_STRUCTURED_QUERY,
        confidence=INTENT_FALLBACK_DEFAULT_CONFIDENCE,
        reason="Structured retrieval/aggregation terms dominate query wording.",
        source="fallback_rules",
    )


def _clamp_confidence(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return INTENT_CONFIDENCE_CLAMP_DEFAULT
    return max(0.0, min(1.0, val))


def _extract_response_text(body: dict[str, Any]) -> str:
    """Extract assistant text from Groq/OpenAI response formats."""
    if "output" in body:
        for item in body["output"]:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text"):
                        return str(part.get("text", ""))

    if "output_text" in body:
        return str(body["output_text"])

    if "choices" in body:
        return str(body["choices"][0]["message"]["content"])

    raise ValueError(f"Unsupported response structure: {body}")
