"""Groq-backed LLM service used by routing, SQL generation, summarization, and signal extraction."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import Groq, RateLimitError

from .config import get_settings

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "groq_analytical_model",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Call Groq with one retry on rate limit and log model usage."""
    settings = get_settings()
    model_name = getattr(settings, model_slot, settings.groq_analytical_model)
    client = _get_client()
    for attempt in range(2):
        try:
            logger.info("Groq call | slot=%s | model=%s", model_slot, model_name)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError as exc:
            error_str = str(exc)
            if "413" in error_str or "tokens" in error_str.lower():
                logger.warning("Groq request too large | model=%s | error=%s", model_name, exc)
                return "[LLM request too large: reduce input size]"
            logger.warning("Groq rate limit | model=%s | attempt=%s | error=%s", model_name, attempt + 1, exc)
            if attempt == 0:
                time.sleep(5)
                continue
            return "[LLM rate limit: please retry]"
        except Exception as exc:
            logger.error("Groq call failed | model=%s | error=%s", model_name, exc)
            return f"[LLM error: {exc}]"
    return "[LLM unavailable]"


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "groq_intent_model",
    max_tokens: int = 256,
) -> dict[str, Any] | None:
    """Call Groq and parse a JSON object."""
    raw = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_slot=model_slot,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.error("Failed to parse LLM JSON: %s | raw=%s", exc, raw[:400])
        return None
