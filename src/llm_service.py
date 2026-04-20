"""LLM service — routes calls to Groq or the VS Code LM API depending on USE_GROQ.

USE_GROQ = True  → Groq API (set GROQ_API_KEY in .env)
USE_GROQ = False → VS Code Language Model API via local extension host on port VSCODE_LM_PORT
                   (open the vscode-lm-extension project in VS Code and press F5 before use)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .app_constants import USE_GROQ
from .config import get_settings

logger = logging.getLogger(__name__)


# ── Groq backend ───────────────────────────────────────────────────────────────

_groq_client = None  # module-level singleton; created once on first call


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=get_settings().groq_api_key)
    return _groq_client


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model_slot: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from groq import RateLimitError  # lazy import — not needed when USE_GROQ=False

    settings = get_settings()
    model_name = getattr(settings, model_slot, settings.groq_analytical_model)
    client = _get_groq_client()

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


# ── VS Code LM backend ─────────────────────────────────────────────────────────

_NOT_RUNNING_MSG = (
    "[LLM unavailable: VS Code extension host is not running. "
    "Open the vscode-lm-extension folder in VS Code and press F5.]"
)


def _call_vscode_lm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import requests  # lazy import

    settings = get_settings()
    url = f"http://127.0.0.1:{settings.vscode_lm_port}/prompt"
    payload = {
        "system": system_prompt,
        "prompt": user_prompt,
        "secret": settings.vscode_lm_secret,
    }
    for attempt in range(2):
        try:
            logger.info("VS Code LM call | attempt=%s", attempt + 1)
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("text", "").strip()
        except requests.exceptions.ConnectionError:
            logger.warning("VS Code LM server not reachable on port %s", settings.vscode_lm_port)
            if attempt == 0:
                time.sleep(2)
                continue
            return _NOT_RUNNING_MSG
        except requests.exceptions.HTTPError as exc:
            logger.error("VS Code LM HTTP error: %s", exc)
            return f"[LLM error: {exc}]"
        except Exception as exc:
            logger.error("VS Code LM call failed: %s", exc)
            return f"[LLM error: {exc}]"
    return _NOT_RUNNING_MSG


# ── Public API (call sites do not need to know which backend is active) ────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "groq_analytical_model",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    if USE_GROQ:
        return _call_groq(system_prompt, user_prompt, model_slot, temperature, max_tokens)
    return _call_vscode_lm(system_prompt, user_prompt, temperature, max_tokens)


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "groq_intent_model",
    max_tokens: int = 256,
) -> dict[str, Any] | None:
    raw = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_slot=model_slot,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if not raw or raw.startswith("[LLM"):
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.error("Failed to parse LLM JSON: %s | raw=%s", exc, raw[:400])
        return None
