"""LLM service backed by the VS Code Language Model API extension host.

The extension (C:/Users/alavu/Projects/vscode-lm-api) must be running in an
Extension Development Host VS Code window before any call is made.
Start it: open that project in VS Code and press F5, or use the
'Start VS Code LM API' launch config in this project (F5 here opens it too).

POST http://127.0.0.1:50234/prompt
  { "system": "...", "prompt": "...", "secret": "abc123" }
  -> { "text": "..." }
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .config import get_settings

logger = logging.getLogger(__name__)

_NOT_RUNNING_MSG = (
    "[LLM unavailable: VS Code extension host is not running. "
    "Open C:/Users/alavu/Projects/vscode-lm-api in VS Code and press F5, "
    "or press F5 in this project using the 'Start VS Code LM API' launch config.]"
)


def _url() -> str:
    return f"http://127.0.0.1:{get_settings().vscode_lm_port}/prompt"


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "",   # kept for call-site compatibility, not used
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Send system + user prompt to the VS Code LM extension and return the text response."""
    settings = get_settings()
    payload = {
        "system": system_prompt,
        "prompt": user_prompt,
        "secret": settings.vscode_lm_secret,
    }
    for attempt in range(2):
        try:
            logger.info("VS Code LM call | attempt=%s", attempt + 1)
            response = requests.post(_url(), json=payload, timeout=120)
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


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model_slot: str = "",   # kept for call-site compatibility, not used
    max_tokens: int = 256,
) -> dict[str, Any] | None:
    """Call the LLM and parse a JSON object from the response."""
    raw = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if not raw or raw.startswith("[LLM"):
        return None
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.error("Failed to parse LLM JSON: %s | raw=%s", exc, raw[:400])
        return None
