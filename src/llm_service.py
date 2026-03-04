# File: src/llm_service.py
"""Shared Groq LLM call helper."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def call_llm(
    prompt: str,
    system_prompt: str = "You are a precise analytical assistant.",
    model_env_var: str = "GROQ_ANALYTICAL_MODEL",
    default_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_output_tokens: int = 300,
) -> str | None:
    """Call Groq Responses API and return assistant text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.getenv(model_env_var, default_model)
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    req = request.Request(
        url="https://api.groq.com/openai/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Compliance-Data-Assistant/1.0",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = _extract_response_text(body).strip()
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, ValueError):
        return None

    return text or None


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
