# File: src/result_summarizer.py
"""Summarize SQL results in plain English using Groq, with local fallback."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def summarize_result(user_query: str, execution_result: dict[str, Any]) -> str:
    summary = _summarize_with_groq(user_query, execution_result)
    if summary:
        return summary
    return _summarize_fallback(execution_result)


def _summarize_with_groq(user_query: str, execution_result: dict[str, Any]) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
    compact_rows = execution_result.get("rows", [])[:25]
    prompt_payload = {
        "user_query": user_query,
        "sql": execution_result.get("sql"),
        "columns": execution_result.get("columns", []),
        "row_count": execution_result.get("row_count", 0),
        "truncated": execution_result.get("truncated", False),
        "rows_sample": compact_rows,
    }
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "input": [
            {
                "role": "system",
                "content": (
                    "You summarize SQL query results for business users.\n"
                    "Rules:\n"
                    "- Be concise (max 3 sentences)\n"
                    "- Highlight top entities if ranking exists\n"
                    "- Mention if results were truncated\n"
                    "- Do not invent data\n"
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, default=str)},
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
        with request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = _extract_response_text(body).strip()
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, ValueError):
        return None

    return text or None


def _summarize_fallback(execution_result: dict[str, Any]) -> str:
    row_count = execution_result.get("row_count", 0)
    columns = execution_result.get("columns", [])
    rows = execution_result.get("rows", [])
    truncated = execution_result.get("truncated", False)

    if row_count == 0:
        return "No matching records were found for the query."

    head = rows[0] if rows else {}
    preview_bits = ", ".join(f"{k}={v}" for k, v in list(head.items())[:4])
    trunc_note = " Results were truncated to the configured row limit." if truncated else ""
    return (
        f"The query returned {row_count} rows with columns: {', '.join(columns)}. "
        f"First row sample: {preview_bits}.{trunc_note}"
    )


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
