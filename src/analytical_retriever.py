# File: src/analytical_retriever.py
"""Retrieve analytical context from vector store capsules."""

from __future__ import annotations

import json
from typing import Any

from .app_constants import (
    ANALYTICAL_FALLBACK_OTHERS_LIMIT,
    ANALYTICAL_FALLBACK_PREVIEW_FIELDS,
    ANALYTICAL_ROWS_SAMPLE_SIZE,
    CAPSULE_TYPE_SUMMARY,
    DEFAULT_ANALYTICAL_TOP_K,
    DEFAULT_ANALYTICAL_TEMPERATURE,
    DEFAULT_ANALYTICAL_MAX_OUTPUT_TOKENS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MIN_SCORE,
    PRIORITY_BOOST_HIGH,
    PRIORITY_BOOST_LOW,
    PRIORITY_BOOST_MEDIUM,
)
from .embedding_service import embed_texts
from .llm_service import call_llm
from .vector_store import DEFAULT_COLLECTION, scroll_capsules_by_sql_hash, search_capsules


def retrieve_analytical_context(
    user_query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = DEFAULT_ANALYTICAL_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    top_k_per_type: int | None = None,
    capsule_type: str | None = None,
    entity: str | None = None,
    capsule_topic: str | None = None,
) -> dict[str, Any]:
    """
    Embed the user query and retrieve the most relevant capsules.
    Retrieval is primarily semantic (vector search) with optional filters.
    """

    vector = embed_texts([user_query])[0]

    raw_hits = search_capsules(
        query_vector=vector,
        limit=max(top_k * 3, top_k),
        collection_name=collection_name,
        capsule_type=capsule_type,
        entity=entity,
        capsule_topic=capsule_topic,
    )

    deduped = _dedupe_hits_by_content_hash(raw_hits)

    filtered = [
        h for h in deduped
        if float(h.get("score", 0)) >= min_score
    ]

    reranked = _rerank_by_priority(filtered)

    hits = _apply_type_limits(
        reranked,
        top_k=top_k,
        top_k_per_type=top_k_per_type
    )

    answer, supporting = _build_analytical_answer(
        user_query,
        hits,
        collection_name=collection_name,
    )

    return {
        "route": "vector_retrieval",
        "hits": hits,
        "answer": answer,
        "supporting": supporting,
    }


def _build_analytical_answer(
    user_query: str,
    hits: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
) -> tuple[str, dict[str, Any]]:
    """
    Build a simple human-readable explanation from the top capsule.
    """

    if not hits:
        return (
            "No analytical context capsules found in the vector store.",
            {"mode": "none", "capsule_names": [], "capsule_count": 0, "source_sql_hash": ""},
        )

    top_hit = hits[0]
    payload = top_hit.get("payload", {})
    score = float(top_hit.get("score", 0.0))
    source_sql_hash = str(payload.get("source_sql_hash", "")).strip()

    if not source_sql_hash:
        #  path: source_sql_hash may be empty by design, so use all retrieved hits.
        rows = _collect_rows_from_hits(hits)
        summary = _summarize_rows(rows, user_query)
        names = _unique_capsule_names(hits)
        payload_name = str(payload.get("capsule_name", "")).strip()
        if not names and payload_name:
            names = [payload_name]
        supporting = {
            "mode": "multi_hit",
            "capsule_names": names,
            "capsule_count": len(hits),
            "source_sql_hash": "",
        }
        return summary or "Capsule retrieved but could not interpret rows.", supporting

    related_capsules = scroll_capsules_by_sql_hash(
        sql_hash=source_sql_hash,
        collection_name=collection_name,
    )
    capsule_names = _unique_capsule_names(related_capsules)
    supporting = {
        "mode": "grouped_by_source_hash",
        "capsule_names": capsule_names,
        "capsule_count": len(related_capsules),
        "source_sql_hash": source_sql_hash,
    }
    rows = _collect_rows_from_hits(related_capsules)

    if not rows:
        return "Capsules retrieved but no usable rows found.", supporting

    summary = _summarize_rows(rows, user_query)
    if summary:
        return (
            f"{summary} (dataset capsules={len(related_capsules)}, score={score:.3f})",
            supporting,
        )

    return "Analytical capsules retrieved but could not interpret rows.", supporting


def _collect_rows_from_hits(hits: list[dict[str, Any]]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for hit in hits:
        payload = hit.get("payload", {})
        rows_json = payload.get("rows_json", "[]")
        try:
            rows = json.loads(rows_json)
        except Exception:
            rows = []
        if not isinstance(rows, list):
            continue
        for row in rows:
            key = json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def _unique_capsule_names(hits: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        payload = hit.get("payload", {})
        name = str(payload.get("capsule_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _summarize_rows(rows: list[Any], question: str) -> str | None:
    """LLM-first summarization with deterministic fallback."""
    llm_summary = _summarize_rows_with_llm(rows, question)
    if llm_summary:
        return llm_summary
    return _summarize_rows_fallback(rows)


def _summarize_rows_with_llm(rows: list[Any], question: str) -> str | None:
    """Use LLM to infer entities/metrics from row sample and question."""
    sample_rows = rows[:ANALYTICAL_ROWS_SAMPLE_SIZE]
    prompt = (
        "User question:\n"
        f"{question}\n\n"
        "Dataset rows (JSON sample):\n"
        f"{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        "Instructions:\n"
        "1. Understand the question.\n"
        "2. Identify relevant columns.\n"
        "3. Determine metric/aggregation.\n"
        "4. Provide a concise analytical answer.\n\n"
        "Rules:\n"
        "- Do not invent data.\n"
        "- If no clear signal exists, say so.\n"
        "- Mention relevant attributes (such as date) when applicable.\n"
        "- Keep response to max 3 sentences.\n"
        "- Output format must be: 'Answer: ... Reason: ...'\n"
        "- Always place the answer first and reason second.\n"
    )
    return call_llm(
        prompt=prompt,
        system_prompt=(
            "You are an analytical assistant for structured data. "
            "Return clear, factual explanations from provided rows."
        ),
        model_env_var="GROQ_ANALYTICAL_MODEL",
        default_model=DEFAULT_GROQ_MODEL,
        temperature=DEFAULT_ANALYTICAL_TEMPERATURE,
        max_output_tokens=DEFAULT_ANALYTICAL_MAX_OUTPUT_TOKENS,
    )


def _summarize_rows_fallback(rows: list[Any]) -> str | None:
    """
    Generic fallback summarization of row data without domain assumptions.
    """

    dict_rows = [r for r in rows if isinstance(r, dict)]

    if not dict_rows:
        return None

    first_row = dict_rows[0]

    # detect numeric columns
    numeric_cols = [
        k for k, v in first_row.items()
        if _is_number(v)
    ]

    if not numeric_cols:
        preview = ", ".join(
            f"{k}={v}" for k, v in list(first_row.items())[:ANALYTICAL_FALLBACK_PREVIEW_FIELDS]
        )
        return f"Top record suggests {preview}"

    metric_col = numeric_cols[0]

    entity_col = next(
        (k for k in first_row.keys() if k != metric_col),
        None
    )

    if not entity_col:
        return None

    normalized = []

    for row in dict_rows:
        if entity_col in row and metric_col in row:
            if _is_number(row[metric_col]):
                normalized.append(
                    (str(row[entity_col]), float(row[metric_col]))
                )

    if not normalized:
        return None

    normalized.sort(key=lambda x: x[1], reverse=True)

    top_entity, top_value = normalized[0]

    others = normalized[1:ANALYTICAL_FALLBACK_OTHERS_LIMIT]

    if others:
        other_text = ", ".join(
            f"{e} ({_fmt(v)})" for e, v in others
        )
        return f"{top_entity} leads with {_fmt(top_value)}. Others include {other_text}."

    return f"{top_entity} leads with {_fmt(top_value)}."


def _dedupe_hits_by_content_hash(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the best scoring capsule per content hash.
    """

    best: dict[str, dict[str, Any]] = {}

    for hit in hits:
        payload = hit.get("payload", {})
        content_hash = payload.get("content_hash")

        key = str(content_hash) if content_hash else str(hit.get("id"))

        prev = best.get(key)

        if prev is None or float(hit.get("score", 0)) > float(prev.get("score", 0)):
            best[key] = hit

    return sorted(
        best.values(),
        key=lambda h: float(h.get("score", 0)),
        reverse=True,
    )


def _apply_type_limits(
    hits: list[dict[str, Any]],
    top_k: int,
    top_k_per_type: int | None,
) -> list[dict[str, Any]]:

    if top_k_per_type is None:
        return hits[:top_k]

    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    for hit in hits:

        ctype = str(
            hit.get("payload", {}).get("capsule_type", CAPSULE_TYPE_SUMMARY)
        )

        used = counts.get(ctype, 0)

        if used >= top_k_per_type:
            continue

        selected.append(hit)
        counts[ctype] = used + 1

        if len(selected) >= top_k:
            break

    return selected


def _rerank_by_priority(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Slightly boost capsules marked as high priority.
    """

    boost = {
        "high": PRIORITY_BOOST_HIGH,
        "medium": PRIORITY_BOOST_MEDIUM,
        "low": PRIORITY_BOOST_LOW,
    }

    def score(hit: dict[str, Any]) -> float:

        base = float(hit.get("score", 0))

        priority = str(
            hit.get("payload", {}).get("capsule_priority", "low")
        ).lower()

        return base + boost.get(priority, 0.0)

    return sorted(hits, key=score, reverse=True)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"
