# File: src/embedding.py
"""Embedding pipeline orchestrator for SQL result capsules."""

from __future__ import annotations

import json
import sys
import hashlib
from typing import Any

from .capsule_builder import build_capsules
from .embedding_service import embed_texts
from .sql_executor import execute_sql
from .text_to_sql import generate_sql_from_question
from .vector_store import DEFAULT_COLLECTION, upsert_capsules


def ingest_from_user_query(
    user_query: str,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 25,
    ingestion_mode: str = "append_unique",
    capsule_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate SQL from a user question, execute it, and index capsules."""
    sql_data = generate_sql_from_question(user_query)
    execution = execute_sql(sql_data["sql"])
    return ingest_from_execution_result(
        execution_result=execution,
        source_sql=sql_data["sql"],
        source_query=user_query,
        collection_name=collection_name,
        batch_size=batch_size,
        ingestion_mode=ingestion_mode,
        capsule_metadata=capsule_metadata,
    )


def ingest_from_sql(
    sql: str,
    source_query: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 25,
    ingestion_mode: str = "append_unique",
    capsule_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute caller-provided SQL, then index capsules."""
    execution = execute_sql(sql)
    return ingest_from_execution_result(
        execution_result=execution,
        source_sql=sql,
        source_query=source_query,
        collection_name=collection_name,
        batch_size=batch_size,
        ingestion_mode=ingestion_mode,
        capsule_metadata=capsule_metadata,
    )


def ingest_from_execution_result(
    execution_result: dict[str, Any],
    source_sql: str,
    source_query: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 25,
    ingestion_mode: str = "append_unique",
    capsule_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store already-fetched SQL result as vectorized context capsules."""
    rows = execution_result.get("rows", [])
    columns = execution_result.get("columns", [])
    if not rows:
        return {
            "collection": collection_name,
            "capsules_indexed": 0,
            "row_count": 0,
            "message": "No rows to index.",
        }

    metadata = dict(capsule_metadata or {})
    if source_query:
        metadata.setdefault("source_query_hash", _sha256_hex(_normalize_text(source_query)))
    capsules = _build_capsules(
        columns=columns, rows=rows, batch_size=batch_size, capsule_metadata=metadata
    )
    texts = [capsule["summary_text"] for capsule in capsules]
    vectors = embed_texts(texts)
    upsert_result = upsert_capsules(
        capsules=capsules,
        vectors=vectors,
        source_sql=source_sql,
        source_query=source_query,
        collection_name=collection_name,
        ingestion_mode=ingestion_mode,
    )

    return {
        "collection": collection_name,
        "capsules_indexed": upsert_result["indexed_count"],
        "capsules_deleted": upsert_result["deleted_count"],
        "source_sql_hash": upsert_result["source_sql_hash"],
        "ingestion_mode": ingestion_mode,
        "row_count": len(rows),
        "vector_dim": len(vectors[0]),
    }


def _build_capsules(
    columns: list[str],
    rows: list[dict[str, Any]],
    batch_size: int,
    capsule_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = capsule_metadata or {}
    return build_capsules(
        columns=columns,
        rows=rows,
        batch_size=batch_size,
        capsule_type=str(metadata.get("capsule_type", "aggregation")),
        capsule_name=metadata.get("capsule_name"),
        entity=str(metadata.get("entity", "")),
        capsule_topic=str(metadata.get("capsule_topic", "")),
        metric_tags=[str(x) for x in metadata.get("metric_tags", [])],
        capsule_version=str(metadata.get("capsule_version", "v1")),
        schema_version=str(metadata.get("schema_version", "v1")),
        source_query_hash=str(metadata.get("source_query_hash", "")),
        capsule_priority=str(metadata.get("capsule_priority", "")),
    )


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            "py -3 -m src.embedding --query \"your question\"\n"
            "py -3 -m src.embedding --sql \"SELECT TOP 100 ...\""
        )

    mode = sys.argv[1].strip().lower()
    value = " ".join(sys.argv[2:]).strip()
    if mode == "--query":
        result = ingest_from_user_query(value)
    elif mode == "--sql":
        result = ingest_from_sql(value)
    else:
        raise SystemExit("First argument must be --query or --sql")

    print(json.dumps(result, indent=2, default=str))
