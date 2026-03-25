# File: src/embedding.py
"""Embedding pipeline orchestrator for SQL result capsules."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .app_constants import (
    CAPSULE_TYPE_SCHEMA_CONTEXT,
    DEFAULT_MAX_GROUP_COLS_PER_TABLE,
    DEFAULT_MAX_RANDOM_PER_TABLE,
    DEFAULT_ROWS_PER_CAPSULE,
    GEN_ROWS_MAX_ALLOWED,
    GEN_ROWS_MAX_EXCLUSIVE,
    GEN_ROWS_MIN,
    INGESTION_MODE_APPEND_UNIQUE,
)
from .capsule_generator import generate_capsules_
from .embedding_service import embed_texts
from .schema_capsule_generator import generate_schema_context_capsules
from .vector_store import DEFAULT_COLLECTION, apply_retention_policies, upsert_capsules


def ingest_capsules(
    capsules: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
    ingestion_mode: str = INGESTION_MODE_APPEND_UNIQUE,
    ingest_run_id: str | None = None,
) -> dict[str, Any]:
    """Embed and index already-built capsules (supports V1/ capsule schemas)."""
    if not capsules:
        return {
            "collection": collection_name,
            "capsules_indexed": 0,
            "row_count": 0,
            "message": "No capsules to index.",
        }

    _validate_capsule_row_limits(capsules, max_rows=GEN_ROWS_MAX_EXCLUSIVE)

    texts = [_capsule_to_embedding_text(c) for c in capsules]
    vectors = embed_texts(texts)

    # Lineage only: if multiple capsules carry different SQLs, store per-capsule payload SQL.
    # Use empty source_sql at batch level to keep ingestion generic.
    upsert_result = upsert_capsules(
        capsules=capsules,
        vectors=vectors,
        source_sql="",
        source_query=None,
        collection_name=collection_name,
        ingestion_mode=ingestion_mode,
        ingest_run_id=ingest_run_id,
    )
    row_count = 0
    for c in capsules:
        try:
            row_count += int(c.get("row_count", 0))
        except (TypeError, ValueError):
            continue

    return {
        "collection": collection_name,
        "capsules_indexed": upsert_result["indexed_count"],
        "capsules_deleted": upsert_result["deleted_count"],
        "source_sql_hash": upsert_result["source_sql_hash"],
        "ingest_run_id": str(upsert_result.get("ingest_run_id", ingest_run_id or "")),
        "ingestion_mode": ingestion_mode,
        "row_count": row_count,
        "vector_dim": len(vectors[0]) if vectors else 0,
    }


def generate_and_ingest_capsules(
    collection_name: str = DEFAULT_COLLECTION,
    target_capsules: int | None = None,
    target_rows: int = DEFAULT_ROWS_PER_CAPSULE,
    max_rows_per_capsule: int = GEN_ROWS_MAX_EXCLUSIVE,
    include_temporal_aggregations: bool = True,
    max_group_cols_per_table: int = DEFAULT_MAX_GROUP_COLS_PER_TABLE,
    use_llm_summaries: bool = False,
    ingestion_mode: str = INGESTION_MODE_APPEND_UNIQUE,
    max_random_per_table: int = DEFAULT_MAX_RANDOM_PER_TABLE,
    replace_similar_capsules: bool = True,
) -> dict[str, Any]:
    """Generate schema-agnostic Capsule  set and index into vector store."""
    _validate_generation_inputs(
        target_capsules=target_capsules,
        target_rows=target_rows,
        max_rows_per_capsule=max_rows_per_capsule,
        max_group_cols_per_table=max_group_cols_per_table,
    )
    with _generation_lock():
        ingest_run_id = str(uuid.uuid4())
        analytical_capsules = generate_capsules_(
            target_capsules=target_capsules,
            target_rows=target_rows,
            max_rows_per_capsule=max_rows_per_capsule,
            include_temporal_aggregations=include_temporal_aggregations,
            max_group_cols_per_table=max_group_cols_per_table,
            use_llm_summaries=use_llm_summaries,
        )
        schema_capsules = generate_schema_context_capsules()
        capsules = [*analytical_capsules, *schema_capsules]
        result = ingest_capsules(
            capsules=capsules,
            collection_name=collection_name,
            ingestion_mode=ingestion_mode,
            ingest_run_id=ingest_run_id,
        )
    result["generated_capsules"] = len(capsules)
    result["generated_analytical_capsules"] = len(analytical_capsules)
    result["generated_schema_context_capsules"] = len(schema_capsules)
    result["capsule_schema"] = ""
    result["ingest_run_id"] = ingest_run_id
    retention = apply_retention_policies(
        collection_name=collection_name,
        max_random_per_table=max_random_per_table,
        replace_similar_capsules=replace_similar_capsules,
    )
    result["retention"] = retention
    return result


def _validate_generation_inputs(
    target_capsules: int | None,
    target_rows: int,
    max_rows_per_capsule: int,
    max_group_cols_per_table: int,
) -> None:
    if target_capsules is not None and int(target_capsules) < 1:
        raise ValueError("Target capsules must be >= 1 when provided.")
    if int(max_rows_per_capsule) < GEN_ROWS_MIN or int(max_rows_per_capsule) >= GEN_ROWS_MAX_EXCLUSIVE:
        raise ValueError(
            f"Rows per capsule must be between {GEN_ROWS_MIN} and {GEN_ROWS_MAX_ALLOWED}."
        )
    if int(target_rows) < GEN_ROWS_MIN:
        raise ValueError("Rows per capsule must be >= 1.")
    if int(target_rows) > int(max_rows_per_capsule):
        raise ValueError("Target rows cannot exceed the strict capsule row limit.")
    if int(max_group_cols_per_table) < 1:
        raise ValueError("Max group columns per table must be >= 1.")


def _capsule_to_embedding_text(capsule: dict[str, Any]) -> str:
    """
    Build embedding text from high-signal capsule fields.

    Preferred format:
    - summary_text + key_columns + tags + tables_used
    """
    summary = str(capsule.get("summary_text", "")).strip()

    key_columns = capsule.get("key_columns", [])
    if not isinstance(key_columns, list):
        key_columns = []
    key_columns_text = ", ".join(str(x) for x in key_columns if str(x).strip())

    tags = capsule.get("tags")
    if tags is None:
        tags = capsule.get("metric_tags", [])
    if not isinstance(tags, list):
        tags = []
    tags_text = ", ".join(str(x) for x in tags if str(x).strip())

    tables_used = capsule.get("tables_used", [])
    if not isinstance(tables_used, list):
        tables_used = []
    tables_text = ", ".join(str(x) for x in tables_used if str(x).strip())

    example_questions = capsule.get("example_questions", [])
    if not isinstance(example_questions, list):
        example_questions = []
    example_questions_text = " | ".join(
        str(x).strip() for x in example_questions[:4] if str(x).strip()
    )

    recommended_joins = capsule.get("recommended_joins", [])
    if not isinstance(recommended_joins, list):
        recommended_joins = []
    joins_text = " | ".join(str(x).strip() for x in recommended_joins[:4] if str(x).strip())

    relevant_columns = capsule.get("relevant_columns", [])
    if not isinstance(relevant_columns, list):
        relevant_columns = []
    relevant_columns_text = ", ".join(str(x).strip() for x in relevant_columns[:8] if str(x).strip())

    sql_template = str(capsule.get("sql_template", "")).strip()

    capsule_type = str(capsule.get("capsule_type", "")).strip()

    parts = [summary]
    if capsule_type:
        parts.append(f"Capsule type: {capsule_type}.")
    if tables_text:
        parts.append(f"Tables used: {tables_text}.")
    if key_columns_text:
        parts.append(f"Key columns: {key_columns_text}.")
    if tags_text:
        parts.append(f"Tags: {tags_text}.")
    if joins_text:
        parts.append(f"Recommended joins: {joins_text}.")
    if relevant_columns_text:
        parts.append(f"Relevant columns: {relevant_columns_text}.")
    if example_questions_text:
        parts.append(f"Example questions: {example_questions_text}.")
    if capsule_type == CAPSULE_TYPE_SCHEMA_CONTEXT and sql_template:
        parts.append(f"SQL template: {sql_template}.")
    return _normalize_text(" ".join(p for p in parts if p))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _validate_capsule_row_limits(capsules: list[dict[str, Any]], max_rows: int) -> None:
    for idx, capsule in enumerate(capsules, start=1):
        row_count_raw = capsule.get("row_count", 0)
        try:
            row_count = int(row_count_raw)
        except (TypeError, ValueError):
            row_count = 0
        if row_count > max_rows:
            raise ValueError(
                f"Capsule at index {idx} has {row_count} rows, which exceeds max {max_rows}. "
                "Capsules must be <= 100 rows."
            )


@contextmanager
def _generation_lock(stale_after_seconds: int = 3600) -> Any:
    """
    Cross-process lock to prevent concurrent capsule generation runs.
    """
    lock_path = Path(__file__).resolve().parent.parent / ".capsule_generation.lock"
    now = time.time()

    if lock_path.exists():
        try:
            age = now - lock_path.stat().st_mtime
        except OSError:
            age = 0
        if age > stale_after_seconds:
            try:
                lock_path.unlink()
            except OSError:
                pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} time={int(now)}".encode("utf-8"))
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(
            "Capsule generation is already running in another session. "
            "Wait for it to finish and try again."
        ) from exc

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "py -3 -m src.embedding --generate-\n"
            "py -3 -m src.embedding --generate- --limit 1000"
        )

    mode = sys.argv[1].strip().lower()
    if mode == "--generate-":
        limit: int | None = None
        if len(sys.argv) >= 4 and sys.argv[2].strip().lower() == "--limit":
            try:
                limit = int(sys.argv[3].strip())
            except ValueError as exc:
                raise SystemExit("--limit must be an integer.") from exc
        result = generate_and_ingest_capsules(target_capsules=limit)
    else:
        raise SystemExit("First argument must be --generate-")

    print(json.dumps(result, indent=2, default=str))
