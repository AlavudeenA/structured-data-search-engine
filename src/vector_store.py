# File: src/vector_store.py
"""Qdrant vector store operations for context capsules."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .app_constants import (
    CAPSULE_TYPE_AGGREGATION,
    CAPSULE_TYPE_DISTRIBUTION,
    CAPSULE_TYPE_RANDOM_SAMPLE,
    CAPSULE_TYPE_SUMMARY,
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_PATH,
    DEFAULT_MAX_RANDOM_PER_TABLE,
    INGESTION_MODE_APPEND_UNIQUE,
    VECTOR_DELETE_SCROLL_LIMIT,
    VECTOR_LIST_LIMIT_DEFAULT,
    VECTOR_SCORE_DEFAULT,
    VECTOR_SCROLL_ALL_POINTS_LIMIT,
    VECTOR_SCROLL_BATCH_LARGE,
    VECTOR_SCROLL_BATCH_SMALL,
    VECTOR_SQL_SCROLL_LIMIT_DEFAULT,
    VECTOR_UPSERT_LOCK_STALE_SECONDS,
)

def upsert_capsules(
    capsules: list[dict[str, Any]],
    vectors: list[list[float]],
    source_sql: str = "",
    source_query: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    ingestion_mode: str = INGESTION_MODE_APPEND_UNIQUE,
    ingest_run_id: str | None = None,
) -> dict[str, Any]:
    """Create collection if needed, then upsert capsule vectors."""
    if not vectors:
        raise ValueError("No vectors provided for upsert.")

    normalized_sql = _normalize_sql(source_sql)
    source_sql_hash = _sha256_hex(normalized_sql) if normalized_sql else ""
    run_id = str(ingest_run_id or "").strip() or str(uuid.uuid4())
    qdrant_path = _get_qdrant_path()
    with _upsert_lock():
        client = QdrantClient(path=qdrant_path)
        try:
            #  behavior: do not rely on source_sql_hash for ingestion logic.
            # Keep hash only as optional lineage/debug payload.
            deleted_count = 0
            _ensure_collection(client, collection_name, len(vectors[0]))
            points = _capsules_to_points(
                capsules=capsules,
                vectors=vectors,
                source_sql=source_sql,
                source_query=source_query,
                source_sql_hash=source_sql_hash,
                ingest_run_id=run_id,
            )
            client.upsert(collection_name=collection_name, points=points)
            return {
                "indexed_count": len(points),
                "deleted_count": deleted_count,
                "source_sql_hash": source_sql_hash,
                "qdrant_path": qdrant_path,
                "ingest_run_id": run_id,
            }
        finally:
            client.close()


def delete_capsules_by_source_sql_hash(
    source_sql_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Delete all points for a source SQL hash."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        return _delete_capsules_by_source_sql_hash(
            client=client,
            source_sql_hash=source_sql_hash,
            collection_name=collection_name,
        )
    finally:
        client.close()


def search_capsules(
    query_vector: list[float],
    limit: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    capsule_type: str | None = None,
    entity: str | None = None,
    capsule_topic: str | None = None,
) -> list[dict[str, Any]]:
    """Search nearest capsules by vector similarity."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return []
        query_filter = _build_search_filter(
            capsule_type=capsule_type, entity=entity, capsule_topic=capsule_topic
        )
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        hits: list[dict[str, Any]] = []
        for p in response.points:
            hits.append(
                {
                    "id": p.id,
                    "score": p.score,
                    "payload": dict(p.payload or {}),
                }
            )
        return hits
    finally:
        client.close()


def reset_collection(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """Delete and recreate a collection, effectively clearing all capsules."""
    qdrant_path = _get_qdrant_path()
    client = QdrantClient(path=qdrant_path)
    try:
        existed = client.collection_exists(collection_name)
        if existed:
            client.delete_collection(collection_name)
        return {
            "collection": collection_name,
            "reset": True,
            "existed_before_reset": existed,
            "qdrant_path": qdrant_path,
        }
    finally:
        client.close()


def reset_all_collections() -> dict[str, Any]:
    """Delete all collections from local Qdrant storage."""
    qdrant_path = _get_qdrant_path()
    client = QdrantClient(path=qdrant_path)
    try:
        collections = [c.name for c in client.get_collections().collections]
        deleted: list[str] = []
        for name in collections:
            client.delete_collection(name)
            deleted.append(name)
        return {
            "reset_all": True,
            "deleted_count": len(deleted),
            "deleted_collections": deleted,
            "qdrant_path": qdrant_path,
        }
    finally:
        client.close()


def purge_local_qdrant_storage() -> dict[str, Any]:
    """Hard reset local Qdrant storage by deleting the storage directory."""
    qdrant_path = _get_qdrant_path()
    path_obj = Path(qdrant_path)
    existed = path_obj.exists()
    if existed:
        shutil.rmtree(path_obj)
    path_obj.mkdir(parents=True, exist_ok=True)
    return {
        "purged": True,
        "path": qdrant_path,
        "existed_before_purge": existed,
    }


def list_capsules(
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = VECTOR_LIST_LIMIT_DEFAULT,
) -> list[dict[str, Any]]:
    """Return all capsules from a collection (up to limit)."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return []
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        offset: Any = None
        batch_size = min(VECTOR_SCROLL_BATCH_SMALL, max(1, limit))
        while len(out) < limit:
            points, offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset,
            )
            for p in points:
                pid = str(p.id)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                out.append(
                    {
                        "id": p.id,
                        "payload": dict(p.payload or {}),
                    }
                )
                if len(out) >= limit:
                    break
            if offset is None:
                break
        return out
    finally:
        client.close()


def count_capsules(
    collection_name: str = DEFAULT_COLLECTION,
    exact: bool = True,
) -> int:
    """Count capsules in a collection using Qdrant count API."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return 0
        return int(client.count(collection_name=collection_name, exact=exact).count)
    finally:
        client.close()


def count_capsules_by_ingest_run_id(
    ingest_run_id: str,
    collection_name: str = DEFAULT_COLLECTION,
    exact: bool = True,
) -> int:
    """Count capsules for a specific ingest run id."""
    if not ingest_run_id:
        return 0
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return 0
        run_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="ingest_run_id",
                    match=models.MatchValue(value=ingest_run_id),
                )
            ]
        )
        return int(
            client.count(
                collection_name=collection_name,
                count_filter=run_filter,
                exact=exact,
            ).count
        )
    finally:
        client.close()


def scroll_capsules_by_sql_hash(
    sql_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = VECTOR_SQL_SCROLL_LIMIT_DEFAULT,
) -> list[dict[str, Any]]:
    """Fetch all capsules matching a source_sql_hash (legacy/debug helper)."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return []
        hits: list[dict[str, Any]] = []
        offset: Any = None
        batch_size = min(VECTOR_SCROLL_BATCH_LARGE, max(1, limit))
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_sql_hash",
                    match=models.MatchValue(value=sql_hash),
                )
            ]
        )
        while len(hits) < limit:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset,
            )
            for point in points:
                hits.append(
                    {
                        "id": point.id,
                        "payload": dict(point.payload or {}),
                        "score": VECTOR_SCORE_DEFAULT,
                    }
                )
                if len(hits) >= limit:
                    break
            if offset is None:
                break
        return hits
    finally:
        client.close()


def delete_capsule_by_id(
    capsule_id: Any,
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """Delete a single capsule by point id."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return False
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[capsule_id]),
        )
        return True
    finally:
        client.close()


def delete_capsules_by_types(
    capsule_types: list[str] | tuple[str, ...],
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """Delete all capsules whose capsule_type is in the provided set."""
    normalized_types = {
        str(capsule_type).strip().lower()
        for capsule_type in capsule_types
        if str(capsule_type).strip()
    }
    if not normalized_types:
        return {"deleted_count": 0, "deleted_ids": [], "capsule_types": []}

    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return {"deleted_count": 0, "deleted_ids": [], "capsule_types": sorted(normalized_types)}

        all_points = _scroll_all_points(client=client, collection_name=collection_name)
        delete_ids: list[Any] = []
        for point in all_points:
            payload = dict(point.payload or {})
            capsule_type = str(payload.get("capsule_type", "")).strip().lower()
            if capsule_type in normalized_types:
                delete_ids.append(point.id)

        if delete_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=delete_ids),
            )

        return {
            "deleted_count": len(delete_ids),
            "deleted_ids": delete_ids,
            "capsule_types": sorted(normalized_types),
        }
    finally:
        client.close()


def apply_retention_policies(
    collection_name: str = DEFAULT_COLLECTION,
    max_random_per_table: int = DEFAULT_MAX_RANDOM_PER_TABLE,
    replace_similar_capsules: bool = True,
) -> dict[str, Any]:
    """
    Apply post-ingestion retention/dedup policies.

    Policies:
    - random_sample: keep only latest N capsules per primary table.
    - aggregation/distribution: if replace_similar_capsules=True, keep latest per signature.
    """
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection_name):
            return {
                "collection": collection_name,
                "deleted_count": 0,
                "deleted_ids": [],
                "max_random_per_table": max_random_per_table,
                "replace_similar_capsules": replace_similar_capsules,
            }

        all_points = _scroll_all_points(client=client, collection_name=collection_name)
        by_id: dict[str, dict[str, Any]] = {}
        for point in all_points:
            pid = str(point.id)
            by_id[pid] = {
                "id": point.id,
                "payload": dict(point.payload or {}),
            }

        to_delete_ids: set[str] = set()

        # Policy 1: random samples per table.
        random_groups: dict[str, list[dict[str, Any]]] = {}
        for item in by_id.values():
            payload = item.get("payload", {})
            if str(payload.get("capsule_type", "")).strip().lower() != CAPSULE_TYPE_RANDOM_SAMPLE:
                continue
            table = _primary_table_from_payload(payload)
            random_groups.setdefault(table, []).append(item)

        keep_n = max(1, int(max_random_per_table))
        for items in random_groups.values():
            ordered = sorted(items, key=_payload_sort_key, reverse=True)
            for old_item in ordered[keep_n:]:
                to_delete_ids.add(str(old_item.get("id")))

        # Policy 2: replace similar agg/distribution capsules.
        if replace_similar_capsules:
            signature_groups: dict[str, list[dict[str, Any]]] = {}
            for item in by_id.values():
                payload = item.get("payload", {})
                ctype = str(payload.get("capsule_type", "")).strip().lower()
                if ctype not in {CAPSULE_TYPE_AGGREGATION, CAPSULE_TYPE_DISTRIBUTION}:
                    continue
                sig = _capsule_signature(payload)
                signature_groups.setdefault(sig, []).append(item)

            for items in signature_groups.values():
                ordered = sorted(items, key=_payload_sort_key, reverse=True)
                for old_item in ordered[1:]:
                    to_delete_ids.add(str(old_item.get("id")))

        deleted_ids: list[Any] = []
        if to_delete_ids:
            delete_points: list[Any] = []
            for pid in sorted(to_delete_ids):
                raw = by_id.get(pid, {}).get("id")
                delete_points.append(raw if raw is not None else pid)
                deleted_ids.append(raw if raw is not None else pid)
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=delete_points),
            )

        return {
            "collection": collection_name,
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "max_random_per_table": keep_n,
            "replace_similar_capsules": replace_similar_capsules,
        }
    finally:
        client.close()


def _ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    if client.collection_exists(collection_name):
        _ensure_payload_indexes(client, collection_name)
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    _ensure_payload_indexes(client, collection_name)


def _capsules_to_points(
    capsules: list[dict[str, Any]],
    vectors: list[list[float]],
    source_sql: str,
    source_query: str | None,
    source_sql_hash: str,
    ingest_run_id: str | None,
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for idx, (capsule, vector) in enumerate(zip(capsules, vectors), start=1):
        capsule_source_sql = str(capsule.get("source_sql", source_sql or "")).strip()
        capsule_source_hash = str(capsule.get("source_sql_hash", "")).strip() or source_sql_hash
        content_hash = _build_content_hash(capsule=capsule, source_sql=capsule_source_sql)
        point_id = _stable_int_id(content_hash)
        metrics = capsule.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows_json = str(capsule.get("rows_json", "[]"))
        key_columns = _normalize_string_list(capsule.get("key_columns", []))
        tags = _normalize_string_list(capsule.get("tags", capsule.get("metric_tags", [])))
        tables_used = _normalize_string_list(capsule.get("tables_used", []))
        relevant_columns = _normalize_string_list(capsule.get("relevant_columns", []))
        recommended_joins = _normalize_string_list(capsule.get("recommended_joins", []))
        recommended_filters = _normalize_string_list(capsule.get("recommended_filters", []))
        example_questions = _normalize_string_list(capsule.get("example_questions", []))
        join_columns = _normalize_string_list(capsule.get("join_columns", []))
        time_columns = _normalize_string_list(capsule.get("time_columns", []))
        entity_columns = _normalize_string_list(capsule.get("entity_columns", []))
        metric_columns = _normalize_string_list(capsule.get("metric_columns", []))
        business_intents = _normalize_string_list(capsule.get("business_intents", []))
        created_at = str(capsule.get("created_at", capsule.get("refreshed_at_utc", "")))
        refreshed_at = str(capsule.get("refreshed_at_utc", created_at))
        payload = {
            "capsule_id": str(capsule.get("capsule_id", point_id)),
            "capsule_index": int(capsule.get("capsule_index", idx)),
            "capsule_name": str(capsule.get("capsule_name", "")),
            "capsule_type": capsule.get("capsule_type", CAPSULE_TYPE_SUMMARY),
            "entity": str(capsule.get("entity", "")),
            "capsule_topic": str(capsule.get("capsule_topic", "")),
            "key_columns": key_columns,
            "tags": tags,
            "tables_used": tables_used,
            "relevant_columns": relevant_columns,
            "recommended_joins": recommended_joins,
            "recommended_filters": recommended_filters,
            "example_questions": example_questions,
            "join_columns": join_columns,
            "time_columns": time_columns,
            "entity_columns": entity_columns,
            "metric_columns": metric_columns,
            "business_intents": business_intents,
            "metric_tags": _normalize_string_list(capsule.get("metric_tags", tags)),
            "capsule_priority": str(capsule.get("capsule_priority", "low")),
            "capsule_version": str(capsule.get("capsule_version", "")),
            "schema_version": str(capsule.get("schema_version", "")),
            "schema_context_json": str(capsule.get("schema_context_json", "")),
            "sql_template": str(capsule.get("sql_template", "")),
            "content_hash": content_hash,
            "source_sql_hash": capsule_source_hash,
            "source_query_hash": str(capsule.get("source_query_hash", "")),
            "ingest_run_id": str(ingest_run_id or ""),
            "summary_text": str(capsule.get("summary_text", "")),
            "metrics_json": json.dumps(metrics, default=str),
            "rows_json": rows_json,
            "source_sql": capsule_source_sql,
            "source_query": source_query or "",
            "row_count": int(capsule.get("row_count", 0)),
            "created_at": created_at,
            "created_at_utc": created_at,
            "refreshed_at_utc": refreshed_at,
        }
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def _stable_int_id(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _build_content_hash(capsule: dict[str, Any], source_sql: str) -> str:
    canonical = {
        "source_sql": _normalize_sql(source_sql),
        "capsule_type": capsule.get("capsule_type", CAPSULE_TYPE_SUMMARY),
        "entity": capsule.get("entity", ""),
        "capsule_topic": capsule.get("capsule_topic", ""),
        "tables_used": _normalize_string_list(capsule.get("tables_used", [])),
        "key_columns": _normalize_string_list(capsule.get("key_columns", [])),
        "tags": _normalize_string_list(capsule.get("tags", capsule.get("metric_tags", []))),
        "relevant_columns": _normalize_string_list(capsule.get("relevant_columns", [])),
        "recommended_joins": _normalize_string_list(capsule.get("recommended_joins", [])),
        "recommended_filters": _normalize_string_list(capsule.get("recommended_filters", [])),
        "example_questions": _normalize_string_list(capsule.get("example_questions", [])),
        "join_columns": _normalize_string_list(capsule.get("join_columns", [])),
        "sql_template": str(capsule.get("sql_template", "")),
        "row_count": int(capsule.get("row_count", 0)),
        "summary_text": capsule.get("summary_text", ""),
        "metrics": capsule.get("metrics", {}),
    }
    return _sha256_hex(json.dumps(canonical, sort_keys=True, default=str))


def _normalize_sql(sql: str) -> str:
    return " ".join((sql or "").strip().lower().split())


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _delete_capsules_by_source_sql_hash(
    client: QdrantClient,
    source_sql_hash: str,
    collection_name: str,
) -> int:
    if not client.collection_exists(collection_name):
        return 0
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_sql_hash",
                match=models.MatchValue(value=source_sql_hash),
            )
        ]
    )
    point_ids: list[Any] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=False,
            with_vectors=False,
            limit=VECTOR_DELETE_SCROLL_LIMIT,
            offset=offset,
        )
        point_ids.extend(point.id for point in points)
        if offset is None:
            break
    if not point_ids:
        return 0
    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=point_ids),
    )
    return len(point_ids)


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    fields = (
        "capsule_type",
        "entity",
        "capsule_topic",
        "tags",
        "tables_used",
        "ingest_run_id",
    )
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index may already exist or backend may not support this operation.
            continue


def _build_search_filter(
    capsule_type: str | None, entity: str | None, capsule_topic: str | None
) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if capsule_type:
        must.append(
            models.FieldCondition(
                key="capsule_type",
                match=models.MatchValue(value=capsule_type),
            )
        )
    if entity:
        must.append(
            models.FieldCondition(
                key="entity",
                match=models.MatchValue(value=entity),
            )
        )
    if capsule_topic:
        must.append(
            models.FieldCondition(
                key="capsule_topic",
                match=models.MatchValue(value=capsule_topic),
            )
        )
    if not must:
        return None
    return models.Filter(must=must)


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        text = str(v).strip()
        if text:
            out.append(text)
    return out


def _scroll_all_points(
    client: QdrantClient,
    collection_name: str,
    limit: int = VECTOR_SCROLL_ALL_POINTS_LIMIT,
) -> list[Any]:
    out: list[Any] = []
    offset: Any = None
    while len(out) < limit:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=min(VECTOR_SCROLL_BATCH_LARGE, max(1, limit - len(out))),
            offset=offset,
        )
        out.extend(points)
        if offset is None:
            break
    return out


def _primary_table_from_payload(payload: dict[str, Any]) -> str:
    tables = payload.get("tables_used", [])
    if isinstance(tables, list) and tables:
        return str(tables[0]).strip().lower() or "unknown"
    return "unknown"


def _capsule_signature(payload: dict[str, Any]) -> str:
    capsule_type = str(payload.get("capsule_type", "")).strip().lower()
    tables = tuple(_normalize_string_list(payload.get("tables_used", [])))
    keys = tuple(_normalize_string_list(payload.get("key_columns", [])))
    tags = tuple(_normalize_string_list(payload.get("tags", [])))
    return json.dumps(
        {
            "capsule_type": capsule_type,
            "tables_used": tables,
            "key_columns": keys,
            "tags": tags,
        },
        sort_keys=True,
    )


def _payload_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    payload = item.get("payload", {})
    created_at = str(payload.get("created_at", "")).strip()
    pid = str(item.get("id", ""))
    return (created_at, pid)


def _get_qdrant_path() -> str:
    configured = os.getenv("QDRANT_PATH")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path(DEFAULT_QDRANT_PATH).resolve())


@contextmanager
def _upsert_lock(stale_after_seconds: int = VECTOR_UPSERT_LOCK_STALE_SECONDS) -> Any:
    """
    Cross-process lock for vector upserts.
    Prevents concurrent writers from inserting into the same local Qdrant path.
    """
    lock_path = Path(__file__).resolve().parent.parent / ".qdrant_upsert.lock"
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
            "Vector DB upsert is already running in another session. "
            "Wait for it to finish and try again."
        ) from exc

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
