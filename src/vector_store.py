# File: src/vector_store.py
"""Qdrant vector store operations for context capsules."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


DEFAULT_COLLECTION = "context_capsules"
DEFAULT_QDRANT_PATH = "qdrant_data"


def upsert_capsules(
    capsules: list[dict[str, Any]],
    vectors: list[list[float]],
    source_sql: str,
    source_query: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    ingestion_mode: str = "append_unique",
) -> dict[str, Any]:
    """Create collection if needed, then upsert capsule vectors."""
    if not vectors:
        raise ValueError("No vectors provided for upsert.")

    source_sql_hash = _sha256_hex(_normalize_sql(source_sql))
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
    try:
        deleted_count = 0
        if ingestion_mode == "replace_source":
            deleted_count = _delete_capsules_by_source_sql_hash(
                client=client,
                source_sql_hash=source_sql_hash,
                collection_name=collection_name,
            )
        _ensure_collection(client, collection_name, len(vectors[0]))
        points = _capsules_to_points(
            capsules=capsules,
            vectors=vectors,
            source_sql=source_sql,
            source_query=source_query,
            source_sql_hash=source_sql_hash,
        )
        client.upsert(collection_name=collection_name, points=points)
        return {
            "indexed_count": len(points),
            "deleted_count": deleted_count,
            "source_sql_hash": source_sql_hash,
        }
    finally:
        client.close()


def delete_capsules_by_source_sql_hash(
    source_sql_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Delete all points for a source SQL hash."""
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
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
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
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
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
    try:
        existed = client.collection_exists(collection_name)
        if existed:
            client.delete_collection(collection_name)
        return {
            "collection": collection_name,
            "reset": True,
            "existed_before_reset": existed,
        }
    finally:
        client.close()


def list_capsules(
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Return all capsules from a collection (up to limit)."""
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
    try:
        if not client.collection_exists(collection_name):
            return []
        out: list[dict[str, Any]] = []
        offset: Any = None
        batch_size = min(250, max(1, limit))
        while len(out) < limit:
            points, offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset,
            )
            for p in points:
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


def scroll_capsules_by_sql_hash(
    sql_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Fetch all capsules matching a source_sql_hash."""
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
    try:
        if not client.collection_exists(collection_name):
            return []
        hits: list[dict[str, Any]] = []
        offset: Any = None
        batch_size = min(1000, max(1, limit))
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
                        "score": 1.0,
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
    client = QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))
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
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for capsule, vector in zip(capsules, vectors):
        content_hash = _build_content_hash(capsule=capsule, source_sql=source_sql)
        point_id = _stable_int_id(content_hash)
        payload = {
            "capsule_index": capsule["capsule_index"],
            "capsule_name": capsule.get("capsule_name", ""),
            "capsule_type": capsule.get("capsule_type", "summary"),
            "entity": capsule.get("entity", ""),
            "capsule_topic": capsule.get("capsule_topic", ""),
            "metric_tags": capsule.get("metric_tags", []),
            "capsule_priority": capsule.get("capsule_priority", "low"),
            "capsule_version": capsule.get("capsule_version", "v1"),
            "schema_version": capsule.get("schema_version", "v1"),
            "content_hash": content_hash,
            "source_sql_hash": source_sql_hash,
            "source_query_hash": capsule.get("source_query_hash", ""),
            "summary_text": capsule["summary_text"],
            "metrics_json": json.dumps(capsule["metrics"], default=str),
            "rows_json": capsule["rows_json"],
            "source_sql": source_sql,
            "source_query": source_query or "",
            "row_count": capsule["row_count"],
            "refreshed_at_utc": capsule["refreshed_at_utc"],
        }
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def _stable_int_id(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _build_content_hash(capsule: dict[str, Any], source_sql: str) -> str:
    canonical = {
        "source_sql": _normalize_sql(source_sql),
        "capsule_type": capsule.get("capsule_type", "summary"),
        "entity": capsule.get("entity", ""),
        "capsule_topic": capsule.get("capsule_topic", ""),
        "summary_text": capsule.get("summary_text", ""),
        "metrics": capsule.get("metrics", {}),
        "rows": _canonical_rows(capsule.get("rows_json", "[]")),
    }
    return _sha256_hex(json.dumps(canonical, sort_keys=True, default=str))


def _canonical_rows(rows_json: str) -> Any:
    try:
        rows = json.loads(rows_json)
    except (TypeError, ValueError):
        return rows_json
    return rows


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
            limit=1000,
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
    fields = ("capsule_type", "entity", "capsule_topic", "source_sql_hash")
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
