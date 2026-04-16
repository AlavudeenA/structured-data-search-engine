"""Shared Qdrant helpers for storing, searching, scrolling, and deleting capsules."""

from __future__ import annotations

import logging
import uuid
import shutil
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .app_constants import ALL_COLLECTIONS, EMBED_DIM
from .config import get_settings

logger = logging.getLogger(__name__)

def _get_qdrant_path() -> str:
    return str(Path(get_settings().qdrant_path).resolve())

def _ensure_collections(client: QdrantClient) -> None:
    existing = {collection.name for collection in client.get_collections().collections}
    for collection_name in ALL_COLLECTIONS:
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=EMBED_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

def upsert_capsule(collection: str, capsule_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    """Upsert one capsule into a collection."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        _ensure_collections(client)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}:{capsule_id}"))
        client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={**payload, "capsule_id": capsule_id},
                )
            ],
        )
    finally:
        client.close()

def upsert_capsules_batch(collection: str, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
    """Upsert many capsules at once."""
    if not items:
        return
    client = QdrantClient(path=_get_qdrant_path())
    try:
        _ensure_collections(client)
        client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}:{capsule_id}")),
                    vector=vector,
                    payload={**payload, "capsule_id": capsule_id},
                )
                for capsule_id, vector, payload in items
            ],
        )
    finally:
        client.close()

def search(
    collection: str,
    vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Search a collection and return normalized hit payloads."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection):
            return []
        res = client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "capsule_id": hit.payload.get("capsule_id", str(hit.id)),
                "score": float(hit.score),
                "collection": collection,
                "payload": hit.payload or {},
            }
            for hit in res.points
        ]
    except Exception as exc:
        logger.error("Qdrant search failed in %s: %s", collection, exc)
        return []
    finally:
        client.close()

def scroll_all(collection: str, limit: int = 10_000) -> list[dict[str, Any]]:
    """Return all payloads from a collection."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection):
            return []
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=min(500, max(1, limit - len(payloads))),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            payloads.extend(point.payload or {} for point in points)
            if next_offset is None or len(payloads) >= limit:
                break
            offset = next_offset
        return payloads[:limit]
    except Exception as exc:
        logger.error("Qdrant scroll failed in %s: %s", collection, exc)
        return []
    finally:
        client.close()

def delete_by_capsule_id(collection: str, capsule_id: str) -> None:
    """Delete one capsule using deterministic point id."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if not client.collection_exists(collection):
            return
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}:{capsule_id}"))
        client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )
    except Exception as exc:
        logger.error("Qdrant delete failed in %s for %s: %s", collection, capsule_id, exc)
    finally:
        client.close()

def clear_collection(collection: str) -> None:
    """Delete and recreate a single collection via API."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection_name=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )
    except Exception as exc:
        logger.warning("Clear collection failed for %s: %s", collection, exc)
    finally:
        client.close()

def reset_all_collections() -> None:
    """Soft delete all collections from local Qdrant storage."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        collections = [c.name for c in client.get_collections().collections]
        for name in collections:
            client.delete_collection(name)
    finally:
        client.close()

def purge_local_qdrant_storage() -> None:
    """Hard reset local Qdrant storage by deleting the storage directory."""
    qdrant_path = _get_qdrant_path()
    path_obj = Path(qdrant_path)
    if path_obj.exists():
        shutil.rmtree(path_obj, ignore_errors=True)
    path_obj.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=qdrant_path)
    try:
        _ensure_collections(client)
    finally:
        client.close()

def collection_counts() -> dict[str, int]:
    """Return point counts for every collection."""
    client = QdrantClient(path=_get_qdrant_path())
    try:
        counts: dict[str, int] = {}
        for collection in ALL_COLLECTIONS:
            try:
                info = client.get_collection(collection)
                counts[collection] = int(info.points_count or 0)
            except Exception:
                counts[collection] = 0
        return counts
    finally:
        client.close()
