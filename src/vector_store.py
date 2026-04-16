"""Shared Qdrant helpers for storing, searching, scrolling, and deleting capsules."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .app_constants import ALL_COLLECTIONS, EMBED_DIM
from .config import get_settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def _safe_get_client() -> QdrantClient | None:
    """Return a Qdrant client or None if local storage is currently locked."""
    try:
        return _get_client()
    except RuntimeError as exc:
        logger.error("Qdrant client unavailable: %s", exc)
        return None


def close_client() -> None:
    """Close the current local Qdrant client and release the file lock."""
    global _client
    if _client is None:
        return
    try:
        local_client = getattr(_client, "_client", None)
        if hasattr(local_client, "close"):
            local_client.close()
    except Exception as exc:
        logger.warning("Failed to close Qdrant client cleanly: %s", exc)
    finally:
        _client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(path=settings.qdrant_path)
        _ensure_collections(_client)
    return _client


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
    client = _safe_get_client()
    if client is None:
        return
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


def upsert_capsules_batch(collection: str, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
    """Upsert many capsules at once."""
    if not items:
        return
    client = _safe_get_client()
    if client is None:
        return
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


def search(
    collection: str,
    vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Search a collection and return normalized hit payloads."""
    client = _safe_get_client()
    if client is None:
        return []
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=vector,
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
            for hit in hits
        ]
    except Exception as exc:
        logger.error("Qdrant search failed in %s: %s", collection, exc)
        return []


def scroll_all(collection: str, limit: int = 10_000) -> list[dict[str, Any]]:
    """Return all payloads from a collection."""
    client = _safe_get_client()
    if client is None:
        return []
    payloads: list[dict[str, Any]] = []
    offset = None
    try:
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
    except Exception as exc:
        logger.error("Qdrant scroll failed in %s: %s", collection, exc)
    return payloads[:limit]


def delete_by_capsule_id(collection: str, capsule_id: str) -> None:
    """Delete one capsule using deterministic point id."""
    client = _safe_get_client()
    if client is None:
        return
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection}:{capsule_id}"))
    try:
        client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )
    except Exception as exc:
        logger.error("Qdrant delete failed in %s for %s: %s", collection, capsule_id, exc)


def clear_collection(collection: str) -> None:
    """Delete and recreate a single collection."""
    client = _safe_get_client()
    if client is None:
        return
    try:
        client.delete_collection(collection_name=collection)
    except Exception as exc:
        logger.warning("Delete collection failed for %s: %s", collection, exc)
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )
    except Exception as exc:
        logger.warning("Create collection failed for %s, retrying after ensure: %s", collection, exc)
        _ensure_collections(client)


def reset_all_collections() -> None:
    """Delete and recreate all collections."""
    close_client()
    import shutil
    qdrant_path = Path(get_settings().qdrant_path)
    if qdrant_path.exists():
        shutil.rmtree(qdrant_path, ignore_errors=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = _safe_get_client()
    if client is None:
        raise RuntimeError(
            f"Qdrant local storage at '{qdrant_path}' is locked by another running process. "
            "Close the other app instance and try again."
        )
    _ensure_collections(client)


def collection_counts() -> dict[str, int]:
    """Return point counts for every collection."""
    client = _safe_get_client()
    if client is None:
        return {collection: 0 for collection in ALL_COLLECTIONS}
    counts: dict[str, int] = {}
    for collection in ALL_COLLECTIONS:
        try:
            info = client.get_collection(collection)
            counts[collection] = int(info.points_count or 0)
        except Exception:
            counts[collection] = 0
    return counts
