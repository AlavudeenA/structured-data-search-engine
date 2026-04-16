"""Vector-search context retrieval across analytical, schema-context, and related collections."""

from __future__ import annotations

import logging

from ..app_constants import (
    ANALYTICAL_MIN_SCORE,
    ANALYTICAL_TOP_K,
    COLLECTION_ANALYTICAL,
    COLLECTION_RELATED,
    COLLECTION_SCHEMA,
    RELATED_MIN_SCORE,
    RELATED_TOP_K,
    SCHEMA_MIN_SCORE,
    SCHEMA_TOP_K,
)
from ..embedding import embed_single
from ..models import CapsuleHit
from ..vector_store import search

logger = logging.getLogger(__name__)


def _wrap_hits(collection: str, raw_hits: list[dict]) -> list[CapsuleHit]:
    return [
        CapsuleHit(
            capsule_id=hit["capsule_id"],
            score=hit["score"],
            collection=collection,
            payload=hit["payload"],
        )
        for hit in raw_hits
        if not hit["payload"].get("is_stale", False)
    ]


def search_all_collections(question: str) -> dict[str, list[CapsuleHit]]:
    """Embed the question and search all three collections."""
    vector = embed_single(question)
    if not vector:
        return {"analytical": [], "schema": [], "related": []}

    analytical = _wrap_hits(
        COLLECTION_ANALYTICAL,
        search(COLLECTION_ANALYTICAL, vector, top_k=ANALYTICAL_TOP_K, score_threshold=ANALYTICAL_MIN_SCORE),
    )
    schema = _wrap_hits(
        COLLECTION_SCHEMA,
        search(COLLECTION_SCHEMA, vector, top_k=SCHEMA_TOP_K, score_threshold=SCHEMA_MIN_SCORE),
    )
    related = _wrap_hits(
        COLLECTION_RELATED,
        search(COLLECTION_RELATED, vector, top_k=RELATED_TOP_K, score_threshold=RELATED_MIN_SCORE),
    )

    logger.info(
        "Context search complete | analytical=%s schema=%s related=%s",
        len(analytical),
        len(schema),
        len(related),
    )
    return {"analytical": analytical, "schema": schema, "related": related}
