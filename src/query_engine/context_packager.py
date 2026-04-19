"""Context package builder that combines primary, graph, linked, and schema capsules."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ..app_constants import COLLECTION_ANALYTICAL, COLLECTION_LINKED, MAX_HOP_DEPTH, MAX_LINKED_CAPSULES
from ..models import CapsuleHit, ContextPackage
from ..vector_store import scroll_all

logger = logging.getLogger(__name__)

_payload_cache: dict[str, dict[str, Any]] = {}


def _warm_cache() -> None:
    if _payload_cache:
        return
    for collection_name in (COLLECTION_ANALYTICAL, COLLECTION_LINKED):
        for payload in scroll_all(collection_name):
            capsule_id = payload.get("capsule_id")
            if capsule_id:
                _payload_cache[capsule_id] = payload


def _get_capsule(capsule_id: str) -> dict[str, Any] | None:
    _warm_cache()
    return _payload_cache.get(capsule_id)


def _follow_links(primary_capsule: dict[str, Any]) -> list[dict[str, Any]]:
    graph_capsules: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque()
    visited = {primary_capsule.get("capsule_id")}

    for linked_id in primary_capsule.get("linked_capsule_ids", []):
        queue.append((linked_id, 1))

    while queue and len(graph_capsules) < MAX_LINKED_CAPSULES:
        capsule_id, depth = queue.popleft()
        if not capsule_id or capsule_id in visited or depth > MAX_HOP_DEPTH:
            continue
        payload = _get_capsule(capsule_id)
        if payload:
            graph_capsules.append(payload)
            visited.add(capsule_id)
            for linked_id in payload.get("linked_capsule_ids", []):
                queue.append((linked_id, depth + 1))
    return graph_capsules


def build_context_package(search_hits: dict[str, list[CapsuleHit]]) -> ContextPackage:
    """Build the final context package used for capsule answers or SQL planning."""
    analytical_hits = search_hits.get("analytical", [])
    schema_hits = search_hits.get("schema", [])
    linked_hits = search_hits.get("linked", [])

    primary_hit = analytical_hits[0] if analytical_hits else None
    primary_capsule = primary_hit.payload if primary_hit else None
    graph_capsules = _follow_links(primary_capsule) if primary_capsule else []
    linked_capsules = [hit.payload for hit in linked_hits]
    schema_capsules = [hit.payload for hit in schema_hits[:3]]

    context_parts: list[str] = []
    if primary_capsule:
        context_parts.append(f"Primary {primary_capsule.get('capsule_id')}: {primary_capsule.get('signal', '')}")
    for graph_cap in graph_capsules:
        context_parts.append(f"Graph {graph_cap.get('capsule_id')}: {graph_cap.get('signal', '')}")
    for linked in linked_capsules:
        context_parts.append(f"Linked {linked.get('capsule_id')}: {linked.get('signal', '')}")
    combined_context = " ".join(part for part in context_parts if part.strip())

    score_values = []
    if primary_hit:
        score_values.append(primary_hit.score)
    score_values.extend(hit.score for hit in analytical_hits[1 : 1 + len(graph_capsules)])
    overall_confidence = round(sum(score_values) / len(score_values), 4) if score_values else 0.0

    return ContextPackage(
        primary_capsule=primary_capsule,
        graph_capsules=graph_capsules,
        linked_capsules=linked_capsules,
        schema_capsules=schema_capsules,
        combined_context=combined_context,
        overall_confidence=overall_confidence,
    )
