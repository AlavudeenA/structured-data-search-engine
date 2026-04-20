"""Build capsule relationships and linked risk capsules from generated analytical capsules."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..app_constants import (
    ANOMALY_DETECTED_THRESHOLD,
    CAPSULE_GRAPH_FILE,
    COLLECTION_ANALYTICAL,
    MIN_SHARED_TAGS_FOR_RELATION,
    REL_AGGREGATES_UP,
    REL_CORROBORATES,
    REL_DRILLS_DOWN,
    REL_SAME_ENTITY,
)
from ..embedding import embed_single, ensure_data_dir
from ..llm_service import call_llm
from ..models import CapsuleGraph, LinkedCapsule, GeneratedCapsule, RelationshipEdge
from ..llm_instructions import LINKED_SIGNAL_SYSTEM, LINKED_SIGNAL_USER
from ..vector_store import scroll_all, set_payload_fields

logger = logging.getLogger(__name__)


def _shared_entity_value(capsule_a: GeneratedCapsule, capsule_b: GeneratedCapsule) -> str | None:
    if not capsule_a.result_rows or not capsule_b.result_rows:
        return None
    cols_a = set(capsule_a.result_rows[0].keys())
    cols_b = set(capsule_b.result_rows[0].keys())
    for field_name in (cols_a & cols_b):
        values_a = {str(row.get(field_name)) for row in capsule_a.result_rows if row.get(field_name)}
        values_b = {str(row.get(field_name)) for row in capsule_b.result_rows if row.get(field_name)}
        shared = values_a & values_b
        if shared:
            return next(iter(shared))
    return None


def _infer_relationship(capsule_a: GeneratedCapsule, capsule_b: GeneratedCapsule) -> tuple[str | None, str | None]:
    tables_a = set(capsule_a.tables_used)
    tables_b = set(capsule_b.tables_used)
    shared_tags = set(capsule_a.tags) & set(capsule_b.tags)
    join_key = _shared_entity_value(capsule_a, capsule_b)

    if tables_a == tables_b and capsule_a.capsule_type != capsule_b.capsule_type:
        return REL_SAME_ENTITY, join_key
    if tables_a.issubset(tables_b) and tables_a != tables_b:
        return REL_DRILLS_DOWN, join_key
    if tables_b.issubset(tables_a) and tables_a != tables_b:
        return REL_AGGREGATES_UP, join_key
    if len(shared_tags) >= MIN_SHARED_TAGS_FOR_RELATION or join_key:
        return REL_CORROBORATES, join_key
    return None, None


def build_graph(capsules: list[GeneratedCapsule]) -> CapsuleGraph:
    """Create graph edges from explicit relationships plus inferred relationships.

    Inferred edges are also written back onto the in-memory capsule objects so
    that when the capsules are persisted to Qdrant their linked_capsule_ids
    includes both explicit and inferred relationships — making them available
    to _follow_links during query answering.
    """
    edges: list[RelationshipEdge] = []
    seen: set[tuple[str, str]] = set()
    capsule_map = {c.capsule_id: c for c in capsules}

    for capsule in capsules:
        for related_id, relationship in zip(capsule.linked_capsule_ids, capsule.relationship_types):
            key = tuple(sorted([capsule.capsule_id, related_id]))
            if key not in seen:
                edges.append(RelationshipEdge(from_id=capsule.capsule_id, to_id=related_id, relationship=relationship))
                seen.add(key)

    for index, capsule_a in enumerate(capsules):
        for capsule_b in capsules[index + 1 :]:
            key = tuple(sorted([capsule_a.capsule_id, capsule_b.capsule_id]))
            if key in seen:
                continue
            relationship, join_key = _infer_relationship(capsule_a, capsule_b)
            if relationship:
                edges.append(
                    RelationshipEdge(
                        from_id=capsule_a.capsule_id,
                        to_id=capsule_b.capsule_id,
                        relationship=relationship,
                        join_key=join_key,
                    )
                )
                seen.add(key)
                # Write inferred edge back onto both capsule objects so Qdrant
                # payloads carry the full relationship set after _persist_analytical.
                if capsule_b.capsule_id not in capsule_a.linked_capsule_ids:
                    capsule_a.linked_capsule_ids.append(capsule_b.capsule_id)
                    capsule_a.relationship_types.append(relationship)
                if capsule_a.capsule_id not in capsule_b.linked_capsule_ids:
                    capsule_b.linked_capsule_ids.append(capsule_a.capsule_id)
                    capsule_b.relationship_types.append(relationship)

    return CapsuleGraph(built_at=datetime.now(timezone.utc).isoformat(), edges=edges, linked_capsule_ids=[])


def _build_linked_signal(capsule_id: str, capsule_what: str, anomaly_score: float, trend_direction: str) -> str:
    prompt = LINKED_SIGNAL_USER.format(
        capsule_id=capsule_id,
        capsule_what=capsule_what,
        anomaly_score=anomaly_score,
        trend_direction=trend_direction,
    )
    text = call_llm(LINKED_SIGNAL_SYSTEM, prompt, model_slot="groq_signal_model", max_tokens=150)
    if not text or text.startswith("[LLM"):
        return (
            f"Anomaly detected in {capsule_what} (score: {anomaly_score:.2f}). "
            f"Review the {capsule_id} capsule for entity-level details."
        )
    return text


def generate_linked_capsules(capsules: list[GeneratedCapsule]) -> list[LinkedCapsule]:
    """Generate pattern-level risk alert capsules for capsules with anomaly scores above zero.
    Alerts describe the statistical pattern only — no entity names are embedded in IDs or signals.
    """
    linked_capsules: list[LinkedCapsule] = []

    for capsule in capsules:
        if capsule.anomaly_score is not None and capsule.anomaly_score >= ANOMALY_DETECTED_THRESHOLD:
            risk_level = "critical" if capsule.anomaly_score >= 0.8 else "high"
            signal = _build_linked_signal(
                capsule.capsule_id,
                capsule.what,
                capsule.anomaly_score,
                capsule.trend_direction or "flat",
            )
            embed_text = f"Anomaly pattern in {capsule.what}. {signal}"
            linked_capsules.append(
                LinkedCapsule(
                    capsule_id=f"alert_{capsule.capsule_id}",
                    linked_from=[capsule.capsule_id],
                    signal=signal,
                    embed_text=embed_text,
                    entity_type=capsule.capsule_type,
                    entity_name=capsule.capsule_id,
                    risk_level=risk_level,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    tags=["linked", "auto_generated", risk_level],
                    vector=embed_single(embed_text),
                )
            )

    return linked_capsules


def save_graph(graph: CapsuleGraph) -> None:
    """Persist graph JSON to disk."""
    ensure_data_dir()
    CAPSULE_GRAPH_FILE.write_text(graph.model_dump_json(indent=2), encoding="utf-8")


def load_graph() -> CapsuleGraph | None:
    """Load graph JSON if available."""
    if not CAPSULE_GRAPH_FILE.exists():
        return None
    try:
        return CapsuleGraph.model_validate_json(CAPSULE_GRAPH_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load capsule graph: %s", exc)
        return None


def _payload_to_capsule(payload: dict) -> GeneratedCapsule | None:
    """Convert a Qdrant payload dict to a GeneratedCapsule for relationship inference.
    Returns None if required fields are missing.
    """
    try:
        return GeneratedCapsule(
            capsule_id=payload["capsule_id"],
            capsule_type=payload.get("capsule_type", "aggregation"),
            priority=payload.get("priority", "P3"),
            what=payload.get("what", ""),
            how=payload.get("how", ""),
            sql=payload.get("sql", ""),
            signal_method=payload.get("signal_method", "rule_based"),
            signal=payload.get("signal", ""),
            embed_text=payload.get("embed_text", ""),
            tables_used=payload.get("tables_used") or [],
            key_columns=payload.get("key_columns") or [],
            tags=payload.get("tags") or [],
            ttl_hours=payload.get("ttl_hours", 24),
            generated_at=payload.get("generated_at", ""),
            expires_at=payload.get("expires_at", ""),
            staleness_trigger=payload.get("staleness_trigger", ""),
            result_rows=payload.get("result_rows") or [],
            anomaly_score=float(payload.get("anomaly_score", 0.0)),
            trend_direction=payload.get("trend_direction", "flat"),
            linked_capsule_ids=payload.get("linked_capsule_ids") or [],
            relationship_types=payload.get("relationship_types") or [],
        )
    except Exception as exc:
        logger.warning("Could not convert payload to capsule (%s): %s", payload.get("capsule_id"), exc)
        return None


def link_user_capsule_to_existing(new_capsule: GeneratedCapsule) -> None:
    """Scan all existing analytical capsules and wire bidirectional links for the new user capsule.

    For each existing capsule that has an inferred relationship with the new capsule:
      - Appends the existing capsule's ID to the new capsule's linked_capsule_ids (via set_payload)
      - Appends the new capsule's ID to the existing capsule's linked_capsule_ids (via set_payload)
      - Appends the new edge to .capsule_graph.json

    Then generates a linked alert capsule if the new capsule has anomaly_score > 0.
    """
    from .store_manager import _persist_linked

    payloads = scroll_all(COLLECTION_ANALYTICAL)
    new_linked_ids: list[str] = list(new_capsule.linked_capsule_ids)
    new_rel_types: list[str] = list(new_capsule.relationship_types)
    new_edges: list[RelationshipEdge] = []

    for payload in payloads:
        existing_id = payload.get("capsule_id", "")
        if not existing_id or existing_id == new_capsule.capsule_id:
            continue

        existing = _payload_to_capsule(payload)
        if existing is None:
            continue

        relationship, join_key = _infer_relationship(new_capsule, existing)
        if not relationship:
            continue

        logger.info(
            "User capsule %s linked to %s via %s",
            new_capsule.capsule_id, existing_id, relationship,
        )

        # Update new capsule's link list (in memory — flushed once at the end)
        if existing_id not in new_linked_ids:
            new_linked_ids.append(existing_id)
            new_rel_types.append(relationship)

        # Update existing capsule's link list in Qdrant immediately (payload-only, no re-embed)
        updated_existing_ids = list(existing.linked_capsule_ids)
        updated_existing_rels = list(existing.relationship_types)
        if new_capsule.capsule_id not in updated_existing_ids:
            updated_existing_ids.append(new_capsule.capsule_id)
            updated_existing_rels.append(relationship)
            set_payload_fields(
                COLLECTION_ANALYTICAL,
                existing_id,
                {
                    "linked_capsule_ids": updated_existing_ids,
                    "relationship_types": updated_existing_rels,
                },
            )

        new_edges.append(
            RelationshipEdge(
                from_id=new_capsule.capsule_id,
                to_id=existing_id,
                relationship=relationship,
                join_key=join_key,
            )
        )

    # Flush updated links for the new capsule itself
    if new_edges:
        set_payload_fields(
            COLLECTION_ANALYTICAL,
            new_capsule.capsule_id,
            {
                "linked_capsule_ids": new_linked_ids,
                "relationship_types": new_rel_types,
            },
        )
        # Append new edges to the saved graph
        graph = load_graph()
        if graph is None:
            graph = CapsuleGraph(
                built_at=datetime.now(timezone.utc).isoformat(),
                edges=[],
                linked_capsule_ids=[],
            )
        graph.edges.extend(new_edges)
        save_graph(graph)
        logger.info(
            "User capsule %s: added %d graph edge(s)", new_capsule.capsule_id, len(new_edges)
        )

    # Generate anomaly alert linked capsule if warranted
    if new_capsule.anomaly_score and new_capsule.anomaly_score >= ANOMALY_DETECTED_THRESHOLD:
        alert_capsules = generate_linked_capsules([new_capsule])
        if alert_capsules:
            _persist_linked(alert_capsules)
