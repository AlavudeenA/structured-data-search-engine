"""Build capsule relationships and related risk capsules from generated analytical capsules."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..app_constants import (
    CAPSULE_GRAPH_FILE,
    MIN_SHARED_TAGS_FOR_RELATION,
    REL_AGGREGATES_UP,
    REL_CORROBORATES,
    REL_DRILLS_DOWN,
    REL_SAME_ENTITY,
)
from ..embedding import embed_single, ensure_data_dir
from ..llm_service import call_llm
from ..models import CapsuleGraph, RelatedCapsule, GeneratedCapsule, RelationshipEdge
from ..llm_instructions import RELATED_SIGNAL_SYSTEM, RELATED_SIGNAL_USER

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
    """Create graph edges from explicit relationships plus inferred relationships."""
    edges: list[RelationshipEdge] = []
    seen: set[tuple[str, str]] = set()

    for capsule in capsules:
        for related_id, relationship in zip(capsule.related_capsule_ids, capsule.relationship_types):
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

    return CapsuleGraph(built_at=datetime.now(timezone.utc).isoformat(), edges=edges, related_capsule_ids=[])


def _build_related_signal(entity_type: str, entity_name: str, signals: list[str]) -> str:
    prompt = RELATED_SIGNAL_USER.format(entity_type=entity_type, entity_name=entity_name, signals="\n".join(signals))
    text = call_llm(RELATED_SIGNAL_SYSTEM, prompt, model_slot="groq_signal_model", max_tokens=150)
    if not text or text.startswith("[LLM"):
        return f"{entity_name} is high risk because {'; '.join(signals[:2])}."
    return text


def generate_related_capsules(capsules: list[GeneratedCapsule]) -> list[RelatedCapsule]:
    """Generate dynamic related risk capsules based on anomaly thresholds."""
    related_capsules: list[RelatedCapsule] = []

    for capsule in capsules:
        if capsule.anomaly_score is not None and capsule.anomaly_score > 0.0:
            risk_level = "critical" if capsule.anomaly_score >= 0.8 else "high"
            first_entity = str(next(iter(capsule.result_rows[0].values()))) if capsule.result_rows else "unknown"
            signal = _build_related_signal(
                "data entity",
                first_entity,
                [
                    f"Anomaly score indicates unusual variance ({capsule.anomaly_score:.2f}).",
                    f"Derived from capsule: {capsule.capsule_id}."
                ]
            )
            embed_text = f"Analytical risk profile for {first_entity}. {signal}"
            related_capsules.append(
                RelatedCapsule(
                    capsule_id=f"alert_{capsule.capsule_id}_{first_entity[:15]}",
                    related_from=[capsule.capsule_id],
                    signal=signal,
                    embed_text=embed_text,
                    entity_type="data entity",
                    entity_name=first_entity,
                    risk_level=risk_level,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    tags=["related", "auto_generated", risk_level],
                    vector=embed_single(embed_text),
                )
            )

    return related_capsules


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
