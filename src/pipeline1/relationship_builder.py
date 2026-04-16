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
    RELATIONSHIP_KEY_FIELDS,
    SYSTEMIC_RISK_ALERT_COUNT,
    SYSTEMIC_RISK_REJECTION_PCT,
)
from ..embedding import embed_single, ensure_data_dir
from ..llm_service import call_llm
from ..models import CapsuleGraph, RelatedCapsule, GeneratedCapsule, RelationshipEdge
from ..prompts import RELATED_SIGNAL_SYSTEM, RELATED_SIGNAL_USER

logger = logging.getLogger(__name__)


def _shared_entity_value(capsule_a: GeneratedCapsule, capsule_b: GeneratedCapsule) -> str | None:
    for field_name in RELATIONSHIP_KEY_FIELDS:
        values_a = {str(row.get(field_name)) for row in capsule_a.result_rows if row.get(field_name) not in (None, "")}
        values_b = {str(row.get(field_name)) for row in capsule_b.result_rows if row.get(field_name) not in (None, "")}
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
    """Generate related broker and employee risk capsules from analytical outputs."""
    capsule_map = {capsule.capsule_id: capsule for capsule in capsules}
    related_capsules: list[RelatedCapsule] = []

    broker_activity = capsule_map.get("trade_requests_by_broker_dealer")
    broker_alerts = capsule_map.get("broker_dealers_high_rejection_and_alerts")
    if broker_activity and broker_alerts:
        rejection_rates = {
            str(row.get("broker_dealer")): float(row.get("rejection_rate_pct", 0.0))
            for row in broker_activity.result_rows
            if row.get("broker_dealer")
        }
        alert_counts = {
            str(row.get("broker_dealer")): int(row.get("total_alerts", 0))
            for row in broker_alerts.result_rows
            if row.get("broker_dealer")
        }
        for broker_name, rejection_rate in rejection_rates.items():
            alert_count = alert_counts.get(broker_name, 0)
            if rejection_rate >= SYSTEMIC_RISK_REJECTION_PCT and alert_count >= SYSTEMIC_RISK_ALERT_COUNT:
                signal = _build_related_signal(
                    "broker",
                    broker_name,
                    [
                        f"Rejection rate is {rejection_rate:.1f}%.",
                        f"Compliance alert count is {alert_count}.",
                    ],
                )
                capsule_id = f"systemic_risk_{broker_name.lower().replace(' ', '_')[:40]}"
                embed_text = f"Broker systemic risk for {broker_name}. {signal}"
                related_capsules.append(
                    RelatedCapsule(
                        capsule_id=capsule_id,
                        related_from=[broker_activity.capsule_id, broker_alerts.capsule_id],
                        signal=signal,
                        embed_text=embed_text,
                        entity_type="broker",
                        entity_name=broker_name,
                        risk_level="high",
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        tags=["related", "broker", "systemic_risk"],
                        vector=embed_single(embed_text),
                    )
                )

    repeat_violators = capsule_map.get("repeat_violators")
    high_severity_alerts = capsule_map.get("high_severity_open_alerts")
    if repeat_violators and high_severity_alerts:
        repeat_map = {
            str(row.get("employee_name")): int(row.get("alert_count", 0))
            for row in repeat_violators.result_rows
            if row.get("employee_name")
        }
        severity_map = {
            str(row.get("employee_name")): str(row.get("severity"))
            for row in high_severity_alerts.result_rows
            if row.get("employee_name")
        }
        for employee_name, alert_count in repeat_map.items():
            severity = severity_map.get(employee_name)
            if severity:
                signal = _build_related_signal(
                    "employee",
                    employee_name,
                    [
                        f"Repeat violator with {alert_count} alerts.",
                        f"Currently has an open {severity} severity alert.",
                    ],
                )
                capsule_id = f"employee_risk_{employee_name.lower().replace(' ', '_')[:40]}"
                embed_text = f"Employee risk profile for {employee_name}. {signal}"
                related_capsules.append(
                    RelatedCapsule(
                        capsule_id=capsule_id,
                        related_from=[repeat_violators.capsule_id, high_severity_alerts.capsule_id],
                        signal=signal,
                        embed_text=embed_text,
                        entity_type="employee",
                        entity_name=employee_name,
                        risk_level="critical",
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        tags=["related", "employee", "repeat_violator", "open_alert"],
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
