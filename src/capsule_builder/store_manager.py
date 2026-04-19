"""Store management and rebuild helpers for analytical, schema, and related capsules."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..app_constants import COLLECTION_ANALYTICAL, COLLECTION_LINKED, COLLECTION_SCHEMA
from ..database_connection import get_fk_relationships, get_schema_metadata
from ..embedding import (
    compute_schema_fingerprint,
    load_refresh_plan,
    load_schema_fingerprint,
    save_refresh_plan,
    save_schema_fingerprint,
)
from ..models import BuildStatus, BuildSummary, CapsuleDefinition, LinkedCapsule, GeneratedCapsule, SchemaContextCapsule
from ..vector_store import clear_collection, collection_counts, purge_local_qdrant_storage, upsert_capsules_batch
from ..business_schema.capsule_definitions import CAPSULE_DEFINITIONS
from .capsule_generator import generate_all_capsules
from .relationship_builder import build_graph, generate_linked_capsules, save_graph
from .schema_capsule_generator import generate_schema_capsules

logger = logging.getLogger(__name__)


def _analytical_payload(capsule: GeneratedCapsule) -> dict:
    return {
        "capsule_id": capsule.capsule_id,
        "capsule_type": capsule.capsule_type,
        "priority": capsule.priority,
        "what": capsule.what,
        "how": capsule.how,
        "signal": capsule.signal,
        "embed_text": capsule.embed_text,
        "tables_used": capsule.tables_used,
        "key_columns": capsule.key_columns,
        "tags": capsule.tags,
        "ttl_hours": capsule.ttl_hours,
        "generated_at": capsule.generated_at,
        "expires_at": capsule.expires_at,
        "is_stale": capsule.is_stale,
        "staleness_trigger": capsule.staleness_trigger,
        "result_rows": capsule.result_rows[:50],
        "sql": capsule.sql,
        "anomaly_score": capsule.anomaly_score,
        "trend_direction": capsule.trend_direction,
        "linked_capsule_ids": capsule.linked_capsule_ids,
        "relationship_types": capsule.relationship_types,
    }


def _schema_payload(capsule: SchemaContextCapsule) -> dict:
    return {
        "capsule_id": capsule.capsule_id,
        "summary": capsule.summary,
        "tables": capsule.tables,
        "relevant_columns": capsule.relevant_columns,
        "recommended_joins": capsule.recommended_joins,
        "join_columns": capsule.join_columns,
        "recommended_filters": capsule.recommended_filters,
        "example_questions": capsule.example_questions,
        "sql_template": capsule.sql_template,
        "tags": capsule.tags,
        "generated_at": capsule.generated_at,
    }


def _related_payload(capsule: LinkedCapsule) -> dict:
    return {
        "capsule_id": capsule.capsule_id,
        "linked_from": capsule.linked_from,
        "signal": capsule.signal,
        "embed_text": capsule.embed_text,
        "entity_type": capsule.entity_type,
        "entity_name": capsule.entity_name,
        "risk_level": capsule.risk_level,
        "generated_at": capsule.generated_at,
        "tags": capsule.tags,
    }


def _persist_analytical(capsules: list[GeneratedCapsule]) -> int:
    items = [(capsule.capsule_id, capsule.vector or [], _analytical_payload(capsule)) for capsule in capsules if capsule.vector]
    upsert_capsules_batch(COLLECTION_ANALYTICAL, items)
    return len(items)


def _persist_schema(capsules: list[SchemaContextCapsule]) -> int:
    items = [(capsule.capsule_id, capsule.vector or [], _schema_payload(capsule)) for capsule in capsules if capsule.vector]
    upsert_capsules_batch(COLLECTION_SCHEMA, items)
    return len(items)


def _persist_linked(capsules: list[LinkedCapsule]) -> int:
    items = [(capsule.capsule_id, capsule.vector or [], _related_payload(capsule)) for capsule in capsules if capsule.vector]
    upsert_capsules_batch(COLLECTION_LINKED, items)
    return len(items)


def _load_definitions(plan_capsule_ids: list[str] | None = None) -> list[CapsuleDefinition]:
    from ..user_capsules import load_user_capsule_defs
    user_defs = load_user_capsule_defs()
    all_defs = CAPSULE_DEFINITIONS + user_defs
    definitions = [CapsuleDefinition(**d) for d in all_defs]
    if plan_capsule_ids:
        # Always include user capsules even if not in the saved plan
        # (they may have been created after the last full build)
        user_ids = {d["capsule_id"] for d in user_defs}
        selected = set(plan_capsule_ids) | user_ids
        definitions = [d for d in definitions if d.capsule_id in selected]
    return definitions



def generate_all_capsule_collections(progress_callback=None) -> BuildSummary:
    """Full build: analytical, schema-context, graph, related, persisted plan and fingerprint."""
    purge_local_qdrant_storage()
    definitions = _load_definitions()
    statuses: list[BuildStatus] = []

    def _progress(capsule_id: str, status: str, signal_preview: str, collection: str = COLLECTION_ANALYTICAL) -> None:
        statuses.append(BuildStatus(capsule_id=capsule_id, status=status, signal_preview=signal_preview))
        if progress_callback:
            progress_callback(capsule_id, status, signal_preview, collection)

    analytical_capsules = generate_all_capsules(definitions, progress_callback=lambda cid, s, p: _progress(cid, s, p, COLLECTION_ANALYTICAL))
    schema_capsules = generate_schema_capsules()
    for sc in schema_capsules:
        _progress(sc.capsule_id, "ok", sc.summary[:80], COLLECTION_SCHEMA)

    graph = build_graph(analytical_capsules)
    linked_capsules = generate_linked_capsules(analytical_capsules)
    for rc in linked_capsules:
        _progress(rc.capsule_id, "ok", rc.signal[:80], COLLECTION_LINKED)

    graph.linked_capsule_ids = [capsule.capsule_id for capsule in linked_capsules]
    save_graph(graph)

    analytical_count = _persist_analytical(analytical_capsules)
    schema_count = _persist_schema(schema_capsules)
    linked_count = _persist_linked(linked_capsules)

    save_refresh_plan([definition.model_dump() for definition in definitions])
    save_schema_fingerprint(get_schema_metadata(), get_fk_relationships())

    return BuildSummary(
        analytical_count=analytical_count,
        schema_count=schema_count,
        linked_count=linked_count,
        graph_edge_count=len(graph.edges),
        statuses=statuses,
        schema_changed=None,
    )


def refresh_targeted_capsules(analytical_ids: list[str], linked_ids: list[str]) -> int:
    """Refresh only the specified analytical capsules and regenerate their linked capsules.
    Uses upsert (not clear) so other capsules in the collection are untouched.
    """
    definitions = _load_definitions(analytical_ids)
    if not definitions:
        return 0
    refreshed_analytical = generate_all_capsules(definitions)
    _persist_analytical(refreshed_analytical)

    if linked_ids and refreshed_analytical:
        new_linked = generate_linked_capsules(refreshed_analytical)
        relevant_linked = [lc for lc in new_linked if lc.capsule_id in set(linked_ids)]
        if relevant_linked:
            _persist_linked(relevant_linked)

    return len(refreshed_analytical)


def refresh_data_only(progress_callback=None) -> BuildSummary:
    """Refresh analytical capsules from the canonical saved plan, keeping schema-context and related untouched."""
    plan = load_refresh_plan()
    definitions = _load_definitions(plan.capsule_ids if plan else None)
    clear_collection(COLLECTION_ANALYTICAL)
    statuses: list[BuildStatus] = []

    def _progress(capsule_id: str, status: str, signal_preview: str) -> None:
        statuses.append(BuildStatus(capsule_id=capsule_id, status=status, signal_preview=signal_preview))
        if progress_callback:
            progress_callback(capsule_id, status, signal_preview)

    analytical_capsules = generate_all_capsules(definitions, progress_callback=_progress)
    analytical_count = _persist_analytical(analytical_capsules)
    return BuildSummary(analytical_count=analytical_count, statuses=statuses)


def schema_refresh(progress_callback=None) -> BuildSummary:
    """Detect schema change and rebuild analytical, schema, and related collections to align with current schema."""
    current_schema = get_schema_metadata()
    current_relationships = get_fk_relationships()
    current_fingerprint = compute_schema_fingerprint(current_schema, current_relationships)
    previous_fingerprint = load_schema_fingerprint()
    schema_changed = previous_fingerprint is None or previous_fingerprint.fingerprint != current_fingerprint

    summary = generate_all_capsule_collections(progress_callback=progress_callback)
    summary.schema_changed = schema_changed
    return summary


def collection_stats() -> dict[str, int]:
    """Return collection counts for the UI."""
    return collection_counts()
