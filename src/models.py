"""Pydantic models used across generation, retrieval, SQL, and UI layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CapsuleDefinition(BaseModel):
    """Static analytical capsule definition."""

    capsule_id: str
    capsule_type: str
    priority: str
    what: str
    how: str
    sql: str
    signal_method: str
    embed_text_template: str
    ttl_hours: int
    tags: list[str]
    tables_used: list[str]
    key_columns: list[str]
    staleness_trigger: str
    related_capsule_ids: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


class GeneratedCapsule(BaseModel):
    """Analytical capsule after SQL execution, enrichment, and embedding."""

    capsule_id: str
    capsule_type: str
    priority: str
    what: str
    how: str
    sql: str
    signal_method: str
    signal: str
    embed_text: str
    tables_used: list[str]
    key_columns: list[str]
    tags: list[str]
    ttl_hours: int
    generated_at: str
    expires_at: str
    is_stale: bool = False
    staleness_trigger: str
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    anomaly_score: float = 0.0
    trend_direction: str = "flat"
    related_capsule_ids: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)
    vector: list[float] | None = None


class SchemaContextCapsule(BaseModel):
    """Schema planning capsule used for SQL guidance."""

    capsule_id: str
    summary: str
    tables: list[str]
    relevant_columns: list[str]
    recommended_joins: list[str]
    join_columns: list[str]
    recommended_filters: list[str]
    example_questions: list[str]
    sql_template: str
    tags: list[str]
    generated_at: str
    vector: list[float] | None = None


class RelatedCapsule(BaseModel):
    """Related risk capsule generated from capsule relationships and signals."""

    capsule_id: str
    related_from: list[str]
    signal: str
    embed_text: str
    entity_type: str
    entity_name: str
    risk_level: str
    generated_at: str
    tags: list[str]
    vector: list[float] | None = None


class IntentResult(BaseModel):
    """Intent detection output."""

    intent: str
    confidence: float
    structured_parts: list[str] = Field(default_factory=list)
    analytical_parts: list[str] = Field(default_factory=list)
    reasoning: str


class CapsuleHit(BaseModel):
    """Single vector-search hit."""

    capsule_id: str
    score: float
    collection: str
    payload: dict[str, Any]


class ContextPackage(BaseModel):
    """Packaged multi-source capsule context for answering or SQL planning."""

    primary_capsule: dict[str, Any] | None = None
    linked_capsules: list[dict[str, Any]] = Field(default_factory=list)
    related_capsules: list[dict[str, Any]] = Field(default_factory=list)
    schema_capsules: list[dict[str, Any]] = Field(default_factory=list)
    combined_context: str = ""
    overall_confidence: float = 0.0


class QueryResponse(BaseModel):
    """Pipeline response returned to UI and CLI."""

    answer: str
    route_taken: str
    intent: str
    confidence: float
    capsules_used: list[str] = Field(default_factory=list)
    sql_generated: str | None = None
    sql_rows: list[dict[str, Any]] | None = None
    context_package: dict[str, Any] | None = None
    autofix_used: bool = False
    error: str | None = None
    answer_ms: int = 0
    sql_reason: str | None = None
    intent_payload: dict[str, Any] | None = None


class RelationshipEdge(BaseModel):
    """Directed graph edge between two capsules."""

    from_id: str
    to_id: str
    relationship: str
    join_key: str | None = None


class CapsuleGraph(BaseModel):
    """Persisted capsule relationship graph."""

    built_at: str
    edges: list[RelationshipEdge] = Field(default_factory=list)
    related_capsule_ids: list[str] = Field(default_factory=list)


class BuildStatus(BaseModel):
    """Per-capsule build progress record."""

    capsule_id: str
    status: str
    signal_preview: str = ""
    error: str | None = None


class BuildSummary(BaseModel):
    """Capsule build or refresh result summary."""

    analytical_count: int = 0
    schema_count: int = 0
    related_count: int = 0
    graph_edge_count: int = 0
    statuses: list[BuildStatus] = Field(default_factory=list)
    schema_changed: bool | None = None


class RefreshPlan(BaseModel):
    """Canonical plan for analytical data refresh."""

    generated_at: str
    capsule_ids: list[str]
    plans: list[dict[str, Any]]


class SchemaFingerprint(BaseModel):
    """Persisted schema fingerprint and supporting metadata."""

    generated_at: str
    fingerprint: str
    tables: dict[str, list[dict[str, str]]]
    relationships: list[dict[str, str]]
