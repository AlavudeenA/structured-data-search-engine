"""Generate schema-context capsules from live schema metadata, FKs, and user configuration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

# Domain-specific schema maps — swap src/business_schema/ to deploy against a new database.
from ..business_schema.capsule_definitions import SCHEMA_DEFINITIONS
from ..database_connection import get_fk_relationships, get_join_paths, get_schema_metadata
from ..embedding import embed_single
from ..models import SchemaContextCapsule

logger = logging.getLogger(__name__)


def _build_capsule(
    capsule_id: str,
    summary: str,
    tables: list[str],
    relevant_columns: list[str],
    recommended_joins: list[str],
    join_columns: list[str],
    recommended_filters: list[str],
    example_questions: list[str],
    sql_template: str,
    tags: list[str],
) -> SchemaContextCapsule:
    generated_at = datetime.now(timezone.utc).isoformat()
    embed_text = (
        f"{summary} Tables: {', '.join(tables)}. Relevant columns: {', '.join(relevant_columns)}. "
        f"Recommended joins: {' | '.join(recommended_joins[:3])}. Example questions: {' | '.join(example_questions[:4])}."
    )
    return SchemaContextCapsule(
        capsule_id=capsule_id,
        summary=summary,
        tables=tables,
        relevant_columns=relevant_columns,
        recommended_joins=recommended_joins,
        join_columns=join_columns,
        recommended_filters=recommended_filters,
        example_questions=example_questions,
        sql_template=sql_template,
        tags=tags,
        generated_at=generated_at,
        vector=embed_single(embed_text),
    )


def generate_schema_capsules() -> list[SchemaContextCapsule]:
    """Build metadata-focused schema capsules for SQL planning."""
    schema = get_schema_metadata()
    fk_relationships = get_fk_relationships()
    join_paths = get_join_paths()

    join_texts = [f"{path['left_table']}.{path['left_column']} = {path['right_table']}.{path['right_column']}" for path in join_paths]
    logger.debug("Schema tables discovered: %s | FK relationships: %s", list(schema), len(fk_relationships))

    capsules: list[SchemaContextCapsule] = []
    
    for definition in SCHEMA_DEFINITIONS:
        # Resolve joins using database FKs if not explicitly defined
        rec_joins = definition.get("recommended_joins", [])
        if not rec_joins and join_texts:
            rec_joins = join_texts[:8]
            
        capsule = _build_capsule(
            capsule_id=definition["capsule_id"],
            summary=definition["summary"],
            tables=definition["tables"],
            relevant_columns=definition["relevant_columns"],
            recommended_joins=rec_joins,
            join_columns=definition["join_columns"],
            recommended_filters=definition.get("recommended_filters", []),
            example_questions=definition.get("example_questions", []),
            sql_template=definition["sql_template"],
            tags=definition["tags"],
        )
        capsules.append(capsule)

    logger.info("Generated %s schema context capsules", len(capsules))
    return capsules

