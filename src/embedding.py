"""Embedding and persistence helpers for vectors, refresh plans, and schema fingerprints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .app_constants import ANALYTICAL_REFRESH_PLAN_FILE, DATA_DIR, SCHEMA_FINGERPRINT_FILE
from .models import RefreshPlan, SchemaFingerprint

logger = logging.getLogger(__name__)

_model: Any = None


def _fastembed_cache_root() -> Path:
    """Return the local fastembed cache root used on this machine."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "Temp" / "fastembed_cache"
    return Path.cwd() / ".fastembed_cache"


def _clear_fastembed_cache() -> None:
    """Delete the local fastembed cache so a broken model can be re-downloaded."""
    cache_root = _fastembed_cache_root()
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
        logger.warning("Cleared corrupted fastembed cache at %s", cache_root)


def _get_model() -> Any:
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        from .config import get_settings

        settings = get_settings()
        logger.info("Loading fastembed model: %s", settings.embed_model)
        try:
            _model = TextEmbedding(model_name=settings.embed_model)
        except ValueError as exc:
            if "tokenizer_config.json" not in str(exc):
                raise
            logger.warning("fastembed cache appears corrupted: %s", exc)
            _clear_fastembed_cache()
            _model = TextEmbedding(model_name=settings.embed_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts with fastembed."""
    if not texts:
        return []
    model = _get_model()
    vectors = list(model.embed(texts))
    return [vector.tolist() if isinstance(vector, np.ndarray) else list(vector) for vector in vectors]


def embed_single(text: str) -> list[float]:
    """Embed one text string."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def ensure_data_dir() -> None:
    """Ensure local data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_refresh_plan(plans: list[dict[str, Any]]) -> None:
    """Persist the canonical analytical refresh plan."""
    ensure_data_dir()
    payload = RefreshPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        capsule_ids=[plan.get("capsule_id", "") for plan in plans],
        plans=plans,
    )
    ANALYTICAL_REFRESH_PLAN_FILE.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def load_refresh_plan() -> RefreshPlan | None:
    """Load the saved analytical refresh plan."""
    if not ANALYTICAL_REFRESH_PLAN_FILE.exists():
        return None
    try:
        return RefreshPlan.model_validate_json(ANALYTICAL_REFRESH_PLAN_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load refresh plan: %s", exc)
        return None


def compute_schema_fingerprint(
    schema: dict[str, list[dict[str, str]]],
    relationships: list[dict[str, str]],
) -> str:
    """Compute a stable hash for schema metadata and relationships."""
    normalized = json.dumps({"schema": schema, "relationships": relationships}, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def save_schema_fingerprint(
    schema: dict[str, list[dict[str, str]]],
    relationships: list[dict[str, str]],
) -> SchemaFingerprint:
    """Persist schema fingerprint and source metadata."""
    ensure_data_dir()
    payload = SchemaFingerprint(
        generated_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=compute_schema_fingerprint(schema, relationships),
        tables=schema,
        relationships=relationships,
    )
    SCHEMA_FINGERPRINT_FILE.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload


def load_schema_fingerprint() -> SchemaFingerprint | None:
    """Load the last persisted schema fingerprint."""
    if not SCHEMA_FINGERPRINT_FILE.exists():
        return None
    try:
        return SchemaFingerprint.model_validate_json(SCHEMA_FINGERPRINT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load schema fingerprint: %s", exc)
        return None
