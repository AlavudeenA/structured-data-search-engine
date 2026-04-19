"""Build a single user-created capsule and upsert it into the analytical collection."""

from __future__ import annotations

import logging

from ..models import CapsuleDefinition
from .capsule_generator import generate_all_capsules
from .store_manager import _persist_analytical

logger = logging.getLogger(__name__)


def build_single_capsule(capsule_def_dict: dict) -> bool:
    """Build and upsert one capsule. Used after a user creates a capsule via the UI."""
    try:
        definition = CapsuleDefinition(**capsule_def_dict)
        capsules = generate_all_capsules([definition])
        if capsules:
            _persist_analytical(capsules)
            return True
        return False
    except Exception as exc:
        logger.error("Failed to build capsule %s: %s", capsule_def_dict.get("capsule_id"), exc)
        return False
