"""CRUD for user-created capsule definitions stored in data/user_capsules.json."""

from __future__ import annotations

import json
import logging

from .app_constants import DATA_DIR

logger = logging.getLogger(__name__)

_USER_CAPSULES_FILE = DATA_DIR / "user_capsules.json"


def load_user_capsule_defs() -> list[dict]:
    if not _USER_CAPSULES_FILE.exists():
        return []
    try:
        raw: list[dict] = json.loads(_USER_CAPSULES_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse user_capsules.json: %s", exc)
        return []
    # Validate each entry against CapsuleDefinition; drop malformed ones with a clear warning.
    from .models import CapsuleDefinition
    valid: list[dict] = []
    for entry in raw:
        try:
            CapsuleDefinition(**entry)
            valid.append(entry)
        except Exception as exc:
            logger.warning("Skipping malformed user capsule '%s': %s", entry.get("capsule_id", "?"), exc)
    return valid


def save_user_capsule_def(definition: dict) -> None:
    existing = load_user_capsule_defs()
    existing_ids = {d["capsule_id"] for d in existing}
    if definition["capsule_id"] in existing_ids:
        existing = [definition if d["capsule_id"] == definition["capsule_id"] else d for d in existing]
    else:
        existing.append(definition)
    _USER_CAPSULES_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def delete_user_capsule_def(capsule_id: str) -> bool:
    existing = load_user_capsule_defs()
    updated = [d for d in existing if d["capsule_id"] != capsule_id]
    if len(updated) == len(existing):
        return False
    _USER_CAPSULES_FILE.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    return True
