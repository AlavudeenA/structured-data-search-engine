"""Capsule snapshot history — save, load, and prune dated snapshots.

Snapshots are stored under:
  data/capsule_history/MM-DD-YYYY/{capsule_id}.json

A folder is created only when a data change is detected and capsules are
refreshed. Within the same day, a later refresh overwrites the earlier
snapshot for the same capsule, keeping only the latest state for that day.
Folders older than CAPSULE_HISTORY_RETENTION_DAYS are pruned automatically.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

from .app_constants import CAPSULE_HISTORY_DIR, CAPSULE_HISTORY_RETENTION_DAYS

logger = logging.getLogger(__name__)

_DATE_FMT = "%m-%d-%Y"


def _today_folder() -> Path:
    return CAPSULE_HISTORY_DIR / datetime.now().strftime(_DATE_FMT)


def save_capsule_snapshot(capsules: list) -> None:
    """Write each capsule to today's history folder.

    Existing files for the same capsule_id are overwritten (latest wins).
    Prunes folders beyond the retention window after writing.
    """
    if not capsules:
        return
    folder = _today_folder()
    folder.mkdir(parents=True, exist_ok=True)
    for capsule in capsules:
        try:
            if hasattr(capsule, "model_dump"):
                data = capsule.model_dump()
            elif isinstance(capsule, dict):
                data = capsule
            else:
                continue
            capsule_id = data.get("capsule_id", "")
            if not capsule_id:
                continue
            (folder / f"{capsule_id}.json").write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to snapshot capsule: %s", exc)
    _prune_old_snapshots()


def load_capsules_in_range(start: date, end: date) -> dict[str, dict]:
    """Return unique capsules (latest version) found in snapshot folders within [start, end].

    Iterates folders in chronological order so a later date always overwrites
    an earlier one for the same capsule_id.
    """
    result: dict[str, dict] = {}
    if not CAPSULE_HISTORY_DIR.exists():
        return result
    for folder in sorted(CAPSULE_HISTORY_DIR.iterdir()):
        if not folder.is_dir():
            continue
        try:
            folder_date = datetime.strptime(folder.name, _DATE_FMT).date()
        except ValueError:
            continue
        if start <= folder_date <= end:
            for json_file in sorted(folder.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    capsule_id = data.get("capsule_id", json_file.stem)
                    result[capsule_id] = data
                except Exception as exc:
                    logger.warning("Failed to load snapshot %s: %s", json_file, exc)
    return result


def available_snapshot_dates() -> list[date]:
    """Return sorted list of dates for which non-empty snapshot folders exist."""
    dates: list[date] = []
    if not CAPSULE_HISTORY_DIR.exists():
        return dates
    for folder in CAPSULE_HISTORY_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            d = datetime.strptime(folder.name, _DATE_FMT).date()
            if any(folder.glob("*.json")):
                dates.append(d)
        except ValueError:
            continue
    return sorted(dates)


def _prune_old_snapshots() -> None:
    cutoff = datetime.now().date() - timedelta(days=CAPSULE_HISTORY_RETENTION_DAYS)
    if not CAPSULE_HISTORY_DIR.exists():
        return
    for folder in CAPSULE_HISTORY_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            folder_date = datetime.strptime(folder.name, _DATE_FMT).date()
            if folder_date < cutoff:
                shutil.rmtree(folder)
                logger.info("Pruned capsule history folder: %s", folder.name)
        except ValueError:
            continue
