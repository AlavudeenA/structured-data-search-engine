"""Compare two sets of capsule snapshots (Baseline vs Comparison Period) via LLM."""

from __future__ import annotations

import logging
from datetime import date

from ..llm_instructions import ACTIVITY_COMPARISON_SYSTEM, ACTIVITY_COMPARISON_USER
from ..llm_service import call_llm

logger = logging.getLogger(__name__)

_MAX_SIGNAL_CHARS = 600
_MAX_ROWS_PER_CAPSULE = 3


def _capsule_block(label: str, capsule: dict) -> str:
    signal = (capsule.get("signal") or capsule.get("what") or "")[:_MAX_SIGNAL_CHARS]
    anomaly = capsule.get("anomaly_score", "n/a")
    trend = capsule.get("trend_direction", "n/a")
    rows = capsule.get("result_rows") or []
    rows_preview = str(rows[:_MAX_ROWS_PER_CAPSULE]) if rows else "no rows"
    return (
        f"[{label}]\n"
        f"  Signal   : {signal}\n"
        f"  Anomaly  : {anomaly}  |  Trend: {trend}\n"
        f"  Top rows : {rows_preview}"
    )


def _fmt_date_range(start: date, end: date) -> str:
    return f"{start.strftime('%m/%d/%Y')} – {end.strftime('%m/%d/%Y')}"


def compare_periods(
    baseline_capsules: dict[str, dict],
    comparison_capsules: dict[str, dict],
    baseline_start: date,
    baseline_end: date,
    comparison_start: date,
    comparison_end: date,
) -> str:
    """Call LLM to compare baseline vs comparison capsule snapshots.

    Each capsule gets a per-capsule diff block. Capsules only in one side are
    noted explicitly. The LLM concludes with an Overall Summary.
    """
    all_ids = sorted(set(baseline_capsules) | set(comparison_capsules))
    diff_blocks: list[str] = []

    for cid in all_ids:
        b = baseline_capsules.get(cid)
        c = comparison_capsules.get(cid)
        block = f"--- Capsule: {cid} ---\n"
        if b and c:
            block += _capsule_block("Baseline", b) + "\n"
            block += _capsule_block("Comparison", c)
        elif b:
            block += _capsule_block("Baseline", b) + "\n"
            block += "[Comparison] Not present — no longer active or not refreshed in Comparison Period."
        else:
            block += "[Baseline] Not present — new or first-changed capsule in Comparison Period.\n"
            block += _capsule_block("Comparison", c)
        diff_blocks.append(block)

    capsule_diffs = "\n\n".join(diff_blocks)
    baseline_label = _fmt_date_range(baseline_start, baseline_end)
    comparison_label = _fmt_date_range(comparison_start, comparison_end)

    user_prompt = ACTIVITY_COMPARISON_USER.format(
        baseline_label=baseline_label,
        comparison_label=comparison_label,
        capsule_diffs=capsule_diffs,
    )

    logger.info(
        "Activity comparison | baseline=%s capsules | comparison=%s capsules",
        len(baseline_capsules),
        len(comparison_capsules),
    )
    return call_llm(
        system_prompt=ACTIVITY_COMPARISON_SYSTEM,
        user_prompt=user_prompt,
        model_slot="groq_analytical_model",
        max_tokens=2048,
    )
