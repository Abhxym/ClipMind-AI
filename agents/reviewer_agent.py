"""
Reviewer Agent — ClipMind AI
=======================================
Pipeline Step 11: Quality Review & Feedback Loop.
  - Checks the edited video exists
  - Validates that at least 70% of trim commands succeeded
  - Probes the actual duration and compares it with the target (±25%)
  - Sets review_passed; the graph routes back to editing_agent while it is
    False and retries remain (MAX_AGENT_RETRIES), otherwise on to export
  - Only problems a re-render can fix (missing output, failed trims) block;
    a duration off target is a warning, since re-editing the same plan
    cannot make the footage longer
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.ffmpeg import probe

DURATION_TOLERANCE_PCT = 0.25   # Allow ±25% of target duration
MIN_EDIT_SUCCESS_RATE = 0.70    # At least 70% of trims must succeed
MAX_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))


def reviewer_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Reviewer / Quality Gatekeeper Agent.
    Reads:  _edited_video_path, editing_commands, target_duration, retry_count
    Writes: review_passed, errors (review issues), logs
    """
    logs: List[str] = ["[reviewer_agent] Reviewing edited video quality"]
    review_errors: List[Dict[str, Any]] = []

    edited_path = state.get("_edited_video_path")
    target_duration = float(state.get("target_duration", 30))
    editing_commands = state.get("editing_commands", [])
    retry_count = state.get("retry_count", {})
    current_retries = retry_count.get("editing_agent", 0)

    # ── Check 1: File exists ─────────────────────────────────────────────────
    if not edited_path or not Path(edited_path).exists():
        review_errors.append({
            "agent": "reviewer_agent",
            "type": "MissingOutput",
            "message": f"Edited video not found: {edited_path}",
        })
        logs.append("[reviewer_agent] ✗ FAIL: Edited video file is missing")
    else:
        # ── Check 2: Duration within tolerance ────────────────────────────────
        meta = probe(edited_path)
        actual_dur = float(meta.get("duration", 0.0))
        min_dur = target_duration * (1 - DURATION_TOLERANCE_PCT)
        max_dur = target_duration * (1 + DURATION_TOLERANCE_PCT)

        if actual_dur < min_dur:
            logs.append(
                f"[reviewer_agent] ⚠ WARN: Too short — {actual_dur:.1f}s < {min_dur:.1f}s "
                "(not enough usable footage for the target; will still export)"
            )
        elif actual_dur > max_dur:
            # Too long is a warning, not a failure
            logs.append(f"[reviewer_agent] ⚠ WARN: Too long — {actual_dur:.1f}s > {max_dur:.1f}s (will still export)")
        else:
            logs.append(f"[reviewer_agent] ✓ Duration OK: {actual_dur:.1f}s (target={target_duration}s)")

        # ── Check 3: Resolution ───────────────────────────────────────────────
        width, height = int(meta.get("width", 0)), int(meta.get("height", 0))
        if width > 0 and height > 0:
            logs.append(f"[reviewer_agent] ✓ Resolution: {width}×{height}")
        else:
            logs.append(f"[reviewer_agent] ⚠ Could not read resolution")

        # ── Check 4: Edit success rate ────────────────────────────────────────
        trim_cmds = [c for c in editing_commands if c.get("step") == "trim"]
        if trim_cmds:
            success_count = sum(1 for c in trim_cmds if c.get("success"))
            success_rate = success_count / len(trim_cmds)
            if success_rate < MIN_EDIT_SUCCESS_RATE:
                review_errors.append({
                    "agent": "reviewer_agent",
                    "type": "LowEditSuccessRate",
                    "message": (
                        f"Only {success_count}/{len(trim_cmds)} trim commands succeeded "
                        f"({success_rate:.0%}, min required: {MIN_EDIT_SUCCESS_RATE:.0%})"
                    ),
                })
                logs.append(f"[reviewer_agent] ✗ FAIL: Edit success rate {success_rate:.0%} below threshold")
            else:
                logs.append(f"[reviewer_agent] ✓ Edit success rate: {success_count}/{len(trim_cmds)} ({success_rate:.0%})")

    # ── Final decision ────────────────────────────────────────────────────────
    if not review_errors:
        logs.append("[reviewer_agent] ✓ PASS — Video approved for export")
        return {**state, "review_passed": True, "current_agent": "reviewer_agent", "errors": [], "logs": logs}

    if current_retries >= MAX_RETRIES:
        logs.append(
            f"[reviewer_agent] ✗ FAIL but max retries ({MAX_RETRIES}) reached — "
            "exporting the best available output"
        )
    else:
        logs.append(
            f"[reviewer_agent] ✗ FAIL — {len(review_errors)} issue(s) found. "
            f"Sending back to editing_agent (retry {current_retries + 1}/{MAX_RETRIES})"
        )
    return {
        **state,
        "review_passed": False,
        "current_agent": "reviewer_agent",
        "errors": review_errors,
        "logs": logs,
    }
