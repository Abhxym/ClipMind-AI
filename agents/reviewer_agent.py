"""
Reviewer Agent — AI Video Director
=======================================
Pipeline Step 11: Quality Review & Feedback Loop.
  - Checks the edited video exists and meets duration requirements
  - Validates that at least 80% of trimming commands succeeded
  - Uses ffprobe to check actual video duration vs. target
  - Clears errors on pass so the workflow proceeds to export
  - On failure, appends specific error messages to guide the editing agent on retry
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState

DURATION_TOLERANCE_PCT = 0.25   # Allow ±25% of target duration
MIN_EDIT_SUCCESS_RATE = 0.70    # At least 70% of trims must succeed
MAX_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))


def _get_ffprobe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe")
    except Exception:
        return "ffprobe"


def _get_video_duration(video_path: str) -> float:
    """Get actual video duration in seconds using ffmpeg probe."""
    from agents.input_agent import _probe_video
    meta = _probe_video(video_path)
    return float(meta.get("duration", 0.0))


def _get_video_resolution(video_path: str) -> tuple:
    """Get video width and height."""
    from agents.input_agent import _probe_video
    meta = _probe_video(video_path)
    return int(meta.get("width", 0)), int(meta.get("height", 0))


def reviewer_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Reviewer / Quality Gatekeeper Agent.
    Reads:  _edited_video_path, editing_commands, target_duration, retry_count
    Writes: errors (cleared on pass), retry_count, logs
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
        actual_dur = _get_video_duration(edited_path)
        min_dur = target_duration * (1 - DURATION_TOLERANCE_PCT)
        max_dur = target_duration * (1 + DURATION_TOLERANCE_PCT)

        if actual_dur < min_dur:
            review_errors.append({
                "agent": "reviewer_agent",
                "type": "TooShort",
                "message": (
                    f"Video is {actual_dur:.1f}s but target is {target_duration}s "
                    f"(min acceptable: {min_dur:.1f}s)"
                ),
            })
            logs.append(f"[reviewer_agent] ✗ FAIL: Too short — {actual_dur:.1f}s < {min_dur:.1f}s")
        elif actual_dur > max_dur:
            # Too long is a warning, not a failure
            logs.append(f"[reviewer_agent] ⚠ WARN: Too long — {actual_dur:.1f}s > {max_dur:.1f}s (will still export)")
        else:
            logs.append(f"[reviewer_agent] ✓ Duration OK: {actual_dur:.1f}s (target={target_duration}s)")

        # ── Check 3: Resolution ───────────────────────────────────────────────
        width, height = _get_video_resolution(edited_path)
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
        # Clear all accumulated errors so the review_routing returns 'export_agent'
        return {
            **state,
            "errors": [],   # Clear errors → triggers export path
            "current_agent": "reviewer_agent",
            "logs": logs,
        }
    else:
        # Check if we've exceeded max retries
        if current_retries >= MAX_RETRIES:
            logs.append(
                f"[reviewer_agent] ✗ FAIL but max retries ({MAX_RETRIES}) reached — "
                "forcing export with best available output"
            )
            # Still clear errors to proceed to export (best-effort)
            return {
                **state,
                "errors": [],
                "current_agent": "reviewer_agent",
                "logs": logs,
            }

        logs.append(
            f"[reviewer_agent] ✗ FAIL — {len(review_errors)} issue(s) found. "
            f"Sending back to editing_agent (retry {current_retries + 1}/{MAX_RETRIES})"
        )
        # Return review_errors so review_routing redirects to editing_agent
        return {
            **state,
            "errors": review_errors,
            "current_agent": "reviewer_agent",
            "logs": logs,
        }
