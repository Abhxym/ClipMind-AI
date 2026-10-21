"""
Evaluation Agent — ClipMind AI
================================
Pipeline Step 13: Run-level evaluation.
  - Duration accuracy of the final video vs. the target
  - Edit success rate (successful trims / planned segments)
  - Counts: scenes, captions, transcript characters, errors, review retries
  - Saves the scores to outputs/<job_id>_eval.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.ffmpeg import get_duration

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))


def evaluation_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Evaluation Agent.
    Reads:  final_video, story_plan, captions, transcript, editing_commands, errors, retry_count
    Writes: evaluation_score
    """
    job_id = state.get("job_id", "unknown")
    logs: List[str] = ["[evaluation_agent] Running evaluation"]

    final_video = state.get("final_video")
    target = state.get("target_duration", 30)
    editing_cmds = state.get("editing_commands", [])
    retry_count = state.get("retry_count", {})

    actual_dur = get_duration(final_video) if final_video and Path(final_video).exists() else 0.0
    dur_score = max(0.0, 1.0 - abs(actual_dur - target) / max(target, 1))
    trim_cmds = [c for c in editing_cmds if c.get("step") == "trim"]
    trim_ok = sum(1 for c in trim_cmds if c.get("success"))
    edit_success_rate = trim_ok / max(len(trim_cmds), 1)

    scores: Dict[str, Any] = {
        "duration_accuracy": round(dur_score, 3),
        "edit_success_rate": round(edit_success_rate, 3),
        "overall_score": round((dur_score + edit_success_rate) / 2, 3),
        "actual_duration_s": round(actual_dur, 2),
        "target_duration_s": target,
        "scene_count": len(state.get("story_plan", [])),
        "caption_count": len(state.get("captions", [])),
        "transcript_chars": sum(len(s.get("text", "")) for s in state.get("transcript", [])),
        "brand_docs_used": len(state.get("retrieved_brand_docs", [])),
        "error_count": len(state.get("errors", [])),
        # editing_agent runs once normally; anything above that is a review retry
        "review_retries": max(0, retry_count.get("editing_agent", 0) - 1),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUT_DIR / f"{job_id}_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    for k, v in scores.items():
        logs.append(f"[evaluation_agent] {k}: {v}")
    logs.append(f"[evaluation_agent] Scores saved to {eval_path}")

    return {**state, "evaluation_score": scores, "current_agent": "evaluation_agent", "errors": [], "logs": logs}
