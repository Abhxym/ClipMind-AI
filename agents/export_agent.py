"""
Export Agent — AI Video Director
==================================
Final step: copies the edited video to the outputs folder,
generates a thumbnail, writes metadata JSON, and computes
evaluation metrics for the pipeline run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from logs.agent_logs import AgentLogger
from memory.langgraph_state import EvaluationMetrics, VideoDirectorState


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _generate_thumbnail(video_path: str, out_path: str) -> bool:
    """Extract the first frame as a thumbnail."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "scale=640:-1",
        out_path,
        "-loglevel", "error",
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _compute_evaluation(state: VideoDirectorState) -> EvaluationMetrics:
    """Aggregate metrics from state for evaluation logging."""
    timings = state.get("agent_timings", {})
    total_ms = sum(timings.values())

    retrieved = len(state.get("retrieved_doc_ids", []))
    reranked = len(state.get("reranked_doc_ids", []))
    retrieval_precision = reranked / retrieved if retrieved > 0 else 0.0

    # Reranker lift: ratio of reduction in result set
    reranker_lift = (
        (retrieved - reranked) / retrieved if retrieved > 0 else 0.0
    )

    captions = state.get("captions", [])
    avg_caption_len = sum(len(c) for c in captions) / max(1, len(captions))

    return EvaluationMetrics(
        scene_selection_count=len(state.get("story_arc", [])),
        scenes_filtered=state.get("filtered_count", 0),
        retrieval_precision_at_k=retrieval_precision,
        reranker_lift=reranker_lift,
        caption_length_avg=avg_caption_len,
        latency_total_ms=total_ms,
        latency_per_agent=timings,
        llm_tokens_used=0,   # populated by LangSmith callback if enabled
        error_count=len(state.get("errors", [])),
    )


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def export_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Export Agent.
    Reads:  state.edited_video_path, state.job_id, state.platform, state.video_title
    Writes: state.final_video_path, state.final_thumbnail_path,
            state.export_metadata, state.evaluation
    """
    job_id = state["job_id"]
    log = AgentLogger("export_agent", job_id)
    log.start("Exporting final reel")

    edited_path = state.get("edited_video_path")
    if not edited_path or not Path(edited_path).exists():
        raise RuntimeError("No edited video found for export")

    # ── Copy to outputs ──────────────────────────────────────
    safe_title = (
        state.get("video_title", "reel")
        .replace(" ", "_")
        .replace("/", "_")[:40]
    )
    final_name = f"{job_id}_{safe_title}.mp4"
    final_path = str(OUTPUT_DIR / final_name)
    shutil.copy2(edited_path, final_path)
    log.info(f"Final video: {final_path}")

    # ── Thumbnail ────────────────────────────────────────────
    thumb_path = str(OUTPUT_DIR / f"{job_id}_thumb.jpg")
    _generate_thumbnail(final_path, thumb_path)

    # ── Metadata JSON ────────────────────────────────────────
    metadata: Dict[str, Any] = {
        "job_id": job_id,
        "title": state.get("video_title", ""),
        "platform": state.get("platform", ""),
        "planned_duration": state.get("planned_duration", 0),
        "scene_count": len(state.get("story_arc", [])),
        "captions": state.get("captions", []),
        "hashtags": state.get("hashtags", []),
        "tone": state.get("overall_tone", ""),
        "brand_style": state.get("brand_style"),
        "agent_timings": state.get("agent_timings", {}),
        "completed_agents": state.get("completed_agents", []),
        "errors": [e for e in state.get("errors", [])],
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = str(OUTPUT_DIR / f"{job_id}_metadata.json")
    Path(meta_path).write_text(json.dumps(metadata, indent=2, default=str))
    log.info(f"Metadata: {meta_path}")

    # ── Evaluation ───────────────────────────────────────────
    evaluation = _compute_evaluation(state)
    eval_path = str(OUTPUT_DIR / f"{job_id}_evaluation.json")
    Path(eval_path).write_text(json.dumps(evaluation, indent=2))

    log.success(
        f"Export complete | "
        f"duration={state.get('planned_duration', 0):.1f}s | "
        f"latency={evaluation['latency_total_ms']:.0f}ms"
    )

    return {
        **state,
        "final_video_path": final_path,
        "final_thumbnail_path": thumb_path,
        "export_metadata": metadata,
        "evaluation": evaluation,
        "current_agent": "export_agent",
        "completed_agents": ["export_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "export_agent": log.elapsed_ms(),
        },
    }
