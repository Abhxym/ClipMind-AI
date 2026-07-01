from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from typing import Any, Dict
from memory.state import VideoState

def _duration(path: str) -> float:
    try:
        import imageio_ffmpeg
        probe = imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg","ffprobe")
        r = subprocess.run([probe,"-v","quiet","-print_format","json","-show_format",path], capture_output=True, text=True, timeout=10)
        return float(json.loads(r.stdout).get("format",{}).get("duration",0))
    except Exception:
        return 0.0

def evaluation_agent_node(state: VideoState) -> VideoState:
    job_id = state.get("job_id", "unknown")
    logs = ["[evaluation_agent] Running evaluation"]
    errors = []
    final_video = state.get("final_video")
    target = state.get("target_duration", 30)
    story_plan = state.get("story_plan", [])
    transcript = state.get("transcript", [])
    captions = state.get("captions", [])
    editing_cmds = state.get("editing_commands", [])
    all_errors = state.get("errors", [])
    retry_count = state.get("retry_count", {})

    actual_dur = _duration(final_video) if final_video and Path(final_video).exists() else 0.0
    dur_score = max(0.0, 1.0 - abs(actual_dur - target) / max(target, 1))
    scene_count = len(story_plan)
    caption_count = len(captions)
    transcript_len = sum(len(s.get("text","")) for s in transcript)
    error_count = len(all_errors)
    total_retries = sum(retry_count.values())
    trim_ok = sum(1 for c in editing_cmds if c.get("step") == "trim" and c.get("success"))
    trim_total = sum(1 for c in editing_cmds if c.get("step") == "trim")
    edit_success_rate = trim_ok / max(trim_total, 1)

    scores = {
        "duration_accuracy": round(dur_score, 3),
        "scene_count": scene_count,
        "caption_count": caption_count,
        "transcript_chars": transcript_len,
        "edit_success_rate": round(edit_success_rate, 3),
        "error_count": error_count,
        "review_retries": total_retries,
        "actual_duration_s": round(actual_dur, 2),
        "target_duration_s": target,
        "overall_score": round((dur_score + edit_success_rate) / 2, 3),
    }

    meta_path = Path("outputs") / f"{job_id}_eval.json"
    meta_path.parent.mkdir(exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(scores, f, indent=2)

    for k, v in scores.items():
        logs.append(f"[evaluation_agent] {k}: {v}")
    logs.append(f"[evaluation_agent] Scores saved to {meta_path}")

    return {**state, "evaluation_score": scores, "current_agent": "evaluation_agent", "errors": [], "logs": logs}
