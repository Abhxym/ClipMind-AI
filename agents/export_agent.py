"""
Export Agent — ClipMind AI
============================
Pipeline Step 12: Final export.
  - Copies the reviewed edit to outputs/<job_id>_final.mp4
  - Extracts a JPEG thumbnail at 20% of the runtime (outputs/<job_id>_thumb.jpg)
  - Writes title, hashtags, tone, music mood and CTA to outputs/<job_id>_meta.json
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.ffmpeg import extract_thumbnail

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))


def export_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Export Agent.
    Reads:  _edited_video_path, caption metadata
    Writes: final_video
    """
    job_id = state.get("job_id", "unknown")
    logs: List[str] = [f"[export_agent] Exporting job {job_id}"]
    errors: List[Dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src = state.get("_edited_video_path")
    if not src or not Path(src).exists():
        errors.append({"agent": "export_agent", "type": "MissingSource", "message": f"No edited video to export ({src})"})
        return {**state, "current_agent": "export_agent", "errors": errors, "logs": logs}

    dest = OUTPUT_DIR / f"{job_id}_final.mp4"
    shutil.copy2(src, str(dest))

    thumb = OUTPUT_DIR / f"{job_id}_thumb.jpg"
    if extract_thumbnail(str(dest), str(thumb)):
        logs.append(f"[export_agent] Thumbnail: {thumb}")
    else:
        logs.append("[export_agent] ⚠ Could not extract thumbnail")

    meta = {
        "job_id": job_id,
        "platform": state.get("platform"),
        "target_duration": state.get("target_duration"),
        "video_title": state.get("_video_title") or "AI Reel",
        "hashtags": state.get("_hashtags", []),
        "tone": state.get("_overall_tone"),
        "music_mood": state.get("_music_mood") or "upbeat",
        "cta": state.get("_cta") or "",
    }
    with open(OUTPUT_DIR / f"{job_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logs.append(f"[export_agent] Done: {dest}")
    return {**state, "final_video": str(dest), "current_agent": "export_agent", "errors": errors, "logs": logs}
