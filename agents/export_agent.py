from __future__ import annotations
import json, os, shutil, subprocess
from pathlib import Path
from typing import Any, Dict
from memory.state import VideoState

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))

def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def _thumb(video_path: str, thumb_path: str) -> bool:
    try:
        probe = [_ffmpeg().replace("ffmpeg","ffprobe"),"-v","quiet","-print_format","json","-show_format",video_path]
        r = subprocess.run(probe, capture_output=True, text=True, timeout=10)
        dur = float(json.loads(r.stdout).get("format",{}).get("duration", 5))
        cmd = [_ffmpeg(),"-y","-ss",str(dur*0.2),"-i",video_path,"-vframes","1","-vf","scale=720:-1",thumb_path,"-loglevel","error"]
        subprocess.run(cmd, timeout=30)
        return Path(thumb_path).exists()
    except Exception:
        return False

def export_agent_node(state: VideoState) -> VideoState:
    job_id = state.get("job_id", "unknown")
    logs = [f"[export_agent] Exporting job {job_id}"]
    errors = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src = state.get("_edited_video_path")
    if not src or not Path(src).exists():
        errors.append({"agent": "export_agent", "type": "MissingSource", "message": f"Source not found: {src}"})
        return {**state, "current_agent": "export_agent", "errors": errors, "logs": logs}
    dest = OUTPUT_DIR / f"{job_id}_final.mp4"
    shutil.copy2(src, str(dest))
    thumb = OUTPUT_DIR / f"{job_id}_thumb.jpg"
    _thumb(str(dest), str(thumb))
    meta = {
        "job_id": job_id, "platform": state.get("platform"),
        "target_duration": state.get("target_duration"),
        "video_title": state.get("_video_title", "AI Reel"),
        "hashtags": state.get("_hashtags", []),
        "music_mood": state.get("_music_mood", "upbeat"),
    }
    with open(OUTPUT_DIR / f"{job_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logs.append(f"[export_agent] Done: {dest}")
    return {**state, "final_video": str(dest), "current_agent": "export_agent", "errors": errors, "logs": logs}
