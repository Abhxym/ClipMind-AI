"""
Input Agent — AI Video Director
================================
Pipeline Step 1: Validate and register uploaded video clips.
  - Confirms all clip paths exist and are readable
  - Uses ffprobe to extract duration, resolution, codec, fps
  - Raises structured errors for missing/corrupt files
  - Populates state with normalised clip metadata
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")) * 1024 * 1024


def _get_ffprobe() -> str:
    import shutil
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        ffp = ff.replace("ffmpeg", "ffprobe")
        if Path(ffp).exists():
            return ffp
    except Exception:
        pass
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    # Fall back to ffmpeg itself — supports all -show_* probing flags
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _probe_video(path: str) -> Dict[str, Any]:
    """Run ffmpeg -i and parse its stderr to extract video metadata."""
    try:
        import re
        cmd = [_get_ffprobe(), "-i", path]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        info = r.stderr.decode("utf-8", errors="replace")

        if not info:
            return {"path": path, "error": "No output from ffmpeg"}

        # Duration: 00:00:19.71
        dur_match = re.search(r"Duration:\s+(\d+):(\d+):([\d.]+)", info)
        duration = 0.0
        if dur_match:
            h, m, s = dur_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)

        # Video stream: 1920x1080, 24 fps
        vid_match = re.search(r"Stream.*Video.*?(\d{2,5})x(\d{2,5}).*?([\d.]+)\s+fps", info)
        width, height, fps = 0, 0, 0.0
        codec = "unknown"
        if vid_match:
            width  = int(vid_match.group(1))
            height = int(vid_match.group(2))
            fps    = float(vid_match.group(3))
        # Codec name from Video stream
        codec_match = re.search(r"Video:\s+(\w+)", info)
        if codec_match:
            codec = codec_match.group(1)

        # Audio stream presence
        has_audio = "Audio:" in info

        size_bytes = Path(path).stat().st_size

        return {
            "path": path,
            "duration": round(duration, 3),
            "size_bytes": size_bytes,
            "width": width,
            "height": height,
            "codec": codec,
            "fps": fps,
            "has_audio": has_audio,
        }
    except Exception as exc:
        return {"path": path, "error": str(exc)}


def input_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Input Validation Agent.
    Reads:  uploaded_videos, user_prompt
    Writes: uploaded_videos (validated), logs, errors
    """
    logs: List[str] = ["[input_agent] Validating uploaded clips"]
    errors: List[Dict[str, Any]] = []
    validated: List[str] = []

    clips = state.get("uploaded_videos", [])
    if not clips:
        errors.append({
            "agent": "input_agent",
            "type": "NoClips",
            "message": "No video clips were uploaded.",
        })
        return {**state, "current_agent": "input_agent", "errors": errors, "logs": logs}

    clip_metadata: List[Dict[str, Any]] = []

    for clip in clips:
        p = Path(clip)

        # Check extension
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            errors.append({
                "agent": "input_agent",
                "type": "InvalidExtension",
                "message": f"{p.name}: unsupported file type '{p.suffix}'",
            })
            continue

        # Check file exists
        if not p.exists():
            errors.append({
                "agent": "input_agent",
                "type": "FileNotFound",
                "message": f"File not found: {clip}",
            })
            continue

        # Check file size
        size = p.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            errors.append({
                "agent": "input_agent",
                "type": "FileTooLarge",
                "message": f"{p.name}: {size // (1024*1024)} MB exceeds limit",
            })
            continue

        # Probe video
        meta = _probe_video(clip)
        if "error" in meta:
            errors.append({
                "agent": "input_agent",
                "type": "ProbeError",
                "message": f"{p.name}: {meta['error']}",
            })
            continue

        duration = meta.get("duration", 0)
        if duration < 0.5:
            errors.append({
                "agent": "input_agent",
                "type": "TooShort",
                "message": f"{p.name}: duration {duration:.2f}s is too short",
            })
            continue

        validated.append(clip)
        clip_metadata.append(meta)
        logs.append(
            f"[input_agent] ✓ {p.name} — {duration:.1f}s | "
            f"{meta['width']}×{meta['height']} | {meta['fps']}fps | "
            f"audio={'yes' if meta['has_audio'] else 'no'}"
        )

    logs.append(f"[input_agent] Validated {len(validated)}/{len(clips)} clips")

    return {
        **state,
        "uploaded_videos": validated,
        "_clip_metadata": clip_metadata,      # extra metadata carried through state
        "current_agent": "input_agent",
        "errors": errors,
        "logs": logs,
    }
