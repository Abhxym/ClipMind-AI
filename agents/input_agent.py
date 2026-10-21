"""
Input Agent — ClipMind AI
================================
Pipeline Step 1: Validate and register uploaded video clips.
  - Confirms all clip paths exist and are readable
  - Probes each clip with FFmpeg to extract duration, resolution, codec, fps
  - Raises structured errors for missing/corrupt files
  - Populates state with normalised clip metadata
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.ffmpeg import probe


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")) * 1024 * 1024


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
        meta = probe(clip)
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
