"""
Preprocess Agent — ClipMind AI
======================================
Pipeline Step 2: Frame extraction and video decomposition.
  - Samples frames at configurable FPS (default 1fps)
  - Saves JPEG keyframes to temp/<job_id>/frames/
  - Computes per-frame blur (Laplacian variance) and brightness scores
  - Builds a list of frame dicts for downstream agents
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.ffmpeg import get_duration, get_ffmpeg

FRAME_SAMPLE_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "1"))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "30.0"))
BRIGHTNESS_MIN = int(os.getenv("BRIGHTNESS_MIN", "20"))
BRIGHTNESS_MAX = int(os.getenv("BRIGHTNESS_MAX", "240"))


def _extract_frames(clip_path: str, out_dir: Path, fps: float) -> List[str]:
    """Extract frames at `fps` rate into out_dir. Returns list of saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "frame_%05d.jpg")
    cmd = [
        get_ffmpeg(), "-y", "-i", clip_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",
        template,
        "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except Exception:
        pass
    return sorted(str(p) for p in out_dir.glob("frame_*.jpg"))


def _score_frame(frame_path: str) -> Dict[str, float]:
    """Compute blur (Laplacian variance) and brightness for a frame."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(frame_path)
        if img is None:
            return {"blur": 0.0, "brightness": 0.0, "quality_ok": False}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        quality_ok = (blur >= BLUR_THRESHOLD and BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX)
        return {"blur": round(blur, 2), "brightness": round(brightness, 2), "quality_ok": quality_ok}
    except ImportError:
        return {"blur": -1.0, "brightness": -1.0, "quality_ok": True}
    except Exception:
        return {"blur": 0.0, "brightness": 0.0, "quality_ok": False}


def preprocess_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Preprocessing Agent.
    Reads:  uploaded_videos
    Writes: frames (list of frame dicts with path, timestamp, scores)
    """
    logs: List[str] = ["[preprocess_agent] Starting frame extraction"]
    errors: List[Dict[str, Any]] = []
    all_frames: List[Dict[str, Any]] = []

    job_id = state.get("job_id", "default")
    base_temp = Path("temp") / job_id

    clips = state.get("uploaded_videos", [])

    for clip_idx, clip_path in enumerate(clips):
        out_dir = base_temp / f"frames_{clip_idx:03d}"
        duration = get_duration(clip_path)

        frame_paths = _extract_frames(clip_path, out_dir, FRAME_SAMPLE_FPS)
        clip_frames = []

        for frame_idx, fp in enumerate(frame_paths):
            # Compute approximate timestamp
            timestamp = round(frame_idx / FRAME_SAMPLE_FPS, 3)
            scores = _score_frame(fp)
            clip_frames.append({
                "frame_path": fp,
                "clip_path": clip_path,
                "clip_index": clip_idx,
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "clip_duration": duration,
                **scores,
            })

        all_frames.extend(clip_frames)
        good = sum(1 for f in clip_frames if f.get("quality_ok", True))
        logs.append(
            f"[preprocess_agent] {Path(clip_path).name}: "
            f"{len(clip_frames)} frames extracted, {good} quality OK"
        )

    logs.append(f"[preprocess_agent] Total frames: {len(all_frames)}")

    return {
        **state,
        "frames": all_frames,
        "current_agent": "preprocess_agent",
        "errors": errors,
        "logs": logs,
    }
