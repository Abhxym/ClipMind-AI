"""
Whisper Agent — AI Video Director
===================================
Pipeline 3: Speech-to-text extraction using OpenAI Whisper.
  - Extracts audio track from each clip (via ffmpeg)
  - Transcribes with Whisper 'base' model
  - Returns time-stamped segments
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.state import VideoState


def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_audio(clip_path: str, out_wav: str) -> bool:
    """Extract a 16kHz mono WAV from the video for Whisper."""
    try:
        cmd = [
            _get_ffmpeg(), "-y", "-i", clip_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            out_wav, "-loglevel", "error",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return r.returncode == 0 and Path(out_wav).exists()
    except Exception:
        return False


def transcribe_audio(wav_path: str, clip_path: str) -> List[Dict[str, Any]]:
    """Transcribe audio using Whisper. Returns time-stamped segments."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, fp16=False, verbose=False)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "clip_path": clip_path,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
            })
        return segments
    except ImportError:
        return [{"clip_path": clip_path, "start": 0.0, "end": 0.0, "text": "[whisper not installed]"}]
    except Exception as exc:
        return [{"clip_path": clip_path, "start": 0.0, "end": 0.0, "text": f"[error: {exc}]"}]


def whisper_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Whisper Speech Extraction Agent.
    Reads:  uploaded_videos
    Writes: transcript (list of time-stamped dicts)
    """
    logs = [f"[whisper_agent] Transcribing {len(state['uploaded_videos'])} clips"]
    errors = []
    all_segments: List[Dict[str, Any]] = []

    job_id = state.get("job_id", "unknown")
    temp_dir = Path("temp") / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, clip_path in enumerate(state["uploaded_videos"]):
        wav_path = str(temp_dir / f"audio_{i:03d}.wav")
        if not extract_audio(clip_path, wav_path):
            logs.append(f"[whisper_agent] No audio in {Path(clip_path).name}, skipping")
            continue

        segments = transcribe_audio(wav_path, clip_path)
        all_segments.extend(segments)
        full_text = " ".join(s["text"] for s in segments)
        logs.append(f"[whisper_agent] {Path(clip_path).name}: {len(segments)} segments — \"{full_text[:80]}...\"")

    logs.append(f"[whisper_agent] Total transcript segments: {len(all_segments)}")

    return {
        **state,
        "transcript": all_segments,
        "current_agent": "whisper_agent",
        "errors": errors,
        "logs": logs,
    }
