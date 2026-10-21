"""
Whisper Tool - ClipMind AI
==================================
Wraps OpenAI Whisper for speech-to-text transcription.
Used by: whisper_agent
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from tools.ffmpeg import get_ffmpeg

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
_model_cache = None


def get_model():
    """Load (and cache) the Whisper model."""
    global _model_cache
    if _model_cache is None:
        import whisper
        _model_cache = whisper.load_model(WHISPER_MODEL)
    return _model_cache


def extract_audio_wav(video_path: str, out_path: str, sample_rate: int = 16000) -> bool:
    """Extract a mono 16kHz WAV from video for Whisper."""
    cmd = [
        get_ffmpeg(), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        out_path, "-loglevel", "error",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and Path(out_path).exists()
    except Exception:
        return False


def transcribe(wav_path: str, clip_path: str = "") -> List[Dict[str, Any]]:
    """
    Transcribe a WAV file using OpenAI Whisper.
    Returns list of: {start, end, text, clip_path}
    """
    try:
        result = get_model().transcribe(wav_path, fp16=False, verbose=False)
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
                "clip_path": clip_path,
            })
        return segments
    except ImportError:
        return [{"start": 0.0, "end": 0.0, "text": "[whisper not installed]", "clip_path": clip_path}]
    except Exception as exc:
        return [{"start": 0.0, "end": 0.0, "text": f"[error: {exc}]", "clip_path": clip_path}]
