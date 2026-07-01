"""
Whisper Tool - AI Video Director
==================================
Wraps OpenAI Whisper for speech-to-text transcription.
Used by: whisper_agent
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_audio_wav(video_path: str, out_path: str, sample_rate: int = 16000) -> bool:
    """Extract a mono 16kHz WAV from video for Whisper."""
    cmd = [
        get_ffmpeg(), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        out_path, "-loglevel", "error",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and Path(out_path).exists()


def transcribe(wav_path: str, clip_path: str = "") -> List[Dict[str, Any]]:
    """
    Transcribe a WAV file using OpenAI Whisper.
    Returns list of: {start, end, text, clip_path}
    """
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, fp16=False, verbose=False)
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
