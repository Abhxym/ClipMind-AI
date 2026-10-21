"""
Whisper Agent — ClipMind AI
===================================
Pipeline 3: Speech-to-text extraction using OpenAI Whisper.
  - Extracts a 16 kHz mono WAV from each clip (via ffmpeg)
  - Transcribes with Whisper (WHISPER_MODEL, default 'base'; loaded once)
  - Returns time-stamped segments
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
from tools.whisper import WHISPER_MODEL, extract_audio_wav, transcribe


def whisper_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Whisper Speech Extraction Agent.
    Reads:  uploaded_videos
    Writes: transcript (list of time-stamped dicts)
    """
    logs = [f"[whisper_agent] Transcribing {len(state['uploaded_videos'])} clips with Whisper '{WHISPER_MODEL}'"]
    errors = []
    all_segments: List[Dict[str, Any]] = []

    job_id = state.get("job_id", "unknown")
    temp_dir = Path("temp") / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, clip_path in enumerate(state["uploaded_videos"]):
        wav_path = str(temp_dir / f"audio_{i:03d}.wav")
        if not extract_audio_wav(clip_path, wav_path):
            logs.append(f"[whisper_agent] No audio in {Path(clip_path).name}, skipping")
            continue

        segments = transcribe(wav_path, clip_path)
        # transcribe() reports failures as a single "[...]" segment; keep those out of the transcript
        if len(segments) == 1 and segments[0]["text"].startswith("["):
            logs.append(f"[whisper_agent] {Path(clip_path).name}: {segments[0]['text']}")
            continue
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
