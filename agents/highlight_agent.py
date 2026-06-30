"""
Highlight Detection Agent — AI Video Director
=============================================
Detects exciting moments in filtered scenes using:
  • OpenCV DNN face detection
  • Haar cascade smile detection
  • Librosa audio onset energy
  • Whisper speech transcription
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from logs.agent_logs import AgentLogger
from memory.langgraph_state import HighlightScore, SceneMetadata, VideoDirectorState


# ── Cascade Classifiers ────────────────────────────────────────────────────────

_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
_SMILE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    global _FACE_CASCADE, _SMILE_CASCADE
    if _FACE_CASCADE is None:
        cc_root = cv2.data.haarcascades
        _FACE_CASCADE = cv2.CascadeClassifier(
            str(Path(cc_root) / "haarcascade_frontalface_default.xml")
        )
        _SMILE_CASCADE = cv2.CascadeClassifier(
            str(Path(cc_root) / "haarcascade_smile.xml")
        )
    return _FACE_CASCADE, _SMILE_CASCADE


def _detect_faces_smiles(frame: np.ndarray) -> Tuple[int, float]:
    """Return (face_count, smile_ratio)."""
    face_cascade, smile_cascade = _load_cascades()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    face_count = len(faces)
    smile_count = 0

    for (x, y, w, h) in faces:
        roi = gray[y + h // 2 : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(
            roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
        )
        if len(smiles) > 0:
            smile_count += 1

    smile_ratio = smile_count / face_count if face_count > 0 else 0.0
    return face_count, smile_ratio


def _extract_audio(clip_path: str, out_wav: str) -> bool:
    """Extract audio from a video using ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        out_wav, "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and Path(out_wav).exists()


def _audio_onset_energy(wav_path: str, start: float, end: float) -> float:
    """
    Compute mean onset strength in the time window [start, end].
    Returns a 0–1 normalised score.
    """
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000, offset=start, duration=end - start)
        if len(y) == 0:
            return 0.0
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return float(np.mean(onset_env)) / 20.0   # normalise
    except Exception:
        return 0.0


def _whisper_transcribe(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper. Returns the transcript string."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        return f"[Transcription unavailable: {e}]"


def analyse_highlights(
    scenes: List[SceneMetadata],
    job_id: str,
) -> Tuple[List[HighlightScore], str, Optional[str]]:
    """
    For each scene, compute face/smile detection and audio excitement.
    Also transcribes the first available clip's audio via Whisper.

    Returns: (highlight_scores, transcript_text, audio_wav_path)
    """
    # Extract audio from the first clip for transcription
    temp_dir = Path("temp") / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Optional[str] = None
    transcript = ""

    if scenes:
        first_clip = scenes[0]["clip_path"]
        wav_path = str(temp_dir / "audio_full.wav")
        if _extract_audio(first_clip, wav_path):
            audio_path = wav_path

    # Transcribe all clips and concatenate
    processed_clips: set = set()
    all_transcripts: List[str] = []
    for scene in scenes:
        if scene["clip_path"] not in processed_clips:
            processed_clips.add(scene["clip_path"])
            wav = str(temp_dir / f"audio_{Path(scene['clip_path']).stem}.wav")
            if _extract_audio(scene["clip_path"], wav):
                txt = _whisper_transcribe(wav)
                if txt and not txt.startswith("["):
                    all_transcripts.append(txt)

    transcript = " ".join(all_transcripts)

    # Per-scene highlight scoring
    highlight_scores: List[HighlightScore] = []

    for scene in scenes:
        cap = cv2.VideoCapture(scene["clip_path"])
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        mid_frame = int(((scene["start_time"] + scene["end_time"]) / 2) * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            face_count, smile_score = _detect_faces_smiles(frame)
            has_face = face_count > 0
        else:
            face_count, smile_score, has_face = 0, 0.0, False

        # Audio peak for this scene window
        audio_wav = str(temp_dir / f"audio_{Path(scene['clip_path']).stem}.wav")
        audio_peak = (
            _audio_onset_energy(audio_wav, scene["start_time"], scene["end_time"])
            if Path(audio_wav).exists()
            else 0.0
        )

        excitement = (
            0.4 * float(has_face)
            + 0.3 * smile_score
            + 0.3 * min(1.0, audio_peak)
        )

        highlight_scores.append(HighlightScore(
            scene_index=scene["scene_index"],
            has_face=has_face,
            face_count=face_count,
            smile_score=smile_score,
            audio_peak_score=audio_peak,
            excitement_score=excitement,
        ))

    return highlight_scores, transcript, audio_path


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def highlight_detection_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Highlight Detection Agent.
    Reads:  state.filtered_scenes
    Writes: state.highlight_scores, state.ranked_scenes, state.transcript, state.audio_path
    """
    job_id = state["job_id"]
    log = AgentLogger("highlight_agent", job_id)
    log.start(f"Detecting highlights in {state['filtered_count']} scenes")

    highlight_scores, transcript, audio_path = analyse_highlights(
        state["filtered_scenes"], job_id
    )

    # Build excitement lookup
    excitement_map: Dict[int, float] = {
        h["scene_index"]: h["excitement_score"] for h in highlight_scores
    }

    # Rank scenes by excitement (descending)
    ranked_scenes = sorted(
        state["filtered_scenes"],
        key=lambda s: excitement_map.get(s["scene_index"], 0.0),
        reverse=True,
    )

    log.info(f"Transcript length: {len(transcript)} chars")
    log.success(
        f"Top highlight: scene {ranked_scenes[0]['scene_index'] if ranked_scenes else 'N/A'}"
    )

    return {
        **state,
        "highlight_scores": highlight_scores,
        "ranked_scenes": ranked_scenes,
        "transcript": transcript,
        "audio_path": audio_path,
        "current_agent": "highlight_agent",
        "completed_agents": ["highlight_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "highlight_agent": log.elapsed_ms(),
        },
    }
