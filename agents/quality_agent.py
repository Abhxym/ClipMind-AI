"""
Quality Analysis Agent — AI Video Director
==========================================
Scores each scene on blur, brightness, and motion.
Filters out unusable scenes before downstream agents process them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from logs.agent_logs import AgentLogger
from memory.langgraph_state import QualityScore, SceneMetadata, VideoDirectorState


# ── Thresholds (configurable via .env) ────────────────────────────────────────

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "80.0"))       # Laplacian variance
BRIGHTNESS_MIN = float(os.getenv("BRIGHTNESS_MIN", "40.0"))
BRIGHTNESS_MAX = float(os.getenv("BRIGHTNESS_MAX", "220.0"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "0.3"))    # Min motion to keep

# Composite score weights
W_BLUR = 0.5
W_BRIGHTNESS = 0.3
W_MOTION = 0.2


def _laplacian_variance(frame: np.ndarray) -> float:
    """Blur metric: higher = sharper. Uses Laplacian filter variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_score(frame: np.ndarray) -> float:
    """Mean brightness of the V channel in HSV (0–255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def _motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Optical flow magnitude between two frames (0–1 normalised)."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(magnitude)) / 20.0   # normalise to ~0–1


def analyse_scene_quality(scene: SceneMetadata) -> QualityScore:
    """
    Sample 3 representative frames from the scene and compute quality scores.
    Returns a QualityScore dict.
    """
    cap = cv2.VideoCapture(scene["clip_path"])
    if not cap.isOpened():
        return _empty_score(scene["scene_index"])

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(scene["start_time"] * native_fps)
    end_frame = int(scene["end_time"] * native_fps)
    total_scene_frames = max(1, end_frame - start_frame)

    # Sample 3 frames: 20%, 50%, 80% into the scene
    sample_offsets = [0.2, 0.5, 0.8]
    frames = []
    for offset in sample_offsets:
        target = start_frame + int(offset * total_scene_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        return _empty_score(scene["scene_index"])

    # Compute metrics
    blur_scores = [_laplacian_variance(f) for f in frames]
    brightness_scores = [_brightness_score(f) for f in frames]
    motion = _motion_score(frames[0], frames[-1]) if len(frames) >= 2 else 0.5

    avg_blur = float(np.mean(blur_scores))
    avg_brightness = float(np.mean(brightness_scores))
    avg_motion = motion

    # Normalise blur to 0–1 (cap at 500 for very sharp frames)
    blur_norm = min(1.0, avg_blur / 500.0)
    # Brightness: penalise over/underexposed
    bri_norm = 1.0 - abs(avg_brightness - 130) / 130.0
    bri_norm = max(0.0, bri_norm)
    # Motion already normalised
    motion_norm = min(1.0, avg_motion)

    composite = (
        W_BLUR * blur_norm
        + W_BRIGHTNESS * bri_norm
        + W_MOTION * motion_norm
    )

    is_usable = (
        avg_blur >= BLUR_THRESHOLD
        and BRIGHTNESS_MIN <= avg_brightness <= BRIGHTNESS_MAX
    )

    return QualityScore(
        scene_index=scene["scene_index"],
        blur_score=avg_blur,
        brightness_score=avg_brightness,
        motion_score=avg_motion,
        overall_score=composite,
        is_usable=is_usable,
    )


def _empty_score(scene_index: int) -> QualityScore:
    return QualityScore(
        scene_index=scene_index,
        blur_score=0.0,
        brightness_score=0.0,
        motion_score=0.0,
        overall_score=0.0,
        is_usable=False,
    )


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def quality_analysis_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Quality Analysis Agent.
    Reads:  state.scenes
    Writes: state.quality_scores, state.filtered_scenes, state.filtered_count
    """
    job_id = state["job_id"]
    log = AgentLogger("quality_agent", job_id)
    log.start(f"Analysing quality for {state['scene_count']} scenes")

    quality_scores: List[QualityScore] = []
    for scene in state["scenes"]:
        score = analyse_scene_quality(scene)
        quality_scores.append(score)
        flag = "✓" if score["is_usable"] else "✗"
        log.debug(
            f"  Scene {scene['scene_index']:03d} {flag} | "
            f"blur={score['blur_score']:.1f} | "
            f"brightness={score['brightness_score']:.1f} | "
            f"motion={score['motion_score']:.2f}"
        )

    # Build lookup: scene_index → is_usable
    usable_indices = {
        qs["scene_index"] for qs in quality_scores if qs["is_usable"]
    }
    filtered_scenes = [
        s for s in state["scenes"] if s["scene_index"] in usable_indices
    ]

    log.success(
        f"Filtered: {len(filtered_scenes)}/{len(state['scenes'])} scenes passed quality check"
    )

    return {
        **state,
        "quality_scores": quality_scores,
        "filtered_scenes": filtered_scenes,
        "filtered_count": len(filtered_scenes),
        "current_agent": "quality_agent",
        "completed_agents": ["quality_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "quality_agent": log.elapsed_ms(),
        },
    }
