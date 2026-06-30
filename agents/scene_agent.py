"""
Scene Extraction Agent — AI Video Director
==========================================
Samples frames from each uploaded clip, detects scene boundaries via
OpenCV histogram comparison, and produces a list of SceneMetadata dicts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from logs.agent_logs import AgentLogger
from memory.langgraph_state import SceneMetadata, VideoDirectorState


THUMBNAIL_DIR = Path("temp/thumbnails")
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# Scene-change detection threshold (0–1): lower = more sensitive
SCENE_CHANGE_THRESHOLD = float(os.getenv("SCENE_CHANGE_THRESHOLD", "0.4"))
FRAME_SAMPLE_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "1"))


def _histogram_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute normalised histogram difference between two BGR frames."""
    hist_a = cv2.calcHist([cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def _save_thumbnail(frame: np.ndarray, scene_index: int, job_id: str) -> str:
    """Save a scene thumbnail as JPEG and return its path."""
    thumb_dir = THUMBNAIL_DIR / job_id
    thumb_dir.mkdir(parents=True, exist_ok=True)
    path = str(thumb_dir / f"scene_{scene_index:04d}.jpg")
    cv2.imwrite(path, cv2.resize(frame, (320, 180)))
    return path


def extract_scenes_from_clip(
    clip_path: str,
    job_id: str,
    base_scene_index: int = 0,
) -> List[SceneMetadata]:
    """
    Extract scenes from a single video clip.

    Strategy:
    - Sample 1 frame per second (configurable)
    - Compare consecutive frames via HSV histogram distance
    - When distance > threshold → new scene boundary detected
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)
    sample_every = max(1, int(native_fps / FRAME_SAMPLE_FPS))

    scenes: List[SceneMetadata] = []
    prev_frame: np.ndarray | None = None
    scene_start_time = 0.0
    scene_idx = base_scene_index
    rep_frame: np.ndarray | None = None

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % sample_every != 0:
            frame_num += 1
            continue

        current_time = frame_num / native_fps

        if prev_frame is not None:
            diff = _histogram_diff(prev_frame, frame)
            if diff > SCENE_CHANGE_THRESHOLD:
                # Scene boundary detected
                thumb = _save_thumbnail(
                    rep_frame if rep_frame is not None else prev_frame,
                    scene_idx, job_id
                )
                scenes.append(SceneMetadata(
                    clip_path=clip_path,
                    scene_index=scene_idx,
                    start_time=scene_start_time,
                    end_time=current_time,
                    duration=current_time - scene_start_time,
                    thumbnail_path=thumb,
                    fps=native_fps,
                    resolution=resolution,
                ))
                scene_idx += 1
                scene_start_time = current_time
                rep_frame = frame.copy()
        else:
            rep_frame = frame.copy()

        prev_frame = frame.copy()
        frame_num += 1

    # Close out the final scene
    if prev_frame is not None:
        end_time = total_frames / native_fps
        thumb = _save_thumbnail(
            rep_frame if rep_frame is not None else prev_frame,
            scene_idx, job_id
        )
        scenes.append(SceneMetadata(
            clip_path=clip_path,
            scene_index=scene_idx,
            start_time=scene_start_time,
            end_time=end_time,
            duration=end_time - scene_start_time,
            thumbnail_path=thumb,
            fps=native_fps,
            resolution=resolution,
        ))

    cap.release()
    return scenes


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def scene_extraction_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Scene Extraction Agent.
    Reads:  state.raw_clips
    Writes: state.scenes, state.scene_count, state.current_agent,
            state.completed_agents, state.agent_timings
    """
    job_id = state["job_id"]
    log = AgentLogger("scene_agent", job_id)
    log.start(f"Extracting scenes from {len(state['raw_clips'])} clips")

    all_scenes: List[SceneMetadata] = []
    global_scene_idx = 0

    for clip_path in state["raw_clips"]:
        log.info(f"Processing: {Path(clip_path).name}")
        try:
            clip_scenes = extract_scenes_from_clip(clip_path, job_id, global_scene_idx)
            all_scenes.extend(clip_scenes)
            global_scene_idx += len(clip_scenes)
            log.debug(f"  → {len(clip_scenes)} scenes detected")
        except Exception as exc:
            log.warning(f"Skipping clip {clip_path}: {exc}")

    log.success(f"Total scenes extracted: {len(all_scenes)}")

    return {
        **state,
        "scenes": all_scenes,
        "scene_count": len(all_scenes),
        "current_agent": "scene_agent",
        "completed_agents": ["scene_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "scene_agent": log.elapsed_ms(),
        },
    }
