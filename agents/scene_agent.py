"""
Scene Extraction Agent — AI Video Director
==========================================
Detects scene boundaries via OpenCV histogram comparison.
Optimised: samples at 2fps max, merges short scenes, skips corrupt clips.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from logs.agent_logs import AgentLogger
from memory.langgraph_state import SceneMetadata, VideoDirectorState


THUMBNAIL_DIR = Path("temp/thumbnails")
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

SCENE_CHANGE_THRESHOLD = float(os.getenv("SCENE_CHANGE_THRESHOLD", "0.35"))
FRAME_SAMPLE_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "2"))
MIN_SCENE_DURATION = float(os.getenv("MIN_SCENE_DURATION", "1.5"))  # seconds
MAX_SCENE_DURATION = float(os.getenv("MAX_SCENE_DURATION", "10.0"))  # seconds


def _histogram_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    hist_a = cv2.calcHist([cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)],
                          [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def _save_thumbnail(frame: np.ndarray, scene_index: int, job_id: str) -> str:
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
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / native_fps
    sample_every = max(1, int(native_fps / FRAME_SAMPLE_FPS))

    # Collect boundary timestamps
    boundaries: List[float] = [0.0]
    prev_frame: np.ndarray | None = None
    frame_num = 0

    while frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
            
        if prev_frame is not None:
            diff = _histogram_diff(prev_frame, frame)
            t = frame_num / native_fps
            time_since_last = t - boundaries[-1]
            
            if diff > SCENE_CHANGE_THRESHOLD:
                # Only add boundary if it's far enough from the last one
                if time_since_last >= MIN_SCENE_DURATION:
                    boundaries.append(t)
            elif time_since_last >= MAX_SCENE_DURATION:
                # Force a cut if the scene is getting too long
                boundaries.append(t)
                
        prev_frame = frame.copy()
        frame_num += sample_every

    cap.release()
    if total_duration - boundaries[-1] < MIN_SCENE_DURATION and len(boundaries) > 1:
        boundaries[-1] = total_duration
    else:
        boundaries.append(total_duration)

    # Build scene list from boundaries
    scenes: List[SceneMetadata] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start
        if duration < MIN_SCENE_DURATION:
            continue

        scene_idx = base_scene_index + len(scenes)

        # Extract representative frame for thumbnail
        cap2 = cv2.VideoCapture(clip_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int((start + duration * 0.3) * native_fps))
        ret, thumb_frame = cap2.read()
        cap2.release()
        thumb_path = _save_thumbnail(thumb_frame if ret else np.zeros((180, 320, 3), np.uint8),
                                     scene_idx, job_id)

        scenes.append(SceneMetadata(
            clip_path=clip_path,
            scene_index=scene_idx,
            start_time=round(start, 3),
            end_time=round(end, 3),
            duration=round(duration, 3),
            thumbnail_path=thumb_path,
            fps=native_fps,
            resolution=(width, height),
        ))

    # Fallback: if no scenes detected, treat whole clip as one scene
    if not scenes and total_duration >= MIN_SCENE_DURATION:
        cap2 = cv2.VideoCapture(clip_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(total_duration * 0.3 * native_fps))
        ret, thumb_frame = cap2.read()
        cap2.release()
        thumb_path = _save_thumbnail(thumb_frame if ret else np.zeros((180, 320, 3), np.uint8),
                                     base_scene_index, job_id)
        scenes.append(SceneMetadata(
            clip_path=clip_path,
            scene_index=base_scene_index,
            start_time=0.0,
            end_time=round(total_duration, 3),
            duration=round(total_duration, 3),
            thumbnail_path=thumb_path,
            fps=native_fps,
            resolution=(width, height),
        ))

    return scenes


def scene_extraction_agent(state: VideoDirectorState) -> VideoDirectorState:
    job_id = state["job_id"]
    log = AgentLogger("scene_agent", job_id)
    log.start(f"Extracting scenes from {len(state['raw_clips'])} clips")

    all_scenes: List[SceneMetadata] = []

    for clip_path in state["raw_clips"]:
        log.info(f"Processing: {Path(clip_path).name}")
        try:
            clip_scenes = extract_scenes_from_clip(clip_path, job_id, len(all_scenes))
            all_scenes.extend(clip_scenes)
            log.debug(f"  → {len(clip_scenes)} scenes")
        except Exception as exc:
            log.warning(f"Skipping {Path(clip_path).name}: {exc}")

    log.success(f"Total scenes extracted: {len(all_scenes)}")

    return {
        **state,
        "scenes": all_scenes,
        "scene_count": len(all_scenes),
        "current_agent": "scene_agent",
        "completed_agents": ["scene_agent"],
        "agent_timings": {**state.get("agent_timings", {}), "scene_agent": log.elapsed_ms()},
    }
