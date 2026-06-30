"""
Video Editing Agent — AI Video Director
========================================
Assembles the final video from the story arc using MoviePy + FFmpeg:
  • Trims each clip to planned start/end
  • Applies transitions between clips
  • Overlays subtitles/captions
  • Mixes in background music
  • Emits progress events via a callback (hooked into WebSocket stream)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional

from logs.agent_logs import AgentLogger
from memory.langgraph_state import ClipPlan, VideoDirectorState


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))


# ── Platform-specific export settings ─────────────────────────────────────────

PLATFORM_SETTINGS = {
    "instagram": {"width": 1080, "height": 1920, "fps": 30, "bitrate": "4M"},  # 9:16
    "tiktok":    {"width": 1080, "height": 1920, "fps": 30, "bitrate": "4M"},
    "youtube":   {"width": 1920, "height": 1080, "fps": 30, "bitrate": "8M"},  # 16:9
    "generic":   {"width": 1280, "height": 720,  "fps": 30, "bitrate": "4M"},
}


def _get_settings(platform: str) -> dict:
    return PLATFORM_SETTINGS.get(platform, PLATFORM_SETTINGS["generic"])


def _trim_clip_ffmpeg(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
    width: int,
    height: int,
    fps: int,
) -> bool:
    """Trim and scale a single clip using ffmpeg."""
    duration = max(0.5, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-ar", "44100",
        "-pix_fmt", "yuv420p",
        output_path,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _apply_transition_ffmpeg(
    clip_a: str,
    clip_b: str,
    output_path: str,
    transition: str,
    fps: int,
    overlap: float = 0.3,
) -> bool:
    """
    Apply a simple transition between two clips.
    Supports: cut (just concat), fade (crossfade), zoom.
    """
    if transition == "cut":
        # Simple concatenation via ffmpeg concat demuxer
        list_file = str(Path(output_path).parent / "concat_list.txt")
        with open(list_file, "w") as f:
            f.write(f"file '{clip_a}'\nfile '{clip_b}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
            "-loglevel", "error",
        ]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    elif transition == "fade":
        # Cross-dissolve: requires re-encoding, approximate with xfade filter
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_a, "-i", clip_b,
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=dissolve:duration={overlap}:offset=0[v];"
            f"[0:a][1:a]acrossfade=d={overlap}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_path,
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            # Fallback to cut
            return _apply_transition_ffmpeg(clip_a, clip_b, output_path, "cut", fps)
        return True

    else:
        # Default: cut
        return _apply_transition_ffmpeg(clip_a, clip_b, output_path, "cut", fps)


def _add_subtitles_ffmpeg(
    input_path: str,
    output_path: str,
    clip_plans: List[ClipPlan],
    font_size: int = 36,
) -> bool:
    """Burn subtitles into the video using ffmpeg drawtext filter."""
    if not clip_plans:
        import shutil
        shutil.copy(input_path, output_path)
        return True

    # Build drawtext filter chain
    filters = []
    cursor = 0.0
    for plan in clip_plans:
        caption = plan["caption"].replace("'", "\\'").replace(":", "\\:")
        dur = plan["duration"]
        if caption:
            filters.append(
                f"drawtext=text='{caption}'"
                f":fontsize={font_size}"
                f":fontcolor=white"
                f":borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h-80"
                f":enable='between(t,{cursor:.2f},{cursor+dur:.2f})'"
            )
        cursor += dur

    if not filters:
        import shutil
        shutil.copy(input_path, output_path)
        return True

    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        output_path,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _mix_music_ffmpeg(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.3,
) -> bool:
    """Mix background music into the video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-shortest",
        output_path,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def video_editing_agent(
    state: VideoDirectorState,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> VideoDirectorState:
    """
    LangGraph node: Video Editing Agent.
    Reads:  state.story_arc, state.selected_music_path, state.platform
    Writes: state.edited_video_path, state.editing_progress
    """
    job_id = state["job_id"]
    log = AgentLogger("editing_agent", job_id)
    log.start(f"Assembling {len(state['story_arc'])} clips")

    settings = _get_settings(state["platform"])
    W, H, FPS = settings["width"], settings["height"], settings["fps"]

    job_temp = TEMP_DIR / job_id / "editing"
    job_temp.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    story_arc = state["story_arc"]
    if not story_arc:
        log.warning("Empty story arc — skipping editing")
        return {**state, "edited_video_path": None, "editing_progress": 0.0}

    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        log.debug(f"[{pct*100:.0f}%] {msg}")

    # Step 1: Trim individual clips
    trimmed_clips: List[str] = []
    for i, plan in enumerate(story_arc):
        out = str(job_temp / f"trimmed_{i:03d}.mp4")
        _progress(i / len(story_arc) * 0.4, f"Trimming clip {i+1}/{len(story_arc)}")
        ok = _trim_clip_ffmpeg(
            plan["clip_path"], out,
            plan["start_time"], plan["end_time"],
            W, H, FPS,
        )
        if ok:
            trimmed_clips.append(out)
        else:
            log.warning(f"Trim failed for plan {i} — skipping")

    if not trimmed_clips:
        raise RuntimeError("All clip trims failed — no clips to assemble")

    _progress(0.4, "Applying transitions")

    # Step 2: Concatenate with transitions
    merged = trimmed_clips[0]
    for i in range(1, len(trimmed_clips)):
        out = str(job_temp / f"merged_{i:03d}.mp4")
        transition = story_arc[i]["transition"] if i < len(story_arc) else "cut"
        ok = _apply_transition_ffmpeg(merged, trimmed_clips[i], out, transition, FPS)
        if ok:
            merged = out
        else:
            log.warning(f"Transition {i} failed — using cut fallback")
            _apply_transition_ffmpeg(merged, trimmed_clips[i], out, "cut", FPS)
            merged = out

    _progress(0.7, "Burning captions")

    # Step 3: Add subtitles
    subtitled = str(job_temp / "with_subs.mp4")
    _add_subtitles_ffmpeg(merged, subtitled, story_arc)

    _progress(0.85, "Mixing background music")

    # Step 4: Mix music
    music_path = state.get("selected_music_path")
    if music_path and Path(music_path).exists():
        final_temp = str(job_temp / "with_music.mp4")
        ok = _mix_music_ffmpeg(subtitled, music_path, final_temp)
        edited_path = final_temp if ok else subtitled
    else:
        edited_path = subtitled

    _progress(1.0, "Editing complete")
    log.success(f"Edited video: {edited_path}")

    return {
        **state,
        "edited_video_path": edited_path,
        "editing_progress": 1.0,
        "current_agent": "editing_agent",
        "completed_agents": ["editing_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "editing_agent": log.elapsed_ms(),
        },
    }
