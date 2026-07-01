"""
Video Editing Agent — AI Video Director
========================================
Assembles the final video from the story arc using FFmpeg.
Handles clips with or without audio streams.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, List, Optional
import imageio_ffmpeg

from logs.agent_logs import AgentLogger
from memory.langgraph_state import ClipPlan, VideoDirectorState


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))

PLATFORM_SETTINGS = {
    "instagram": {"width": 1080, "height": 1920, "fps": 30},
    "tiktok":    {"width": 1080, "height": 1920, "fps": 30},
    "youtube":   {"width": 1920, "height": 1080, "fps": 30},
    "generic":   {"width": 1280, "height": 720,  "fps": 30},
}

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _has_audio(path: str) -> bool:
    """Check if a video file has an audio stream."""
    r = _run([FFMPEG, "-i", path, "-hide_banner"], timeout=10)
    return "Audio:" in r.stderr


def _trim_clip(input_path: str, output_path: str, start: float, end: float,
               width: int, height: int, fps: int) -> bool:
    duration = max(0.5, end - start)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
          f"format=yuv420p")
    # Check if source has audio
    has_src_audio = _has_audio(input_path)
    if has_src_audio:
        inputs = ["-ss", str(start), "-i", input_path, "-t", str(duration)]
        audio_args = ["-c:a", "aac", "-ar", "44100", "-ac", "2"]
    else:
        inputs = [
            "-ss", str(start), "-i", input_path, "-t", str(duration),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"
        ]
        audio_args = ["-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest"]
        
    cmd = [
        FFMPEG, "-y",
        *inputs,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast",
        *audio_args,
        output_path, "-loglevel", "error",
    ]
    r = _run(cmd)
    return r.returncode == 0 and Path(output_path).exists()


def _add_silent_audio(input_path: str, output_path: str) -> bool:
    """Add a silent audio track to a video that has no audio."""
    cmd = [
        FFMPEG, "-y",
        "-i", input_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-loglevel", "error",
    ]
    r = _run(cmd)
    return r.returncode == 0 and Path(output_path).exists()


def _concat_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate clips using concat demuxer (most reliable for same-format clips)."""
    if len(clip_paths) == 1:
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return True

    # Write concat list file
    list_file = str(Path(output_path).parent / "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # Use forward slashes and absolute path
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy",
        output_path, "-loglevel", "error",
    ]
    r = _run(cmd, timeout=300)
    return r.returncode == 0 and Path(output_path).exists()


def _add_subtitles(input_path: str, output_path: str,
                   clip_plans: List[ClipPlan], font_size: int = 36) -> bool:
    if not any(p["caption"] for p in clip_plans):
        import shutil
        shutil.copy2(input_path, output_path)
        return True

    filters = []
    cursor = 0.0
    for plan in clip_plans:
        caption = plan["caption"].replace("'", "\\'").replace(":", "\\:")
        dur = plan["duration"]
        if caption:
            filters.append(
                f"drawtext=text='{caption}':fontsize={font_size}"
                f":fontcolor=white:borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h-80"
                f":enable='between(t,{cursor:.2f},{cursor+dur:.2f})'"
            )
        cursor += dur

    if not filters:
        import shutil
        shutil.copy2(input_path, output_path)
        return True

    cmd = [
        FFMPEG, "-y", "-i", input_path,
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "fast", "-c:a", "copy",
        output_path, "-loglevel", "error",
    ]
    r = _run(cmd, timeout=120)
    if r.returncode != 0:
        import shutil
        shutil.copy2(input_path, output_path)
    return True


def _mix_music(video_path: str, music_path: str, output_path: str,
               volume: float = 0.3) -> bool:
    cmd = [
        FFMPEG, "-y", "-i", video_path, "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={volume}[m];[0:a][m]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-loglevel", "error",
    ]
    r = _run(cmd, timeout=120)
    return r.returncode == 0 and Path(output_path).exists()


def video_editing_agent(
    state: VideoDirectorState,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> VideoDirectorState:
    job_id = state["job_id"]
    log = AgentLogger("editing_agent", job_id)
    log.start(f"Assembling {len(state['story_arc'])} clips")

    settings = PLATFORM_SETTINGS.get(state["platform"], PLATFORM_SETTINGS["generic"])
    W, H, FPS = settings["width"], settings["height"], settings["fps"]

    job_temp = TEMP_DIR / job_id / "editing"
    job_temp.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    story_arc = state["story_arc"]
    if not story_arc:
        log.warning("Empty story arc — skipping editing")
        return {**state, "edited_video_path": None, "editing_progress": 0.0}

    # Step 1: Trim clips
    trimmed: List[str] = []
    for i, plan in enumerate(story_arc):
        out = str(job_temp / f"trimmed_{i:03d}.mp4")
        ok = _trim_clip(plan["clip_path"], out, plan["start_time"], plan["end_time"], W, H, FPS)
        if ok:
            trimmed.append(out)
            log.debug(f"Trimmed clip {i+1}/{len(story_arc)}")
        else:
            log.warning(f"Trim failed for clip {i} — skipping")

    if not trimmed:
        raise RuntimeError("All clip trims failed")

    # Step 2: Ensure all clips have audio
    # Skipped: already handled during _trim_clip
    ready = trimmed

    # Step 3: Concatenate
    concatenated = str(job_temp / "concatenated.mp4")
    ok = _concat_clips(ready, concatenated)
    if not ok:
        raise RuntimeError("Concatenation failed")
    log.debug("Concatenation done")

    # Step 4: Subtitles
    subtitled = str(job_temp / "with_subs.mp4")
    _add_subtitles(concatenated, subtitled, story_arc)

    # Step 5: Music
    music_path = state.get("selected_music_path")
    if music_path and Path(music_path).exists():
        with_music = str(job_temp / "with_music.mp4")
        edited_path = with_music if _mix_music(subtitled, music_path, with_music) else subtitled
    else:
        edited_path = subtitled

    log.success(f"Edited video: {edited_path}")

    return {
        **state,
        "edited_video_path": edited_path,
        "editing_progress": 1.0,
        "current_agent": "editing_agent",
        "completed_agents": ["editing_agent"],
        "agent_timings": {**state.get("agent_timings", {}), "editing_agent": log.elapsed_ms()},
    }
