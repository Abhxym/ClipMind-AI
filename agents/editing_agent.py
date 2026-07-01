"""
Editing Agent — AI Video Director
=====================================
Pipeline Step 10: Video Editing — Trim, Caption Burn, Concatenate.
  - Reads the story_plan and captions from state
  - For each story entry, trims the clip using tools/ffmpeg.py
  - Burns the caption text into the frame via drawtext filter
  - Concatenates all trimmed clips into a single output video
  - Records editing_commands for reviewer and evaluation
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState
import tools.ffmpeg as ffmpeg_tool

PLATFORM_DIMS = {
    "instagram": (1080, 1920),
    "tiktok": (1080, 1920),
    "youtube": (1920, 1080),
    "generic": (1080, 1920),
}


def editing_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Video Editing Agent.
    Reads:  story_plan, captions, uploaded_videos, platform, job_id
    Writes: editing_commands, _edited_video_path
    """
    logs: List[str] = ["[editing_agent] Starting video editing pipeline"]
    errors: List[Dict[str, Any]] = []
    editing_commands: List[Dict[str, Any]] = []

    job_id = state.get("job_id", "default")
    platform = state.get("platform", "instagram")
    story_plan = state.get("story_plan", [])
    captions = state.get("captions", [])
    clips = state.get("uploaded_videos", [])
    retry_count = state.get("retry_count", {})

    # Track retries for the review loop
    current_retries = retry_count.get("editing_agent", 0) + 1
    retry_count = {**retry_count, "editing_agent": current_retries}

    if not story_plan:
        errors.append({
            "agent": "editing_agent",
            "type": "NoStoryPlan",
            "message": "Story plan is empty — cannot edit video",
        })
        return {**state, "current_agent": "editing_agent", "errors": errors, "logs": logs, "retry_count": retry_count}

    if not clips:
        errors.append({
            "agent": "editing_agent",
            "type": "NoClips",
            "message": "No validated clips found in state",
        })
        return {**state, "current_agent": "editing_agent", "errors": errors, "logs": logs, "retry_count": retry_count}

    width, height = PLATFORM_DIMS.get(platform, (1080, 1920))

    temp_dir = Path("temp") / job_id / "edit"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Build a caption lookup: scene_index -> caption text
    caption_map: Dict[int, str] = {
        cap.get("scene_index", i): cap.get("caption", "")
        for i, cap in enumerate(captions)
    }

    trimmed_clips: List[str] = []

    for i, entry in enumerate(story_plan):
        clip_idx = entry.get("clip_index", 0)
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start + 5.0))
        caption_text = caption_map.get(i, "")

        if clip_idx >= len(clips):
            logs.append(f"[editing_agent] Scene {i}: clip_index {clip_idx} out of range — skipping")
            continue

        src_clip = clips[clip_idx]
        if not Path(src_clip).exists():
            errors.append({
                "agent": "editing_agent",
                "type": "MissingClip",
                "message": f"Scene {i}: source clip not found: {src_clip}",
            })
            continue

        out_path = str(temp_dir / f"segment_{i:03d}.mp4")
        t_start = time.time()

        # Detect audio presence from earlier metadata
        clip_meta = state.get("_clip_metadata", [])
        has_audio = True
        if clip_meta and clip_idx < len(clip_meta):
            has_audio = clip_meta[clip_idx].get("has_audio", True)

        success = ffmpeg_tool.trim(
            input_path=src_clip,
            start=start,
            end=end,
            output_path=out_path,
            width=width,
            height=height,
            caption=caption_text,
            has_audio=has_audio,
        )

        elapsed = round((time.time() - t_start) * 1000, 1)

        cmd_record = {
            "step": "trim",
            "scene_index": i,
            "clip_index": clip_idx,
            "clip_path": src_clip,
            "start": start,
            "end": end,
            "output": out_path,
            "caption": caption_text,
            "success": success,
            "elapsed_ms": elapsed,
        }
        editing_commands.append(cmd_record)

        if success:
            trimmed_clips.append(out_path)
            logs.append(f"[editing_agent] ✓ Segment {i}: {Path(src_clip).name} {start:.1f}s→{end:.1f}s | {elapsed}ms")
        else:
            errors.append({
                "agent": "editing_agent",
                "type": "TrimFailed",
                "message": f"Scene {i}: FFmpeg trim failed for {Path(src_clip).name} {start:.1f}s→{end:.1f}s",
            })
            logs.append(f"[editing_agent] ✗ Segment {i} FAILED: {Path(src_clip).name}")

    if not trimmed_clips:
        errors.append({
            "agent": "editing_agent",
            "type": "NoSegments",
            "message": "No segments were successfully trimmed — cannot concatenate",
        })
        return {
            **state,
            "editing_commands": editing_commands,
            "current_agent": "editing_agent",
            "errors": errors,
            "logs": logs,
            "retry_count": retry_count,
        }

    # Concatenate
    concat_output = str(temp_dir / "concat_output.mp4")
    t_start = time.time()
    concat_ok = ffmpeg_tool.concat(trimmed_clips, concat_output)
    elapsed_concat = round((time.time() - t_start) * 1000, 1)

    editing_commands.append({
        "step": "concat",
        "input_count": len(trimmed_clips),
        "output": concat_output,
        "success": concat_ok,
        "elapsed_ms": elapsed_concat,
    })

    if concat_ok:
        logs.append(f"[editing_agent] ✓ Concatenated {len(trimmed_clips)} segments → {concat_output} | {elapsed_concat}ms")
    else:
        errors.append({
            "agent": "editing_agent",
            "type": "ConcatFailed",
            "message": f"FFmpeg concat failed with {len(trimmed_clips)} segments",
        })

    edited_path = concat_output if concat_ok and Path(concat_output).exists() else None
    logs.append(
        f"[editing_agent] Editing complete. Success={concat_ok}, "
        f"segments={len(trimmed_clips)}/{len(story_plan)}, retries={current_retries}"
    )

    return {
        **state,
        "editing_commands": editing_commands,
        "_edited_video_path": edited_path,
        "current_agent": "editing_agent",
        "errors": errors,
        "logs": logs,
        "retry_count": retry_count,
    }
