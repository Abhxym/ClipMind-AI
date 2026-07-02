"""
Jobs Routes — AI Video Director API
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from api.models import (
    CreateJobRequest,
    JobCreateResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    EnhanceVideoRequest,
    EnhanceVideoResponse,
)


router = APIRouter(prefix="/api/jobs", tags=["jobs"])

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))

# ── In-memory job store (replace with Redis/DB in production) ──────────────────
_jobs: Dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


def _run_pipeline_sync(job_id: str, clips: list, request: CreateJobRequest, loop: asyncio.AbstractEventLoop):
    """Runs the pipeline in a background thread."""
    from api.routes.ws import manager
    from graphs.workflow import run_pipeline
    from memory.state import create_initial_state
    import datetime

    AGENT_ORDER = [
        "input_agent", "preprocess_agent", "whisper_agent",
        "scene_agent", "metadata_agent", "embedding_agent",
        "rag_agent", "planner_agent", "caption_agent",
        "editing_agent", "reviewer_agent", "export_agent",
        "evaluation_agent",
    ]

    def progress_callback(agent_name: str, status: str, attempt: int):
        if job_id not in _jobs:
            return
        idx = AGENT_ORDER.index(agent_name) if agent_name in AGENT_ORDER else 0
        pct = ((idx + (1 if status == "done" else 0)) / len(AGENT_ORDER)) * 100
        _jobs[job_id]["current_agent"] = agent_name
        _jobs[job_id]["agent_statuses"][agent_name] = {
            "status": status, "attempt": attempt
        }
        _jobs[job_id]["progress_pct"] = pct

        # Broadcast over WebSocket — fire-and-forget from sync thread using main loop
        payload = {
            "type": "progress",
            "job_id": job_id,
            "agent": agent_name,
            "status": status,
            "attempt": attempt,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "progress_pct": pct,
            "message": "",
        }
        try:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(job_id, payload), loop)
        except Exception:
            pass

    _jobs[job_id]["status"] = JobStatus.running

    try:
        initial_state = create_initial_state(
            job_id=job_id,
            user_prompt=request.user_prompt,
            uploaded_videos=clips,
            target_duration=request.target_duration,
            platform=request.platform.value,
            brand_pdf_path=request.brand_pdf_path,
        )
        final_state = run_pipeline(job_id=job_id, initial_state=initial_state)
        _jobs[job_id]["status"] = JobStatus.done
        _jobs[job_id]["final_state"] = final_state
        _jobs[job_id]["progress_pct"] = 100.0
        
        try:
            if loop and loop.is_running():
                payload = {
                    "type": "progress",
                    "job_id": job_id,
                    "agent": "pipeline",
                    "status": "done",
                    "attempt": 0,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "progress_pct": 100.0,
                    "message": "Pipeline completed successfully",
                }
                asyncio.run_coroutine_threadsafe(manager.broadcast(job_id, payload), loop)
        except Exception:
            pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[job_id]["status"] = JobStatus.failed
        _jobs[job_id]["error"] = str(e)


@router.post("", response_model=JobCreateResponse)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create and enqueue a new video generation job.
    Clip paths are provided in the request body as clip_paths.
    """
    clip_paths = [p.strip() for p in request.clip_paths if p.strip()]
    if not clip_paths:
        raise HTTPException(400, "At least one clip path required")

    # Resolve to absolute paths so the background thread can find files
    # regardless of working directory
    resolved_clips = []
    for p in clip_paths:
        abs_path = Path(p) if Path(p).is_absolute() else (Path.cwd() / p).resolve()
        if not abs_path.exists():
            raise HTTPException(400, f"Clip file not found: {p}")
        resolved_clips.append(str(abs_path))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.pending,
        "current_agent": None,
        "agent_statuses": {},
        "progress_pct": 0.0,
        "final_state": None,
        "error": None,
        "created_at": time.time(),
    }

    loop = asyncio.get_running_loop()
    background_tasks.add_task(_run_pipeline_sync, job_id, resolved_clips, request, loop)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.pending,
        message="Job created. Connect to /api/ws/{job_id} for live progress.",
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = _get_job(job_id)
    final_state = job.get("final_state") or {}
    errors = final_state.get("errors", [])

    AGENT_ORDER = [
        "input_agent", "preprocess_agent", "whisper_agent",
        "scene_agent", "metadata_agent", "embedding_agent",
        "rag_agent", "planner_agent", "caption_agent",
        "editing_agent", "reviewer_agent", "export_agent",
        "evaluation_agent",
    ]
    agent_statuses = []
    for name in AGENT_ORDER:
        info = job["agent_statuses"].get(name, {})
        agent_statuses.append({
            "name": name,
            "status": info.get("status", "pending"),
            "elapsed_ms": final_state.get("agent_timings", {}).get(name),
        })

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        current_agent=job.get("current_agent"),
        completed_agents=final_state.get("completed_agents", []),
        agent_statuses=agent_statuses,
        progress_pct=job.get("progress_pct", 0.0),
        error_count=len(errors),
        errors=errors,
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    job = _get_job(job_id)
    if job["status"] not in (JobStatus.done, JobStatus.failed):
        raise HTTPException(202, "Job still processing")

    state = job.get("final_state") or {}

    def _url(path: Optional[str]) -> Optional[str]:
        if not path or not Path(path).exists():
            return None
        return f"/outputs/{Path(path).name}"

    # Extract captions as a flat list of strings for the response
    raw_captions = state.get("captions", [])
    caption_texts = [
        c.get("caption", "") if isinstance(c, dict) else str(c)
        for c in raw_captions
    ]

    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        video_title=state.get("_video_title"),
        final_video_url=_url(state.get("final_video")),
        final_thumbnail_url=_url(
            state.get("final_video", "").replace("_final.mp4", "_thumb.jpg")
            if state.get("final_video") else None
        ),
        captions=caption_texts,
        hashtags=state.get("_hashtags", []),
        tone=state.get("_overall_tone"),
        planned_duration=sum(e.get("duration", 0) for e in state.get("story_plan", [])),
        evaluation=state.get("evaluation_score"),
        agent_timings={},
        metadata_url=_url(
            str(Path("outputs") / f"{job_id}_meta.json")
            if (Path("outputs") / f"{job_id}_meta.json").exists() else None
        ),
    )


@router.get("/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Serve final video or thumbnail for download."""
    _get_job(job_id)
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), filename=filename)


@router.post("/{job_id}/enhance", response_model=EnhanceVideoResponse)
async def enhance_video(job_id: str, request: EnhanceVideoRequest):
    import subprocess
    job = _get_job(job_id)
    if job["status"] != JobStatus.done:
        raise HTTPException(400, "Job must be completed to enhance")
    
    state = job.get("final_state")
    if not state or not state.get("final_video"):
        raise HTTPException(400, "No final video found for this job")
        
    input_video = Path(state["final_video"])
    if not input_video.exists():
        raise HTTPException(404, "Original video file not found")
        
    output_video = OUTPUT_DIR / f"{job_id}_enhanced.mp4"
    
    ffmpeg_cmd = [
        str(Path.cwd() / ".venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"),
        "-y", 
        "-i", str(input_video)
    ]
    
    filter_complex = []
    
    # Video filters
    v_filter = ""
    if request.filter_type == "vivid":
        v_filter = "eq=saturation=1.5:contrast=1.1"
    elif request.filter_type == "cinematic":
        v_filter = "colorbalance=rs=.05:gs=.05:bs=-.1"
    elif request.filter_type == "bw":
        v_filter = "hue=s=0"
    elif request.filter_type == "vintage":
        v_filter = "eq=saturation=0.5:contrast=0.8:gamma=1.2"
        
    if v_filter:
        filter_complex.append(f"[0:v]{v_filter}[vout]")
        
    # Audio mixing
    audio_inputs = 1
    if request.music_track != "none":
        music_path = Path.cwd() / f"api/assets/music/{request.music_track}.mp3"
        if music_path.exists():
            ffmpeg_cmd.extend(["-i", str(music_path)])
            # Mix original audio (0:a) with lowered background music (1:a)
            # volume=0.2 for music so it doesn't overpower
            filter_complex.append(f"[1:a]volume=0.2[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]")
            audio_inputs = 2
            
    if filter_complex:
        ffmpeg_cmd.extend(["-filter_complex", ";".join(filter_complex)])
        if v_filter:
            ffmpeg_cmd.extend(["-map", "[vout]"])
        else:
            ffmpeg_cmd.extend(["-map", "0:v"])
            
        if audio_inputs == 2:
            ffmpeg_cmd.extend(["-map", "[aout]"])
        else:
            ffmpeg_cmd.extend(["-map", "0:a"])
    else:
        # Just copy if no filters or music (though why call enhance?)
        ffmpeg_cmd.extend(["-c", "copy"])
        
    # Re-encode settings if filtered
    if v_filter or audio_inputs == 2:
        ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac", "-b:a", "128k"])
        
    ffmpeg_cmd.append(str(output_video))
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("FFMPEG ERROR:", e.stderr.decode())
        raise HTTPException(500, f"FFmpeg processing failed: {e.stderr.decode()}")
        
    # Update state to point to new enhanced video for future downloads/views
    job["final_state"]["final_video"] = str(output_video)
        
    return EnhanceVideoResponse(
        job_id=job_id,
        status="done",
        enhanced_video_url=f"/outputs/{output_video.name}",
        message="Video enhanced successfully"
    )

