"""
Jobs Routes — ClipMind AI API
"""

from __future__ import annotations

import asyncio
import datetime
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
from graphs.workflow import AGENT_ORDER


router = APIRouter(prefix="/api/jobs", tags=["jobs"])

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", "api/assets/music"))

VIDEO_FILTERS = {
    "none": "",
    "vivid": "eq=saturation=1.5:contrast=1.1",
    "cinematic": "colorbalance=rs=.05:gs=.05:bs=-.1",
    "bw": "hue=s=0",
    "vintage": "eq=saturation=0.5:contrast=0.8:gamma=1.2",
}
MUSIC_TRACKS = {"none", "track1", "track2", "track3", "track4", "track5"}

# ── In-memory job store (replace with Redis/DB in production) ──────────────────
_jobs: Dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


def _progress_pct(agent: str, status: str) -> float:
    idx = AGENT_ORDER.index(agent) if agent in AGENT_ORDER else 0
    return round(((idx + (1 if status == "done" else 0)) / len(AGENT_ORDER)) * 100, 1)


def _run_pipeline_sync(job_id: str, clips: list, request: CreateJobRequest, loop: asyncio.AbstractEventLoop):
    """Runs the pipeline in a background thread and streams progress over WebSocket."""
    from api.routes.ws import manager
    from graphs.workflow import run_pipeline
    from memory.state import create_initial_state

    def broadcast(agent: str, status: str, attempt: int, pct: float, message: str = ""):
        payload = {
            "type": "progress",
            "job_id": job_id,
            "agent": agent,
            "status": status,
            "attempt": attempt,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "progress_pct": pct,
            "message": message,
        }
        try:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(job_id, payload), loop)
        except Exception:
            pass

    def on_progress(agent_name: str, status: str, attempt: int):
        job = _jobs.get(job_id)
        if job is None:
            return
        pct = _progress_pct(agent_name, status)
        job["current_agent"] = agent_name
        job["agent_statuses"][agent_name] = {"status": status, "attempt": attempt}
        job["progress_pct"] = pct
        broadcast(agent_name, status, attempt, pct)

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
        final_state = run_pipeline(job_id=job_id, initial_state=initial_state, on_progress=on_progress)
        _jobs[job_id]["final_state"] = final_state

        final_video = final_state.get("final_video")
        if final_video and Path(final_video).exists():
            _jobs[job_id]["status"] = JobStatus.done
            _jobs[job_id]["progress_pct"] = 100.0
            broadcast("pipeline", "done", 0, 100.0, "Pipeline completed successfully")
        else:
            errors = final_state.get("errors", [])
            reason = errors[-1]["message"] if errors else "No video was produced"
            _jobs[job_id]["status"] = JobStatus.failed
            _jobs[job_id]["error"] = reason
            broadcast("pipeline", "error", 0, _jobs[job_id]["progress_pct"], reason)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[job_id]["status"] = JobStatus.failed
        _jobs[job_id]["error"] = str(e)
        broadcast("pipeline", "error", 0, _jobs[job_id]["progress_pct"], str(e))


@router.post("", response_model=JobCreateResponse)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create and enqueue a new video generation job.
    Clip paths (from /api/upload/video) are provided in the request body as clip_paths.
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

    if request.brand_pdf_path and not Path(request.brand_pdf_path).exists():
        raise HTTPException(400, f"Brand PDF not found: {request.brand_pdf_path}")

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
        message=f"Job created. Connect to /api/ws/{job_id} for live progress.",
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = _get_job(job_id)
    final_state = job.get("final_state") or {}
    errors = final_state.get("errors", [])
    timings = final_state.get("agent_timings", {})

    agent_statuses = []
    for name in AGENT_ORDER:
        info = job["agent_statuses"].get(name, {})
        agent_statuses.append({
            "name": name,
            "status": info.get("status", "pending"),
            "elapsed_ms": timings.get(name),
        })

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        current_agent=job.get("current_agent"),
        completed_agents=[a["name"] for a in agent_statuses if a["status"] == "done"],
        agent_statuses=agent_statuses,
        progress_pct=job.get("progress_pct", 0.0),
        error_count=len(errors),
        errors=errors,
        error=job.get("error"),
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Final result. While the job is still pending/running only job_id and status are set."""
    job = _get_job(job_id)
    if job["status"] not in (JobStatus.done, JobStatus.failed):
        return JobResultResponse(job_id=job_id, status=job["status"])

    state = job.get("final_state") or {}

    def _url(path: Optional[str]) -> Optional[str]:
        if not path or not Path(path).exists():
            return None
        return f"/outputs/{Path(path).name}"

    raw_captions = state.get("captions", [])
    caption_texts = [
        c.get("caption", "") if isinstance(c, dict) else str(c)
        for c in raw_captions
    ]

    final_video = state.get("final_video")
    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        video_title=state.get("_video_title"),
        final_video_url=_url(final_video),
        final_thumbnail_url=_url(final_video.replace("_final.mp4", "_thumb.jpg")) if final_video else None,
        captions=caption_texts,
        hashtags=state.get("_hashtags", []),
        tone=state.get("_overall_tone"),
        music_mood=state.get("_music_mood"),
        cta=state.get("_cta"),
        planned_duration=sum(e.get("duration", 0) for e in state.get("story_plan", [])),
        evaluation=state.get("evaluation_score"),
        agent_timings=state.get("agent_timings", {}),
        metadata_url=_url(str(OUTPUT_DIR / f"{job_id}_meta.json")),
    )


@router.get("/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Serve one of this job's output files (video, thumbnail, metadata) for download."""
    _get_job(job_id)
    if Path(filename).name != filename or not filename.startswith(job_id):
        raise HTTPException(400, "Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), filename=filename)


@router.post("/{job_id}/enhance", response_model=EnhanceVideoResponse)
def enhance_video(job_id: str, request: EnhanceVideoRequest):
    """
    Re-render the finished reel with a colour filter and/or background music.
    Always starts from the original export, so enhancements never stack.
    Runs in FastAPI's threadpool (sync def) so FFmpeg does not block the event loop.
    """
    import subprocess
    from tools.ffmpeg import get_ffmpeg

    job = _get_job(job_id)
    if job["status"] != JobStatus.done:
        raise HTTPException(400, "Job must be completed to enhance")
    if request.filter_type not in VIDEO_FILTERS:
        raise HTTPException(400, f"Unknown filter '{request.filter_type}'. Options: {', '.join(VIDEO_FILTERS)}")
    if request.music_track not in MUSIC_TRACKS:
        raise HTTPException(400, f"Unknown music track '{request.music_track}'")
    if request.filter_type == "none" and request.music_track == "none":
        raise HTTPException(400, "Choose a filter or a music track")

    input_video = OUTPUT_DIR / f"{job_id}_final.mp4"
    if not input_video.exists():
        raise HTTPException(404, "Original video file not found")

    music_path = None
    if request.music_track != "none":
        music_path = MUSIC_DIR / f"{request.music_track}.mp3"
        if not music_path.exists():
            raise HTTPException(
                400,
                f"Music file {music_path} is missing. Add your own royalty-free "
                f"{request.music_track}.mp3 to {MUSIC_DIR}/ (see README).",
            )

    # Unique name per combination: /outputs is served with an immutable cache header
    output_video = OUTPUT_DIR / f"{job_id}_enhanced_{request.filter_type}_{request.music_track}.mp4"

    cmd = [get_ffmpeg(), "-y", "-i", str(input_video)]
    if music_path:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    v_filter = VIDEO_FILTERS[request.filter_type]
    filters = []
    if v_filter:
        filters.append(f"[0:v]{v_filter}[vout]")
    if music_path:
        # Background music at 20% under the original audio, cut to the video length
        filters.append("[1:a]volume=0.2[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]")

    cmd += ["-filter_complex", ";".join(filters)]
    cmd += ["-map", "[vout]" if v_filter else "0:v", "-map", "[aout]" if music_path else "0:a"]
    cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "fast"] if v_filter else ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if music_path else ["-c:a", "copy"]
    cmd += [str(output_video), "-loglevel", "error"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")
        raise HTTPException(500, f"FFmpeg processing failed: {stderr[-500:]}")

    return EnhanceVideoResponse(
        job_id=job_id,
        status="done",
        enhanced_video_url=f"/outputs/{output_video.name}",
        message="Video enhanced successfully",
    )
