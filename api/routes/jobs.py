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
    from pipeline.graph import run_pipeline
    import datetime

    AGENT_ORDER = [
        "scene_agent", "quality_agent", "highlight_agent",
        "caption_agent", "brand_rag_agent", "story_planner_agent",
        "editing_agent", "export_agent",
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
        final_state = run_pipeline(
            job_id=job_id,
            raw_clips=clips,
            user_prompt=request.user_prompt,
            target_duration=request.target_duration,
            platform=request.platform.value,
            brand_pdf_path=request.brand_pdf_path,
            progress_callback=progress_callback,
        )
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

    for p in clip_paths:
        if not Path(p).exists():
            raise HTTPException(400, f"Clip file not found: {p}")

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
    background_tasks.add_task(_run_pipeline_sync, job_id, clip_paths, request, loop)

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
        "scene_agent", "quality_agent", "highlight_agent",
        "caption_agent", "brand_rag_agent", "story_planner_agent",
        "editing_agent", "export_agent",
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

    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        video_title=state.get("video_title"),
        final_video_url=_url(state.get("final_video_path")),
        final_thumbnail_url=_url(state.get("final_thumbnail_path")),
        captions=state.get("captions", []),
        hashtags=state.get("hashtags", []),
        tone=state.get("overall_tone"),
        planned_duration=state.get("planned_duration"),
        evaluation=state.get("evaluation"),
        agent_timings=state.get("agent_timings", {}),
    )


@router.get("/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Serve final video or thumbnail for download."""
    _get_job(job_id)
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), filename=filename)
