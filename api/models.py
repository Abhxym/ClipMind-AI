"""
Pydantic Models — AI Video Director API
=========================================
Request and response schemas for all FastAPI routes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    instagram = "instagram"
    tiktok = "tiktok"
    youtube = "youtube"
    generic = "generic"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


# ── Request Models ─────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    user_prompt: str = Field(..., description="Natural language brief for the reel")
    target_duration: int = Field(30, ge=10, le=300, description="Target output duration in seconds")
    platform: Platform = Platform.instagram
    brand_pdf_path: Optional[str] = Field(None, description="Server path to uploaded brand PDF")
    clip_paths: List[str] = Field(default_factory=list, description="Server paths to uploaded video clips")


class EnhanceVideoRequest(BaseModel):
    filter_type: str = Field("none", description="Filter to apply: none, vivid, cinematic, bw, vintage")
    music_track: str = Field("none", description="Music track to mix: none, track1, track2, track3, track4, track5")


# ── Response Models ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    server_path: str


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class AgentStatus(BaseModel):
    name: str
    status: str         # pending | running | done | error
    elapsed_ms: Optional[float] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_agent: Optional[str] = None
    completed_agents: List[str] = []
    agent_statuses: List[AgentStatus] = []
    progress_pct: float = 0.0
    error_count: int = 0
    errors: List[Dict[str, Any]] = []


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_title: Optional[str] = None
    final_video_url: Optional[str] = None
    final_thumbnail_url: Optional[str] = None
    captions: List[str] = []
    hashtags: List[str] = []
    tone: Optional[str] = None
    planned_duration: Optional[float] = None
    evaluation: Optional[Dict[str, Any]] = None
    agent_timings: Dict[str, float] = {}
    metadata_url: Optional[str] = None


class EnhanceVideoResponse(BaseModel):
    job_id: str
    status: str
    enhanced_video_url: str
    message: str


# ── WebSocket Messages ─────────────────────────────────────────────────────────

class WSProgressMessage(BaseModel):
    type: str = "progress"
    job_id: str
    agent: str
    status: str         # running | done | error
    attempt: int = 0
    timestamp: str
    progress_pct: float = 0.0
    message: Optional[str] = None
