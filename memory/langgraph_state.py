"""
LangGraph State Schema — AI Video Director
==========================================
The single source of truth that flows through every agent node.
Every agent reads from and writes back to this TypedDict.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


# ── Sub-schemas ────────────────────────────────────────────────────────────────

class SceneMetadata(TypedDict):
    """Represents one extracted scene from a source clip."""
    clip_path: str
    scene_index: int
    start_time: float           # seconds into the original clip
    end_time: float
    duration: float
    thumbnail_path: str
    fps: float
    resolution: tuple           # (width, height)


class QualityScore(TypedDict):
    """Per-scene quality analysis results."""
    scene_index: int
    blur_score: float           # Laplacian variance — higher = sharper
    brightness_score: float     # Mean HSV-V channel (0–255)
    motion_score: float         # Optical flow mean magnitude
    overall_score: float        # Weighted composite
    is_usable: bool             # Passes all thresholds


class HighlightScore(TypedDict):
    """Per-scene highlight / excitement analysis."""
    scene_index: int
    has_face: bool
    face_count: int
    smile_score: float          # 0.0 – 1.0
    audio_peak_score: float     # Librosa onset energy
    excitement_score: float     # Composite highlight score


class ClipPlan(TypedDict):
    """Story planner's ordered cut plan for the final reel."""
    order: int
    scene_index: int
    clip_path: str
    start_time: float
    end_time: float
    duration: float
    transition: str             # cut | fade | zoom | wipe
    caption: str
    caption_start: float        # seconds into final video


class BrandStyle(TypedDict):
    """Brand style guide retrieved from RAG."""
    primary_color: str
    secondary_color: str
    caption_tone: str           # energetic | professional | casual | cinematic
    font_preference: str
    transition_style: str       # fast-cut | smooth | dynamic
    music_preference: str       # upbeat | lo-fi | cinematic | pop
    logo_position: str          # top-left | top-right | bottom-right
    raw_chunks: List[str]       # Raw retrieved text chunks


class EvaluationMetrics(TypedDict):
    """Collected evaluation data for this job run."""
    scene_selection_count: int
    scenes_filtered: int
    retrieval_precision_at_k: float
    reranker_lift: float        # P@K improvement from reranker
    caption_length_avg: float
    latency_total_ms: float
    latency_per_agent: Dict[str, float]
    llm_tokens_used: int
    error_count: int


class AgentError(TypedDict):
    """Structured error record written on agent failure."""
    agent_name: str
    error_type: str
    error_message: str
    retry_attempt: int
    timestamp: str
    recovered: bool


# ── Main State ─────────────────────────────────────────────────────────────────

class VideoDirectorState(TypedDict):
    """
    Central LangGraph state — all agents read from and write to this.
    Uses Annotated reducers so concurrent writes merge correctly.
    """

    # ── Job Identity ──────────────────────────────────────────
    job_id: str
    created_at: str

    # ── User Inputs ───────────────────────────────────────────
    raw_clips: List[str]                        # Absolute paths to uploaded videos
    user_prompt: str                            # e.g. "30-second Instagram reel for hackathon"
    target_duration: int                        # Target output seconds (default: 30)
    platform: str                               # instagram | youtube | tiktok | generic
    brand_pdf_path: Optional[str]               # Optional uploaded brand guide PDF

    # ── Scene Extraction Agent ────────────────────────────────
    scenes: List[SceneMetadata]                 # All extracted scenes
    scene_count: int

    # ── Quality Analysis Agent ────────────────────────────────
    quality_scores: List[QualityScore]
    filtered_scenes: List[SceneMetadata]        # Scenes passing quality threshold
    filtered_count: int

    # ── Highlight Detection Agent ─────────────────────────────
    highlight_scores: List[HighlightScore]
    ranked_scenes: List[SceneMetadata]          # Scenes sorted by excitement score
    transcript: str                             # Whisper transcription of audio
    audio_path: Optional[str]                   # Extracted audio track

    # ── Caption Generator Agent ───────────────────────────────
    video_title: str
    captions: List[str]                         # Per-scene caption suggestions
    hashtags: List[str]
    overall_tone: str                           # Detected mood from Gemini

    # ── Brand RAG Agent ───────────────────────────────────────
    brand_style: Optional[BrandStyle]
    retrieved_doc_ids: List[str]               # IDs of retrieved chunks
    reranked_doc_ids: List[str]                # After BGE reranking
    rag_query: str                              # Query sent to vector DB

    # ── Story Planner Agent ───────────────────────────────────
    story_arc: List[ClipPlan]                  # The ordered edit plan
    selected_music_path: Optional[str]
    planned_duration: float                    # Actual planned duration

    # ── Video Editing Agent ───────────────────────────────────
    edited_video_path: Optional[str]
    editing_progress: float                    # 0.0 – 1.0

    # ── Export Agent ──────────────────────────────────────────
    final_video_path: Optional[str]
    final_thumbnail_path: Optional[str]
    export_metadata: Dict[str, Any]

    # ── Pipeline Orchestration ────────────────────────────────
    current_agent: str
    completed_agents: Annotated[List[str], operator.add]    # Accumulates
    errors: Annotated[List[AgentError], operator.add]       # Accumulates all errors
    retry_counts: Dict[str, int]
    agent_timings: Dict[str, float]             # Agent name → ms elapsed

    # ── Evaluation ────────────────────────────────────────────
    evaluation: Optional[EvaluationMetrics]


# ── Default State Factory ──────────────────────────────────────────────────────

def create_initial_state(
    job_id: str,
    raw_clips: List[str],
    user_prompt: str,
    target_duration: int = 30,
    platform: str = "instagram",
    brand_pdf_path: Optional[str] = None,
    created_at: Optional[str] = None,
) -> VideoDirectorState:
    """Build an initial state dict for a new video director job."""
    import datetime
    return VideoDirectorState(
        job_id=job_id,
        created_at=created_at or datetime.datetime.utcnow().isoformat(),
        raw_clips=raw_clips,
        user_prompt=user_prompt,
        target_duration=target_duration,
        platform=platform,
        brand_pdf_path=brand_pdf_path,
        scenes=[],
        scene_count=0,
        quality_scores=[],
        filtered_scenes=[],
        filtered_count=0,
        highlight_scores=[],
        ranked_scenes=[],
        transcript="",
        audio_path=None,
        video_title="",
        captions=[],
        hashtags=[],
        overall_tone="",
        brand_style=None,
        retrieved_doc_ids=[],
        reranked_doc_ids=[],
        rag_query="",
        story_arc=[],
        selected_music_path=None,
        planned_duration=0.0,
        edited_video_path=None,
        editing_progress=0.0,
        final_video_path=None,
        final_thumbnail_path=None,
        export_metadata={},
        current_agent="",
        completed_agents=[],
        errors=[],
        retry_counts={},
        agent_timings={},
        evaluation=None,
    )
