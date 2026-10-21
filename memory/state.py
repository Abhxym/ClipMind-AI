"""
VideoState Schema
=================
Central state object for the 13-agent ClipMind AI LangGraph workflow.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


def merge_timings(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Reducer for agent_timings: adds up time when a node runs more than once (review loop)."""
    merged = dict(left or {})
    for agent, ms in (right or {}).items():
        merged[agent] = round(merged.get(agent, 0.0) + ms, 1)
    return merged


class VideoState(TypedDict):
    """
    Shared LangGraph state representing the entire lifecycle of a video project.
    """
    # ── Job Identity ──
    job_id: str
    
    # ── Input Agent ──
    user_prompt: str
    uploaded_videos: List[str]
    target_duration: int
    platform: str
    brand_pdf_path: Optional[str]
    
    # ── Preprocessing Agent ──
    frames: List[Dict[str, Any]]                # Extracted frame metadata
    
    # ── Whisper Agent ──
    transcript: List[Dict[str, Any]]            # Time-stamped transcript
    
    # ── Scene Understanding Agent ──
    scene_descriptions: List[Dict[str, Any]]    # Gemini Vision (BLIP fallback)
    
    # ── Object Detection & Emotion Agent ──
    objects: List[Dict[str, Any]]               # Gemini / YOLO objects, emotion, excitement
    
    # ── Embedding & RAG Agents ──
    retrieved_brand_docs: List[Dict[str, Any]]  # Docs from vector DB (RAG)
    
    # ── Story Planner Agent ──
    story_plan: List[Dict[str, Any]]            # LLM-generated scene sequence
    
    # ── Caption Agent ──
    captions: List[Dict[str, Any]]              # Time-stamped captions
    
    # ── Editing Agent ──
    editing_commands: List[Dict[str, Any]]      # FFmpeg instructions generated
    
    # ── Reviewer & Export Agents ──
    final_video: Optional[str]                  # Path to exported MP4
    
    # ── Evaluation Agent ──
    evaluation_score: Dict[str, Any]            # Duration accuracy, edit success, etc.
    
    # ── Orchestration & Observability ──
    logs: Annotated[List[str], operator.add]
    current_agent: str
    errors: Annotated[List[Dict[str, Any]], operator.add]
    retry_count: Dict[str, int]                 # Track retries for Review->Edit loops
    review_passed: bool                         # Set by reviewer_agent, read by the router
    agent_timings: Annotated[Dict[str, float], merge_timings]  # ms per agent

    # ── Internal Agent State ──
    _clip_metadata: List[Dict[str, Any]]
    _video_title: Optional[str]
    _hashtags: List[str]
    _overall_tone: Optional[str]
    _music_mood: Optional[str]
    _cta: Optional[str]
    _edited_video_path: Optional[str]


def create_initial_state(
    job_id: str,
    user_prompt: str,
    uploaded_videos: List[str],
    target_duration: int = 30,
    platform: str = "instagram",
    brand_pdf_path: Optional[str] = None
) -> VideoState:
    return VideoState(
        job_id=job_id,
        user_prompt=user_prompt,
        uploaded_videos=uploaded_videos,
        target_duration=target_duration,
        platform=platform,
        brand_pdf_path=brand_pdf_path,
        frames=[],
        transcript=[],
        scene_descriptions=[],
        objects=[],
        retrieved_brand_docs=[],
        story_plan=[],
        captions=[],
        editing_commands=[],
        final_video=None,
        evaluation_score={},
        logs=[],
        current_agent="",
        errors=[],
        retry_count={},
        review_passed=False,
        agent_timings={},
        _clip_metadata=[],
        _video_title=None,
        _hashtags=[],
        _overall_tone=None,
        _music_mood=None,
        _cta=None,
        _edited_video_path=None
    )
