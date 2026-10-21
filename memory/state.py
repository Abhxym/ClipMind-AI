"""
VideoState Schema
=================
Central state object for the 14-agent AI Video Director workflow.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


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
    scene_descriptions: List[Dict[str, Any]]    # Florence-2 / BLIP descriptions
    
    # ── Object Detection & Emotion Agent ──
    objects: List[Dict[str, Any]]               # YOLO detected objects & Emotion scores
    
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
    evaluation_score: Dict[str, float]          # Quality/Ragas scores
    
    # ── Orchestration & Observability ──
    logs: Annotated[List[str], operator.add]
    current_agent: str
    errors: Annotated[List[Dict[str, Any]], operator.add]
    retry_count: Dict[str, int]                 # Track retries for Review->Edit loops

    # ── Internal Agent State ──
    _clip_metadata: List[Dict[str, Any]]
    _video_title: Optional[str]
    _hashtags: List[str]
    _overall_tone: Optional[str]
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
        _clip_metadata=[],
        _video_title=None,
        _hashtags=[],
        _overall_tone=None,
        _edited_video_path=None
    )
