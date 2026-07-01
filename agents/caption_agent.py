"""
Caption Generator Agent — AI Video Director
============================================
Uses Google Gemini Flash to generate:
  • A compelling video title
  • Per-scene captions
  • Relevant hashtags
  • Tone analysis of the content
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from logs.agent_logs import AgentLogger
from memory.langgraph_state import SceneMetadata, VideoDirectorState


GEMINI_CAPTION_MODEL = os.getenv("GEMINI_MODEL_CAPTION", "gemini-1.5-flash")
MAX_SCENES_TO_SEND = 5   # Send top N thumbnails to Gemini to control cost


def _encode_image(path: str) -> Optional[str]:
    """Base64-encode a JPEG thumbnail for Gemini multimodal input."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _build_caption_prompt(
    user_prompt: str,
    platform: str,
    target_duration: int,
    transcript: str,
    scene_count: int,
) -> str:
    return f"""
You are an expert social media video director AI.

User's video brief: "{user_prompt}"
Platform: {platform}
Target duration: {target_duration} seconds
Number of selected scenes: {scene_count}
Audio transcript excerpt: "{transcript[:800] if transcript else 'N/A'}"

Based on the video thumbnails provided and the brief above, generate:

1. A compelling, platform-native VIDEO TITLE (max 60 chars)
2. {min(scene_count, 5)} SHORT SCENE CAPTIONS (max 8 words each, punchy and engaging)
3. 10–15 RELEVANT HASHTAGS including trending ones for {platform}
4. The overall TONE of the video (one of: energetic, cinematic, casual, professional, emotional, humorous)

Respond ONLY with valid JSON in this exact format:
{{
  "title": "...",
  "captions": ["...", "...", "..."],
  "hashtags": ["#...", "#...", "..."],
  "tone": "..."
}}
""".strip()


def generate_captions(
    user_prompt: str,
    platform: str,
    target_duration: int,
    transcript: str,
    top_scenes: List[SceneMetadata],
) -> dict:
    """
    Call Gemini Flash with scene thumbnails + context.
    Returns parsed JSON with title, captions, hashtags, tone.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_CAPTION_MODEL,
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Build multimodal message content
    content: list = [
        {
            "type": "text",
            "text": _build_caption_prompt(
                user_prompt, platform, target_duration, transcript, len(top_scenes)
            ),
        }
    ]

    # Attach top-N thumbnails
    for i, scene in enumerate(top_scenes[:MAX_SCENES_TO_SEND]):
        thumb = scene.get("thumbnail_path", "")
        encoded = _encode_image(thumb) if thumb else None
        if encoded:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                },
            })

    try:
        response = llm.invoke([HumanMessage(content=content)])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        # Graceful fallback
        return {
            "title": f"Epic {platform.title()} Reel",
            "captions": [f"Scene {i+1}" for i in range(len(top_scenes))],
            "hashtags": ["#viral", "#reel", "#video"],
            "tone": "energetic",
        }


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def caption_generator_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Caption Generator Agent.
    Reads:  state.ranked_scenes, state.transcript, state.user_prompt
    Writes: state.video_title, state.captions, state.hashtags, state.overall_tone
    """
    job_id = state["job_id"]
    log = AgentLogger("caption_agent", job_id)
    log.start("Generating captions with Gemini Flash")

    top_scenes = state["ranked_scenes"][:MAX_SCENES_TO_SEND]

    # Fallback if no scenes available
    if not top_scenes:
        log.warning("No ranked scenes available — using fallback captions")
        result = {
            "title": f"Epic {state['platform'].title()} Reel",
            "captions": [],
            "hashtags": ["#viral", "#reel", "#video"],
            "tone": "energetic",
        }
    else:
        result = generate_captions(
            user_prompt=state["user_prompt"],
            platform=state["platform"],
            target_duration=state["target_duration"],
            transcript=state["transcript"],
            top_scenes=top_scenes,
        )

    title = result.get("title", "")
    captions = result.get("captions", [])
    hashtags = result.get("hashtags", [])
    tone = result.get("tone", "energetic")

    log.info(f'Title: "{title}"')
    log.info(f'Tone: {tone}')
    log.info(f'Hashtags: {" ".join(hashtags[:5])} ...')
    log.success(f"Generated {len(captions)} captions")

    return {
        **state,
        "video_title": title,
        "captions": captions,
        "hashtags": hashtags,
        "overall_tone": tone,
        "current_agent": "caption_agent",
        "completed_agents": ["caption_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "caption_agent": log.elapsed_ms(),
        },
    }
