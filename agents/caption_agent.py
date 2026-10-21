"""
Caption Agent — ClipMind AI
=====================================
Pipeline Step 9: Caption & Hashtag Generation using Gemini (GEMINI_MODEL_CAPTION).
  - Generates an engaging video title based on the story plan and user brief
  - Writes per-scene captions (burned into individual clip segments)
  - Generates platform-optimized hashtags
  - Determines overall tone and music mood for the editing agent
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from memory.state import VideoState

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_CAPTION = os.getenv("GEMINI_MODEL_CAPTION", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

PLATFORM_HASHTAG_COUNT = {
    "instagram": 25,
    "tiktok": 10,
    "youtube": 15,
    "generic": 10,
}

PLATFORM_CAPTION_STYLE = {
    "instagram": "punchy, emoji-rich, lifestyle-focused",
    "tiktok": "casual, energetic, trending slang OK",
    "youtube": "descriptive, keyword-rich, informative",
    "generic": "neutral and engaging",
}


def _build_caption_prompt(state: VideoState) -> str:
    platform = state.get("platform", "instagram")
    user_prompt = state.get("user_prompt", "Create an engaging video")
    story_plan = state.get("story_plan", [])
    scene_descriptions = state.get("scene_descriptions", [])
    transcript = state.get("transcript", [])

    # Summarize scene captions
    plan_summary = ""
    for i, entry in enumerate(story_plan):
        clip_idx = entry.get("clip_index", 0)
        desc_match = next((d for d in scene_descriptions if d.get("clip_index") == clip_idx), {})
        mood = desc_match.get("mood", "neutral")
        hint = entry.get("caption_hint", desc_match.get("description", ""))[:100]
        plan_summary += f"Scene {i+1}: mood={mood}, hint=\"{hint}\"\n"

    transcript_excerpt = " ".join(s.get("text", "") for s in transcript[:10])[:300]

    hashtag_count = PLATFORM_HASHTAG_COUNT.get(platform, 10)
    style = PLATFORM_CAPTION_STYLE.get(platform, "engaging")

    prompt = f"""You are a social media content creator specializing in {platform} video content.

USER BRIEF: {user_prompt}
PLATFORM: {platform} (caption style: {style})
TRANSCRIPT EXCERPT: {transcript_excerpt}

STORY PLAN ({len(story_plan)} scenes):
{plan_summary}

Generate the following in valid JSON:
1. "video_title": A catchy, compelling title for the video (max 60 chars)
2. "overall_tone": One word describing the video's tone (e.g., "inspiring", "energetic", "emotional")
3. "music_mood": Music style recommendation (e.g., "upbeat electronic", "cinematic orchestral", "lo-fi chill")
4. "scene_captions": List of {len(story_plan)} captions — one per scene, max 8 words each, in the same order as the story plan
5. "hashtags": List of exactly {hashtag_count} relevant hashtags (include # prefix)
6. "cta": A call-to-action line for the video description (max 100 chars)

Respond ONLY with valid JSON:
{{
  "video_title": "...",
  "overall_tone": "...",
  "music_mood": "...",
  "scene_captions": ["...", "..."],
  "hashtags": ["#...", "#..."],
  "cta": "..."
}}"""

    return prompt


def _generate_with_gemini(state: VideoState) -> Tuple[Dict[str, Any], str]:
    """Use Gemini to generate captions. Returns (result, failure_reason)."""
    if not GOOGLE_API_KEY:
        return {}, "GOOGLE_API_KEY not set"
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)

        prompt = _build_caption_prompt(state)
        response = client.models.generate_content(
            model=GEMINI_MODEL_CAPTION,
            contents=prompt
        )
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        if isinstance(result, dict):
            return result, ""
        return {}, "response was not a JSON object"
    except ImportError:
        return {}, "google-genai not installed"
    except json.JSONDecodeError:
        return {}, "response was not valid JSON"
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {str(exc)[:120]}"


def _short_caption(text: str, max_words: int = 8) -> str:
    """First sentence of a description, cut to max_words on a word boundary."""
    first = text.split(". ")[0].strip().rstrip(".")
    words = first.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def _short_title(text: str, max_chars: int = 60) -> str:
    """Brief → title: cut on a word boundary instead of mid-word."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "..."


def _fallback_captions(state: VideoState) -> Dict[str, Any]:
    """Generate basic captions without Gemini."""
    story_plan = state.get("story_plan", [])
    scene_descriptions = state.get("scene_descriptions", [])
    platform = state.get("platform", "instagram")
    user_prompt = state.get("user_prompt", "")

    scene_captions = []
    for entry in story_plan:
        clip_idx = entry.get("clip_index", 0)
        hint = entry.get("caption_hint", "")
        if not hint:
            desc_match = next((d for d in scene_descriptions if d.get("clip_index") == clip_idx), {})
            hint = desc_match.get("description", "")
        scene_captions.append(_short_caption(hint))  # "" = no caption burned in

    default_hashtags = {
        "instagram": ["#reels", "#viral", "#content", "#video", "#creative", "#trending"],
        "tiktok": ["#fyp", "#viral", "#trending", "#video", "#content"],
        "youtube": ["#youtube", "#video", "#content", "#watch"],
        "generic": ["#video", "#content", "#creative"],
    }

    return {
        "video_title": _short_title(user_prompt) or "AI Generated Reel",
        "overall_tone": "dynamic",
        "music_mood": "upbeat electronic",
        "scene_captions": scene_captions,
        "hashtags": default_hashtags.get(platform, default_hashtags["generic"]),
        "cta": "Like and follow for more amazing content!",
    }


def caption_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Caption Generation Agent.
    Reads:  story_plan, scene_descriptions, user_prompt, platform
    Writes: captions, _video_title, _hashtags, _music_mood, _overall_tone
    """
    logs: List[str] = ["[caption_agent] Generating captions and hashtags"]
    errors: List[Dict[str, Any]] = []

    # Try Gemini
    result, reason = _generate_with_gemini(state)

    if result:
        logs.append(f"[caption_agent] Gemini ({GEMINI_MODEL_CAPTION}) generated captions successfully")
    else:
        logs.append(f"[caption_agent] Gemini unavailable ({reason}) — using fallback caption generator")
        result = _fallback_captions(state)

    video_title = result.get("video_title", "AI Generated Reel")
    overall_tone = result.get("overall_tone", "dynamic")
    music_mood = result.get("music_mood", "upbeat")
    scene_captions_raw = result.get("scene_captions", [])
    hashtags = result.get("hashtags", [])
    cta = result.get("cta", "")

    # Build structured captions list aligned with story plan
    story_plan = state.get("story_plan", [])
    captions: List[Dict[str, Any]] = []
    for i, entry in enumerate(story_plan):
        caption_text = scene_captions_raw[i] if i < len(scene_captions_raw) else ""
        captions.append({
            "scene_index": i,
            "clip_index": entry.get("clip_index", 0),
            "start": entry.get("start", 0.0),
            "end": entry.get("end", 0.0),
            "caption": caption_text,
        })

    logs.append(f"[caption_agent] Title: \"{video_title}\"")
    logs.append(f"[caption_agent] Tone: {overall_tone} | Music: {music_mood}")
    logs.append(f"[caption_agent] Hashtags: {' '.join(hashtags[:5])}...")
    logs.append(f"[caption_agent] Scene captions: {len(captions)}")

    return {
        **state,
        "captions": captions,
        "_video_title": video_title,
        "_hashtags": hashtags,
        "_music_mood": music_mood,
        "_overall_tone": overall_tone,
        "_cta": cta,
        "current_agent": "caption_agent",
        "errors": errors,
        "logs": logs,
    }
