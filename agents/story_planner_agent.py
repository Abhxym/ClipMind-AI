"""
Story Planner Agent — AI Video Director
=======================================
Uses Gemini Pro to generate a narrative story arc:
  • Scene ordering
  • Per-scene duration allocation
  • Transition types between scenes
  • Background music selection
  • Caption assignment

Respects: target_duration, platform, brand_style, ranked_scenes
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from logs.agent_logs import AgentLogger
from memory.langgraph_state import (
    BrandStyle,
    ClipPlan,
    SceneMetadata,
    VideoDirectorState,
)


GEMINI_PLANNER_MODEL = os.getenv("GEMINI_MODEL_PLANNER", "gemini-1.5-pro")
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", "assets/sample_music"))


def _select_music(brand_style: Optional[BrandStyle], tone: str) -> Optional[str]:
    """Select a music track from the assets directory matching the mood."""
    if not MUSIC_DIR.exists():
        return None

    pref = (brand_style or {}).get("music_preference", tone)
    tracks = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))

    # Try to match filename keywords
    mood_map = {
        "upbeat":    ["upbeat", "energetic", "hype", "pop"],
        "cinematic": ["cinematic", "epic", "dramatic", "orchestral"],
        "lo-fi":     ["lofi", "lo-fi", "chill", "ambient"],
        "corporate": ["corporate", "professional", "business"],
        "ambient":   ["ambient", "soft", "calm", "peaceful"],
        "pop":       ["pop", "catchy", "fun"],
    }
    keywords = mood_map.get(pref, ["upbeat"])

    for track in tracks:
        if any(kw in track.stem.lower() for kw in keywords):
            return str(track)

    # Fallback: pick any available track
    return str(random.choice(tracks)) if tracks else None


def _build_planner_prompt(
    user_prompt: str,
    platform: str,
    target_duration: int,
    tone: str,
    brand_style: Optional[BrandStyle],
    ranked_scenes: List[SceneMetadata],
    captions: List[str],
) -> str:
    scenes_summary = []
    for i, sc in enumerate(ranked_scenes[:15]):  # send at most 15 scenes
        scenes_summary.append(
            f"  Scene {sc['scene_index']} | duration={sc['duration']:.1f}s | clip={Path(sc['clip_path']).name}"
        )

    brand_context = ""
    if brand_style:
        brand_context = (
            f"Brand style: transitions={brand_style['transition_style']}, "
            f"music={brand_style['music_preference']}, "
            f"tone={brand_style['caption_tone']}"
        )

    return f"""
You are an expert AI video editor creating a {platform} reel.

User brief: "{user_prompt}"
Target duration: {target_duration} seconds
Tone: {tone}
{brand_context}

Available scenes (ranked by excitement, best first):
{chr(10).join(scenes_summary)}

Available captions: {captions[:10]}

Create a STORY ARC for the final {target_duration}-second reel.
Rules:
- Select exactly the scenes needed to fill ~{target_duration} seconds
- Each scene should be 2–6 seconds long
- Transitions: cut | fade | zoom | wipe
- Assign one caption per scene
- Start with the most exciting scene
- End with a strong closing scene

Respond ONLY with valid JSON:
{{
  "story_arc": [
    {{
      "order": 1,
      "scene_index": <int>,
      "start_time": <float>,
      "end_time": <float>,
      "transition": "<cut|fade|zoom|wipe>",
      "caption": "<caption text>"
    }},
    ...
  ],
  "planned_duration": <float>
}}
""".strip()


def plan_story(
    state: VideoDirectorState,
) -> tuple[List[ClipPlan], float]:
    """Call Gemini Pro to generate the story arc. Returns (clip_plans, total_duration)."""
    # Fallback if no ranked scenes
    if not state["ranked_scenes"]:
        return [], 0.0

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_PLANNER_MODEL,
        temperature=0.4,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    prompt = _build_planner_prompt(
        user_prompt=state["user_prompt"],
        platform=state["platform"],
        target_duration=state["target_duration"],
        tone=state.get("overall_tone", "energetic"),
        brand_style=state.get("brand_style"),
        ranked_scenes=state["ranked_scenes"],
        captions=state.get("captions", []),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception:
        # Graceful fallback: use top scenes with equal split
        data = _fallback_story_arc(state)

    # Build scene index lookup
    scene_map = {s["scene_index"]: s for s in state["ranked_scenes"]}
    used_scenes = set()

    clip_plans: List[ClipPlan] = []
    cursor = 0.0
    for item in data.get("story_arc", []):
        scene = scene_map.get(item["scene_index"])
        if not scene:
            # Fallback: pick the highest ranked scene not yet used
            for fb_scene in state["ranked_scenes"]:
                if fb_scene["scene_index"] not in used_scenes:
                    scene = fb_scene
                    item["scene_index"] = scene["scene_index"]
                    break
            if not scene:
                continue
        
        used_scenes.add(scene["scene_index"])
        start = float(item.get("start_time", scene["start_time"]))
        end = float(item.get("end_time", scene["end_time"]))
        duration = end - start

        clip_plans.append(ClipPlan(
            order=item["order"],
            scene_index=item["scene_index"],
            clip_path=scene["clip_path"],
            start_time=start,
            end_time=end,
            duration=duration,
            transition=item.get("transition", "cut"),
            caption=item.get("caption", ""),
            caption_start=cursor,
        ))
        cursor += duration

    return clip_plans, float(data.get("planned_duration", cursor))


def _fallback_story_arc(state: VideoDirectorState) -> dict:
    """Simple fallback when Gemini is unavailable."""
    target = state["target_duration"]
    captions = state.get("captions", [])
    scenes = state["ranked_scenes"]
    clips_needed = max(1, target // 4)

    arc = []
    cursor = 0.0
    order = 1

    for sc in scenes:
        starts = []
        cur_st = sc["start_time"]
        while cur_st < sc["end_time"]:
            clip_dur = min(4.0, sc["end_time"] - cur_st)
            if clip_dur >= 2.0:
                starts.append((cur_st, clip_dur))
            cur_st += 4.0
            
        for st, cdur in starts:
            arc.append({
                "order": order,
                "scene_index": sc["scene_index"],
                "start_time": st,
                "end_time": st + cdur,
                "transition": "cut",
                "caption": captions[order-1] if order-1 < len(captions) else f"Scene {order}",
            })
            cursor += cdur
            order += 1
            if len(arc) >= clips_needed:
                break
        if len(arc) >= clips_needed:
            break

    if arc:
        arc[-1]["transition"] = "fade"

    return {"story_arc": arc, "planned_duration": cursor}


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def story_planner_agent(state: VideoDirectorState) -> VideoDirectorState:
    """
    LangGraph node: Story Planner Agent.
    Reads:  state.ranked_scenes, state.brand_style, state.captions, state.overall_tone
    Writes: state.story_arc, state.selected_music_path, state.planned_duration
    """
    job_id = state["job_id"]
    log = AgentLogger("story_planner_agent", job_id)
    log.start("Planning story arc with Gemini Pro")

    story_arc, planned_duration = plan_story(state)
    music_path = _select_music(state.get("brand_style"), state.get("overall_tone", "energetic"))

    log.info(f"Story arc: {len(story_arc)} clips | planned duration: {planned_duration:.1f}s")
    log.info(f"Music: {Path(music_path).name if music_path else 'none'}")
    log.success(f"Transitions used: {set(c['transition'] for c in story_arc)}")

    return {
        **state,
        "story_arc": story_arc,
        "selected_music_path": music_path,
        "planned_duration": planned_duration,
        "current_agent": "story_planner_agent",
        "completed_agents": ["story_planner_agent"],
        "agent_timings": {
            **state.get("agent_timings", {}),
            "story_planner_agent": log.elapsed_ms(),
        },
    }
