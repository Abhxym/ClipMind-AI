"""
Planner Agent — AI Video Director
=====================================
Pipeline Step 8: Story Planning using Gemini Pro.
  - Takes scene descriptions, transcript, object metadata, and brand docs
  - Uses Gemini 1.5 Pro to compose a structured story arc
  - Selects which clips to use, in what order, for what duration
  - Outputs a story_plan list: [{clip_index, start, end, reason, duration, caption_hint}]
  - Robust fallback if Gemini is unavailable
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memory.state import VideoState

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_PLANNER = os.getenv("GEMINI_MODEL_PLANNER", "gemini-2.5-pro")


def _build_planner_prompt(state: VideoState) -> str:
    """Construct the structured prompt for the story planner."""
    user_prompt = state.get("user_prompt", "Create an engaging reel")
    target_duration = state.get("target_duration", 30)
    platform = state.get("platform", "instagram")
    scene_descriptions = state.get("scene_descriptions", [])
    objects = state.get("objects", [])
    transcript = state.get("transcript", [])
    brand_docs = state.get("retrieved_brand_docs", [])

    # Build scene summary table
    scenes_text = ""
    for i, desc in enumerate(scene_descriptions):
        obj_data = next((o for o in objects if o.get("clip_index") == i), {})
        clip_path = desc.get("clip_path", f"clip_{i}")
        duration = obj_data.get("end", 0) - obj_data.get("start", 0)
        scenes_text += (
            f"Clip {i}: path={Path(clip_path).name}, duration={duration:.1f}s, "
            f"mood={desc.get('mood','?')}, description=\"{desc.get('description','')[:120]}\", "
            f"objects={obj_data.get('object_labels',[])[:5]}, "
            f"excitement={obj_data.get('excitement_score', 5):.1f}/10\n"
        )

    # Transcript summary (first 500 chars)
    transcript_text = " ".join(s.get("text", "") for s in transcript[:20])[:500] if transcript else "No speech detected"

    # Brand context
    brand_text = ""
    if brand_docs:
        brand_text = "\n".join(d.get("text", "")[:150] for d in brand_docs[:3])
    else:
        brand_text = "No brand guidelines available. Use a dynamic, engaging style."

    prompt = f"""You are a professional video editor creating a {platform} reel.

USER BRIEF: {user_prompt}
TARGET DURATION: {target_duration} seconds
PLATFORM: {platform}

AVAILABLE CLIPS:
{scenes_text}

TRANSCRIPT EXCERPT:
{transcript_text}

BRAND/STYLE CONTEXT:
{brand_text}

TASK: Create an optimal story plan. Select clips that together form a compelling narrative arc.
The total duration of selected clips MUST be as close to {target_duration} seconds as possible.

For each selected clip, specify:
- clip_index: integer (0-based index from the available clips list)
- start: float (start time within the clip, in seconds — use 0.0 if the full clip is desired)
- end: float (end time within the clip, in seconds — use the clip's actual duration if full clip)
- duration: float (end - start)
- reason: string (why this clip is included in the story)
- caption_hint: string (a brief caption idea for this scene)
- transition: string (cut | fade | dissolve — default "cut")

Respond ONLY with valid JSON array:
[
  {{"clip_index": 0, "start": 0.0, "end": 5.0, "duration": 5.0, "reason": "...", "caption_hint": "...", "transition": "cut"}},
  ...
]

Rules:
1. Total sum of durations MUST equal approximately {target_duration} seconds.
2. Pick the BEST clips by excitement, mood match, and narrative flow.
3. Each clip segment should be between 2 and 10 seconds.
4. If one clip isn't long enough, split it into multiple entries with different start/end.
5. Create a story arc: opening hook → build → climax → resolution.
"""
    return prompt


def _plan_with_gemini(state: VideoState) -> List[Dict[str, Any]]:
    """Call Gemini Pro to generate the story plan."""
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)

        prompt = _build_planner_prompt(state)
        response = client.models.generate_content(
            model=GEMINI_MODEL_PLANNER,
            contents=prompt
        )
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        plan = json.loads(text)
        if isinstance(plan, list):
            return plan
        return []
    except ImportError:
        return []
    except json.JSONDecodeError:
        return []
    except Exception:
        return []


def _fallback_plan(state: VideoState) -> List[Dict[str, Any]]:
    """Build a simple story plan without Gemini."""
    target = state.get("target_duration", 30)
    objects = state.get("objects", [])
    scene_descriptions = state.get("scene_descriptions", [])

    if not objects and not scene_descriptions:
        return []

    # Sort clips by excitement score
    scored_clips = sorted(
        objects,
        key=lambda o: o.get("excitement_score", 5.0),
        reverse=True
    )

    plan = []
    total = 0.0
    target_f = float(target)

    for clip_data in scored_clips:
        if total >= target_f:
            break
        idx = clip_data.get("clip_index", clip_data.get("global_index", 0))
        clip_end = clip_data.get("end", 10.0)
        clip_start = clip_data.get("start", 0.0)
        available = clip_end - clip_start

        if available <= 0:
            available = 10.0
            clip_end = 10.0

        # Take up to 10 seconds, or what remains of the target
        segment_dur = min(available, 10.0, target_f - total)
        if segment_dur < 1.0:
            break

        desc_match = next((d for d in scene_descriptions if d.get("clip_index") == idx), {})

        plan.append({
            "clip_index": idx,
            "start": round(clip_start, 3),
            "end": round(clip_start + segment_dur, 3),
            "duration": round(segment_dur, 3),
            "reason": f"High excitement score ({clip_data.get('excitement_score', 5):.1f}/10)",
            "caption_hint": desc_match.get("description", "")[:80],
            "transition": "cut",
        })
        total += segment_dur

    # If we still don't have enough, pad with first clips
    if total < target_f * 0.7 and scene_descriptions:
        for i, desc in enumerate(scene_descriptions):
            if total >= target_f:
                break
            if any(p["clip_index"] == i for p in plan):
                continue
            segment_dur = min(10.0, target_f - total)
            plan.append({
                "clip_index": i,
                "start": 0.0,
                "end": round(segment_dur, 3),
                "duration": round(segment_dur, 3),
                "reason": "Fallback: padding to reach target duration",
                "caption_hint": desc.get("description", "")[:80],
                "transition": "cut",
            })
            total += segment_dur

    return plan


def planner_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Story Planner Agent.
    Reads:  scene_descriptions, objects, transcript, retrieved_brand_docs, user_prompt
    Writes: story_plan
    """
    logs: List[str] = ["[planner_agent] Building story plan with Gemini Pro"]
    errors: List[Dict[str, Any]] = []

    target_duration = state.get("target_duration", 30)

    # Try Gemini first
    plan = _plan_with_gemini(state)

    if plan:
        logs.append(f"[planner_agent] Gemini generated {len(plan)} scene entries")
    else:
        logs.append("[planner_agent] Gemini unavailable — using fallback planner")
        plan = _fallback_plan(state)

    # Validate and sanitize the plan
    clips = state.get("uploaded_videos", [])
    objects = state.get("objects", [])
    validated_plan = []

    for entry in plan:
        clip_idx = entry.get("clip_index", 0)

        # Clamp to available clips
        if clip_idx >= len(clips):
            clip_idx = clip_idx % max(len(clips), 1)
            entry["clip_index"] = clip_idx

        # Ensure start < end
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start + 5.0))
        if end <= start:
            end = start + 5.0
        entry["start"] = round(start, 3)
        entry["end"] = round(end, 3)
        entry["duration"] = round(end - start, 3)

        validated_plan.append(entry)

    total_planned = sum(e.get("duration", 0) for e in validated_plan)
    logs.append(
        f"[planner_agent] Final plan: {len(validated_plan)} segments, "
        f"total={total_planned:.1f}s (target={target_duration}s)"
    )

    for i, entry in enumerate(validated_plan):
        clip_name = Path(clips[entry['clip_index']]).name if entry['clip_index'] < len(clips) else "?"
        logs.append(
            f"[planner_agent] [{i+1}] {clip_name} "
            f"{entry['start']:.1f}s→{entry['end']:.1f}s ({entry['duration']:.1f}s) — {entry.get('reason','')[:60]}"
        )

    return {
        **state,
        "story_plan": validated_plan,
        "current_agent": "planner_agent",
        "errors": errors,
        "logs": logs,
    }
