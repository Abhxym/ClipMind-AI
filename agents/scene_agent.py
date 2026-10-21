"""
Scene Agent — ClipMind AI
=================================
Pipeline Step 4: Scene Understanding using Gemini Vision.
  - Samples one representative frame per clip
  - Sends the frame to Gemini (GEMINI_MODEL) for a rich scene description
  - Falls back to BLIP-2 or a generic description if Gemini unavailable
  - Produces scene_descriptions list consumed by metadata and planner agents
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.state import VideoState

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _encode_image(path: str) -> Optional[str]:
    """Base64-encode an image file."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _describe_with_gemini(frame_path: str, clip_index: int) -> Dict[str, Any]:
    """Use Gemini Vision to describe a scene frame."""
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)

        import PIL.Image
        img = PIL.Image.open(frame_path)

        prompt = (
            "You are an expert video editor. Describe this video frame concisely in 2-3 sentences. "
            "Focus on: main subject/action, mood/emotion, visual style, setting/background. "
            "Also rate on a scale 0-10: visual_quality, emotional_impact, cinematic_value. "
            "Respond ONLY with valid JSON: "
            "{\"description\": \"...\", \"mood\": \"...\", \"setting\": \"...\", "
            "\"visual_quality\": 7, \"emotional_impact\": 6, \"cinematic_value\": 7}"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, img]
        )
        text = response.text.strip()

        # Parse JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        return {
            "clip_index": clip_index,
            "frame_path": frame_path,
            "description": parsed.get("description", ""),
            "mood": parsed.get("mood", "neutral"),
            "setting": parsed.get("setting", ""),
            "visual_quality": float(parsed.get("visual_quality", 5)),
            "emotional_impact": float(parsed.get("emotional_impact", 5)),
            "cinematic_value": float(parsed.get("cinematic_value", 5)),
            "source": "gemini",
        }
    except ImportError:
        return _fallback_description(frame_path, clip_index, "google-genai not installed")
    except json.JSONDecodeError:
        # Gemini responded but not valid JSON — extract text response as description
        try:
            raw = response.text.strip()
            return {
                "clip_index": clip_index,
                "frame_path": frame_path,
                "description": raw[:300],
                "mood": "neutral",
                "setting": "",
                "visual_quality": 5.0,
                "emotional_impact": 5.0,
                "cinematic_value": 5.0,
                "source": "gemini-raw",
            }
        except Exception:
            return _fallback_description(frame_path, clip_index, "JSON parse error")
    except Exception as exc:
        return _fallback_description(frame_path, clip_index, str(exc))


def _fallback_description(frame_path: str, clip_index: int, reason: str) -> Dict[str, Any]:
    """Fallback: attempt BLIP-2 or use generic placeholder."""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        from PIL import Image

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        img = Image.open(frame_path).convert("RGB")
        inputs = processor(img, return_tensors="pt")
        out = model_blip.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return {
            "clip_index": clip_index,
            "frame_path": frame_path,
            "description": caption,
            "mood": "neutral",
            "setting": "",
            "visual_quality": 5.0,
            "emotional_impact": 5.0,
            "cinematic_value": 5.0,
            "source": "blip",
        }
    except Exception:
        pass

    # Final fallback: generic
    return {
        "clip_index": clip_index,
        "frame_path": frame_path,
        "description": f"Video scene from clip {clip_index}. [Gemini unavailable: {reason[:80]}]",
        "mood": "neutral",
        "setting": "unknown",
        "visual_quality": 5.0,
        "emotional_impact": 5.0,
        "cinematic_value": 5.0,
        "source": "fallback",
    }


def _pick_representative_frame(frames: List[Dict], clip_index: int) -> Optional[str]:
    """Pick the best quality frame from a clip."""
    clip_frames = [f for f in frames if f.get("clip_index") == clip_index and f.get("quality_ok", True)]
    if not clip_frames:
        clip_frames = [f for f in frames if f.get("clip_index") == clip_index]
    if not clip_frames:
        return None
    # Pick the middle frame as most representative
    return clip_frames[len(clip_frames) // 2].get("frame_path")


def scene_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Scene Understanding Agent.
    Reads:  frames, uploaded_videos
    Writes: scene_descriptions
    """
    logs: List[str] = ["[scene_agent] Analyzing scenes with Gemini Vision"]
    errors: List[Dict[str, Any]] = []
    descriptions: List[Dict[str, Any]] = []

    frames = state.get("frames", [])
    clips = state.get("uploaded_videos", [])

    if not clips:
        logs.append("[scene_agent] No clips to analyze")
        return {**state, "scene_descriptions": [], "current_agent": "scene_agent", "errors": [], "logs": logs}

    for clip_idx, clip_path in enumerate(clips):
        frame_path = _pick_representative_frame(frames, clip_idx)

        if not frame_path or not Path(frame_path).exists():
            # Try any frame with this clip_path
            matching = [f for f in frames if f.get("clip_path") == clip_path]
            if matching:
                frame_path = matching[len(matching) // 2].get("frame_path")

        if not frame_path or not Path(frame_path).exists():
            errors.append({
                "agent": "scene_agent",
                "type": "NoFrame",
                "message": f"No valid frame found for clip {clip_idx}: {Path(clip_path).name}",
            })
            descriptions.append(_fallback_description("", clip_idx, "no frame available"))
            continue

        desc = _describe_with_gemini(frame_path, clip_idx)
        desc["clip_path"] = clip_path
        descriptions.append(desc)
        logs.append(
            f"[scene_agent] Clip {clip_idx} ({Path(clip_path).name}): "
            f"\"{desc['description'][:80]}...\" | mood={desc['mood']} | "
            f"source={desc['source']}"
        )

    logs.append(f"[scene_agent] Generated {len(descriptions)} scene descriptions")

    return {
        **state,
        "scene_descriptions": descriptions,
        "current_agent": "scene_agent",
        "errors": errors,
        "logs": logs,
    }
