"""
Metadata Agent — ClipMind AI
=====================================
Pipeline Step 5: Object Detection & Emotion Analysis.
  - Uses Gemini Vision (or a YOLOv8 fallback, if ultralytics is installed) to detect objects,
    identify people, label emotions, and compute an excitement score
  - Merges scene descriptions with object metadata
  - Output feeds the Embedding Agent for vector search
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.state import VideoState

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _analyze_with_gemini(frame_path: str, clip_index: int, scene_desc: str) -> Dict[str, Any]:
    """Use Gemini to detect objects and estimate emotional tone."""
    try:
        from google import genai
        import PIL.Image

        client = genai.Client(api_key=GOOGLE_API_KEY)

        img = PIL.Image.open(frame_path)

        prompt = (
            f"Scene context: {scene_desc}\n\n"
            "Analyze this video frame and respond ONLY with valid JSON containing:\n"
            "- object_labels: list of detected objects/people/animals (max 10 items)\n"
            "- dominant_emotion: the primary emotional tone (e.g. joy, excitement, calm, sadness, tension)\n"
            "- people_count: estimated number of visible people (int)\n"
            "- excitement_score: 0-10 float rating of how exciting/energetic this frame is\n"
            "- action_detected: true if there is visible motion/action\n"
            "- color_palette: list of 3 dominant color descriptors\n\n"
            "Example: {\"object_labels\": [\"person\", \"car\"], \"dominant_emotion\": \"excitement\", "
            "\"people_count\": 2, \"excitement_score\": 7.5, \"action_detected\": true, "
            "\"color_palette\": [\"golden\", \"blue\", \"white\"]}"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, img]
        )
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        return {
            "clip_index": clip_index,
            "frame_path": frame_path,
            "object_labels": parsed.get("object_labels", []),
            "dominant_emotion": parsed.get("dominant_emotion", "neutral"),
            "people_count": int(parsed.get("people_count", 0)),
            "excitement_score": float(parsed.get("excitement_score", 5.0)),
            "action_detected": bool(parsed.get("action_detected", False)),
            "color_palette": parsed.get("color_palette", []),
            "description": scene_desc,
            "source": "gemini",
        }
    except ImportError:
        return _fallback_metadata(frame_path, clip_index, scene_desc, "google-genai not installed")
    except json.JSONDecodeError:
        return _fallback_metadata(frame_path, clip_index, scene_desc, "JSON parse error")
    except Exception as exc:
        return _fallback_metadata(frame_path, clip_index, scene_desc, str(exc))


def _analyze_with_yolo(frame_path: str, clip_index: int, scene_desc: str) -> Optional[Dict[str, Any]]:
    """Attempt YOLO object detection as a fallback."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # nano model, fast
        results = model(frame_path, verbose=False)
        labels = []
        for r in results:
            for cls_id in r.boxes.cls.cpu().numpy().astype(int):
                label = r.names.get(int(cls_id), str(cls_id))
                if label not in labels:
                    labels.append(label)

        people_count = labels.count("person")
        excitement = min(10.0, len(labels) * 1.5)
        return {
            "clip_index": clip_index,
            "frame_path": frame_path,
            "object_labels": labels[:10],
            "dominant_emotion": "neutral",
            "people_count": people_count,
            "excitement_score": round(excitement, 2),
            "action_detected": len(labels) > 2,
            "color_palette": [],
            "description": scene_desc,
            "source": "yolo",
        }
    except Exception:
        return None


def _fallback_metadata(frame_path: str, clip_index: int, scene_desc: str, reason: str) -> Dict[str, Any]:
    """Try YOLO, otherwise return generic metadata."""
    yolo_result = _analyze_with_yolo(frame_path, clip_index, scene_desc)
    if yolo_result:
        return yolo_result

    return {
        "clip_index": clip_index,
        "frame_path": frame_path,
        "object_labels": [],
        "dominant_emotion": "neutral",
        "people_count": 0,
        "excitement_score": 5.0,
        "action_detected": False,
        "color_palette": [],
        "description": scene_desc,
        "source": f"fallback:{reason[:60]}",
    }


def metadata_agent_node(state: VideoState) -> VideoState:
    """
    LangGraph Node: Metadata / Object Detection Agent.
    Reads:  frames, scene_descriptions
    Writes: objects (list of metadata dicts per clip)
    """
    logs: List[str] = ["[metadata_agent] Detecting objects and emotions"]
    errors: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []

    frames = state.get("frames", [])
    scene_descriptions = state.get("scene_descriptions", [])
    clips = state.get("uploaded_videos", [])

    # Build a lookup: clip_index -> scene description
    desc_map: Dict[int, str] = {
        d.get("clip_index", i): d.get("description", "")
        for i, d in enumerate(scene_descriptions)
    }

    for clip_idx, clip_path in enumerate(clips):
        scene_desc = desc_map.get(clip_idx, "")

        # Pick the best frame for this clip
        clip_frames = [f for f in frames if f.get("clip_index") == clip_idx and f.get("quality_ok", True)]
        if not clip_frames:
            clip_frames = [f for f in frames if f.get("clip_index") == clip_idx]

        if not clip_frames:
            errors.append({
                "agent": "metadata_agent",
                "type": "NoFrame",
                "message": f"No frames found for clip {clip_idx}",
            })
            objects.append(_fallback_metadata("", clip_idx, scene_desc, "no frame"))
            continue

        # Use the middle frame
        frame_path = clip_frames[len(clip_frames) // 2].get("frame_path", "")
        clip_duration = clip_frames[0].get("clip_duration", 0)

        if not frame_path or not Path(frame_path).exists():
            objects.append(_fallback_metadata("", clip_idx, scene_desc, "frame file missing"))
            continue

        meta = _analyze_with_gemini(frame_path, clip_idx, scene_desc)
        meta["clip_path"] = clip_path
        meta["global_index"] = clip_idx
        meta["start"] = 0.0
        meta["end"] = clip_duration
        objects.append(meta)

        logs.append(
            f"[metadata_agent] Clip {clip_idx}: objects={meta['object_labels'][:5]}, "
            f"emotion={meta['dominant_emotion']}, excitement={meta['excitement_score']}"
        )

    logs.append(f"[metadata_agent] Processed {len(objects)} clips")

    return {
        **state,
        "objects": objects,
        "current_agent": "metadata_agent",
        "errors": errors,
        "logs": logs,
    }
