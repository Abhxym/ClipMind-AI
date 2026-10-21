"""
Evaluation Metrics — ClipMind AI
=======================================
Computes and logs all evaluation metrics from a finished pipeline run.
Outputs CSV rows for benchmark comparison.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from memory.state import VideoState


EVAL_OUTPUT_DIR = Path("evaluation/results")


# ── Individual Metric Functions ────────────────────────────────────────────────

def scene_selection_accuracy(
    selected_indices: List[int],
    ground_truth_indices: List[int],
) -> float:
    """
    IoU-based scene selection accuracy.
    ground_truth_indices = manually labelled 'good' scenes.
    """
    if not ground_truth_indices:
        return 0.0
    selected_set = set(selected_indices)
    truth_set = set(ground_truth_indices)
    intersection = len(selected_set & truth_set)
    union = len(selected_set | truth_set)
    return intersection / union if union > 0 else 0.0


def caption_quality_score(generated: List[str], reference: List[str]) -> dict:
    """
    ROUGE-L score between generated and reference captions.
    Returns {"rouge_l": float, "avg_length": float}
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for gen, ref in zip(generated, reference):
            if gen and ref:
                score = scorer.score(ref, gen)
                scores.append(score["rougeL"].fmeasure)
        return {
            "rouge_l": float(sum(scores) / len(scores)) if scores else 0.0,
            "avg_length": float(sum(len(c) for c in generated) / max(1, len(generated))),
        }
    except ImportError:
        return {"rouge_l": 0.0, "avg_length": 0.0}


def retrieval_precision_at_k(
    retrieved_docs: List[str],
    relevant_keywords: List[str],
    k: int,
) -> float:
    """
    Precision@K: fraction of top-K docs containing a relevant keyword.
    """
    top = retrieved_docs[:k]
    hits = sum(
        1 for doc in top
        if any(kw.lower() in doc.lower() for kw in relevant_keywords)
    )
    return hits / k if k > 0 else 0.0


def latency_breakdown(agent_timings: Dict[str, float]) -> dict:
    """Break down total latency by agent and identify bottleneck."""
    total = sum(agent_timings.values())
    sorted_agents = sorted(agent_timings.items(), key=lambda x: x[1], reverse=True)
    bottleneck = sorted_agents[0][0] if sorted_agents else "none"
    return {
        "total_ms": total,
        "total_s": total / 1000,
        "bottleneck_agent": bottleneck,
        "per_agent_ms": agent_timings,
        "per_agent_pct": {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in agent_timings.items()
        },
    }


def hallucination_check(captions: List[str], transcript: str) -> float:
    """
    Heuristic hallucination rate: fraction of captions containing
    terms NOT present in the transcript. A proxy for factual grounding.
    """
    if not transcript or not captions:
        return 0.0
    transcript_words = set(transcript.lower().split())
    hallucinated = 0
    for caption in captions:
        caption_words = set(caption.lower().split())
        # Allow common stop words; penalise named entities absent from transcript
        key_words = {w for w in caption_words if len(w) > 5}
        if key_words and not any(w in transcript_words for w in key_words):
            hallucinated += 1
    return hallucinated / len(captions)


# ── Full Run Report ────────────────────────────────────────────────────────────

def generate_report(state: VideoState, config: Optional[Dict] = None) -> dict:
    """
    Aggregate metrics from a completed pipeline run (the final VideoState)
    into one flat report row.
    config: optional benchmark config (chunk_size, backend, top_k, model)
    """
    config = config or {}
    timings = state.get("agent_timings", {})
    lat = latency_breakdown(timings)
    ev = state.get("evaluation_score") or {}
    captions = [c.get("caption", "") for c in state.get("captions", [])]
    transcript = " ".join(s.get("text", "") for s in state.get("transcript", []))

    report = {
        # ── Identity
        "job_id": state.get("job_id", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        # ── Benchmark config
        "vector_db_backend": config.get("backend", os.getenv("VECTOR_DB_BACKEND", "faiss")),
        "chunk_size": config.get("chunk_size", int(os.getenv("CHUNK_SIZE", "512"))),
        "top_k_retrieve": config.get("top_k_retrieve", int(os.getenv("TOP_K_RETRIEVE", "10"))),
        "top_k_rerank": config.get("top_k_rerank", int(os.getenv("TOP_K_RERANK", "3"))),
        "gemini_model": config.get("model", os.getenv("GEMINI_MODEL_PLANNER", "gemini-2.5-pro")),

        # ── Scene metrics
        "total_clips": len(state.get("uploaded_videos", [])),
        "frames_extracted": len(state.get("frames", [])),
        "frames_quality_ok": sum(1 for f in state.get("frames", []) if f.get("quality_ok")),
        "selected_scenes": len(state.get("story_plan", [])),

        # ── RAG / captions
        "retrieved_docs": len(state.get("retrieved_brand_docs", [])),
        "caption_count": len(captions),
        "avg_caption_length": round(sum(len(c) for c in captions) / max(1, len(captions)), 1),
        "hallucination_rate": hallucination_check(captions, transcript),

        # ── Output quality
        "duration_accuracy": ev.get("duration_accuracy", 0.0),
        "edit_success_rate": ev.get("edit_success_rate", 0.0),

        # ── Latency
        "latency_total_ms": lat["total_ms"],
        "latency_total_s": lat["total_s"],
        "bottleneck_agent": lat["bottleneck_agent"],
        **{f"latency_{k}_ms": v for k, v in timings.items()},

        "error_count": len(state.get("errors", [])),
        "success": bool(state.get("final_video")),
    }
    return report


def append_to_csv(report: dict, csv_path: str = "evaluation/results/benchmark.csv"):
    """Append one report row to the benchmark CSV."""
    path = Path(csv_path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(report)


def load_results(csv_path: str = "evaluation/results/benchmark.csv") -> pd.DataFrame:
    """Load all benchmark results as a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path))


def print_summary(report: dict):
    """Pretty-print evaluation summary to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"📊 Evaluation Report — Job {report['job_id'][:8]}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    key_metrics = [
        ("Clips", f"{report['total_clips']}"),
        ("Frames (quality OK)", f"{report['frames_quality_ok']}/{report['frames_extracted']}"),
        ("Selected for Reel", f"{report['selected_scenes']}"),
        ("Brand Docs Retrieved", f"{report['retrieved_docs']}"),
        ("Duration Accuracy", f"{report['duration_accuracy']:.3f}"),
        ("Edit Success Rate", f"{report['edit_success_rate']:.0%}"),
        ("Hallucination Rate", f"{report['hallucination_rate']:.2%}"),
        ("Total Latency", f"{report['latency_total_s']:.1f}s"),
        ("Bottleneck", report["bottleneck_agent"]),
        ("Errors", str(report["error_count"])),
        ("Success", "✅" if report["success"] else "❌"),
    ]
    for metric, value in key_metrics:
        table.add_row(metric, value)

    console.print(table)
