"""
Evaluation Metrics — AI Video Director
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

from memory.langgraph_state import VideoDirectorState


EVAL_OUTPUT_DIR = Path("evaluation/results")
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

def generate_report(state: VideoDirectorState, config: Optional[Dict] = None) -> dict:
    """
    Aggregate all metrics from a completed pipeline run into a report dict.
    config: optional benchmark config (chunk_size, backend, top_k, model)
    """
    config = config or {}
    timings = state.get("agent_timings", {})
    lat = latency_breakdown(timings)

    report = {
        # ── Identity
        "job_id": state.get("job_id", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        # ── Benchmark config
        "vector_db_backend": config.get("backend", os.getenv("VECTOR_DB_BACKEND", "faiss")),
        "chunk_size": config.get("chunk_size", int(os.getenv("CHUNK_SIZE", "512"))),
        "top_k_retrieve": config.get("top_k_retrieve", int(os.getenv("TOP_K_RETRIEVE", "10"))),
        "top_k_rerank": config.get("top_k_rerank", int(os.getenv("TOP_K_RERANK", "3"))),
        "gemini_model": config.get("model", os.getenv("GEMINI_MODEL_PLANNER", "gemini-1.5-pro")),
        "use_reranker": config.get("use_reranker", True),

        # ── Scene metrics
        "total_scenes": state.get("scene_count", 0),
        "filtered_scenes": state.get("filtered_count", 0),
        "selected_scenes": len(state.get("story_arc", [])),
        "filter_rate_pct": round(
            (1 - state.get("filtered_count", 0) / max(1, state.get("scene_count", 1))) * 100, 1
        ),

        # ── RAG metrics
        "retrieved_docs": len(state.get("retrieved_doc_ids", [])),
        "reranked_docs": len(state.get("reranked_doc_ids", [])),
        "retrieval_precision_at_k": state.get("evaluation", {}).get("retrieval_precision_at_k", 0.0)
            if state.get("evaluation") else 0.0,
        "reranker_lift": state.get("evaluation", {}).get("reranker_lift", 0.0)
            if state.get("evaluation") else 0.0,

        # ── Caption metrics
        "caption_count": len(state.get("captions", [])),
        "avg_caption_length": state.get("evaluation", {}).get("caption_length_avg", 0.0)
            if state.get("evaluation") else 0.0,
        "hallucination_rate": hallucination_check(
            state.get("captions", []), state.get("transcript", "")
        ),

        # ── Latency
        "latency_total_ms": lat["total_ms"],
        "latency_total_s": lat["total_s"],
        "bottleneck_agent": lat["bottleneck_agent"],
        **{f"latency_{k}_ms": v for k, v in timings.items()},

        # ── Cost (approximate)
        "llm_tokens_used": state.get("evaluation", {}).get("llm_tokens_used", 0)
            if state.get("evaluation") else 0,
        "error_count": len(state.get("errors", [])),
        "success": state.get("final_video_path") is not None,
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
        ("Total Scenes", f"{report['total_scenes']}"),
        ("After Quality Filter", f"{report['filtered_scenes']}"),
        ("Selected for Reel", f"{report['selected_scenes']}"),
        ("Retrieval P@K", f"{report['retrieval_precision_at_k']:.3f}"),
        ("Reranker Lift", f"{report['reranker_lift']:+.3f}"),
        ("Hallucination Rate", f"{report['hallucination_rate']:.2%}"),
        ("Total Latency", f"{report['latency_total_s']:.1f}s"),
        ("Bottleneck", report["bottleneck_agent"]),
        ("LLM Tokens", str(report["llm_tokens_used"])),
        ("Errors", str(report["error_count"])),
        ("Success", "✅" if report["success"] else "❌"),
    ]
    for metric, value in key_metrics:
        table.add_row(metric, value)

    console.print(table)
