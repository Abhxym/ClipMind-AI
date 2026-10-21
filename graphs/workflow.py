"""
Workflow — ClipMind AI
======================
The LangGraph StateGraph connecting all 13 agents:

  input → preprocess → whisper → scene → metadata → embedding → rag
        → planner → caption → editing → reviewer ─┬→ export → evaluation → END
                                     ▲            │
                                     └── retry ───┘  (review failed, retries left)

Every node is wrapped so that it reports "running"/"done" to an optional
progress callback (used by the API's WebSocket stream) and records its
wall-clock time in state["agent_timings"].
"""

import os
import time
from typing import Callable, Optional

from langgraph.graph import StateGraph, END

from memory.state import VideoState
from agents.input_agent import input_agent_node
from agents.preprocess_agent import preprocess_agent_node
from agents.whisper_agent import whisper_agent_node
from agents.scene_agent import scene_agent_node
from agents.metadata_agent import metadata_agent_node
from agents.embedding_agent import embedding_agent_node
from agents.rag_agent import rag_agent_node
from agents.planner_agent import planner_agent_node
from agents.caption_agent import caption_agent_node
from agents.editing_agent import editing_agent_node
from agents.reviewer_agent import reviewer_agent_node, MAX_RETRIES
from agents.export_agent import export_agent_node
from agents.evaluation_agent import evaluation_agent_node

# (agent_name, status, attempt) — status is "running" | "done" | "error"
ProgressCallback = Callable[[str, str, int], None]

AGENT_NODES = [
    ("input_agent", input_agent_node),
    ("preprocess_agent", preprocess_agent_node),
    ("whisper_agent", whisper_agent_node),
    ("scene_agent", scene_agent_node),
    ("metadata_agent", metadata_agent_node),
    ("embedding_agent", embedding_agent_node),
    ("rag_agent", rag_agent_node),
    ("planner_agent", planner_agent_node),
    ("caption_agent", caption_agent_node),
    ("editing_agent", editing_agent_node),
    ("reviewer_agent", reviewer_agent_node),
    ("export_agent", export_agent_node),
    ("evaluation_agent", evaluation_agent_node),
]
AGENT_ORDER = [name for name, _ in AGENT_NODES]

CHECKPOINT_DB = os.getenv("SQLITE_CHECKPOINT_PATH", "pipeline/checkpoints.db")


def review_routing(state: VideoState) -> str:
    """Router for the review feedback loop."""
    if state.get("review_passed"):
        return "export_agent"
    if state.get("retry_count", {}).get("editing_agent", 0) < MAX_RETRIES:
        return "editing_agent"
    return "export_agent"


def _instrument(name: str, fn, on_progress: Optional[ProgressCallback]):
    """Wrap a node: report progress and record its elapsed time."""
    attempts = {"n": 0}

    def node(state: VideoState) -> VideoState:
        attempts["n"] += 1
        if on_progress:
            on_progress(name, "running", attempts["n"])
        t0 = time.perf_counter()
        try:
            result = fn(state)
        except Exception:
            if on_progress:
                on_progress(name, "error", attempts["n"])
            raise
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Override the agent_timings copied in by {**state}; the reducer adds it up
        result = {**result, "agent_timings": {name: elapsed_ms}}
        if on_progress:
            on_progress(name, "done", attempts["n"])
        return result

    return node


def build_workflow(on_progress: Optional[ProgressCallback] = None) -> StateGraph:
    workflow = StateGraph(VideoState)

    for name, fn in AGENT_NODES:
        workflow.add_node(name, _instrument(name, fn, on_progress))

    workflow.set_entry_point("input_agent")
    linear = AGENT_ORDER[: AGENT_ORDER.index("reviewer_agent") + 1]
    for src, dst in zip(linear, linear[1:]):
        workflow.add_edge(src, dst)

    workflow.add_conditional_edges(
        "reviewer_agent",
        review_routing,
        {"editing_agent": "editing_agent", "export_agent": "export_agent"},
    )
    workflow.add_edge("export_agent", "evaluation_agent")
    workflow.add_edge("evaluation_agent", END)

    return workflow


def get_compiled_workflow(checkpointer=None, on_progress: Optional[ProgressCallback] = None):
    return build_workflow(on_progress).compile(checkpointer=checkpointer)


def run_pipeline(
    job_id: str,
    initial_state: VideoState,
    on_progress: Optional[ProgressCallback] = None,
) -> VideoState:
    """Run the full graph for one job, checkpointing every step to SQLite."""
    import sqlite3
    from pathlib import Path
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    try:
        app = get_compiled_workflow(SqliteSaver(conn), on_progress)
        config = {
            "configurable": {"thread_id": job_id},
            # 13 nodes + up to MAX_RETRIES extra editing/review passes
            "recursion_limit": len(AGENT_ORDER) + 2 * MAX_RETRIES + 5,
        }
        return app.invoke(initial_state, config=config)
    finally:
        conn.close()
