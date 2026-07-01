"""
LangGraph Pipeline — AI Video Director
========================================
Assembles all 8 agent nodes into a StateGraph with:
  • Sequential execution with conditional edges
  • SQLite checkpointing for recovery
  • LangSmith tracing (when API key present)
  • Max-3-retry wrapper around each agent
  • Progress callback hooks for WebSocket streaming
"""

from __future__ import annotations

import datetime
import os
import time
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.scene_agent import scene_extraction_agent
from agents.quality_agent import quality_analysis_agent
from agents.highlight_agent import highlight_detection_agent
from agents.caption_agent import caption_generator_agent
from agents.brand_rag_agent import brand_rag_agent
from agents.story_planner_agent import story_planner_agent
from agents.editing_agent import video_editing_agent
from agents.export_agent import export_agent
from logs.agent_logs import AgentLogger
from memory.langgraph_state import AgentError, VideoDirectorState, create_initial_state


MAX_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))
CHECKPOINT_PATH = os.getenv("SQLITE_CHECKPOINT_PATH", "pipeline/checkpoints.db")


# ── Configure LangSmith tracing ────────────────────────────────────────────────

def _setup_langsmith():
    if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
        os.environ.setdefault("LANGCHAIN_PROJECT", "ai-video-director")
        print("LangSmith tracing enabled")


_setup_langsmith()


# ── Retry-wrapped node factory ─────────────────────────────────────────────────

def _with_retry(
    agent_fn: Callable,
    agent_name: str,
    progress_callback: Optional[Callable] = None,
) -> Callable:
    """
    Wraps an agent function with:
      1. Retry logic (up to MAX_RETRIES attempts)
      2. Structured error logging on failure
      3. State update with error record if all retries exhausted
    """
    @wraps(agent_fn)
    def wrapper(state: VideoDirectorState) -> VideoDirectorState:
        log = AgentLogger(agent_name, state["job_id"])
        retry_counts = state.get("retry_counts", {})
        attempt = retry_counts.get(agent_name, 0)

        for attempt in range(MAX_RETRIES):
            try:
                if progress_callback:
                    progress_callback(agent_name, "running", attempt)

                if agent_name == "editing_agent":
                    # Editing agent takes optional progress callback
                    result = agent_fn(state, progress_callback=None)
                else:
                    result = agent_fn(state)

                if progress_callback:
                    progress_callback(agent_name, "done", attempt)
                return result

            except Exception as exc:
                log.failure(exc, retry_attempt=attempt)
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"Retrying {agent_name} (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(1.5 ** attempt)   # exponential back-off
                else:
                    # All retries exhausted — record error and continue pipeline
                    error = AgentError(
                        agent_name=agent_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        retry_attempt=attempt,
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        recovered=False,
                    )
                    if progress_callback:
                        progress_callback(agent_name, "error", attempt)
                    return {
                        **state,
                        "errors": [error],
                        "current_agent": agent_name,
                        "completed_agents": [f"{agent_name}:FAILED"],
                        "retry_counts": {
                            **retry_counts,
                            agent_name: attempt + 1,
                        },
                    }
        return state  # Should not reach here

    return wrapper


# ── Conditional edges ──────────────────────────────────────────────────────────

def _should_continue_after_quality(state: VideoDirectorState) -> str:
    """If no scenes passed quality filter, fall back to all scenes ranked by composite score."""
    if state.get("filtered_count", 0) == 0:
        return "highlight_agent"  # fallback: let highlight agent work on all scenes
    return "highlight_agent"


def _should_continue_after_editing(state: VideoDirectorState) -> str:
    """Abort export if editing produced no output."""
    if not state.get("edited_video_path"):
        return "export_error"
    return "export_agent"


def _export_error_node(state: VideoDirectorState) -> VideoDirectorState:
    """Terminal node for pipeline failures."""
    error = AgentError(
        agent_name="pipeline",
        error_type="PipelineAborted",
        error_message="Pipeline aborted — insufficient usable scenes or editing failed",
        retry_attempt=0,
        timestamp=datetime.datetime.utcnow().isoformat(),
        recovered=False,
    )
    return {**state, "errors": [error], "current_agent": "ABORTED"}


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_graph(
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        progress_callback: Optional fn(agent_name, status, attempt) for WS streaming

    Returns:
        Compiled LangGraph app (invoke with state dict + config)
    """
    graph = StateGraph(VideoDirectorState)

    # Add nodes (all retry-wrapped)
    graph.add_node(
        "scene_agent",
        _with_retry(scene_extraction_agent, "scene_agent", progress_callback)
    )
    graph.add_node(
        "quality_agent",
        _with_retry(quality_analysis_agent, "quality_agent", progress_callback)
    )
    graph.add_node(
        "highlight_agent",
        _with_retry(highlight_detection_agent, "highlight_agent", progress_callback)
    )
    graph.add_node(
        "caption_agent",
        _with_retry(caption_generator_agent, "caption_agent", progress_callback)
    )
    graph.add_node(
        "brand_rag_agent",
        _with_retry(brand_rag_agent, "brand_rag_agent", progress_callback)
    )
    graph.add_node(
        "story_planner_agent",
        _with_retry(story_planner_agent, "story_planner_agent", progress_callback)
    )
    graph.add_node(
        "editing_agent",
        _with_retry(video_editing_agent, "editing_agent", progress_callback)
    )
    graph.add_node(
        "export_agent",
        _with_retry(export_agent, "export_agent", progress_callback)
    )
    graph.add_node("export_error", _export_error_node)

    # Entry point
    graph.set_entry_point("scene_agent")

    # Sequential edges
    graph.add_edge("scene_agent", "quality_agent")

    # Conditional: quality → highlight OR error
    graph.add_conditional_edges(
        "quality_agent",
        _should_continue_after_quality,
        {
            "highlight_agent": "highlight_agent",
            "export_error": "export_error",
        },
    )

    graph.add_edge("highlight_agent", "caption_agent")
    graph.add_edge("caption_agent", "brand_rag_agent")
    graph.add_edge("brand_rag_agent", "story_planner_agent")
    graph.add_edge("story_planner_agent", "editing_agent")

    # Conditional: editing → export OR error
    graph.add_conditional_edges(
        "editing_agent",
        _should_continue_after_editing,
        {
            "export_agent": "export_agent",
            "export_error": "export_error",
        },
    )

    graph.add_edge("export_agent", END)
    graph.add_edge("export_error", END)

    return graph


def compile_pipeline(
    progress_callback: Optional[Callable] = None,
    use_checkpointing: bool = True,
):
    """
    Compile the LangGraph with SQLite checkpointing.
    Returns the compiled app ready for invoke().
    """
    import os
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    graph = build_graph(progress_callback)

    if use_checkpointing:
        import sqlite3
        conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        app = graph.compile(checkpointer=checkpointer)
    else:
        app = graph.compile()

    return app


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_pipeline(
    job_id: str,
    raw_clips: list,
    user_prompt: str,
    target_duration: int = 30,
    platform: str = "instagram",
    brand_pdf_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> VideoDirectorState:
    """
    End-to-end convenience runner.
    Returns the final VideoDirectorState.
    """
    initial = create_initial_state(
        job_id=job_id,
        raw_clips=raw_clips,
        user_prompt=user_prompt,
        target_duration=target_duration,
        platform=platform,
        brand_pdf_path=brand_pdf_path,
    )

    app = compile_pipeline(progress_callback=progress_callback)
    config = {"configurable": {"thread_id": job_id}}
    final_state = app.invoke(initial, config=config)
    return final_state
