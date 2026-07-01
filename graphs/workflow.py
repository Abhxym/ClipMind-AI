"""
Workflow — AI Video Director
============================
The main LangGraph definition connecting all 14 agents.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

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
from agents.reviewer_agent import reviewer_agent_node
from agents.export_agent import export_agent_node
from agents.evaluation_agent import evaluation_agent_node

def review_routing(state: VideoState):
    """Router for the review feedback loop."""
    errors = state.get("errors", [])
    if errors and sum(state.get("retry_count", {}).values()) < 3:
        return "editing_agent"
    return "export_agent"

def build_workflow() -> StateGraph:
    workflow = StateGraph(VideoState)

    # 1. Add Nodes
    workflow.add_node("input_agent", input_agent_node)
    workflow.add_node("preprocess_agent", preprocess_agent_node)
    workflow.add_node("whisper_agent", whisper_agent_node)
    workflow.add_node("scene_agent", scene_agent_node)
    workflow.add_node("metadata_agent", metadata_agent_node)
    workflow.add_node("embedding_agent", embedding_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("caption_agent", caption_agent_node)
    workflow.add_node("editing_agent", editing_agent_node)
    workflow.add_node("reviewer_agent", reviewer_agent_node)
    workflow.add_node("export_agent", export_agent_node)
    workflow.add_node("evaluation_agent", evaluation_agent_node)

    # 2. Add Edges (Linear for now, with review loop)
    workflow.set_entry_point("input_agent")
    workflow.add_edge("input_agent", "preprocess_agent")
    workflow.add_edge("preprocess_agent", "whisper_agent")
    workflow.add_edge("whisper_agent", "scene_agent")
    workflow.add_edge("scene_agent", "metadata_agent")
    workflow.add_edge("metadata_agent", "embedding_agent")
    workflow.add_edge("embedding_agent", "rag_agent")
    workflow.add_edge("rag_agent", "planner_agent")
    workflow.add_edge("planner_agent", "caption_agent")
    workflow.add_edge("caption_agent", "editing_agent")
    workflow.add_edge("editing_agent", "reviewer_agent")
    
    # Review loop
    workflow.add_conditional_edges(
        "reviewer_agent",
        review_routing,
        {
            "editing_agent": "editing_agent",
            "export_agent": "export_agent"
        }
    )
    
    workflow.add_edge("export_agent", "evaluation_agent")
    workflow.add_edge("evaluation_agent", END)

    return workflow

def get_compiled_workflow(memory: SqliteSaver = None):
    return build_workflow().compile(checkpointer=memory)

def run_pipeline(job_id: str, initial_state: VideoState) -> VideoState:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    import os

    db_path = "pipeline/checkpoints.db"
    os.makedirs("pipeline", exist_ok=True)
    
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        app = get_compiled_workflow(memory)
        config = {"configurable": {"thread_id": job_id}}
        
        # Run graph
        result = app.invoke(initial_state, config=config)
        return result

