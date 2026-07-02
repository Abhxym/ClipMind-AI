import type { AgentInfo } from '../types'

const AGENT_COLORS: Record<string, string> = {
  input_agent:          'var(--agent-input, #64748b)',
  preprocess_agent:     'var(--agent-preprocess, #0ea5e9)',
  whisper_agent:        'var(--agent-whisper, #f59e0b)',
  scene_agent:          'var(--agent-scene, #ec4899)',
  metadata_agent:       'var(--agent-metadata, #8b5cf6)',
  embedding_agent:      'var(--agent-embedding, #14b8a6)',
  rag_agent:            'var(--agent-rag, #3b82f6)',
  planner_agent:        'var(--agent-planner, #f43f5e)',
  caption_agent:        'var(--agent-caption, #eab308)',
  editing_agent:        'var(--agent-editing, #10b981)',
  reviewer_agent:       'var(--agent-reviewer, #ef4444)',
  export_agent:         'var(--agent-export, #6366f1)',
  evaluation_agent:     'var(--agent-evaluation, #8b5cf6)',
}

const AGENT_META: Record<string, { label: string; icon: string; description: string }> = {
  input_agent:          { label: 'Validation',         icon: '📥', description: 'ffprobe validation · extension/size checks' },
  preprocess_agent:     { label: 'Frame Extraction',   icon: '🎞️', description: 'FFmpeg extraction · OpenCV quality scores' },
  whisper_agent:        { label: 'Transcription',      icon: '🎙️', description: 'OpenAI Whisper · speech-to-text' },
  scene_agent:          { label: 'Scene Extraction',   icon: '🎬', description: 'Gemini Vision · scene understanding' },
  metadata_agent:       { label: 'Metadata & Emotion', icon: '🔍', description: 'Gemini/YOLO · object detection · excitement score' },
  embedding_agent:      { label: 'Vector Build',       icon: '🔢', description: 'BGE embeddings · FAISS index' },
  rag_agent:            { label: 'Brand RAG',          icon: '🧠', description: 'FAISS retrieval · BGE Reranker' },
  planner_agent:        { label: 'Story Planner',      icon: '📖', description: 'Gemini Pro narrative arc · clip timing' },
  caption_agent:        { label: 'Caption Generator',  icon: '💬', description: 'Gemini Flash · titles · hashtags' },
  editing_agent:        { label: 'Video Editing',      icon: '✂️',  description: 'FFmpeg trim · subtitles · concat' },
  reviewer_agent:       { label: 'Quality Review',     icon: '✅', description: 'Duration check · edit success rate gating' },
  export_agent:         { label: 'Export',             icon: '📤', description: 'Final file save · thumbnail generation' },
  evaluation_agent:     { label: 'Evaluation',         icon: '📊', description: 'Metrics scoring · duration accuracy' },
}

interface AgentGraphProps {
  agents: AgentInfo[]
}

function StatusDot({ status }: { status: string }) {
  const map: Record<string, string> = {
    pending: '⬜',
    running: '🔄',
    done: '✅',
    error: '❌',
  }
  return <span>{map[status] ?? '⬜'}</span>
}

function AgentNode({ agent }: { agent: AgentInfo }) {
  const meta = AGENT_META[agent.name] || { label: agent.name, icon: '🤖', description: '' }
  const color = AGENT_COLORS[agent.name] || '#8b5cf6'
  const isRunning = agent.status === 'running'
  const isDone = agent.status === 'done'

  return (
    <div
      className={`agent-node ${isRunning ? 'agent-node--running' : ''} ${isDone ? 'agent-node--done' : ''} ${agent.status === 'error' ? 'agent-node--error' : ''}`}
      style={{ '--agent-color': color } as React.CSSProperties}
    >
      <div className="agent-node__header">
        <span className="agent-node__icon">{meta.icon}</span>
        <div className="agent-node__info">
          <div className="agent-node__label">{meta.label}</div>
          <div className="agent-node__desc">{meta.description}</div>
        </div>
        <StatusDot status={agent.status} />
      </div>
      {agent.elapsedMs && isDone && (
        <div className="agent-node__timing">
          {agent.elapsedMs < 1000
            ? `${agent.elapsedMs.toFixed(0)}ms`
            : `${(agent.elapsedMs / 1000).toFixed(1)}s`}
        </div>
      )}
      {isRunning && (
        <div className="agent-node__progress-bar">
          <div className="agent-node__progress-fill" />
        </div>
      )}
    </div>
  )
}

export function AgentGraph({ agents }: AgentGraphProps) {
  return (
    <div className="agent-graph">
      <div className="agent-graph__title">
        <span>🤖</span> Agent Pipeline
      </div>
      <div className="agent-graph__nodes">
        {agents.map((agent, i) => (
          <div key={agent.name} className="agent-graph__item">
            <AgentNode agent={agent} />
            {i < agents.length - 1 && (
              <div className={`agent-graph__connector ${agent.status === 'done' ? 'agent-graph__connector--active' : ''}`}>
                <div className="agent-graph__connector-line" />
                <div className="agent-graph__connector-arrow">▼</div>
              </div>
            )}
          </div>
        ))}
      </div>

      <style>{`
        .agent-graph {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--r-xl);
          padding: 24px;
        }
        .agent-graph__title {
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .agent-graph__nodes { display: flex; flex-direction: column; gap: 0; }
        .agent-graph__item { display: flex; flex-direction: column; align-items: center; }

        .agent-node {
          width: 100%;
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--r-md);
          padding: 12px 14px;
          transition: all 0.3s;
          position: relative;
          overflow: hidden;
        }
        .agent-node::before {
          content: '';
          position: absolute;
          left: 0; top: 0; bottom: 0;
          width: 3px;
          background: var(--agent-color, #8b5cf6);
          opacity: 0.4;
          transition: opacity 0.3s;
        }
        .agent-node--running { border-color: var(--agent-color, #8b5cf6); }
        .agent-node--running::before { opacity: 1; }
        .agent-node--running { animation: pulse-glow 2s infinite; }
        .agent-node--done { border-color: rgba(16,185,129,0.3); }
        .agent-node--done::before { background: #10b981; opacity: 1; }
        .agent-node--error { border-color: rgba(239,68,68,0.4); }
        .agent-node--error::before { background: #ef4444; opacity: 1; }

        .agent-node__header {
          display: flex; align-items: center; gap: 10px;
        }
        .agent-node__icon { font-size: 1.2rem; flex-shrink: 0; }
        .agent-node__info { flex: 1; min-width: 0; }
        .agent-node__label { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }
        .agent-node__desc { font-size: 0.7rem; color: var(--text-muted); margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .agent-node__timing {
          font-size: 0.7rem;
          font-family: 'JetBrains Mono', monospace;
          color: #10b981;
          text-align: right;
          margin-top: 4px;
        }
        .agent-node__progress-bar {
          height: 2px;
          background: var(--color-surface-3);
          border-radius: 1px;
          margin-top: 8px;
          overflow: hidden;
        }
        .agent-node__progress-fill {
          height: 100%;
          background: var(--agent-color, #8b5cf6);
          border-radius: 1px;
          animation: shimmer 1.5s infinite;
          background-size: 200% 100%;
          background-image: linear-gradient(90deg, transparent 25%, rgba(255,255,255,0.3) 50%, transparent 75%);
          width: 100%;
        }

        .agent-graph__connector {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 2px 0;
          opacity: 0.3;
          transition: opacity 0.3s;
        }
        .agent-graph__connector--active { opacity: 1; }
        .agent-graph__connector-line {
          width: 2px; height: 10px;
          background: linear-gradient(to bottom, #8b5cf6, #3b82f6);
        }
        .agent-graph__connector-arrow {
          font-size: 0.6rem;
          color: #3b82f6;
          margin-top: -2px;
        }
      `}</style>
    </div>
  )
}
