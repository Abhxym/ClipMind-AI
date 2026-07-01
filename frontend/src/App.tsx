import { useState, useEffect, useCallback, useRef } from 'react'
import { UploadZone } from './components/UploadZone'
import { AgentGraph } from './components/AgentGraph'
import { VideoPreview } from './components/VideoPreview'
import { MetricsPanel } from './components/MetricsPanel'
import { createJob, getJobResult, createWSConnection } from './api'
import type { UploadedClip, AgentInfo, WSMessage, JobResult, Platform } from './types'
import './App.css'

const AGENT_ORDER = [
  'scene_agent', 'quality_agent', 'highlight_agent', 'caption_agent',
  'brand_rag_agent', 'story_planner_agent', 'editing_agent', 'export_agent',
]

const AGENT_META_LABELS: Record<string, string> = {
  scene_agent:         'Scene Extraction',
  quality_agent:       'Quality Analysis',
  highlight_agent:     'Highlight Detection',
  caption_agent:       'Caption Generator',
  brand_rag_agent:     'Brand RAG',
  story_planner_agent: 'Story Planner',
  editing_agent:       'Video Editing',
  export_agent:        'Exporting',
}

const defaultAgents = (): AgentInfo[] =>
  AGENT_ORDER.map(name => ({
    name,
    label: name,
    description: '',
    color: '#8b5cf6',
    icon: '🤖',
    status: 'pending',
  }))

type AppStage = 'upload' | 'processing' | 'done' | 'error'

export default function App() {
  const [stage, setStage] = useState<AppStage>('upload')
  const [clips, setClips] = useState<UploadedClip[]>([])
  const [brandPdfPath, setBrandPdfPath] = useState<string | undefined>()
  const [prompt, setPrompt] = useState('Create a 30-second energetic Instagram reel for a college hackathon')
  const [duration, setDuration] = useState(30)
  const [platform, setPlatform] = useState<Platform>('instagram')
  const [jobId, setJobId] = useState<string | null>(null)
  const [agents, setAgents] = useState<AgentInfo[]>(defaultAgents())
  const [progress, setProgress] = useState(0)
  const [currentAgent, setCurrentAgent] = useState<string | null>(null)
  const [result, setResult] = useState<JobResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const showToast = useCallback((msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }, [])

  const updateAgent = useCallback((agentName: string, status: AgentInfo['status'], elapsedMs?: number) => {
    setAgents(prev => prev.map(a =>
      a.name === agentName ? { ...a, status, elapsedMs: elapsedMs ?? a.elapsedMs } : a
    ))
  }, [])

  const handleWSMessage = useCallback((msg: unknown) => {
    const m = msg as WSMessage
    if (m.type === 'progress' && m.agent) {
      setCurrentAgent(m.agent)
      if (m.status === 'running') {
        updateAgent(m.agent, 'running')
        // Set progress to start of this agent's slice
        const idx = AGENT_ORDER.indexOf(m.agent)
        if (idx >= 0) setProgress((idx / AGENT_ORDER.length) * 100)
      }
      if (m.status === 'done') {
        updateAgent(m.agent, 'done')
        // Set progress to end of this agent's slice
        const idx = AGENT_ORDER.indexOf(m.agent)
        if (idx >= 0) setProgress(((idx + 1) / AGENT_ORDER.length) * 100)
      }
      if (m.status === 'error') updateAgent(m.agent, 'error')
      // Also accept explicit progress_pct if server sends it
      if (m.progress_pct !== undefined && m.progress_pct > 0) setProgress(m.progress_pct)
    }
  }, [updateAgent])

  const startJob = async () => {
    if (clips.length === 0) return
    setError(null)
    setStage('processing')
    setAgents(defaultAgents())
    setProgress(0)

    try {
      const res = await createJob({
        clips: clips.map(c => c.serverPath),
        userPrompt: prompt,
        targetDuration: duration,
        platform,
        brandPdfPath,
      })
      const id: string = res.job_id
      setJobId(id)

      // Connect WebSocket
      const ws = createWSConnection(id, handleWSMessage)
      wsRef.current = ws

      // Poll for completion every 5s as backup
      const poller = setInterval(async () => {
        try {
          const r = await getJobResult(id)
          if (r.status === 'done') {
            clearInterval(poller)
            setResult(r)
            setProgress(100)
            setStage('done')
            ws.close()
          } else if (r.status === 'failed') {
            clearInterval(poller)
            setError('Pipeline failed — check logs for details')
            setStage('error')
            ws.close()
          }
        } catch (e: unknown) {
          // 202 means still processing — ignore; anything else surface it
          const msg = e instanceof Error ? e.message : String(e)
          if (!msg.includes('202')) console.warn('Poll error:', msg)
        }
      }, 5000)
    } catch (e: unknown) {
      const err = e instanceof Error ? e.message : String(e)
      setError(`Failed to start job: ${err}`)
      setStage('error')
    }
  }

  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  const reset = () => {
    setStage('upload')
    setClips([])
    setBrandPdfPath(undefined)
    setJobId(null)
    setAgents(defaultAgents())
    setProgress(0)
    setResult(null)
    setError(null)
    setToast(null)
    wsRef.current?.close()
  }

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────── */}
      <header className="app-header">
        <div className="container">
          <div className="header-inner">
            <div className="header-brand">
              <span className="header-logo">🎬</span>
              <div>
                <div className="header-title gradient-text">AI Video Director</div>
                <div className="header-sub">Multi-agent reel generation · LangGraph · RAG · Gemini</div>
              </div>
            </div>
            <div className="header-badges">
              <span className="badge badge-purple">LangGraph</span>
              <span className="badge badge-blue">RAG + Reranker</span>
              <span className="badge badge-green">Gemini</span>
              <span className="badge badge-yellow">FFmpeg</span>
            </div>
          </div>
        </div>
        {/* ── Navbar progress bar (only during processing) ── */}
        {stage === 'processing' && (
          <div className="navbar-progress">
            <div
              className="navbar-progress-fill"
              style={{ width: `${progress}%` }}
            />
            <div className="navbar-progress-label">
              {AGENT_ORDER.indexOf(currentAgent ?? '') >= 0
                ? `${AGENT_META_LABELS[currentAgent!]} — ${progress.toFixed(0)}%`
                : `Starting… ${progress.toFixed(0)}%`}
            </div>
          </div>
        )}
      </header>

      <main className="app-main container">
        {/* ── Upload Stage ─────────────────────────────── */}
        {stage === 'upload' && (
          <div className="stage-upload animate-fade-up">
            <div className="hero-text">
              <h1>Turn raw clips into a <span className="gradient-text">polished reel</span></h1>
              <p>Upload 10–20 clips. Describe your vision. Watch 8 AI agents do the rest.</p>
            </div>

            <div className="upload-layout">
              {/* Left: Upload */}
              <div className="upload-col">
                <div className="clip-guide card">
                  <div className="clip-guide__title">📋 What clips work best</div>
                  <div className="clip-guide__grid">
                    <div className="clip-guide__item clip-guide__item--good">
                      <div className="clip-guide__item-icon">✅</div>
                      <div>
                        <div className="clip-guide__item-label">Good lighting</div>
                        <div className="clip-guide__item-desc">Bright, evenly lit scenes — natural light or studio</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--good">
                      <div className="clip-guide__item-icon">✅</div>
                      <div>
                        <div className="clip-guide__item-label">3–15 sec clips</div>
                        <div className="clip-guide__item-desc">Short punchy clips get the best scene cuts</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--good">
                      <div className="clip-guide__item-icon">✅</div>
                      <div>
                        <div className="clip-guide__item-label">People &amp; faces</div>
                        <div className="clip-guide__item-desc">Face/smile detection boosts highlight scores</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--good">
                      <div className="clip-guide__item-icon">✅</div>
                      <div>
                        <div className="clip-guide__item-label">Clear audio</div>
                        <div className="clip-guide__item-desc">Whisper transcribes speech for smarter captions</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--bad">
                      <div className="clip-guide__item-icon">❌</div>
                      <div>
                        <div className="clip-guide__item-label">Blurry / shaky</div>
                        <div className="clip-guide__item-desc">Low Laplacian score — filtered out by quality agent</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--bad">
                      <div className="clip-guide__item-icon">❌</div>
                      <div>
                        <div className="clip-guide__item-label">Very dark / overexposed</div>
                        <div className="clip-guide__item-desc">Brightness outside 20–240 range gets low scores</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--bad">
                      <div className="clip-guide__item-icon">❌</div>
                      <div>
                        <div className="clip-guide__item-label">Clips &gt; 500 MB</div>
                        <div className="clip-guide__item-desc">Split large files before uploading</div>
                      </div>
                    </div>
                    <div className="clip-guide__item clip-guide__item--tip">
                      <div className="clip-guide__item-icon">💡</div>
                      <div>
                        <div className="clip-guide__item-label">10–20 clips ideal</div>
                        <div className="clip-guide__item-desc">Gives the story planner enough variety to build a narrative</div>
                      </div>
                    </div>
                  </div>
                  <div className="clip-guide__formats">
                    <span className="badge badge-purple">MP4</span>
                    <span className="badge badge-blue">MOV</span>
                    <span className="badge badge-green">AVI</span>
                    <span className="badge badge-yellow">MKV</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>· up to 500 MB each</span>
                  </div>
                </div>
                <UploadZone
                  onClipsReady={setClips}
                  onBrandPdf={setBrandPdfPath}
                />
              </div>

              {/* Right: Config */}
              <div className="config-col">
                <div className="card">
                  <h3 style={{ marginBottom: 16 }}>🎯 Director's Brief</h3>

                  <label className="form-label" htmlFor="prompt-input">Your Vision</label>
                  <textarea
                    id="prompt-input"
                    className="textarea"
                    rows={3}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    placeholder="Create a 30-second energetic Instagram reel for a college hackathon..."
                    title="Your Vision"
                  />

                  <div className="form-row" style={{ marginTop: 16 }}>
                    <div style={{ flex: 1 }}>
                      <label className="form-label" htmlFor="platform-select">Platform</label>
                      <select
                        id="platform-select"
                        className="select"
                        value={platform}
                        onChange={e => setPlatform(e.target.value as Platform)}
                        title="Platform"
                      >
                        <option value="instagram">📱 Instagram</option>
                        <option value="tiktok">🎵 TikTok</option>
                        <option value="youtube">▶️ YouTube</option>
                        <option value="generic">🎬 Generic</option>
                      </select>
                    </div>
                    <div style={{ width: 120 }}>
                      <label className="form-label" htmlFor="duration-input">Duration (s)</label>
                      <input
                        id="duration-input"
                        type="number"
                        className="input"
                        value={duration}
                        min={10}
                        max={300}
                        onChange={e => setDuration(Number(e.target.value))}
                        title="Duration in seconds"
                      />
                    </div>
                  </div>

                  {brandPdfPath && (
                    <div className="brand-indicator">
                      ✅ Brand guide indexed
                    </div>
                  )}

                  <button
                    id="generate-btn"
                    className="btn btn-primary generate-btn"
                    onClick={startJob}
                    disabled={clips.length === 0 || !prompt.trim()}
                  >
                    {clips.length === 0
                      ? '⬆️ Upload clips first'
                      : `🚀 Generate Reel from ${clips.length} clips`}
                  </button>

                  {clips.length === 0 && (
                    <p style={{ textAlign: 'center', fontSize: '0.8rem', marginTop: 8, color: 'var(--text-muted)' }}>
                      Upload at least 1 video clip to start
                    </p>
                  )}
                </div>

                {/* Pipeline overview */}
                <div className="card pipeline-overview">
                  <div className="po-title">🤖 8-Agent Pipeline</div>
                  <div className="po-agents">
                    {['Scene Extract', 'Quality Check', 'Highlights', 'Captions', 'Brand RAG', 'Story Plan', 'Edit', 'Export'].map((name, i) => (
                      <div key={i} className="po-agent">
                        <div className="po-agent-dot" />
                        <span>{name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Processing Stage ─────────────────────────── */}
        {stage === 'processing' && (
          <div className="stage-processing animate-fade-up">
            <div className="processing-header">
              <h2>🎬 Generating your reel<span className="dot-anim">...</span></h2>
              <p>Job ID: <span className="mono" style={{ color: '#8b5cf6', fontSize: '0.8rem' }}>{jobId}</span></p>
              <div className="overall-progress">
                <div className="progress-bar" style={{ height: 8 }}>
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                </div>
                <div className="progress-label">{progress.toFixed(0)}% · {currentAgent?.replace('_', ' ') || 'starting…'}</div>
              </div>
            </div>

            <div className="processing-layout">
              <div className="graph-col">
                <AgentGraph agents={agents} />
              </div>
              <div className="processing-info card">
                <div className="pi-title">What's happening</div>
                <div className="pi-steps">
                  {[
                    "🎬 Extracting scenes from your clips using OpenCV",
                    "🔍 Scoring blur, brightness, and motion quality",
                    "✨ Detecting faces, smiles, and audio peaks",
                    "💬 Generating captions via Gemini Flash",
                    "🧠 Retrieving brand style via FAISS + BGE Reranker",
                    "📖 Planning narrative arc with Gemini Pro",
                    "✂️ Editing with FFmpeg — transitions, music, subs",
                    "📤 Exporting final reel + evaluation metrics"
                  ].map((step, idx) => {
                    const currentIdx = currentAgent ? AGENT_ORDER.indexOf(currentAgent) : -1;
                    let className = "pi-step";
                    if (idx < currentIdx || currentAgent === 'pipeline') className += " pi-step--done";
                    else if (idx === currentIdx) className += " pi-step--active";
                    
                    return <div key={idx} className={className}>{step}</div>
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Done Stage ───────────────────────────────── */}
        {stage === 'done' && result && (
          <div className="stage-done animate-fade-up">
            <div className="done-header">
              <h2>✅ Your reel is ready!</h2>
              <button className="btn btn-ghost" id="new-reel-btn" onClick={reset}>+ New Reel</button>
            </div>

            <div className="done-layout">
              <div className="done-left">
                <VideoPreview result={result} />
              </div>
              <div className="done-right">
                <AgentGraph agents={agents} />
                <MetricsPanel result={result} onToast={showToast} />
              </div>
            </div>
          </div>
        )}

        {/* ── Error Stage ──────────────────────────────── */}
        {stage === 'error' && (
          <div className="stage-error animate-fade-up">
            <div className="error-card card">
              <div style={{ fontSize: '3rem' }}>❌</div>
              <h2>Pipeline Error</h2>
              <p>{error}</p>
              <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
                <button className="btn btn-primary" id="retry-btn" onClick={startJob}>Retry</button>
                <button className="btn btn-ghost" onClick={reset}>Start Over</button>
              </div>
            </div>
          </div>
        )}

        {/* ── Toast Notification ───────────────────────── */}
        {toast && (
          <div className={`toast toast--${toast.type}`}>
            {toast.type === 'success' ? '✅ ' : '❌ '}
            {toast.msg}
          </div>
        )}
      </main>
    </div>
  )
}
