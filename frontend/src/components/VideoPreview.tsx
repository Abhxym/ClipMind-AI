import { useRef, useState } from 'react'
import type { JobResult, Platform } from '../types'
import { enhanceVideo } from '../api'

interface VideoPreviewProps {
  result: JobResult
  platform?: Platform | string
}

export function VideoPreview({ result, platform }: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  const [selectedFilter, setSelectedFilter] = useState('none')
  const [selectedMusic, setSelectedMusic] = useState('none')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhancedVideoUrl, setEnhancedVideoUrl] = useState<string | null>(null)

  const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const originalVideoUrl = result.final_video_url ? new URL(result.final_video_url, backendUrl).href : null;
  const videoUrl = enhancedVideoUrl ? new URL(enhancedVideoUrl, backendUrl).href : originalVideoUrl;
  const thumbUrl = result.final_thumbnail_url ? new URL(result.final_thumbnail_url, backendUrl).href : null;

  const handleEnhance = async () => {
    try {
      setIsEnhancing(true)
      const res = await enhanceVideo(result.job_id, selectedFilter, selectedMusic)
      setEnhancedVideoUrl(res.enhanced_video_url)
    } catch (err) {
      console.error('Enhance failed:', err)
      alert('Failed to enhance video.')
    } finally {
      setIsEnhancing(false)
    }
  }

  return (
    <div className="video-preview">
      <div className="vp-header">
        <div>
          <h2 className="vp-title gradient-text">
            {result.video_title || 'Your AI-Generated Reel'}
          </h2>
          <div className="vp-meta">
            {result.tone && <span className="badge badge-purple">{result.tone}</span>}
            {result.planned_duration && (
              <span className="badge badge-blue">⏱ {result.planned_duration.toFixed(1)}s</span>
            )}
          </div>
        </div>

        {videoUrl && (
          <a
            href={videoUrl}
            download
            className="btn btn-primary"
            id="download-video-btn"
          >
            ⬇️ Download
          </a>
        )}
      </div>

      {/* ── Video Player ─────────────────────────────── */}
      <div className="vp-player">
        {videoUrl ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              poster={thumbUrl || undefined}
              className="vp-video"
              controls
            />
          </>
        ) : thumbUrl ? (
          <img src={thumbUrl} alt="Reel thumbnail" className="vp-thumb" />
        ) : (
          <div className="vp-placeholder">
            <div style={{ fontSize: '3rem' }}>🎬</div>
            <div>Video processing...</div>
          </div>
        )}
      </div>

      {/* ── Instagram Studio ─────────────────────────────── */}
      {platform === 'instagram' && videoUrl && (
        <div className="vp-studio">
          <div className="vp-studio-header">
            <h3>✨ Instagram Studio</h3>
            <span className="badge badge-purple">Post-processing</span>
          </div>
          
          <div className="vp-studio-controls">
            <div className="vp-studio-group">
              <label>Cinematic Filters</label>
              <div className="vp-studio-options">
                {['none', 'vivid', 'cinematic', 'bw', 'vintage'].map(f => (
                  <button
                    key={f}
                    className={`btn-studio-opt ${selectedFilter === f ? 'active' : ''}`}
                    onClick={() => setSelectedFilter(f)}
                  >
                    {f === 'bw' ? 'B&W' : f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="vp-studio-group">
              <label>Background Music</label>
              <div className="vp-studio-options">
                {['none', 'track1', 'track2', 'track3', 'track4', 'track5'].map(m => (
                  <button
                    key={m}
                    className={`btn-studio-opt ${selectedMusic === m ? 'active' : ''}`}
                    onClick={() => setSelectedMusic(m)}
                  >
                    {m === 'none' ? 'None' : m.replace('track', 'Track ')}
                  </button>
                ))}
              </div>
            </div>

            <button
              className="btn btn-primary btn-enhance"
              onClick={handleEnhance}
              disabled={isEnhancing || (selectedFilter === 'none' && selectedMusic === 'none')}
            >
              {isEnhancing ? '⏳ Rendering...' : 'Apply & Render'}
            </button>
          </div>
        </div>
      )}

      {/* ── Captions ─────────────────────────────────── */}
      {result.captions.length > 0 && (
        <div className="vp-captions">
          <div className="vp-section-label">💬 Scene Captions</div>
          <div className="vp-caption-list">
            {result.captions.map((cap, i) => (
              <div key={i} className="vp-caption-item">
                <span className="vp-caption-num">{i + 1}</span>
                <span className="vp-caption-text">{cap}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .video-preview {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--r-xl);
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .vp-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
        .vp-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
        .vp-meta { display: flex; gap: 8px; flex-wrap: wrap; }

        .vp-player {
          border-radius: var(--r-lg);
          overflow: hidden;
          background: black;
          position: relative;
          aspect-ratio: 9/16;
          max-height: 500px;
          margin: 0 auto;
          width: 100%;
          max-width: 280px;
        }
        .vp-video, .vp-thumb {
          width: 100%; height: 100%;
          object-fit: cover;
          display: block;
        }
        .vp-placeholder {
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 12px;
          color: var(--text-muted);
          font-size: 0.9rem;
        }

        .vp-section-label {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 8px;
        }
        .vp-caption-list { display: flex; flex-direction: column; gap: 6px; }
        .vp-caption-item {
          display: flex; align-items: flex-start; gap: 10px;
          background: var(--color-surface-2);
          border-radius: var(--r-sm);
          padding: 8px 12px;
        }
        .vp-caption-num {
          font-size: 0.7rem;
          font-weight: 700;
          color: #8b5cf6;
          background: rgba(139,92,246,0.1);
          border-radius: 4px;
          padding: 2px 6px;
          flex-shrink: 0;
          font-family: 'JetBrains Mono', monospace;
        }
        .vp-caption-text { font-size: 0.85rem; color: var(--text-secondary); }

        /* ── Instagram Studio ── */
        .vp-studio {
          background: rgba(220, 158, 28, 0.03);
          border: 1px solid rgba(220, 158, 28, 0.15);
          border-radius: var(--r-lg);
          padding: 20px;
          margin-top: 10px;
        }
        .vp-studio-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 16px;
        }
        .vp-studio-header h3 {
          font-size: 1.1rem;
          color: #f0ece4;
          font-weight: 600;
        }
        .vp-studio-controls {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .vp-studio-group label {
          display: block;
          font-size: 0.8rem;
          color: var(--text-secondary);
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .vp-studio-options {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .btn-studio-opt {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: var(--text-secondary);
          padding: 6px 12px;
          border-radius: var(--r-md);
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-studio-opt:hover {
          background: rgba(255, 255, 255, 0.1);
          color: #f0ece4;
        }
        .btn-studio-opt.active {
          background: rgba(220, 158, 28, 0.15);
          border-color: rgba(220, 158, 28, 0.4);
          color: #f0b429;
          font-weight: 500;
        }
        .btn-enhance {
          margin-top: 8px;
          width: 100%;
          justify-content: center;
        }
        .btn-enhance:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  )
}
