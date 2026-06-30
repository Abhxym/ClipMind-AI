import { useState, useRef } from 'react'
import { JobResult } from '../types'

interface VideoPreviewProps {
  result: JobResult
}

export function VideoPreview({ result }: VideoPreviewProps) {
  const [playing, setPlaying] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) { videoRef.current.pause() }
    else { videoRef.current.play() }
    setPlaying(!playing)
  }

  const videoUrl = result.final_video_url

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
              poster={result.final_thumbnail_url || undefined}
              className="vp-video"
              controls
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
          </>
        ) : result.final_thumbnail_url ? (
          <img src={result.final_thumbnail_url} alt="Reel thumbnail" className="vp-thumb" />
        ) : (
          <div className="vp-placeholder">
            <div style={{ fontSize: '3rem' }}>🎬</div>
            <div>Video processing...</div>
          </div>
        )}
      </div>

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
      `}</style>
    </div>
  )
}
