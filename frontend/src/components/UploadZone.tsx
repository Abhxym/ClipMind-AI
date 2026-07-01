import { useState, useCallback } from 'react'
import type { UploadedClip } from '../types'
import { uploadVideo, uploadBrandPdf } from '../api'

interface UploadZoneProps {
  onClipsReady: (clips: UploadedClip[]) => void
  onBrandPdf: (path: string) => void
}

export function UploadZone({ onClipsReady, onBrandPdf }: UploadZoneProps) {
  const [clips, setClips] = useState<UploadedClip[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [brandDragOver, setBrandDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})

  const handleVideoFiles = useCallback(async (files: File[]) => {
    const videoFiles = files.filter(f => f.type.startsWith('video/'))
    if (videoFiles.length === 0) return
    setUploading(true)

    const newClips: UploadedClip[] = []
    for (const file of videoFiles) {
      try {
        setUploadProgress(p => ({ ...p, [file.name]: 50 }))
        const data = await uploadVideo(file)
        const previewUrl = URL.createObjectURL(file)
        newClips.push({
          fileId: data.file_id,
          filename: data.filename,
          sizeBytes: data.size_bytes,
          serverPath: data.server_path,
          previewUrl
        })
        setUploadProgress(p => ({ ...p, [file.name]: 100 }))
      } catch (err) {
        console.error('Upload failed:', err)
        setUploadProgress(p => ({ ...p, [file.name]: -1 }))
      }
    }

    const updated = [...clips, ...newClips]
    setClips(updated)
    onClipsReady(updated)
    setUploading(false)
  }, [clips, onClipsReady])

  const handleBrandFile = useCallback(async (file: File) => {
    if (!file.name.endsWith('.pdf')) return
    try {
      const data = await uploadBrandPdf(file)
      onBrandPdf(data.server_path)
    } catch (err) {
      console.error('Brand PDF upload failed:', err)
    }
  }, [onBrandPdf])

  const removeClip = (fileId: string) => {
    const updated = clips.filter(c => c.fileId !== fileId)
    setClips(updated)
    onClipsReady(updated)
  }

  return (
    <div className="upload-zone-wrapper animate-fade-up">
      {/* ── Video Upload ─────────────────────────────── */}
      <div
        className={`drop-zone ${dragOver ? 'drop-zone--active' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault(); setDragOver(false)
          handleVideoFiles(Array.from(e.dataTransfer.files))
        }}
        onClick={() => document.getElementById('video-input')!.click()}
        id="video-drop-zone"
      >
        <div className="drop-zone__icon">🎬</div>
        <div className="drop-zone__title">Drop your video clips here</div>
        <div className="drop-zone__sub">MP4, MOV, AVI, MKV • up to 500MB each • 10–20 clips</div>
        <input
          id="video-input"
          type="file"
          accept="video/*"
          multiple
          style={{ display: 'none' }}
          onChange={e => handleVideoFiles(Array.from(e.target.files || []))}
          title="Upload video files"
        />
        {uploading && <div className="drop-zone__spinner" />}
      </div>

      {/* ── Clip Grid ─────────────────────────────────── */}
      {clips.length > 0 && (
        <div className="clip-grid">
          {clips.map(clip => (
            <div key={clip.fileId} className="clip-card">
              {clip.previewUrl && (
                <video
                  src={clip.previewUrl}
                  className="clip-thumb"
                  muted
                  onMouseEnter={e => (e.target as HTMLVideoElement).play()}
                  onMouseLeave={e => { const v = e.target as HTMLVideoElement; v.pause(); v.currentTime = 0 }}
                />
              )}
              <div className="clip-info">
                <span className="clip-name">{clip.filename.slice(0, 22)}{clip.filename.length > 22 ? '…' : ''}</span>
                <span className="clip-size">{(clip.sizeBytes / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <button className="clip-remove" onClick={() => removeClip(clip.fileId)}>✕</button>
              {uploadProgress[clip.filename] === -1 && (
                <div className="clip-error">Upload failed</div>
              )}
            </div>
          ))}
        </div>
      )}

      {clips.length > 0 && (
        <div className="clip-count">
          <span className="badge badge-purple">🎥 {clips.length} clip{clips.length !== 1 ? 's' : ''} ready</span>
        </div>
      )}

      {/* ── Brand PDF Upload ─────────────────────────── */}
      <div
        className={`brand-drop ${brandDragOver ? 'brand-drop--active' : ''}`}
        onDragOver={e => { e.preventDefault(); setBrandDragOver(true) }}
        onDragLeave={() => setBrandDragOver(false)}
        onDrop={e => {
          e.preventDefault(); setBrandDragOver(false)
          const file = e.dataTransfer.files[0]
          if (file) handleBrandFile(file)
        }}
        onClick={() => document.getElementById('pdf-input')!.click()}
        id="brand-pdf-drop"
      >
        <span>📄</span>
        <span>Drop brand guide PDF <span style={{ color: 'var(--text-muted)' }}>(optional)</span></span>
        <input
          id="pdf-input"
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) handleBrandFile(f) }}
          title="Upload brand guide PDF"
        />
      </div>

      <style>{`
        .upload-zone-wrapper { display: flex; flex-direction: column; gap: 16px; }
        .drop-zone {
          border: 2px dashed var(--color-border);
          border-radius: var(--r-xl);
          padding: 48px 32px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s;
          background: var(--color-surface);
          position: relative;
          overflow: hidden;
        }
        .drop-zone::before {
          content: '';
          position: absolute;
          inset: 0;
          background: var(--grad-glow);
          opacity: 0;
          transition: opacity 0.3s;
        }
        .drop-zone--active, .drop-zone:hover { border-color: #8b5cf6; }
        .drop-zone--active::before, .drop-zone:hover::before { opacity: 1; }
        .drop-zone__icon { font-size: 3rem; margin-bottom: 12px; }
        .drop-zone__title { font-size: 1.2rem; font-weight: 600; margin-bottom: 6px; }
        .drop-zone__sub { color: var(--text-muted); font-size: 0.85rem; }
        .drop-zone__spinner {
          width: 28px; height: 28px; margin: 12px auto 0;
          border: 3px solid var(--color-surface-3);
          border-top-color: #8b5cf6;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        .clip-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 12px;
        }
        .clip-card {
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--r-md);
          overflow: hidden;
          position: relative;
          transition: border-color 0.2s;
        }
        .clip-card:hover { border-color: var(--color-border-glow); }
        .clip-thumb {
          width: 100%; height: 90px; object-fit: cover; display: block;
          background: var(--color-surface-3);
        }
        .clip-info {
          padding: 8px;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .clip-name { font-size: 0.75rem; font-weight: 500; color: var(--text-primary); }
        .clip-size { font-size: 0.7rem; color: var(--text-muted); }
        .clip-remove {
          position: absolute;
          top: 4px; right: 4px;
          background: rgba(239,68,68,0.85);
          border: none; border-radius: 50%;
          width: 20px; height: 20px;
          color: white; font-size: 0.65rem;
          cursor: pointer; display: flex;
          align-items: center; justify-content: center;
          opacity: 0; transition: opacity 0.2s;
        }
        .clip-card:hover .clip-remove { opacity: 1; }
        .clip-error {
          position: absolute; inset: 0;
          background: rgba(239,68,68,0.5);
          display: flex; align-items: center; justify-content: center;
          font-size: 0.7rem; font-weight: 600;
        }
        .clip-count { text-align: center; }
        .brand-drop {
          border: 1px dashed var(--color-border);
          border-radius: var(--r-md);
          padding: 14px 24px;
          display: flex;
          align-items: center;
          gap: 10px;
          cursor: pointer;
          font-size: 0.9rem;
          color: var(--text-secondary);
          transition: all 0.2s;
        }
        .brand-drop:hover, .brand-drop--active { border-color: #f59e0b; color: #fbbf24; }
      `}</style>
    </div>
  )
}
