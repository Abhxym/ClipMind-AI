// API client for AI Video Director backend

const BASE = '/api'

export async function uploadVideo(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload/video`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function uploadBrandPdf(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload/brand-pdf`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function createJob(params: {
  clips: string[]
  userPrompt: string
  targetDuration: number
  platform: string
  brandPdfPath?: string
}) {
  const url = `${BASE}/jobs`
  const body = {
    user_prompt: params.userPrompt,
    target_duration: params.targetDuration,
    platform: params.platform,
    brand_pdf_path: params.brandPdfPath,
    clip_paths: params.clips,
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getJobResult(jobId: string) {
  const res = await fetch(`${BASE}/jobs/${jobId}/result`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getJobStatus(jobId: string) {
  const res = await fetch(`${BASE}/jobs/${jobId}/status`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function createWSConnection(jobId: string, onMessage: (msg: unknown) => void) {
  const wsUrl = `ws://localhost:8000/api/ws/${jobId}`
  const ws = new WebSocket(wsUrl)
  ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)) } catch (_) {} }
  ws.onerror = () => console.warn('WS error')
  return ws
}

export async function enhanceVideo(jobId: string, filterType: string, musicTrack: string) {
  const url = `${BASE}/jobs/${jobId}/enhance`
  const body = {
    filter_type: filterType,
    music_track: musicTrack,
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
