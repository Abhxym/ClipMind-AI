// Shared types for the AI Video Director frontend

export type Platform = 'instagram' | 'tiktok' | 'youtube' | 'generic'
export type JobStatus = 'pending' | 'running' | 'done' | 'failed'
export type AgentStatusType = 'pending' | 'running' | 'done' | 'error'

export interface UploadedClip {
  fileId: string
  filename: string
  sizeBytes: number
  serverPath: string
  previewUrl?: string
}

export interface AgentInfo {
  name: string
  label: string
  description: string
  color: string
  icon: string
  status: AgentStatusType
  elapsedMs?: number
}

export interface WSMessage {
  type: 'progress' | 'connected' | 'heartbeat' | 'pong'
  job_id?: string
  agent?: string
  status?: string
  attempt?: number
  timestamp?: string
  progress_pct?: number
  message?: string
}

export interface JobResult {
  job_id: string
  status: JobStatus
  video_title?: string
  final_video_url?: string
  final_thumbnail_url?: string
  captions: string[]
  hashtags: string[]
  tone?: string
  planned_duration?: number
  evaluation?: Record<string, unknown>
  agent_timings: Record<string, number>
}
