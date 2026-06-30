import { JobResult } from '../types'

interface MetricsPanelProps {
  result: JobResult
}

function MetricCard({ label, value, sub, color }: {
  label: string; value: string; sub?: string; color?: string
}) {
  return (
    <div className="metric-card">
      <div className="metric-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      <div className="metric-label">{label}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

export function MetricsPanel({ result }: MetricsPanelProps) {
  const ev = result.evaluation as Record<string, number> | undefined
  const timings = result.agent_timings
  const totalMs = Object.values(timings).reduce((a, b) => a + b, 0)
  const bottleneck = Object.entries(timings).sort((a, b) => b[1] - a[1])[0]

  return (
    <div className="metrics-panel">
      <h3 className="metrics-title">📊 Evaluation Report</h3>

      {/* ── Top metrics ─────────────────────────────── */}
      <div className="metrics-grid">
        <MetricCard
          label="Total Latency"
          value={totalMs < 1000 ? `${totalMs.toFixed(0)}ms` : `${(totalMs / 1000).toFixed(1)}s`}
          sub="end-to-end"
          color="#8b5cf6"
        />
        <MetricCard
          label="Retrieval P@K"
          value={ev ? ev['retrieval_precision_at_k']?.toFixed(3) ?? '—' : '—'}
          sub="after reranking"
          color="#06b6d4"
        />
        <MetricCard
          label="Reranker Lift"
          value={ev ? (ev['reranker_lift'] >= 0 ? '+' : '') + ev['reranker_lift']?.toFixed(3) : '—'}
          sub="BGE cross-encoder"
          color="#10b981"
        />
        <MetricCard
          label="Scenes Selected"
          value={String(result.evaluation ? (ev?.['scene_selection_count'] ?? '—') : result.captions.length)}
          sub="in final reel"
          color="#f59e0b"
        />
        <MetricCard
          label="Errors"
          value={ev ? String(ev['error_count'] ?? 0) : '0'}
          color={ev && ev['error_count'] > 0 ? '#ef4444' : '#10b981'}
        />
        <MetricCard
          label="Bottleneck"
          value={bottleneck ? bottleneck[0].replace('_agent', '') : '—'}
          sub={bottleneck ? `${(bottleneck[1] / 1000).toFixed(1)}s` : ''}
          color="#ec4899"
        />
      </div>

      {/* ── Agent Timings Bar chart ─────────────────── */}
      {Object.keys(timings).length > 0 && (
        <div className="timing-chart">
          <div className="timing-chart__title">Agent Latency Breakdown</div>
          {Object.entries(timings)
            .sort((a, b) => b[1] - a[1])
            .map(([agent, ms]) => (
              <div key={agent} className="timing-row">
                <div className="timing-label">{agent.replace('_agent', '').replace('_', ' ')}</div>
                <div className="timing-bar-wrap">
                  <div
                    className="timing-bar"
                    style={{ width: `${Math.min(100, (ms / totalMs) * 100)}%` }}
                  />
                </div>
                <div className="timing-ms mono">
                  {ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`}
                </div>
              </div>
            ))}
        </div>
      )}

      {/* ── Hashtags ────────────────────────────────── */}
      {result.hashtags.length > 0 && (
        <div className="hashtag-section">
          <div className="timing-chart__title">Generated Hashtags</div>
          <div className="hashtag-list">
            {result.hashtags.slice(0, 15).map(tag => (
              <span key={tag} className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .metrics-panel {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--r-xl);
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .metrics-title { font-size: 1rem; font-weight: 700; }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }
        .metric-card {
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--r-md);
          padding: 14px;
          text-align: center;
        }
        .metric-value { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
        .metric-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; }
        .metric-sub { font-size: 0.68rem; color: var(--text-muted); }

        .timing-chart { display: flex; flex-direction: column; gap: 8px; }
        .timing-chart__title {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 4px;
        }
        .timing-row { display: flex; align-items: center; gap: 10px; }
        .timing-label { font-size: 0.72rem; color: var(--text-secondary); width: 80px; flex-shrink: 0; }
        .timing-bar-wrap { flex: 1; height: 6px; background: var(--color-surface-3); border-radius: 3px; overflow: hidden; }
        .timing-bar { height: 100%; background: var(--grad-primary); border-radius: 3px; transition: width 0.6s ease; }
        .timing-ms { font-size: 0.68rem; width: 44px; text-align: right; flex-shrink: 0; }

        .hashtag-section { display: flex; flex-direction: column; gap: 8px; }
        .hashtag-list { display: flex; flex-wrap: wrap; gap: 6px; }
      `}</style>
    </div>
  )
}
