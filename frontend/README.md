# ClipMind AI — dashboard

React 19 + Vite + TypeScript front end for the ClipMind AI pipeline. See the [main README](../README.md).

```bash
npm install
npm run dev      # http://localhost:5173 — proxies /api and /outputs to http://localhost:8000
                 # (override with API_PROXY_TARGET=http://localhost:<port>)
npm run build    # type-check + production build into dist/
npm run lint     # oxlint
```

| Path | What |
|---|---|
| `src/App.tsx` | upload → processing → result flow, WebSocket + polling |
| `src/api.ts` | REST + WebSocket client (same-origin, proxied) |
| `src/components/UploadZone.tsx` | clip + brand-PDF upload |
| `src/components/AgentGraph.tsx` | live 13-agent status graph |
| `src/components/VideoPreview.tsx` | player, captions, Studio filters/music |
| `src/components/MetricsPanel.tsx` | evaluation scores + per-agent latency |
| `nginx.conf` | production proxy used by `Dockerfile.frontend` |
