import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend the dev server proxies /api and /outputs to (override if port 8000 is taken)
const apiTarget = process.env.API_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        ws: true,
      },
      '/outputs': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
