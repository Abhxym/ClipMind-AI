import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

// ── Cinematic mouse parallax on background ──
let rafId: number | null = null
document.addEventListener('mousemove', (e) => {
  if (rafId) return
  rafId = requestAnimationFrame(() => {
    const bg = document.querySelector('.cinematic-bg') as HTMLElement | null
    if (bg) {
      const xPct = (e.clientX / window.innerWidth  - 0.5) * 14
      const yPct = (e.clientY / window.innerHeight - 0.5) * 9
      bg.style.transform = `scale(1.08) translate(${xPct}px, ${yPct}px)`
    }
    rafId = null
  })
})
document.addEventListener('mouseleave', () => {
  const bg = document.querySelector('.cinematic-bg') as HTMLElement | null
  if (bg) bg.style.transform = 'scale(1.08) translate(0px, 0px)'
})
