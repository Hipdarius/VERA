# Regoscan Web — Mission Console

Next.js 14 + Tailwind + Recharts + Framer Motion frontend backed by a FastAPI/ONNX Python backend. Deploys to Vercel as a single project.

## Components

| Component | Description |
|-----------|-------------|
| `Hero.tsx` | Header with status pills, theme toggle, loading state |
| `ScanButton.tsx` | Scan trigger with animation |
| `UploadPanel.tsx` | CSV file upload and parsing |
| `SpectrumChart.tsx` | 288-point spectrum with endmember overlays |
| `IlmeniteGauge.tsx` | Circular SVG gauge for ilmenite mass % |
| `ProbabilityBars.tsx` | Horizontal bars for 5 mineral classes |
| `ScanHistory.tsx` | Clickable list of past scans (max 10) |
| `MissionPanel.tsx` | Reusable themed panel wrapper |
| `ThemeProvider.tsx` | Dark/light mode context with localStorage |

## Local Development

Two terminals required:

```bash
# Terminal 1 — FastAPI backend (from repo root)
uv run uvicorn apps.api:app --reload --port 8000

# Terminal 2 — Next.js (from repo root)
cd web && npm run dev
```

The `.env.local` file sets `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000` to proxy API calls to the backend.

Open http://localhost:3000.

## Deploying to Vercel

1. Copy the trained model: `cp runs/cnn_v2/model.onnx web/api/model.onnx`
2. Push to GitHub
3. Link to Vercel: `cd web && vercel link`
4. Deploy: `vercel --prod`

In production, do not set `NEXT_PUBLIC_API_BASE` — Vercel rewrites route `/api/*` to the Python serverless function.
