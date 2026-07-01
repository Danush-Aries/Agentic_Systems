# Agentic Systems Playground

Interactive live demo for the Agentic Systems ReAct framework.

- **Backend**: FastAPI, streams ReAct trace events over a WebSocket at `/ws/run`.
- **Frontend**: Vite + React + TypeScript + Tailwind + shadcn/ui — a single-page
  two-panel UI (config on the left, live trace on the right).
- **Adapter**: `backend/adapters.py` wraps the framework's `ReActAgent`. Currently
  emits a mocked trace so the pipeline can be tested without an LLM key
  (see the `TODO: wire real agent` comment).

## Run locally

```bash
cd playground
docker compose up --build
# open http://localhost:5173
```

## Deploy

Backend and frontend are packaged as independent Docker images so you can push
them to Fly.io / Railway / Render (backend) and Vercel / Netlify / Cloudflare
Pages (frontend). Wire the frontend's `VITE_BACKEND_WS_URL` to the deployed
backend's WebSocket URL (`wss://...`).
