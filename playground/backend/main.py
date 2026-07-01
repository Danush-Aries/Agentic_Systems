"""
FastAPI backend for the Agentic Systems playground.

Endpoints
---------
GET  /health                → liveness probe
GET  /templates             → list available agent templates
WS   /ws/run                → stream a ReAct trace as JSON events

Rate limiting is in-memory (per-IP, sliding window). No Redis, no DB — this
service is deliberately stateless and free-tier friendly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Make the parent framework importable when running from the playground dir.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adapters import iter_events, list_templates  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("playground")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Agentic Systems Playground", version="0.1.0")

_ALLOWED_ORIGINS = os.getenv(
    "PLAYGROUND_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limit
# ---------------------------------------------------------------------------

_RATE_WINDOW_SEC = 60
_RATE_MAX_REQUESTS = 10
_rate_buckets: Dict[str, Deque[float]] = {}


def _rate_limited(client_key: str) -> bool:
    now = time.time()
    bucket = _rate_buckets.setdefault(client_key, deque())
    while bucket and now - bucket[0] > _RATE_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _RATE_MAX_REQUESTS:
        return True
    bucket.append(now)
    return False


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/templates")
async def templates():
    return {"templates": list_templates()}


# ---------------------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------------------

@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await ws.accept()

    client_host = ws.client.host if ws.client else "unknown"
    if _rate_limited(client_host):
        await ws.send_json({
            "type": "error",
            "content": "Rate limit exceeded. Try again in a minute.",
            "ts": time.time(),
        })
        await ws.close(code=1013)
        return

    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        await ws.send_json({
            "type": "error",
            "content": "Timed out waiting for run request.",
            "ts": time.time(),
        })
        await ws.close(code=1011)
        return

    try:
        payload = json.loads(raw)
        template_id = str(payload.get("template", "assistant"))
        task = str(payload.get("task", "")).strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        await ws.send_json({
            "type": "error",
            "content": "Invalid JSON payload. Expected {template, task}.",
            "ts": time.time(),
        })
        await ws.close(code=1003)
        return

    if not task:
        await ws.send_json({
            "type": "error",
            "content": "Task must be a non-empty string.",
            "ts": time.time(),
        })
        await ws.close(code=1003)
        return

    logger.info("Run request from %s: template=%s task=%r", client_host, template_id, task)

    try:
        async for event in iter_events(template_id, task):
            await ws.send_json(event.to_dict())
    except WebSocketDisconnect:
        logger.info("Client disconnected mid-run")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while streaming events")
        try:
            await ws.send_json({
                "type": "error",
                "content": f"Server error: {exc}",
                "ts": time.time(),
            })
        except Exception:  # noqa: BLE001
            pass

    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass
