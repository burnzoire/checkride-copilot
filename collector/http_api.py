"""
Minimal FastAPI HTTP server exposing the current LiveState.

  GET /state   → full LiveState dict (JSON)
  GET /health  → {"ok": true, "data_age_ms": <int>}

Run standalone:  uvicorn collector.http_api:app --host 127.0.0.1 --port 7779
Or launch via collector/main.py which starts UDP + HTTP together.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from collector import state_store

app = FastAPI(title="DCS Copilot Collector", version="0.1.0")


@app.get("/state")
def get_state():
    state = state_store.get()
    if state is None:
        raise HTTPException(status_code=503, detail="No DCS state received yet.")
    return JSONResponse(content=state)


@app.get("/health")
def health():
    age = state_store.get_age_ms()
    return {"ok": age >= 0, "data_age_ms": age}


@app.get("/debug")
def debug():
    """Full last raw packet for development — shows exactly what DCS is sending."""
    raw = state_store.get_raw_packet()
    if raw is None:
        raise HTTPException(status_code=503, detail="No packet received yet.")
    return JSONResponse(content=raw)


@app.get("/raw_cp")
def get_raw_cp():
    """Raw CockpitParams dict for Phase 0 index discovery."""
    raw = state_store.get_raw_cp()
    if raw is None:
        raise HTTPException(status_code=503, detail="No raw CockpitParams received yet.")
    return JSONResponse(content=raw)
