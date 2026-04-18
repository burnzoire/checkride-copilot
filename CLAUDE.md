# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Checkride Copilot is a fully local, voice-first AI copilot for Digital Combat Simulator (DCS). It runs entirely on the pilot's machine — no cloud, no data exfiltration. The pilot speaks a query via push-to-talk; the system queries live cockpit state from DCS, searches procedure documents, and synthesizes a spoken answer via a local LLM. MVP airframe: F/A-18C Hornet.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (once they exist)
pytest

# Run a single test file
pytest tests/path/to/test_file.py

# Run a single test
pytest tests/path/to/test_file.py::test_function_name
```

> This project is in early scaffold. No runnable entry points exist yet. See `docs/build-plan.md` for the full implementation roadmap.

## Architecture

Six components run as separate local processes, communicating over loopback TCP:

| Component | Entry point | Responsibility |
|-----------|-------------|----------------|
| `collector/` | `dcs_collector.py` | UDP server on `localhost:7778`; ingests JSON from DCS `Export.lua`; writes `current_state.json`; exposes `GET /state` HTTP endpoint |
| `mcp_server/` | `mcp_server.py` | FastMCP server; exposes `get_current_state`, `diagnose_action_blockers`, `search_procedures`, `get_next_procedure_step` tools; stateless between calls |
| `orchestrator/` | `orchestrator.py` | Receives transcribed text; routes to MCP tools; assembles prompt; calls Ollama; streams reply to TTS |
| `retrieval/` | — | ChromaDB + BM25 index over F/A-18C procedure documents; queried by the MCP server |
| `voice/` | `voice_controller.py` | Push-to-talk capture (pynput + PyAudio → faster-whisper STT); feeds transcript to orchestrator; plays Piper TTS output to headset |
| `docs/` | — | Build plan and reference PDFs (`.training/` holds source docs, gitignored) |

**Data flow:**
```
Push-to-talk → PyAudio → faster-whisper → orchestrator
    → MCP server → collector HTTP / ChromaDB
    → Ollama (localhost:11434) → Piper TTS → headset
```

**Ollama** runs as a sidecar (`localhost:11434`); call via plain HTTP POST — no Ollama SDK needed. Target models: `mistral-nemo` or `llama3`.

**DCS side:** `Export.lua` lives in `D:/DCS World/Scripts/Export.lua` and sends UDP packets every 0.1–0.5 s.

## Key Design Constraints

- **Fully local**: no network calls to external services ever. Every LLM, STT, TTS, and embedding call must stay on `localhost`.
- **Latency budget**: end-to-end response must be ≤ 3 seconds. STT target ≤ 200 ms (`base.en` on GPU). TTS target ~50 ms (Piper). Use streaming where possible.
- **Read-only**: the system never controls the aircraft, never sends radio calls, never writes DCS state.
- **Python 3.11+** throughout.
- **Whisper model cap**: never use `large` — latency is unacceptable. Use `base.en` (MVP), `small.en` (fallback).

## Stack Versions / Model Notes

- STT: `faster-whisper`, model `base.en` (~150 MB), fallback `small.en`
- TTS: `piper-tts`, voice `en_US-ryan-medium`
- Embeddings: Ollama embeddings via `sentence-transformers`
- Vector store: ChromaDB (local), BM25 (`rank_bm25`) as semantic fallback
- MCP: `fastmcp` (decorator-based, simpler than `mcp` SDK)

## Reference Material

The `.training/` directory (gitignored) contains source PDFs used to build the RAG index. The DCS FA-18C Hornet guide is the primary V1 document source.
