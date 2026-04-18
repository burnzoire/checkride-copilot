# Checkride Copilot

A fully local, voice-first copilot for Digital Combat Simulator (DCS). Part of the **Checkride** suite.

## Overview

Checkride Copilot puts an AI weapons systems officer in your ear — no cloud, no latency, no data leaving your machine. You fly; it briefs you.

**MVP aircraft:** F/A-18C Hornet

## Architecture

| Component | Purpose |
|-----------|---------|
| `collector/` | DCS-BIOS / Telemachus data ingestion |
| `mcp_server/` | Model Context Protocol server exposing aircraft state as tool calls |
| `orchestrator/` | Conversation loop and tool-call routing |
| `retrieval/` | Local RAG over NATOPS / Chuck's guides (Ollama embeddings) |
| `voice/` | STT (Whisper) and TTS (Piper) pipeline |
| `docs/` | Build plan and reference material |

## Stack

- **LLM:** Ollama (local, e.g. `mistral-nemo` or `llama3`)
- **Context protocol:** MCP (Model Context Protocol)
- **STT:** Faster-Whisper
- **TTS:** Piper TTS
- **Retrieval:** ChromaDB + Ollama embeddings
- **Data source:** DCS-BIOS / Export.lua

## Status

Early scaffolding. See [`docs/build-plan.md`](docs/build-plan.md) for the full roadmap.

---

*Checkride — train like you fly.*
