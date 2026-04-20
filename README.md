# Checkride Copilot

A fully local, voice-first AI copilot for Digital Combat Simulator (DCS). Part of the **Checkride** suite.

Put an AI instructor pilot in your ear — no cloud, no subscription, no data leaving your machine. You fly; it briefs you.

**MVP aircraft:** F/A-18C Hornet

---

## Developer Setup

### 1. Clone

```powershell
git clone https://github.com/your-org/checkride-copilot.git
cd checkride-copilot

```

### 2. One-command installer (recommended, Windows)

This installer auto-detects your GPU and recommends/installs the correct PyTorch build:
- **RTX 50xx** → CUDA 12.8 nightly (`cu128`)
- **RTX 10xx–40xx** → CUDA 12.1 (`cu121`)
- **No NVIDIA GPU** → CPU torch wheel

It also installs/pulls everything else needed (uv, project deps, Ollama, model, DCS hook).

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Optional flags:

```powershell
.\scripts\setup.ps1 -SkipDcsHook
.\scripts\setup.ps1 -SkipModelPull
.\scripts\setup.ps1 -SkipOllama
```

### 3. Run the voice demo

```powershell
uv run start-demo
```

**Default push-to-talk key:** `Caps Lock`  
Hold to speak, release to transcribe.

```powershell
# Options:
uv run start-demo --ptt scroll_lock   # change PTT key
uv run start-demo --mic 2             # select audio input device index
uv run start-demo --list-mics         # list available input devices
uv run start-demo --tts-device 1      # select audio output device
uv run start-demo --no-tts            # print replies only, no audio
```

For the orchestrator entry point and tests, use:

```powershell
uv run start
uv run test
```

**First run:** Whisper (`small.en`, ~480 MB) and Kokoro TTS download automatically. This takes a minute on first launch; subsequent starts are fast.

> DCS World is optional for voice-only Q&A. For live cockpit state, install the Export.lua hook (included in `setup.ps1`) and restart DCS.

---

## Project Structure

```
checkride-copilot/
├── collector/          DCS telemetry ingestion (Export.lua → HTTP)
├── data/
│   └── airframes/fa18c/
│       ├── procedures/ Per-procedure JSON files + index.json
│       ├── cockpit/    Switch locations and panel layout
│       └── facts/      Weapon and system reference facts
├── dcs/                Export.lua snippet installed into DCS Saved Games
├── docs/               Build plan and reference material
├── models/
│   └── piper/          Piper TTS voice model (bundled)
├── mcp_server/         MCP server exposing aircraft state as tool calls
├── orchestrator/       Conversation loop and LLM routing
├── retrieval/          BM25 RAG over procedure documents
├── scripts/
│   ├── voice_demo.py   Main entry point
│   ├── install.py      DCS Export.lua installer
│   └── monitor.py      Live cockpit state monitor
└── voice/              STT (faster-whisper) and TTS (Kokoro/Piper) pipeline
```

---

## Architecture

Six components, all on loopback:

| Component | Responsibility |
|-----------|----------------|
| `collector/` | UDP server on `localhost:7778`; ingests JSON from DCS `Export.lua`; exposes `GET /state` |
| `mcp_server/` | FastMCP server; `get_current_state`, `search_procedures`, `get_next_procedure_step` tools |
| `orchestrator/` | Routes transcribed text → MCP tools → Ollama → TTS |
| `retrieval/` | BM25 index over F/A-18C procedures; queried by MCP server |
| `voice/` | PTT capture → faster-whisper STT; Kokoro/Piper TTS → headset |
| `scripts/voice_demo.py` | Integrated single-process demo (no separate MCP/collector required) |

**Data flow:**
```
PTT held → PyAudio capture → faster-whisper STT
  → orchestrator → BM25 retrieval / cockpit state
  → Ollama (qwen2.5:7b) → Kokoro TTS → headset
```

---

## Stack

| Layer | Library | Model |
|-------|---------|-------|
| STT | faster-whisper | `small.en` (~480 MB, auto-downloaded) |
| TTS | kokoro-onnx | `am_adam` (auto-downloaded), Piper fallback |
| LLM | Ollama | `qwen2.5:7b` (pull manually) |
| Retrieval | rank-bm25 | procedure JSON index |
| Vector store | ChromaDB | local, no external service |

---

## Running Tests

```powershell
pytest
```

---

## Rewriting Procedure Steps (Instructor Voice)

Procedure JSON files include a `"voiced"` field with instructor-voice rewrites for TTS delivery. To regenerate after editing procedures:

```powershell
# Preview all rewrites without saving:
python -m data.airframes.fa18c.voice_procedures --dry-run

# Rewrite a single procedure:
python -m data.airframes.fa18c.voice_procedures --proc agm_65f_missile_ir_seeker_only

# Rewrite with per-procedure approval:
python -m data.airframes.fa18c.voice_procedures --review
```

---

## Installer Builds

Tagged releases (`v*.*.*`) trigger a GitHub Actions matrix build producing two Windows ZIPs — `*-gpu.zip` (CUDA 12.1) and `*-cpu.zip` — attached to the GitHub Release. `setup.ps1` in each ZIP auto-detects your GPU and offers to swap variants if needed.

To cut a release:

```powershell
git tag v0.1.0 && git push origin v0.1.0
```

For manual packaging:

```powershell
# CPU (portable, works everywhere):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pyinstaller
pyinstaller scripts/voice_demo.py --name CheckrideCopilot --onedir --add-data "data;data" --add-data "models;models"

# GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt pyinstaller
pyinstaller scripts/voice_demo.py --name CheckrideCopilot --onedir --add-data "data;data" --add-data "models;models"
```

---

*Checkride — train like you fly.*
