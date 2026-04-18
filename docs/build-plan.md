# DCS Copilot — Full Build Plan

*Target Airframe: F/A-18C Hornet | Architecture: Fully Local | Voice-First*

---

## Table of Contents

1. [Product Definition](#1-product-definition)
2. [User Stories](#2-user-stories)
3. [System Architecture](#3-system-architecture)
4. [Recommended Tech Stack](#4-recommended-tech-stack)
5. [MCP Design](#5-mcp-design)
6. [Canonical Data Model](#6-canonical-data-model)
7. [Truth and Precedence Rules](#7-truth-and-precedence-rules)
8. [Voice Interaction Design](#8-voice-interaction-design)
9. [Retrieval and Document Strategy](#9-retrieval-and-document-strategy)
10. [Mission Script Analysis Strategy](#10-mission-script-analysis-strategy)
11. [Diagnostic Engine Design](#11-diagnostic-engine-design)
12. [API and Process Boundaries](#12-api-and-process-boundaries)
13. [Repository Structure](#13-repository-structure)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Example Prompts and System Instructions](#15-example-prompts-and-system-instructions)
16. [Risks and Mitigations](#16-risks-and-mitigations)
17. [First Coding Slice](#17-first-coding-slice)
18. [Deliverables](#18-deliverables)

---

## 1. Product Definition

### Summary

DCS Copilot is a fully local, voice-first advisory assistant for Digital Combat Simulator that runs entirely on the pilot's machine. It listens for push-to-talk queries during flight, queries live cockpit state exported from DCS, searches mission Lua scripts for objective and trigger logic, and searches curated F/A-18C procedure documents, then synthesizes a concise spoken answer using a local LLM via Ollama. The system never controls the aircraft, never makes autonomous radio calls, and never sends data off-machine. It is designed to reduce pilot cognitive load during complex in-flight situations — answering questions like "why can't I fire?" or "what's my next step?" in under three seconds.

### MVP Definition

The MVP delivers a single complete vertical: push-to-talk voice input → live DCS state query → procedure document search → LLM synthesis → spoken reply, for the F/A-18C only.

**MVP includes:**
- Push-to-talk STT (Whisper, local)
- Live DCS state via Export.lua UDP collector
- MCP server exposing: `get_current_state`, `diagnose_action_blockers`, `search_procedures`, `get_next_procedure_step`
- ChromaDB-backed procedure document index (F/A-18C SOP, checklists, weapons employment)
- Ollama integration with a single model (Mistral 7B or Llama 3 8B)
- Local TTS output (Piper TTS)
- Simple push-to-talk controller script
- Rule-based diagnostic engine for weapon employment and radar/TGP blockers

**MVP does NOT include:**
- Mission script analysis (V2)
- Wake word activation (V2)
- Any GUI beyond a minimal console window
- Multi-airframe support
- Network/cloud integration of any kind
- Autonomous actions or radio calls
- Persistent session memory or conversation history beyond current query

### Out of Scope for V1

- GUI cockpit overlay or HUD integration
- Any second airframe (F-16, A-10, etc.)
- DCS-BIOS integration (Export.lua is sufficient for MVP)
- Autonomous radio comms (SRS, VAICOM replacement)
- Mission editor integration
- Networked/multiplayer operation
- Mobile or tablet interface
- Automatic checklist progression
- Training/quiz mode
- Any cloud STT, TTS, or LLM service (optional addon only, explicitly not built)

---

## 2. User Stories

### MVP Stories (V1)

**US-01: Weapon Release Blocker**
*As a pilot in a BVR engagement, when I try to fire an AIM-120 and nothing happens, I want to ask "why can't I fire?" and receive a spoken answer identifying the specific blocker (e.g., "Master Arm is safe. Set Master Arm to ARM.") within 3 seconds.*

**US-02: TGP Troubleshooting**
*As a pilot trying to lase a ground target, when the TGP won't track, I want to ask "why won't my TGP track?" and be told the specific missing condition (e.g., "TGP is powered but LST/NFLIR is not in TDC priority. Slew TDC to TGP DDI first.") without leaving the cockpit scan.*

**US-03: Next Checklist Step**
*As a pilot running a cold start in a rush, I want to say "next step" and have the copilot read me the next checklist step I haven't yet completed, based on current cockpit state.*

**US-04: LGB Employment Query**
*As a pilot ingressing on a target with a GBU-12, I want to ask "what do I need to lase a GBU-12?" and receive a spoken 3–5 step answer covering master arm, TGP designation, and laser trigger, sourced from the procedure docs.*

**US-05: State Snapshot**
*As a pilot who needs a quick cockpit status, I want to ask "what's my current state?" and receive a concise spoken summary of fuel, master arm, selected weapon, autopilot, and radar mode.*

**US-06: Procedure Recall**
*As a pilot who forgot the AAR procedure, I want to ask "how do I do aerial refueling?" and receive a spoken step-by-step summary of the key actions for the F/A-18C AAR procedure.*

**US-07: Radar Mode Help**
*As a pilot whose radar isn't showing targets, I want to ask "why is my radar not picking up contacts?" and get a spoken diagnosis based on live radar mode state and procedure guidance.*

**US-08: Weapon Selection Help**
*As a pilot not sure which weapon to use, I want to ask "what weapon do I have selected?" and get an immediate spoken answer from live state.*

**US-09: Unexplained Cockpit Behavior**
*As a pilot confused by an unexpected caution light, I want to ask "what does the FUEL LOW caution mean?" and receive a spoken explanation sourced from the procedures document.*

**US-10: Startup Guidance**
*As a pilot doing a hot start or partial startup, I want to ask "am I ready to taxi?" and receive a spoken yes/no with the first unresolved checklist item blocking readiness.*

### Later Stories (V2+)

**US-11: Mission Objective Query**
*As a pilot not sure what the mission wants me to do next, I want to ask "what does this mission want?" and have the copilot analyze the mission Lua scripts to identify active triggers and objectives.*

**US-12: Mission Script Explanation**
*As a pilot in a scripted mission with unexpected behavior, I want to ask "what is this mission script doing?" and receive an explanation of key triggers and callbacks from the actual Lua code.*

**US-13: Multi-Airframe Support**
*As a pilot flying an F-16C, I want the copilot to automatically detect my current airframe from DCS state and load the correct procedure documents.*

**US-14: Persistent Checklist Tracking**
*As a pilot who pauses mid-checklist, I want the copilot to remember which steps I've completed within the session and pick up where I left off.*

**US-15: Threat Identification**
*As a pilot getting an RWR spike, I want to ask "what's spiking me?" and receive an advisory about the threat platform if the mission scripts contain spawn or threat type data.*

---

## 3. System Architecture

### Architecture Overview

The system has six distinct layers that run as separate local processes. Each layer has one clear responsibility. They communicate via loopback TCP or shared file where noted.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PILOT INPUT LAYER                            │
│                                                                       │
│   Push-to-Talk Key (configurable)  →  PyAudio audio capture          │
│   Python voice_controller.py                                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ WAV audio
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SPEECH-TO-TEXT LAYER                            │
│                                                                       │
│   faster-whisper (local, GPU or CPU)                                  │
│   Returns: transcript string + confidence                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ transcript
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MCP CLIENT / ORCHESTRATOR                          │
│                                                                       │
│   orchestrator.py  — decides which MCP tools to call, assembles      │
│   context, calls Ollama, returns spoken answer                        │
└──────┬─────────────────────┬──────────────────────┬─────────────────┘
       │ MCP calls            │ MCP calls             │ HTTP
       ▼                      ▼                       ▼
┌────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐
│  MCP SERVER    │  │  RETRIEVAL LAYER     │  │  OLLAMA              │
│  (FastMCP/py)  │  │                      │  │                      │
│                │  │  ChromaDB            │  │  Mistral 7B or       │
│ Tools:         │  │  (procedure docs)    │  │  Llama 3 8B          │
│ get_state      │  │  + BM25 fallback      │  │                      │
│ diagnose       │  │                      │  │  POST /api/chat      │
│ search_procs   │  │  Text chunked,        │  │  (streaming)         │
│ next_step      │  │  metadata-tagged      │  │                      │
│ search_scripts │  │                      │  └──────────────────────┘
│                │  └─────────────────────┘
└───────┬────────┘
        │ UDP read
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DCS STATE COLLECTOR                             │
│                                                                       │
│   dcs_collector.py — UDP server on localhost:7778                    │
│   Receives JSON from DCS Export.lua, writes current_state.json       │
│   Also: reads mission .miz files from D:/DCS World/...               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Export.lua UDP packets
                           ▼
                 ┌──────────────────┐
                 │   DCS PROCESS    │
                 │   D:/DCS World   │
                 │   Export.lua     │
                 │   sends JSON     │
                 │   on every frame │
                 └──────────────────┘

After answer is synthesized:

┌─────────────────────────────────────────────────────────────────────┐
│                      TEXT-TO-SPEECH LAYER                            │
│                                                                       │
│   Piper TTS (local, ~50ms)                                           │
│   Plays WAV via sounddevice to headset                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**DCS State Collector** (`dcs_collector.py`): Listens on `localhost:7778` UDP for JSON packets sent by `Export.lua`. Normalizes the raw export data into the canonical state schema. Writes `current_state.json` to a shared temp path. Also exposes a minimal HTTP endpoint (`GET /state`) for the MCP server to poll. Separate process, always running when DCS is running.

**Export.lua** (DCS side): Custom script placed in `D:/DCS World/Scripts/Export.lua`. Runs inside DCS's Lua environment. Calls `LoGetSelfData()`, `LoGetNavigationInfo()`, `LoGetMechInfo()`, and other DCS export APIs every 0.1–0.5 seconds. Serializes to JSON and sends via UDP socket to localhost:7778.

**MCP Server** (`mcp_server.py`): A Python FastMCP server. Exposes tools as JSON-Schema-defined functions. Called by the orchestrator. Reads current state from the collector's HTTP endpoint, runs the diagnostic engine, proxies retrieval calls to ChromaDB. Stateless between calls.

**Retrieval Layer** (ChromaDB + BM25): ChromaDB instance running locally. Contains chunked, metadata-tagged procedure documents. The MCP server queries it via the `chromadb` Python client. BM25 via `rank_bm25` provides a fallback when semantic search returns low-confidence results.

**Orchestrator** (`orchestrator.py`): The brain. Receives transcript, decides which MCP tools to call (initially rule-based routing, later LLM-driven), assembles a structured prompt with retrieved context, sends to Ollama, streams response, hands text to TTS. Runs in the same process as the voice controller.

**Ollama**: Running as a local HTTP server (`localhost:11434`). Serves the chosen model. Called via simple HTTP POST. No special library needed — `httpx` or `requests` is sufficient.

**Voice Controller** (`voice_controller.py`): Monitors a configurable keyboard key (e.g., Right Ctrl) via `pynput`. On keydown, starts audio capture via `PyAudio`. On keyup, stops capture, sends WAV to Whisper for transcription, hands transcript to orchestrator.

**TTS Layer**: `piper_tts` Python wrapper or CLI. Takes the final text response, synthesizes speech, plays via `sounddevice` on the selected audio output device (pilot's headset).

---

## 4. Recommended Tech Stack

### Language: Python 3.11+

**Justification**: The entire ecosystem — faster-whisper, ChromaDB, Piper TTS bindings, pynput, sounddevice — is Python-native. Node would require bridging to Python for almost everything. Solo developer context makes a polyglot stack a maintenance liability. Python is the obvious and correct choice.

### MCP Server: FastMCP (Python)

`fastmcp` is a lightweight Python library for building MCP servers. It uses decorators to define tools and handles JSON-Schema generation automatically. It's simpler than the official `mcp` SDK for a solo developer and is compatible with any MCP-capable client. Serves over stdio or SSE.

```
pip install fastmcp
```

### Local STT: faster-whisper

`faster-whisper` is a reimplementation of OpenAI Whisper using CTranslate2. It runs significantly faster than vanilla Whisper — on an RTX 3080, `base.en` transcribes 5 seconds of audio in under 200ms. Use `base.en` for MVP (English only, fastest). Use `small.en` if accuracy is insufficient. Never use `large` — latency is unacceptable in-flight.

```
pip install faster-whisper
Model: base.en (~150MB, ~180ms transcription on GPU)
Fallback: small.en (~500MB, ~400ms)
```

### Local TTS: Piper

Piper is a fast, local neural TTS engine. It produces natural-sounding speech at ~50ms on CPU. Voices are small (~50MB). Use `en_US-lessac-medium` or `en_US-ryan-medium` — both are clear and neutral. Piper runs as a subprocess or via Python bindings.

```
pip install piper-tts
Voice: en_US-ryan-medium (clear, neutral, masculine)
Latency: ~50ms for typical 10-word response
```

**Rejected alternatives**: Coqui TTS (slower, heavier), Azure TTS (cloud), gTTS (requires internet).

### Document Indexing / Retrieval: ChromaDB + rank_bm25

ChromaDB is a local vector database with a simple Python API. It requires no separate server process — it runs embedded. Use `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings (runs fully locally, ~25ms per query on CPU). BM25 via `rank_bm25` handles keyword fallback for exact terminology (e.g., "HARM", "JHMCS") that semantic search handles poorly.

```
pip install chromadb sentence-transformers rank-bm25
Embedding model: all-MiniLM-L6-v2 (~80MB)
```

**Rejected alternatives**: Weaviate (needs Docker), Pinecone (cloud), LanceDB (less mature Python API).

### DCS Communication: Export.lua → UDP JSON

DCS's built-in `Export.lua` hook runs inside the game and can send data via LuaSocket (bundled with DCS). This is the most reliable and lowest-latency method for getting cockpit state out of DCS. No DCS-BIOS is needed for MVP (BIOS adds complexity without benefit for a voice advisory system). The collector listens on UDP localhost:7778.

**Export frequency**: Every 0.25 seconds (4Hz) for the MVP. This is sufficient — the LLM call will take longer than the state update interval.

### Local LLM: Ollama with Mistral 7B Instruct or Llama 3.1 8B Instruct

**Primary recommendation**: `mistral:7b-instruct` (GGUF Q4_K_M). Mistral 7B Instruct handles short, grounded Q&A well, follows instructions reliably, and runs at ~20 tok/s on an RTX 3080 at Q4 quantization. A 15-token spoken answer takes under 1 second to generate.

**Alternative**: `llama3.1:8b-instruct-q4_k_m`. Slightly better instruction following, similar latency. Use this if Mistral's outputs feel too verbose.

**Do not use**: 13B or larger models for MVP — latency is too high for in-flight use. 70B is entirely out of scope.

```
ollama pull mistral:7b-instruct
ollama pull llama3.1:8b
```

Set `num_predict: 80` as a hard cap. Voice answers must not exceed ~20 words.

### IPC: HTTP (localhost only)

All inter-process communication uses simple HTTP on loopback. The DCS collector exposes `GET /state` on port 7779. The MCP server communicates with the orchestrator over stdio (standard MCP transport) or a local SSE endpoint. Ollama already speaks HTTP. No message queue, no gRPC, no sockets except for the initial DCS UDP export.

**Rejected alternatives**: ZeroMQ (overkill), Redis (unnecessary dependency), named pipes (platform-specific complexity).

### Audio I/O: PyAudio + sounddevice

`PyAudio` for capture (well-tested, cross-platform). `sounddevice` for playback (simpler API, non-blocking). Configure both to target the headset device by name.

---

## 5. MCP Design

The MCP server exposes tools that the orchestrator calls to gather context before querying the LLM. Each tool is deterministic and returns structured JSON. The LLM never calls tools directly in MVP — the orchestrator does routing.

### Tool: `get_current_state`

**Purpose**: Returns a snapshot of the current DCS cockpit state for the active airframe.

**Input schema**:
```json
{
  "type": "object",
  "properties": {
    "fields": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Optional list of field names to return. If omitted, returns all fields.",
      "default": []
    }
  },
  "required": []
}
```

**Output schema**:
```json
{
  "type": "object",
  "properties": {
    "airframe": { "type": "string" },
    "timestamp_ms": { "type": "integer" },
    "master_arm": { "type": "string", "enum": ["SAFE", "ARM", "SIMULATE"] },
    "selected_weapon": { "type": "string" },
    "weapon_stations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "station": { "type": "integer" },
          "store": { "type": "string" },
          "ready": { "type": "boolean" }
        }
      }
    },
    "radar_mode": { "type": "string" },
    "radar_elevation": { "type": "number" },
    "tgp_powered": { "type": "boolean" },
    "tgp_mode": { "type": "string" },
    "tgp_tdc_priority": { "type": "boolean" },
    "laser_armed": { "type": "boolean" },
    "altitude_ft": { "type": "number" },
    "airspeed_kts": { "type": "number" },
    "aoa_deg": { "type": "number" },
    "fuel_lbs": { "type": "number" },
    "autopilot_engaged": { "type": "boolean" },
    "gear_position": { "type": "string", "enum": ["UP", "DOWN", "IN_TRANSIT"] },
    "flaps_position": { "type": "string" },
    "engine_state": {
      "type": "object",
      "properties": {
        "left_rpm": { "type": "number" },
        "right_rpm": { "type": "number" }
      }
    },
    "hmd_enabled": { "type": "boolean" },
    "jhmcs_mode": { "type": "string" },
    "data_age_ms": { "type": "integer", "description": "Milliseconds since last DCS export packet" }
  }
}
```

**Example call**:
```json
{
  "name": "get_current_state",
  "arguments": { "fields": ["master_arm", "selected_weapon", "weapon_stations"] }
}
```

**Example response**:
```json
{
  "master_arm": "SAFE",
  "selected_weapon": "AIM-120C",
  "weapon_stations": [
    { "station": 1, "store": "AIM-120C", "ready": true },
    { "station": 9, "store": "AIM-120C", "ready": true }
  ],
  "data_age_ms": 120
}
```

---

### Tool: `diagnose_action_blockers`

**Purpose**: Given an intended action, returns the list of blocking conditions and their current state, sourced from the rule engine and live state.

**Input schema**:
```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "description": "The action the pilot is trying to perform",
      "examples": ["fire_aim120", "release_gbu12", "tgp_track", "lase_target", "start_engine"]
    }
  },
  "required": ["action"]
}
```

**Output schema**:
```json
{
  "type": "object",
  "properties": {
    "action": { "type": "string" },
    "can_execute": { "type": "boolean" },
    "blockers": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "condition": { "type": "string" },
          "required_value": { "type": "string" },
          "current_value": { "type": "string" },
          "fix": { "type": "string" },
          "severity": { "type": "string", "enum": ["blocking", "warning"] }
        }
      }
    },
    "warnings": { "type": "array", "items": { "type": "string" } },
    "state_age_ms": { "type": "integer" }
  }
}
```

**Example call**:
```json
{
  "name": "diagnose_action_blockers",
  "arguments": { "action": "fire_aim120" }
}
```

**Example response**:
```json
{
  "action": "fire_aim120",
  "can_execute": false,
  "blockers": [
    {
      "condition": "master_arm",
      "required_value": "ARM",
      "current_value": "SAFE",
      "fix": "Set Master Arm switch to ARM on the left console.",
      "severity": "blocking"
    }
  ],
  "warnings": ["Radar not in STT — target not locked"],
  "state_age_ms": 95
}
```

---

### Tool: `search_procedures`

**Purpose**: Semantic + keyword search of the local curated procedure document index. Returns relevant chunks with source metadata.

**Input schema**:
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "airframe": { "type": "string", "default": "fa18c" },
    "top_k": { "type": "integer", "default": 4, "maximum": 8 },
    "category": {
      "type": "string",
      "description": "Optional filter: 'weapons', 'startup', 'navigation', 'emergency', 'sensors', 'general'",
      "default": null
    }
  },
  "required": ["query"]
}
```

**Output schema**:
```json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "chunk_id": { "type": "string" },
          "text": { "type": "string" },
          "source_doc": { "type": "string" },
          "section": { "type": "string" },
          "page": { "type": "integer" },
          "category": { "type": "string" },
          "score": { "type": "number" }
        }
      }
    },
    "query_used": { "type": "string" }
  }
}
```

**Example call**:
```json
{
  "name": "search_procedures",
  "arguments": {
    "query": "AIM-120 employment preconditions master arm radar",
    "airframe": "fa18c",
    "top_k": 3,
    "category": "weapons"
  }
}
```

**Example response**:
```json
{
  "results": [
    {
      "chunk_id": "fa18c_weapons_aim120_003",
      "text": "Prior to AIM-120C employment: Master Arm must be ARM. Radar must be in TWS or STT with target designated. Ensure selected weapon is AIM-120C on SMS page. Fire with trigger — two-stage pickle button.",
      "source_doc": "FA18C_Weapons_Employment_SOP_v2.pdf",
      "section": "4.2 AIM-120 AMRAAM Employment",
      "page": 47,
      "category": "weapons",
      "score": 0.91
    }
  ],
  "query_used": "AIM-120 employment preconditions master arm radar"
}
```

---

### Tool: `get_next_procedure_step`

**Purpose**: Given a named procedure and current cockpit state, returns the next incomplete step.

**Input schema**:
```json
{
  "type": "object",
  "properties": {
    "procedure_name": {
      "type": "string",
      "description": "e.g. 'cold_start', 'before_takeoff', 'carrier_trap', 'aar'",
      "examples": ["cold_start", "before_takeoff", "aar", "carrier_cat_launch"]
    },
    "airframe": { "type": "string", "default": "fa18c" }
  },
  "required": ["procedure_name"]
}
```

**Output schema**:
```json
{
  "type": "object",
  "properties": {
    "procedure_name": { "type": "string" },
    "current_step_index": { "type": "integer" },
    "total_steps": { "type": "integer" },
    "next_step": {
      "type": "object",
      "properties": {
        "index": { "type": "integer" },
        "description": { "type": "string" },
        "action": { "type": "string" },
        "check_field": { "type": "string" },
        "required_value": { "type": "string" },
        "current_value": { "type": "string" },
        "complete": { "type": "boolean" }
      }
    },
    "completed_steps": { "type": "integer" },
    "blocking_reason": { "type": "string" }
  }
}
```

**Example call**:
```json
{
  "name": "get_next_procedure_step",
  "arguments": {
    "procedure_name": "cold_start",
    "airframe": "fa18c"
  }
}
```

**Example response**:
```json
{
  "procedure_name": "cold_start",
  "current_step_index": 4,
  "total_steps": 22,
  "next_step": {
    "index": 4,
    "description": "Battery switch",
    "action": "Set battery switch to ON on the left console",
    "check_field": "battery_switch",
    "required_value": "ON",
    "current_value": "OFF",
    "complete": false
  },
  "completed_steps": 4,
  "blocking_reason": null
}
```

---

### Tool: `search_mission_scripts` (V2, defined now for interface completeness)

**Purpose**: Searches the current mission's Lua scripts for triggers, callbacks, objectives, and capability flags relevant to the query.

**Input schema**:
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "search_type": {
      "type": "string",
      "enum": ["text", "triggers", "callbacks", "objectives", "zones", "spawns"],
      "default": "text"
    },
    "mission_path": { "type": "string", "description": "Optional override. Defaults to last-detected .miz." }
  },
  "required": ["query"]
}
```

**Output schema**:
```json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file": { "type": "string" },
          "line": { "type": "integer" },
          "match_text": { "type": "string" },
          "context": { "type": "string" },
          "type": { "type": "string" }
        }
      }
    },
    "mission_file": { "type": "string" },
    "script_count": { "type": "integer" }
  }
}
```

---

### Additional Recommended Tools

**`get_weapon_inventory`**: Returns all loaded stores by station with type, quantity, and ready state. Useful for "what weapons do I have?" queries without parsing the full state blob.

**`get_procedure_list`**: Returns all available named procedures for the current airframe. Lets the LLM know what procedure queries are valid.

**`explain_caution_light`**: Takes a caution light name (e.g., "FUEL LOW", "L GEN") and returns a text explanation sourced from procedures. Avoids full RAG when the query is a known-key lookup.

**`get_mission_context`** (V2): Returns high-level mission metadata — mission name, theater, weather, time of day — from the .miz file without requiring full script analysis.

---

## 6. Canonical Data Model

The system uses a normalized internal schema that all components read from and write to. This schema is the shared language between the DCS collector, the MCP server, the diagnostic engine, and the LLM prompts.

### Core Entities

```python
# models.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Aircraft:
    airframe_id: str          # "fa18c"
    display_name: str         # "F/A-18C Hornet"
    active: bool              # Is this the currently flown airframe?
    supported_procedures: list[str] = field(default_factory=list)
    supported_weapons: list[str] = field(default_factory=list)

@dataclass
class SubSystem:
    name: str                 # "radar", "tgp", "sms", "mfd_left"
    powered: bool
    mode: str | None          # Current mode string
    fault: bool
    fault_code: str | None

@dataclass
class Control:
    name: str                 # "master_arm_switch"
    position: str             # "ARM", "SAFE", "SIMULATE"
    expected_type: str        # "enum", "bool", "float", "int"

@dataclass
class Weapon:
    station: int
    store_type: str           # "AIM-120C", "GBU-12", "AIM-9X"
    category: str             # "AA", "AG", "GUN"
    quantity: int
    ready: bool
    guidance: str | None      # "radar", "ir", "laser", "gps", None

@dataclass
class Mode:
    subsystem: str            # "radar"
    current_mode: str         # "RWS", "TWS", "STT"
    sub_mode: str | None

@dataclass
class Action:
    action_id: str            # "fire_aim120"
    display_name: str         # "Fire AIM-120C"
    required_conditions: list[str]  # references to BlockingCondition.condition_id

@dataclass
class BlockingCondition:
    condition_id: str         # "master_arm_armed"
    description: str
    check_field: str          # field in LiveState to inspect
    required_value: Any
    comparator: str           # "eq", "ne", "gte", "lte", "in"
    fix_instruction: str      # "Set Master Arm to ARM on left console."
    severity: str             # "blocking", "warning"

@dataclass
class ProcedureStep:
    procedure_id: str         # "cold_start"
    step_index: int
    description: str
    action: str
    check_field: str | None   # maps to LiveState field
    required_value: Any | None
    notes: str | None

@dataclass
class ScriptCapability:       # V2
    source_file: str
    capability_type: str      # "trigger", "callback", "zone", "spawn"
    name: str
    condition: str | None
    effect: str | None
    raw_lua: str

@dataclass
class Evidence:
    source_type: str          # "live_state", "procedure_doc", "mission_script"
    source_id: str            # collector, doc chunk_id, or script file
    field: str
    value: Any
    confidence: float         # 0.0 - 1.0
    timestamp_ms: int
```

### LiveState Schema

The `LiveState` is a flat dict of named fields written by the collector and consumed by the diagnostic engine and MCP tools:

```python
# The authoritative field list for the normalized LiveState dict
LIVE_STATE_FIELDS = {
    # Aircraft identity
    "airframe": str,            # "FA-18C_hornet"
    "airframe_normalized": str, # "fa18c"

    # Power and master switches
    "battery_switch": str,      # "ON" / "OFF"
    "master_arm": str,          # "SAFE" / "ARM" / "SIMULATE"
    "external_power": str,

    # Engines
    "engine_left_rpm": float,
    "engine_right_rpm": float,
    "engine_left_state": str,   # "off", "starting", "running"
    "engine_right_state": str,

    # Navigation
    "altitude_ft": float,
    "airspeed_kts": float,
    "mach": float,
    "aoa_deg": float,
    "heading_deg": float,
    "lat": float,
    "lon": float,

    # Fuel
    "fuel_internal_lbs": float,
    "fuel_total_lbs": float,

    # Gear / Flaps
    "gear_position": str,       # "UP" / "DOWN" / "IN_TRANSIT"
    "flaps_position": str,      # "UP" / "HALF" / "FULL"
    "hook_position": str,       # "UP" / "DOWN"

    # Autopilot
    "autopilot_engaged": bool,
    "autopilot_mode": str,

    # Radar
    "radar_powered": bool,
    "radar_mode": str,          # "RWS", "TWS", "STT", "STBY", "OFF"
    "radar_elevation_deg": float,
    "radar_az_scan": float,
    "radar_locked_target": bool,
    "radar_tdc_priority": bool,

    # TGP / Sensors
    "tgp_powered": bool,
    "tgp_mode": str,            # "STBY", "SLAVE", "A/G", "TV", "FLIR"
    "tgp_tdc_priority": bool,
    "tgp_tracking": bool,
    "tgp_lasing": bool,

    # JHMCS
    "hmd_enabled": bool,
    "jhmcs_mode": str,

    # Weapons / SMS
    "selected_weapon": str,
    "weapon_stations": list,    # list of Weapon dicts
    "gun_rounds": int,
    "gun_selected": bool,

    # Laser
    "laser_armed": bool,
    "laser_code": int,

    # MFD
    "mfd_left_page": str,
    "mfd_right_page": str,
    "mfd_center_page": str,

    # Data freshness
    "timestamp_ms": int,
    "data_age_ms": int,
}
```

### How Sources Map to Schema

| Source | Maps to | Confidence |
|--------|---------|------------|
| DCS Export.lua packet | `LiveState` fields directly | 1.0 |
| Procedure doc chunk | `ProcedureStep` and `BlockingCondition.fix_instruction` | 0.8 (curated but not live) |
| Mission Lua script text match | `ScriptCapability` | 0.6 (inferred from code) |
| LLM model memory | Never stored in schema | 0.0 (not trusted) |

Evidence objects are assembled per query and attached to the LLM prompt. The LLM is instructed to cite its evidence source in responses.

---

## 7. Truth and Precedence Rules

### Precedence Hierarchy

```
1. Live state (DCS Export.lua) — ground truth for current cockpit condition
2. Curated procedure documents — ground truth for correct procedures and preconditions
3. Mission Lua script analysis — ground truth for mission-specific capabilities/restrictions
4. LLM model memory — never trusted; used only for phrasing and synthesis
```

### Conflict Resolution Rules

**Rule 1: Live state beats everything for current-state questions.**
If the pilot asks "what is my master arm setting?", the answer comes only from `LiveState.master_arm`. The LLM must not override or second-guess this, even if it "knows" the answer.

**Rule 2: When live state and procedure disagree on a precondition, live state wins and the procedure explains the fix.**
Example: Live state says `master_arm = SAFE`. Procedure says "Master Arm must be ARM to fire." Output: "Master Arm is SAFE [from live state]. Set to ARM [per procedures]."

**Rule 3: When state data is stale (>2000ms since last export packet), the system must say so.**
All responses where `data_age_ms > 2000` must include a spoken caveat: "Note: cockpit data may be out of date." The LLM is instructed to always emit this caveat when `data_age_ms` is flagged.

**Rule 4: When procedures conflict with each other (two doc chunks disagree), surface both and prefer the chunk with higher specificity.**
Example: A generic SOP chunk says "Master Arm ARM before weapons employment" and a specific AIM-120 chunk adds "radar in STT or TWS also required." Both are returned; the orchestrator passes both to the LLM; the LLM uses the more specific one and is prompted to prefer specificity.

**Rule 5: When mission scripts suggest a capability that live state or procedures contradict, treat script as advisory.**
Example: Mission script defines a no-weapons zone flag, but live state shows full weapon readiness. Output: "Your systems show weapon ready, but the mission script defines a restricted zone — check mission rules before firing."

**Rule 6: Model memory is never cited as a source.**
The system prompt explicitly forbids the LLM from saying "I know that..." or "Typically..." without grounding. If no evidence was retrieved, the LLM must say "I don't have a document source for this — check your kneeboard."

### Confidence and Uncertainty Surfacing

The orchestrator computes a `context_confidence` score for each query:

```python
def compute_context_confidence(
    live_state_age_ms: int,
    procedure_results: list,
    script_results: list
) -> float:
    score = 1.0
    if live_state_age_ms > 2000:
        score -= 0.4
    if not procedure_results:
        score -= 0.3
    elif max(r["score"] for r in procedure_results) < 0.5:
        score -= 0.2
    return max(0.0, score)
```

| Score | Behavior |
|-------|----------|
| ≥ 0.8 | Confident answer, no caveat |
| 0.5–0.8 | Answer with source attribution |
| 0.3–0.5 | Answer with "low confidence" spoken prefix |
| < 0.3 | "I'm not sure — verify manually." |

---

## 8. Voice Interaction Design

### Push-to-Talk vs Wake Word

**Decision: Push-to-Talk (PTT).** Wake word (e.g., "Hey Copilot") introduces false activations from radio chatter, game audio, ATC voices, and weapons effects. In a DCS session, the background audio environment is unpredictable and loud. PTT eliminates false activations entirely, is already familiar to DCS pilots from comms setup, and adds zero latency overhead. Bind PTT to a spare HOTAS button or keyboard key (default: Right Ctrl).

Wake word may be explored in V3 only if a hardware push-to-talk solution is impractical.

### Latency Targets

| Stage | Target | Notes |
|-------|--------|-------|
| PTT keydown → audio capture start | <10ms | pynput polling |
| PTT keyup → Whisper transcript | <300ms | faster-whisper base.en |
| Transcript → MCP tool calls complete | <500ms | DCS state cached; ChromaDB query |
| MCP context assembled → Ollama first token | <800ms | Mistral 7B Q4 on GPU |
| Ollama last token → TTS playback start | <100ms | Piper TTS |
| **Total PTT release → first word spoken** | **<2000ms** | **Target** |
| **Acceptable maximum** | **3000ms** | Beyond this, pilot trust degrades |

If the system is running on CPU-only, targets shift: Whisper +500ms, Ollama +1500ms. For CPU-only setups, use `tiny.en` for STT and hard-cap LLM response at 50 tokens.

### Response Style Rules

1. **Maximum 25 words** for spoken output. The LLM is instructed via system prompt to enforce this.
2. **Lead with the answer**, not the explanation. "Master Arm is SAFE — set it to ARM." not "There are several preconditions for AIM-120 employment..."
3. **No filler phrases.** No "Great question", "Of course", "Certainly". Voice is flat and functional.
4. **Cite source in one word.** "Procedures say: ..." or "Live state shows: ..." before the answer.
5. **Numbers in standard aviation format.** "Two-seven-zero degrees" not "270°". Altitude in "feet" not "ft".
6. **If uncertain, say "Unclear — verify manually."** Never fabricate.
7. **Critical blockers first.** When multiple blockers exist, speak the most critical first and say "plus N more" if needed.

### Example Voice Interactions

#### "Why can't I fire?"

**Pilot says**: "Why can't I fire?"

**System actions**:
1. Transcribe: "Why can't I fire?"
2. Route to: `diagnose_action_blockers(action="fire")` + `get_current_state()`
3. Diagnostic engine identifies: `master_arm = SAFE` (blocking), `radar_mode = RWS` (warning for AIM-120)
4. `search_procedures(query="AIM-120 fire master arm radar", top_k=2)` confirms both conditions
5. LLM assembles response from structured blockers + procedure chunk

**Spoken response**: "Master Arm is Safe — set to ARM. Radar also in RWS, not STT — lock a target. That's your fire."

---

#### "Next step"

**Pilot says**: "Next step."

**System actions**:
1. Transcribe: "Next step"
2. Detect active procedure context. If no procedure was recently mentioned, ask: "Next step of which procedure? Say the procedure name."
3. If procedure is known (e.g., cold start was mentioned last): `get_next_procedure_step(procedure_name="cold_start")`
4. Returns: step 4, "Battery switch — set to ON"
5. No LLM needed for single-step response — template fill.

**Spoken response**: "Step four. Battery switch — set to ON."

---

#### "Explain"

**Pilot says**: "Explain."

**System actions**:
1. Transcribe: "Explain"
2. Follow-on to the previous query. Orchestrator keeps a one-query memory buffer.
3. Previous query was "why can't I fire?" — previous blockers are in context.
4. `search_procedures(query="AIM-120 master arm radar STT employment detail")` with `top_k=5`
5. LLM synthesizes a slightly expanded explanation (still capped at 50 words for "explain" mode).

**Spoken response**: "The AIM-120 needs Master Arm on ARM and radar in STT with a locked target. In RWS, the missile has no guidance uplink. Lock a target first, then pickle."

---

#### "What does this mission want?" (V2)

**Pilot says**: "What does this mission want?"

**System actions**:
1. Transcribe: "What does this mission want?"
2. `search_mission_scripts(query="objective trigger complete win condition", search_type="objectives")`
3. Returns matched trigger blocks from mission Lua
4. `get_current_state(fields=["lat", "lon", "altitude_ft"])` to contextualize
5. LLM synthesizes from script findings

**Spoken response**: "Mission objective: destroy the bridge at grid one-two four-four. Trigger fires on bridge destruction. No timer found."

---

## 9. Retrieval and Document Strategy

### Document Ingestion

**Source documents for F/A-18C MVP**:
- DCS F/A-18C Early Access Manual (PDF, public)
- Community-written SOP documents (PDF/text)
- Checklists (cold start, before takeoff, before landing, emergency)
- Weapons employment guides (AIM-120, AIM-9X, GBU-12, HARM, Maverick)
- Sensor employment guides (TGP, JHMCS, FLIR)

Place all documents in `data/docs/fa18c/`. The ingestion script processes them once and populates ChromaDB.

**Ingestion pipeline**:

```python
# retrieval/ingest/ingest_docs.py

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF extraction

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Word-based overlap chunking."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_pdf(path: Path, airframe: str, category: str, collection):
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_{page_num}_{i}"
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{
                    "source_doc": path.name,
                    "page": page_num + 1,
                    "airframe": airframe,
                    "category": category,
                    "chunk_index": i,
                }]
            )

def run_ingestion(docs_root: Path, airframe: str, chroma_path: Path):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name="procedures",
        metadata={"hnsw:space": "cosine"}
    )
    category_dirs = ["weapons", "sensors", "startup", "navigation", "emergency", "general"]
    for cat in category_dirs:
        cat_dir = docs_root / airframe / cat
        if not cat_dir.exists():
            continue
        for pdf_path in cat_dir.glob("*.pdf"):
            print(f"Ingesting {pdf_path.name} ({cat})...")
            ingest_pdf(pdf_path, airframe, cat, collection)
    print(f"Ingestion complete. Total docs: {collection.count()}")
```

**Chunk size rationale**: 300 words. Smaller than 300 loses procedure context. Larger than 400 dilutes relevance. 50-word overlap ensures procedure steps that span chunk boundaries are still retrievable.

### Metadata Tagging Strategy

Every chunk gets these metadata fields:
- `airframe`: `"fa18c"` — critical for future multi-airframe filtering
- `category`: one of `["startup", "weapons", "sensors", "navigation", "emergency", "general"]`
- `source_doc`: filename
- `page`: page number
- `section`: extracted from heading text if available (regex-matched on PDF structure)

**Category assignment**: The ingestion script takes `category` as an argument per directory. Organize docs by category in subdirectories: `data/docs/fa18c/weapons/`, `data/docs/fa18c/startup/`, etc. Do not rely on LLMs to auto-categorize during ingestion — it adds latency and errors.

### Retrieval Strategy

**Primary**: Dense retrieval via `all-MiniLM-L6-v2` embeddings in ChromaDB. Query with `collection.query(query_texts=[query], n_results=top_k, where={"airframe": "fa18c"})`.

**Secondary (fallback)**: BM25 keyword search over the same document corpus. Use `rank_bm25`. Activated when the top ChromaDB result has `score < 0.5`. BM25 is particularly better for exact callouts like "JHMCS", "MIDS", specific weapon designations.

**Result fusion**: When both methods return results, merge and deduplicate by `chunk_id`, keeping the highest score from either method.

**Hallucination prevention**:
1. System prompt explicitly says: "Use only the provided document excerpts. Do not add information not in the excerpts."
2. Retrieved chunks are passed verbatim in the prompt — no summarization by the orchestrator before the LLM sees them.
3. If `search_procedures` returns zero results, the orchestrator sends a no-retrieval prompt variant that instructs the LLM to say "I don't have a procedure document for this — check your kneeboard."

### Multi-Airframe and SOP Variants (Later)

When a second airframe is added, the `airframe` metadata field filters results automatically. The ingestion pipeline requires only that new documents be added to the correct subdirectory and ingested with the correct `airframe` tag. No schema changes required.

SOP variants (e.g., squadron SOP vs. the DCS manual) can coexist in the same collection. Add a `sop_variant` metadata field and allow filtering by it. The orchestrator selects the preferred variant via config.

---

## 10. Mission Script Analysis Strategy

### What We're Analyzing

DCS missions ship as `.miz` files — ZIP archives containing:
- `mission` — Lua table defining the scenario (weather, units, waypoints)
- `options` — mission options
- `dictionary` — string ID mappings
- `triggers/` — Lua trigger scripts
- `l10n/DEFAULT/` — localized script files

The scripts directory contains the actual game logic: trigger conditions, callbacks, objective checks, spawn events, and capability flags.

### V1: Simple Text Search (Stubbed in MVP, Usable Immediately)

```python
# mission/reader.py

import zipfile
import re
from pathlib import Path

def extract_mission_scripts(miz_path: Path) -> dict[str, str]:
    """Extract all Lua files from a .miz archive."""
    scripts = {}
    with zipfile.ZipFile(miz_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.lua'):
                scripts[name] = zf.read(name).decode('utf-8', errors='replace')
    return scripts

def text_search_scripts(scripts: dict[str, str], query: str) -> list[dict]:
    """Naive text search across all script files."""
    results = []
    query_lower = query.lower()
    for filename, content in scripts.items():
        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            if query_lower in line.lower():
                # Get 3 lines of context
                start = max(0, line_num - 2)
                end = min(len(lines), line_num + 2)
                context = "\n".join(lines[start:end])
                results.append({
                    "file": filename,
                    "line": line_num,
                    "match_text": line.strip(),
                    "context": context,
                    "type": "text_match"
                })
    return results[:20]  # Cap results

def find_latest_miz(dcs_root: Path) -> Path | None:
    """Find the most recently modified .miz in DCS Missions folder."""
    missions_dir = dcs_root / "Missions"
    if not missions_dir.exists():
        # Also check saved games
        return None
    miz_files = list(missions_dir.glob("**/*.miz"))
    if not miz_files:
        return None
    return max(miz_files, key=lambda p: p.stat().st_mtime)
```

### V2: Structured Script-Aware Analysis

V2 adds a proper Lua-aware parser. Key V2 targets:

```python
# mission/parser.py — V2

import re

TRIGGER_PATTERNS = {
    "condition": re.compile(
        r'trigger\.condition\.(.*?)\s*=', re.DOTALL
    ),
    "action": re.compile(
        r'trigger\.action\.(.*?)\s*\(', re.DOTALL
    ),
    "flag_set": re.compile(
        r'trigger\.action\.setUserFlag\s*\(\s*["\']?(\w+)["\']?\s*,\s*(true|false|\d+)',
        re.IGNORECASE
    ),
    "zone_ref": re.compile(
        r'trigger\.misc\.inZone\s*\(\s*["\']([^"\']+)["\']'
    ),
    "spawn": re.compile(
        r'coalition\.addGroup\s*\(|mist\.dynAdd\s*\(|trigger\.action\.activateGroup\s*\('
    ),
    "win_condition": re.compile(
        r'trigger\.action\.mission(?:End|Accomplished|Failed)\s*\('
    ),
    "callback": re.compile(
        r'world\.addEventHandler\s*\(|mist\.flagFunc\s*\(|timer\.scheduleFunction\s*\('
    ),
}

def extract_triggers(lua_content: str) -> list[dict]:
    """Extract structured trigger information from Lua content."""
    results = []
    for pattern_name, pattern in TRIGGER_PATTERNS.items():
        for match in pattern.finditer(lua_content):
            line_num = lua_content[:match.start()].count('\n') + 1
            results.append({
                "type": pattern_name,
                "match": match.group(0),
                "line": line_num,
                "groups": list(match.groups()),
            })
    return results
```

**V2 detection targets summary**:

| Target | Method |
|--------|--------|
| Triggers | Parse `trigger.current` and `trigger.condition` tables |
| Callbacks | Regex for `addEventHandler`, `flagFunc`, `schedule` patterns |
| Objective logic | Detect `missionEnd`, `missionAccomplished`, `missionFailed` calls |
| Spawn logic | Detect `addGroup`, `dynAdd`, `activateGroup` calls |
| Zone references | Extract `inZone` and `outZone` strings |
| Capability flags | Detect boolean assignments in global scope |

---

## 11. Diagnostic Engine Design

The diagnostic engine is the most important non-LLM component. It answers "why can't I do X?" using explicit rules, not LLM inference. Rules are deterministic, auditable, and easily updated.

### Core Engine

```python
# diagnostic/engine.py

from dataclasses import dataclass
from typing import Any

@dataclass
class DiagnosticRule:
    condition_id: str
    description: str
    check_field: str
    required_value: Any
    comparator: str          # "eq", "ne", "gte", "lte", "in", "not_in"
    fix_instruction: str
    severity: str            # "blocking" or "warning"


def check_condition(rule: DiagnosticRule, live_state: dict) -> bool:
    """Returns True if condition is SATISFIED (i.e., not blocking)."""
    val = live_state.get(rule.check_field)
    if val is None:
        return False  # Unknown field = blocking (conservative)
    if rule.comparator == "eq":
        return val == rule.required_value
    elif rule.comparator == "ne":
        return val != rule.required_value
    elif rule.comparator == "gte":
        return float(val) >= float(rule.required_value)
    elif rule.comparator == "lte":
        return float(val) <= float(rule.required_value)
    elif rule.comparator == "in":
        return val in rule.required_value
    elif rule.comparator == "not_in":
        return val not in rule.required_value
    return False


def diagnose_action(action_id: str, live_state: dict) -> dict:
    from diagnostic.action_map import normalize_action
    from diagnostic.rules import get_rules_for_action

    normalized = normalize_action(action_id, live_state)
    rules = get_rules_for_action(normalized)

    if not rules:
        return {
            "action": action_id,
            "can_execute": None,
            "blockers": [],
            "warnings": [f"No diagnostic rules defined for action '{action_id}'."],
            "state_age_ms": live_state.get("data_age_ms", -1),
        }

    blockers = []
    warnings = []
    for rule in rules:
        satisfied = check_condition(rule, live_state)
        if not satisfied:
            entry = {
                "condition": rule.condition_id,
                "description": rule.description,
                "required_value": str(rule.required_value),
                "current_value": str(live_state.get(rule.check_field, "unknown")),
                "fix": rule.fix_instruction,
                "severity": rule.severity,
            }
            if rule.severity == "blocking":
                blockers.append(entry)
            else:
                warnings.append(entry)

    return {
        "action": normalized,
        "can_execute": len(blockers) == 0,
        "blockers": blockers,
        "warnings": [w["fix"] for w in warnings],
        "state_age_ms": live_state.get("data_age_ms", -1),
    }
```

### Rules: Weapons (F/A-18C)

```python
# diagnostic/rules/fa18c_weapons.py

from diagnostic.engine import DiagnosticRule

FIRE_AIM120_RULES = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm switch must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm switch to ARM on the left console.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="aim120_selected",
        description="AIM-120C must be the selected weapon on SMS",
        check_field="selected_weapon",
        required_value="AIM-120C",
        comparator="eq",
        fix_instruction="Select AIM-120C on the SMS DDI page.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="radar_on",
        description="Radar must be powered and not in STBY or OFF",
        check_field="radar_mode",
        required_value=["OFF", "STBY"],
        comparator="not_in",
        fix_instruction="Power the radar — select RWS, TWS, or STT mode.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="radar_tracking_mode",
        description="Radar should be in STT or TWS for active AMRAAM guidance",
        check_field="radar_mode",
        required_value=["STT", "TWS"],
        comparator="in",
        fix_instruction="Radar is in RWS — lock a target in STT or use TWS for pitbull AMRAAM.",
        severity="warning"
    ),
]

RELEASE_GBU12_RULES = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm to ARM.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="gbu12_selected",
        description="GBU-12 must be selected",
        check_field="selected_weapon",
        required_value="GBU-12",
        comparator="eq",
        fix_instruction="Select GBU-12 on the SMS DDI page.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="tgp_powered_lgb",
        description="TGP must be powered for laser guidance",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP on the right DDI.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="tgp_tracking_lgb",
        description="TGP must have a tracked designation",
        check_field="tgp_tracking",
        required_value=True,
        comparator="eq",
        fix_instruction="Designate a target in the TGP DDI before releasing.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="laser_armed_lgb",
        description="Laser should be armed before release point",
        check_field="laser_armed",
        required_value=True,
        comparator="eq",
        fix_instruction="Arm the laser on the TGP DDI — fire laser before impact.",
        severity="warning"
    ),
    DiagnosticRule(
        condition_id="safe_release_altitude",
        description="Altitude should be above minimum safe release altitude",
        check_field="altitude_ft",
        required_value=1500,
        comparator="gte",
        fix_instruction="You may be below safe release altitude for GBU-12 fuze arming.",
        severity="warning"
    ),
]

TGP_TRACK_RULES = [
    DiagnosticRule(
        condition_id="tgp_powered",
        description="TGP must be powered",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP on the right DDI.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="tgp_tdc_priority",
        description="TGP DDI must have TDC priority",
        check_field="tgp_tdc_priority",
        required_value=True,
        comparator="eq",
        fix_instruction="Slew TDC to the TGP DDI to give it priority.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="tgp_not_stby",
        description="TGP must not be in STBY",
        check_field="tgp_mode",
        required_value="STBY",
        comparator="ne",
        fix_instruction="TGP is in STBY — switch to A/G mode on the TGP DDI.",
        severity="blocking"
    ),
]

LASE_TARGET_RULES = [
    DiagnosticRule(
        condition_id="tgp_powered_lase",
        description="TGP must be powered",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="laser_armed",
        description="Laser must be armed",
        check_field="laser_armed",
        required_value=True,
        comparator="eq",
        fix_instruction="Arm the laser on the TGP DDI.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="tgp_tracking_lase",
        description="TGP must be tracking a target to lase it",
        check_field="tgp_tracking",
        required_value=True,
        comparator="eq",
        fix_instruction="Designate a target in the TGP before lasing.",
        severity="blocking"
    ),
]
```

### Action Normalization

```python
# diagnostic/action_map.py

from difflib import get_close_matches

# Maps normalized aliases → canonical action_id
# List values indicate ambiguity requiring state disambiguation
ACTION_ALIASES: dict[str, str | list[str]] = {
    "fire":              ["fire_aim120", "fire_aim9x", "fire_gun"],  # ambiguous
    "fire aim-120":      "fire_aim120",
    "fire amraam":       "fire_aim120",
    "shoot missile":     "fire_aim120",
    "fire missile":      ["fire_aim120", "fire_aim9x"],
    "drop bomb":         "release_gbu12",
    "release bomb":      "release_gbu12",
    "track target":      "tgp_track",
    "tgp track":         "tgp_track",
    "lase":              "lase_target",
    "laser":             "lase_target",
    "start engine":      "start_engine",
    "engine start":      "start_engine",
    "fire_aim120":       "fire_aim120",
    "tgp_track":         "tgp_track",
    "release_gbu12":     "release_gbu12",
    "lase_target":       "lase_target",
    "start_engine":      "start_engine",
}

def normalize_action(raw: str, live_state: dict | None = None) -> str:
    raw_lower = raw.lower().strip()

    result = ACTION_ALIASES.get(raw_lower)
    if result is None:
        # Fuzzy match
        matches = get_close_matches(raw_lower, ACTION_ALIASES.keys(), n=1, cutoff=0.6)
        result = ACTION_ALIASES.get(matches[0]) if matches else raw_lower

    # Resolve ambiguity using live state
    if isinstance(result, list) and live_state:
        selected = live_state.get("selected_weapon", "").upper()
        if "AIM-120" in selected:
            return "fire_aim120"
        elif "AIM-9" in selected:
            return "fire_aim9x"
        elif "GBU" in selected or "BOMB" in selected:
            return "release_gbu12"
        return result[0]  # Default to first

    return result if isinstance(result, str) else (result[0] if result else raw_lower)
```

### Worked Examples

**AIM-120 release** with `master_arm=SAFE`, `radar_mode=RWS`:
- Rule `master_arm_armed` fails (SAFE ≠ ARM) → blocking
- Rule `radar_tracking_mode` fails (RWS not in [STT, TWS]) → warning
- Output: 1 blocker, 1 warning

**TGP tracking** with `tgp_powered=True`, `tgp_tdc_priority=False`, `tgp_mode=A/G`:
- Rule `tgp_powered` passes
- Rule `tgp_tdc_priority` fails → blocking
- Output: 1 blocker

**GBU-12 release** with `master_arm=ARM`, `selected_weapon=GBU-12`, `tgp_powered=True`, `tgp_tracking=False`, `laser_armed=False`, `altitude_ft=8000`:
- master_arm passes, gbu12_selected passes, tgp_powered passes
- `tgp_tracking_lgb` fails → blocking
- `laser_armed_lgb` fails → warning
- `safe_release_altitude` passes
- Output: 1 blocker, 1 warning

---

## 12. API and Process Boundaries

### Process Map

| Process | Language | Entry Point | Transport In | Transport Out |
|---------|----------|-------------|--------------|---------------|
| DCS (game) | — | DCS.exe | — | UDP :7778 |
| Export.lua | Lua (in-process) | Export.lua | DCS API | UDP send |
| dcs_collector | Python | `collector/main.py` | UDP :7778 | HTTP :7779 |
| mcp_server | Python | `mcp/server.py` | stdio (MCP) | stdio (MCP) |
| orchestrator + voice | Python | `app/main.py` | stdio (MCP child) | — |
| ollama | Go binary | `ollama serve` | HTTP :11434 | HTTP :11434 |
| piper_tts | subprocess | `piper` binary | stdin text | WAV stdout |

### Interface Definitions

**DCS → Collector (UDP :7778)**:
```json
{
  "event": "export",
  "ts": 1712345678.123,
  "data": {
    "SelfData": {},
    "MechInfo": {},
    "NavInfo": {},
    "PayloadInfo": {}
  }
}
```

**Collector HTTP API (localhost:7779)**:
```
GET /state    → LiveState dict (JSON)
GET /health   → {"ok": true, "data_age_ms": 180}
```

**Orchestrator → MCP Server (stdio)**:
```json
// Request:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "diagnose_action_blockers",
    "arguments": { "action": "fire_aim120" }
  }
}

// Response:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{ "type": "text", "text": "{...json...}" }]
  }
}
```

**Orchestrator → Ollama (HTTP)**:
```python
import httpx

response = httpx.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "mistral:7b-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": assembled_prompt}
        ],
        "stream": False,
        "options": {
            "num_predict": 80,
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    timeout=8.0
)
answer = response.json()["message"]["content"]
```

**Orchestrator → Piper TTS (subprocess)**:
```python
import subprocess, sounddevice, soundfile, io

def speak(text: str, voice_model: str = "en_US-ryan-medium"):
    result = subprocess.run(
        ["piper", "--model", voice_model, "--output_raw"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=3.0
    )
    audio_data, samplerate = soundfile.read(
        io.BytesIO(result.stdout),
        dtype="float32"
    )
    sounddevice.play(audio_data, samplerate)
    sounddevice.wait()
```

---

## 13. Repository Structure

```
dcs-copilot/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.yaml                  # User config: keybind, model, audio device, paths
├── .env.example
│
├── dcs/
│   └── Export.lua               # Place in D:/DCS World/Scripts/
│
├── collector/
│   ├── __init__.py
│   ├── main.py                  # Launches UDP server + HTTP API
│   ├── udp_server.py            # Receives DCS export packets
│   ├── normalizer.py            # Raw DCS API → LiveState dict
│   ├── http_api.py              # GET /state, GET /health (Flask or FastAPI)
│   └── state_store.py           # Thread-safe in-memory state with age tracking
│
├── mcp/
│   ├── __init__.py
│   ├── server.py                # FastMCP server entry point
│   └── tools/
│       ├── state_tool.py        # get_current_state, get_weapon_inventory
│       ├── diagnostic_tool.py   # diagnose_action_blockers
│       ├── procedure_tool.py    # search_procedures, get_next_procedure_step, get_procedure_list
│       ├── script_tool.py       # search_mission_scripts (stub → V2 impl)
│       └── caution_tool.py      # explain_caution_light
│
├── diagnostic/
│   ├── __init__.py
│   ├── engine.py                # diagnose_action(), check_condition(), DiagnosticRule
│   ├── action_map.py            # normalize_action(), ACTION_ALIASES
│   ├── procedure_tracker.py     # get_next_incomplete_step() from YAML definitions
│   └── rules/
│       ├── __init__.py          # get_rules_for_action() dispatcher
│       ├── fa18c_weapons.py     # fire_aim120, release_gbu12, fire_aim9x
│       ├── fa18c_sensors.py     # tgp_track, lase_target, radar rules
│       └── fa18c_startup.py     # cold_start, start_engine rules
│
├── retrieval/
│   ├── __init__.py
│   ├── chroma_store.py          # ChromaDB client wrapper, search()
│   ├── bm25_store.py            # BM25 index build and search
│   ├── fusion.py                # Merge + deduplicate results from both methods
│   └── ingest/
│       ├── ingest_docs.py       # Full ingestion pipeline
│       └── chunk.py             # Text chunking utilities
│
├── data/
│   ├── docs/
│   │   └── fa18c/
│   │       ├── weapons/         # AIM-120, GBU-12, HARM docs (PDF)
│   │       ├── sensors/         # TGP, radar, JHMCS docs (PDF)
│   │       ├── startup/         # Cold start, shutdown checklists (PDF)
│   │       ├── navigation/
│   │       └── emergency/
│   ├── procedures/
│   │   └── fa18c/
│   │       ├── cold_start.yaml
│   │       ├── before_takeoff.yaml
│   │       ├── aar.yaml
│   │       └── carrier_cat_launch.yaml
│   └── chroma_db/               # ChromaDB persistent storage (gitignored)
│
├── voice/
│   ├── __init__.py
│   ├── controller.py            # PTT key listener, audio capture loop
│   ├── stt.py                   # faster-whisper wrapper
│   └── tts.py                   # Piper TTS wrapper + sounddevice playback
│
├── orchestrator/
│   ├── __init__.py
│   ├── main.py                  # Main app loop, wires all components
│   ├── router.py                # Maps transcript → tool call sequence
│   ├── context_builder.py       # Assembles structured LLM prompt from tool results
│   ├── ollama_client.py         # HTTP client for Ollama
│   └── prompts.py               # System prompt + query templates
│
├── mission/
│   ├── __init__.py
│   ├── reader.py                # .miz ZIP extraction
│   ├── searcher.py              # Text search across extracted scripts
│   └── parser.py                # V2: structured Lua trigger analysis
│
├── tests/
│   ├── test_diagnostic.py
│   ├── test_normalizer.py
│   ├── test_retrieval.py
│   ├── test_router.py
│   └── fixtures/
│       ├── sample_state_safe.json     # master_arm SAFE, radar RWS
│       ├── sample_state_ready.json    # master_arm ARM, radar STT
│       └── sample_export_packet.json  # Raw DCS UDP packet
│
└── scripts/
    ├── start_collector.py       # python scripts/start_collector.py
    ├── start_mcp.py             # python scripts/start_mcp.py
    ├── start_all.py             # Launches collector + orchestrator together
    ├── ingest_all.py            # Runs full document ingestion
    ├── test_voice.py            # Test STT + TTS round-trip without DCS
    └── slice1_test.py           # First coding slice end-to-end test
```

### Structured Procedure YAML

```yaml
# data/procedures/fa18c/cold_start.yaml
procedure_id: cold_start
display_name: "F/A-18C Cold Start"
airframe: fa18c
steps:
  - index: 0
    description: "Seat and harness"
    action: "Adjust seat, stow mirrors, connect harness"
    check_field: null
    required_value: null
    notes: "Manual check — not verifiable from export data"

  - index: 1
    description: "Battery switch"
    action: "Set battery switch to ON"
    check_field: "battery_switch"
    required_value: "ON"
    notes: null

  - index: 2
    description: "APU start"
    action: "Press APU START on left console. Wait for stabilization."
    check_field: "apu_running"
    required_value: true
    notes: "Typical APU spin-up: 30 seconds"

  - index: 3
    description: "Left engine start"
    action: "Set left throttle to IDLE when JFS engages"
    check_field: "engine_left_state"
    required_value: "running"
    notes: null
```

---

## 14. Implementation Roadmap

### Phase 0: DCS State Export (Week 1–2)

**Objective**: Prove live DCS state flows into Python.

**Deliverables**:
- `dcs/Export.lua` sending JSON to localhost:7778 at 4Hz
- `collector/udp_server.py` receiving and storing latest packet
- `collector/normalizer.py` covering the 12 fields needed for diagnostic MVP
- `collector/http_api.py` with `GET /state` and `GET /health`
- `sample_export_packet.json` captured from a live DCS F/A-18C session

**Technical risks**: DCS F/A-18C export API coverage is incomplete for some cockpit indicators. `LoGetCockpitParams()` uses numeric indices that require research for the Hornet. Budget 2–3 days to identify correct indices for master_arm, radar_mode, tgp_powered, tgp_tdc_priority. Use DCS's own `LoGetPayloadInfo()` for weapon stores.

**Effort**: 10–15 hours

---

### Phase 1: Diagnostic Engine (Week 2–3)

**Objective**: "Why can't I fire?" answered from live state with no LLM.

**Deliverables**:
- `diagnostic/engine.py` with `DiagnosticRule`, `diagnose_action()`, `check_condition()`
- Rules for: `fire_aim120`, `tgp_track`, `release_gbu12`, `lase_target`, `start_engine`
- `diagnostic/action_map.py` with initial aliases
- Test suite: `tests/test_diagnostic.py` with fixture states
- CLI test: `python -m diagnostic.engine --action fire_aim120 --state tests/fixtures/sample_state_safe.json`

**Technical risks**: Some DCS export fields may not resolve cleanly (e.g., TDC priority per-DDI). Design graceful degradation: missing fields return `"unknown"` and the rule conservatively returns a caveat, not a false clear.

**Effort**: 12–16 hours

---

### Phase 2: Document Retrieval (Week 3–4)

**Objective**: Answer procedure questions from local documents.

**Deliverables**:
- `retrieval/ingest/ingest_docs.py` pipeline operational
- ChromaDB collection populated with F/A-18C docs (minimum: weapons, TGP, cold start, AAR)
- `retrieval/chroma_store.py` with `search(query, airframe, top_k, category)` method
- BM25 fallback in `retrieval/bm25_store.py`
- Fusion in `retrieval/fusion.py`
- Manual test: verify top-3 chunks for queries like "AIM-120 preconditions", "cold start battery"

**Technical risks**: Some DCS PDFs use image-based pages. Use PyMuPDF first; fall back to `pytesseract` OCR for image-only pages. Test extraction quality before assuming chunks are usable.

**Effort**: 10–14 hours

---

### Phase 3: MCP Server (Week 4–5)

**Objective**: All tools accessible over MCP protocol.

**Deliverables**:
- `mcp/server.py` with FastMCP and all 6 tools registered
- `search_mission_scripts` stubbed (returns `{"results": [], "note": "V2 feature"}`)
- Manual test via `mcp dev mcp/server.py` in MCP inspector

**Technical risks**: FastMCP version compatibility. Pin version. Test both stdio and SSE transport.

**Effort**: 8–12 hours

---

### Phase 4: Orchestrator + Ollama (Week 5–6)

**Objective**: Text query → spoken answer with no voice layer yet.

**Deliverables**:
- `orchestrator/router.py` with intent detection
- `orchestrator/context_builder.py` assembling structured prompt
- `orchestrator/ollama_client.py` HTTP client with timeout and retry
- `orchestrator/prompts.py` — system prompt and 3 templates tested
- CLI test: `python orchestrator/main.py --query "why can't I fire?" --use-sample-state`
- Per-stage latency logging

**Technical risks**: LLM may ignore brevity instruction and return verbose answers. Test 20+ example queries. Tune temperature to 0.1. If still verbose, add a post-processing step that truncates to the first sentence ending with a period.

**Effort**: 16–20 hours

---

### Phase 5: Voice Layer (Week 6–7)

**Objective**: PTT voice input → spoken output.

**Deliverables**:
- `voice/controller.py` — PTT key monitoring with configurable key
- `voice/stt.py` — faster-whisper wrapper with aviation initial prompt
- `voice/tts.py` — Piper TTS + sounddevice playback
- Audio device configuration in `config.yaml`
- End-to-end voice test without DCS (using sample state)
- Latency measurements per stage

**Technical risks**: PyAudio installation on Windows requires specific DLL. Document exact install steps. Test Whisper with a recording of cockpit jargon through the headset before declaring done.

**Effort**: 12–16 hours

---

### Phase 6: Integration + Polish (Week 7–8)

**Objective**: Stable MVP for real DCS sessions.

**Deliverables**:
- `scripts/start_all.py` launching collector + orchestrator in one command
- Stale state handling (>2000ms caveat in spoken response)
- Confidence scoring in orchestrator
- "No source found" fallback path tested
- `scripts/ingest_all.py` for one-command re-ingestion
- README with complete setup guide including DCS Export.lua install path

**Effort**: 8–12 hours

---

### Phase 7: V2 Features (Post-MVP, unscheduled)

- `mission/reader.py` and `mission/searcher.py` for .miz text search
- V2 trigger/callback parser (`mission/parser.py`)
- One-query session memory buffer in orchestrator
- Named procedure tracking within a session
- Second airframe support scaffold

---

## 15. Example Prompts and System Instructions

### System Prompt

```
You are DCS Copilot, a concise in-flight advisory assistant for the F/A-18C Hornet.

ABSOLUTE RULES:
1. Respond in 25 words or fewer. Lead with the answer. No filler words.
2. Use ONLY the live state data and document excerpts provided below. Never add facts from your training.
3. If live state data is provided, it is the ground truth for current cockpit conditions.
4. If a document excerpt answers the question, cite it: "Procedures say: ..."
5. If live state shows a blocker, state it: "BLOCKED: [reason]. Fix: [action]."
6. If data_age_ms > 2000 (flagged as STALE), add: "Note: data may be out of date."
7. If you have no relevant evidence, say exactly: "No source found — check your kneeboard."
8. Never use: "I think", "I believe", "typically", "usually", "in most cases", "generally."
9. Format numbers in aviation style: "two-seven-zero degrees", "eight thousand feet."
10. If multiple blockers exist, speak the most critical first, then "plus N more."
```

---

### Prompt Template: Live State Only

```python
LIVE_STATE_ONLY_TEMPLATE = """\
PILOT QUERY: {query}

LIVE COCKPIT STATE (data age: {data_age_ms}ms{stale_flag}):
{live_state_summary}

DIAGNOSTIC RESULT:
{diagnostic_json}

Answer the pilot's question using only the live state and diagnostic data above.
Maximum 25 words. Lead with the answer.
"""
```

---

### Prompt Template: Live State + Procedures

```python
LIVE_STATE_PROCEDURES_TEMPLATE = """\
PILOT QUERY: {query}

LIVE COCKPIT STATE (data age: {data_age_ms}ms{stale_flag}):
{live_state_summary}

DIAGNOSTIC RESULT:
{diagnostic_json}

PROCEDURE DOCUMENT EXCERPTS (cite as "Procedures say:" if used):
{procedure_chunks}

Answer the pilot's question using the live state, diagnostic result, and procedure excerpts above.
Maximum 25 words. Lead with the answer. Never add facts not present in the above.
"""
```

---

### Prompt Template: Live State + Scripts + Procedures

```python
FULL_CONTEXT_TEMPLATE = """\
PILOT QUERY: {query}

LIVE COCKPIT STATE (data age: {data_age_ms}ms{stale_flag}):
{live_state_summary}

DIAGNOSTIC RESULT:
{diagnostic_json}

MISSION SCRIPT FINDINGS (cite as "Mission scripts show:" if used):
{script_findings}

PROCEDURE DOCUMENT EXCERPTS (cite as "Procedures say:" if used):
{procedure_chunks}

PRECEDENCE: live state > procedures > mission scripts. Never override live state with other sources.
If live state contradicts procedures, trust live state for current condition, cite procedures for the fix.
Maximum 25 words. Lead with the answer. Never fabricate.
"""
```

---

### Orchestrator Query Routing

```python
# orchestrator/router.py

import re

INTENT_PATTERNS = {
    "diagnose": [
        r"why can't i", r"why won't", r"why isn't", r"why doesn't",
        r"not working", r"won't fire", r"can't fire", r"blocked",
        r"not tracking", r"won't track", r"can't release", r"won't lock"
    ],
    "next_step": [
        r"next step", r"what's next", r"next item", r"continue checklist",
        r"what do i do next", r"proceed"
    ],
    "procedure_lookup": [
        r"how do i", r"how to", r"what do i need", r"procedure for",
        r"steps for", r"walk me through", r"explain how"
    ],
    "state_summary": [
        r"what's my", r"current state", r"status", r"what is my",
        r"am i ready", r"what weapon"
    ],
    "mission_query": [
        r"mission", r"objective", r"what does this mission",
        r"what am i supposed to", r"mission script"
    ],
}

def detect_intent(transcript: str) -> str:
    lower = transcript.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                return intent
    return "general"  # Fallback: procedure_lookup + state context
```

---

## 16. Risks and Mitigations

### Latency Overrun

**Risk**: Total response time exceeds 3 seconds, breaking pilot trust.

**Mitigations**: Log per-stage timing from day one. Pre-load Whisper model at startup (not on demand — cold load is ~2s). Cache ChromaDB embedding model. Hard-cap `num_predict: 80` in Ollama. If GPU is unavailable, switch to `tiny.en` Whisper and reduce LLM to Phi-3 mini (faster on CPU). Consider running Ollama with `--num-gpu 99` to force full GPU offload.

### STT Errors on Aviation Vocabulary

**Risk**: Whisper mishears "AIM-120" as "aim one twenty", "JHMCS" as "Jemex", "HARM" as "arm".

**Mitigations**: Use Whisper `initial_prompt` parameter: `"DCS F/A-18C Hornet cockpit. Aviation terms: AIM-120 AMRAAM, GBU-12 Paveway, TGP Targeting Pod, JHMCS helmet, HARM anti-radiation missile, Master Arm, SMS stores, DDI display, HOTAS controls."` This biases Whisper decoding heavily toward aviation vocabulary. Also: enable Whisper VAD (voice activity detection) to prevent capturing ambient game audio during PTT hold.

### Retrieval Failure and Hallucination

**Risk**: ChromaDB returns irrelevant chunks; LLM invents procedure steps.

**Mitigations**: Score threshold of 0.45 — reject chunks below this. BM25 fallback for terminology that semantic search misses. "No source found" fallback path: when both ChromaDB and BM25 return nothing, the orchestrator uses the no-retrieval template and the LLM is instructed to say it has no source. Do not use the LLM's training knowledge as a fallback.

### Stale DCS State

**Risk**: Export.lua stops sending during mission pause, DCS loading screen, or crash. Copilot answers from old state.

**Mitigations**: `state_store.py` tracks `last_updated_ms`. All tool responses include `data_age_ms`. Orchestrator flags stale state (>2000ms) and the system prompt instructs the LLM to emit the stale caveat. If `data_age_ms > 10000`, the orchestrator should not call `diagnose_action_blockers` and instead respond: "Cannot reach DCS — check if DCS is running."

### Ambiguous Pilot Queries

**Risk**: "Why can't I fire?" is ambiguous when pilot has multiple weapon types loaded.

**Mitigations**: `normalize_action()` uses `selected_weapon` from live state to disambiguate. If `selected_weapon` is null, default to the most commonly used weapon type in F/A-18C combat (AIM-120) and state the assumption in the response: "Assuming AIM-120: Master Arm is Safe — set to ARM."

### LLM Instruction Non-Compliance

**Risk**: Mistral 7B ignores brevity constraints or adds hallucinated details.

**Mitigations**: `temperature: 0.1` (near-deterministic). `num_predict: 80` hard cap. Post-processing: if response length > 50 words, truncate at the first sentence boundary after word 20. Run 20+ adversarial test cases before MVP. If Mistral proves unreliable, switch to Phi-3 Mini (3.8B) which tends to be more instruction-following on constrained tasks.

### DCS Export API Coverage Gaps

**Risk**: The fields needed for diagnostic rules aren't all available through standard DCS export APIs.

**Mitigations**: Research `LoGetCockpitParams()` numeric indices for the F/A-18C (community resources exist for this). Where a field is not exportable, mark it as `"unknown"` in LiveState and design the rule to return a caveat rather than a false clear: "Cannot verify [field] from DCS export — check manually."

### Maintenance Burden

**Risk**: DCS updates break export API. Document ingestion is manual recurring work. Diagnostic rules get out of date with DCS patches.

**Mitigations**: Diagnostic rules in YAML files (not hardcoded) so non-developer updates require no deploy. Ingestion is a single command (`python scripts/ingest_all.py`) documented in README. Keep a `CHANGELOG.md` noting which DCS version each rule was verified against.

---

## 17. First Coding Slice

### Scope

The smallest useful vertical slice: **DCS live state → diagnostic → spoken output with no LLM and no voice input**. This is a CLI script that:
1. Polls the collector HTTP endpoint (or uses a sample state if collector is offline)
2. Runs `diagnose_action("fire_aim120", state)`
3. Formats the result into a spoken sentence
4. Plays it through Piper TTS

This validates: DCS export API coverage, normalizer correctness, diagnostic rule logic, TTS audio routing — the three riskiest and most foundational unknowns.

### Acceptance Criteria

- [ ] `Export.lua` installed in `D:/DCS World/Scripts/` and verified: UDP packets arrive in Python when DCS runs with the F/A-18C in the cockpit
- [ ] `GET /state` returns a dict containing at minimum: `master_arm`, `selected_weapon`, `radar_mode`, `tgp_powered`, `tgp_tdc_priority`, `data_age_ms`
- [ ] `diagnose_action("fire_aim120", state)` returns correct blockers for 3 distinct states:
  - `master_arm=SAFE` → 1 blocking
  - `master_arm=ARM, radar_mode=RWS` → 0 blocking, 1 warning
  - `master_arm=ARM, radar_mode=STT` → 0 blocking, 0 warnings
- [ ] Piper TTS speaks the diagnostic result clearly through the headset with no audio errors
- [ ] End-to-end (state poll → audio) completes in under 500ms

### Implementation

```python
#!/usr/bin/env python3
# scripts/slice1_test.py
"""First coding slice: DCS state → diagnose → speak. No LLM required."""

import sys
import time
sys.path.insert(0, ".")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from diagnostic.engine import diagnose_action
from voice.tts import speak


SAMPLE_STATE = {
    "master_arm": "SAFE",
    "selected_weapon": "AIM-120C",
    "radar_mode": "RWS",
    "tgp_powered": False,
    "tgp_tdc_priority": False,
    "data_age_ms": 200,
    "airframe": "fa18c",
}


def get_state() -> dict:
    if HAS_HTTPX:
        try:
            r = httpx.get("http://localhost:7779/state", timeout=1.0)
            r.raise_for_status()
            print("[collector] Live state received.")
            return r.json()
        except Exception as e:
            print(f"[collector] Not reachable ({e}) — using sample state.")
    else:
        print("[collector] httpx not installed — using sample state.")
    return SAMPLE_STATE


def format_diagnostic_for_speech(result: dict) -> str:
    stale = result.get("state_age_ms", 0) > 2000
    stale_suffix = " Note: data may be out of date." if stale else ""

    if result.get("can_execute") is None:
        return f"No diagnostic rules found for this action.{stale_suffix}"

    if result["can_execute"] and not result.get("warnings"):
        return f"Clear to execute.{stale_suffix}"

    if result["can_execute"] and result.get("warnings"):
        w = result["warnings"][0]
        extra = f" Plus {len(result['warnings']) - 1} more." if len(result["warnings"]) > 1 else ""
        return f"Clear with caution: {w}{extra}{stale_suffix}"

    blockers = result["blockers"]
    primary = blockers[0]["fix"]
    extra = f" Plus {len(blockers) - 1} more blockers." if len(blockers) > 1 else ""
    return f"Blocked. {primary}{extra}{stale_suffix}"


def main():
    print("=== DCS Copilot: Slice 1 Test ===\n")

    # 1. Get state
    state = get_state()
    print(f"State: master_arm={state.get('master_arm')}, "
          f"radar={state.get('radar_mode')}, "
          f"tgp_powered={state.get('tgp_powered')}")

    # 2. Diagnose
    t0 = time.monotonic()
    result = diagnose_action("fire_aim120", state)
    diag_ms = int((time.monotonic() - t0) * 1000)

    print(f"\nDiagnostic ({diag_ms}ms):")
    print(f"  can_execute: {result['can_execute']}")
    print(f"  blockers: {len(result['blockers'])}")
    for b in result["blockers"]:
        print(f"    BLOCKING: {b['fix']}")
    for w in result.get("warnings", []):
        print(f"    WARNING: {w}")

    # 3. Speak
    text = format_diagnostic_for_speech(result)
    print(f"\nSpeaking: \"{text}\"")

    t1 = time.monotonic()
    speak(text)
    tts_ms = int((time.monotonic() - t1) * 1000)

    print(f"\nTTS playback: {tts_ms}ms")
    print(f"Total (from state): {diag_ms + tts_ms}ms")


if __name__ == "__main__":
    main()
```

---

## 18. Deliverables

### Recommended Final MVP Scope Statement

> DCS Copilot MVP is a fully local, push-to-talk voice advisory system for the F/A-18C Hornet in DCS. It ingests live cockpit state via Export.lua UDP, indexes curated F/A-18C procedure documents locally in ChromaDB, runs a rule-based diagnostic engine for weapon and sensor employment questions, and uses Ollama (Mistral 7B) to synthesize concise spoken answers via Piper TTS. Response target: under 3 seconds from PTT release to first spoken word. Fully local. No cloud. No aircraft control.

---

### Prioritized Task List

**P0 — Foundation (nothing works without these)**
1. Export.lua → UDP collector → normalized LiveState dict working with DCS running
2. `diagnostic/engine.py` with rules for `fire_aim120`, `tgp_track`, `release_gbu12`
3. Piper TTS installed and producing audio on the headset
4. faster-whisper installed and transcribing aviation vocabulary correctly

**P1 — MVP Core**
5. Collector HTTP API (`GET /state`)
6. MCP server with `get_current_state` and `diagnose_action_blockers`
7. F/A-18C procedure docs ingested into ChromaDB
8. `search_procedures` and `get_next_procedure_step` tools working
9. Orchestrator routing + context builder + Ollama integration
10. PTT voice controller loop integrated with orchestrator

**P2 — Polish Before First Real Session**
11. Stale state detection and spoken caveat
12. Confidence scoring and "no source found" fallback
13. Action alias normalization with live-state disambiguation
14. Procedure YAML files for cold_start, before_takeoff, AAR
15. Additional rules: `start_engine`, `lase_target`
16. Per-stage latency logging and measurement

**P3 — V2 (Post-MVP)**
17. `.miz` extraction and text search for mission scripts
18. One-query session memory in orchestrator
19. V2 Lua trigger/callback parser
20. Second airframe support scaffold

---

### Sample MCP Tool Definition in Code

```python
# mcp/tools/diagnostic_tool.py

import json
from fastmcp import FastMCP
from diagnostic.engine import diagnose_action
from collector.state_store import get_current_state_dict


def register_diagnostic_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def diagnose_action_blockers(action: str) -> str:
        """
        Given an intended pilot action, return the list of blocking conditions
        checked against the current live DCS cockpit state.

        action: Canonical action ID or natural language alias.
                Examples: 'fire_aim120', 'tgp_track', 'release_gbu12',
                          'lase_target', 'start_engine', 'fire amraam'

        Returns JSON string with can_execute bool, blockers list, warnings list.
        """
        live_state = get_current_state_dict()

        if live_state is None:
            result = {
                "action": action,
                "can_execute": False,
                "blockers": [{
                    "condition": "collector_offline",
                    "description": "DCS state collector is not reachable",
                    "required_value": "running",
                    "current_value": "offline",
                    "fix": "Start the collector: python scripts/start_collector.py",
                    "severity": "blocking"
                }],
                "warnings": [],
                "state_age_ms": -1
            }
        else:
            result = diagnose_action(action, live_state)

        return json.dumps(result, indent=2)
```

```python
# mcp/server.py

from fastmcp import FastMCP
from mcp.tools.state_tool import register_state_tools
from mcp.tools.diagnostic_tool import register_diagnostic_tools
from mcp.tools.procedure_tool import register_procedure_tools
from mcp.tools.caution_tool import register_caution_tools

mcp = FastMCP(
    name="dcs-copilot",
    version="0.1.0",
    description="DCS F/A-18C voice copilot MCP server — fully local"
)

register_state_tools(mcp)
register_diagnostic_tools(mcp)
register_procedure_tools(mcp)
register_caution_tools(mcp)

if __name__ == "__main__":
    mcp.run()  # Default: stdio transport
```

---

### Sample Diagnostic Rule in Code

```python
# diagnostic/rules/fa18c_weapons.py
# All rules for F/A-18C weapon employment diagnostics.
# Add new rules here. No code changes needed in engine.py.

from diagnostic.engine import DiagnosticRule

FIRE_AIM120_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm switch must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm switch to ARM on the left console.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="aim120_selected",
        description="AIM-120C must be selected on SMS page",
        check_field="selected_weapon",
        required_value="AIM-120C",
        comparator="eq",
        fix_instruction="Select AIM-120C on the SMS DDI page.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="radar_not_off",
        description="Radar must be powered and not in STBY or OFF",
        check_field="radar_mode",
        required_value=["OFF", "STBY"],
        comparator="not_in",
        fix_instruction="Power the radar — select RWS, TWS, or STT.",
        severity="blocking"
    ),
    DiagnosticRule(
        condition_id="radar_lock_mode",
        description="Radar should be in STT or TWS for active AMRAAM uplink",
        check_field="radar_mode",
        required_value=["STT", "TWS"],
        comparator="in",
        fix_instruction="Radar in RWS — lock a target in STT or use TWS for AMRAAM guidance.",
        severity="warning"
    ),
]


# diagnostic/rules/__init__.py

from diagnostic.rules.fa18c_weapons import (
    FIRE_AIM120_RULES,
    RELEASE_GBU12_RULES,
    TGP_TRACK_RULES,
    LASE_TARGET_RULES,
)
from diagnostic.rules.fa18c_startup import START_ENGINE_RULES
from diagnostic.engine import DiagnosticRule

_ACTION_RULE_MAP: dict[str, list[DiagnosticRule]] = {
    "fire_aim120":    FIRE_AIM120_RULES,
    "release_gbu12":  RELEASE_GBU12_RULES,
    "tgp_track":      TGP_TRACK_RULES,
    "lase_target":    LASE_TARGET_RULES,
    "start_engine":   START_ENGINE_RULES,
}

def get_rules_for_action(action_id: str) -> list[DiagnosticRule]:
    return _ACTION_RULE_MAP.get(action_id, [])
```

---

### Sample End-to-End Request Flow

```
Time      Event
────────────────────────────────────────────────────────────────────────
T+0ms     Pilot holds Right Ctrl (PTT key)
T+10ms    voice/controller.py: keydown detected → PyAudio capture starts
T+2800ms  Pilot releases Right Ctrl → audio capture stops
T+2810ms  WAV buffer (2.8s) passed to voice/stt.py
T+2990ms  faster-whisper base.en returns: "why can't I fire?" (conf: 0.94)
T+2995ms  orchestrator/router.py: detect_intent → "diagnose"

T+3000ms  Orchestrator calls MCP: diagnose_action_blockers(action="fire")
           → action_map: "fire" + selected_weapon="AIM-120C" → "fire_aim120"
           → collector: GET http://localhost:7779/state (5ms)
           → LiveState: master_arm="SAFE", radar_mode="RWS" (age: 180ms)
           → engine: 4 rules checked
               master_arm_armed: SAFE ≠ ARM → BLOCKING
               aim120_selected: AIM-120C = AIM-120C → PASS
               radar_not_off: RWS not in [OFF, STBY] → PASS
               radar_lock_mode: RWS not in [STT, TWS] → WARNING
           → result: {can_execute: false, blockers: [master_arm], warnings: [radar]}
T+3040ms  MCP returns diagnostic JSON

T+3045ms  Orchestrator calls MCP: search_procedures(
             query="AIM-120 fire master arm radar preconditions",
             airframe="fa18c", top_k=2
           )
           → ChromaDB query + embedding (25ms)
           → score: 0.91 → above threshold → BM25 not needed
           → returns 2 chunks from FA18C_Weapons_Employment_SOP_v2.pdf
T+3210ms  MCP returns procedure chunks

T+3215ms  context_builder.py assembles prompt:
           - System prompt (brevity rules)
           - PILOT QUERY: "why can't I fire?"
           - LIVE STATE: master_arm=SAFE, selected_weapon=AIM-120C, radar=RWS
           - DIAGNOSTIC: 1 blocker (master_arm), 1 warning (radar)
           - PROCEDURE EXCERPTS: [SOP chunk text]

T+3220ms  ollama_client.py: POST http://localhost:11434/api/chat
           model=mistral:7b-instruct, stream=false, num_predict=80, temp=0.1

T+3650ms  Ollama returns:
           "Master Arm is Safe — set to ARM. Radar in RWS, not STT — lock a target."
           (14 words — within 25-word limit ✓)

T+3655ms  Orchestrator validates response: 14 words OK, no forbidden phrases OK

T+3660ms  voice/tts.py: subprocess call to piper binary with response text
T+3710ms  Piper synthesizes WAV (~50ms)
T+3715ms  sounddevice.play() → audio starts in headset

─────────────────────────────────────────────────────────────────────────
PTT release (T+2800ms) → first word spoken (T+3715ms) = 915ms ✓
Total wall time from PTT press = 3715ms (includes 2.8s of the pilot speaking)
─────────────────────────────────────────────────────────────────────────
```

---

### Export.lua Reference Implementation

```lua
-- dcs/Export.lua
-- Place this file at: D:/DCS World/Scripts/Export.lua
-- DCS loads this automatically on mission start.

local socket = require("socket")
local JSON = loadfile("Scripts/JSON.lua")()  -- DCS ships JSON.lua

local host = "127.0.0.1"
local port = 7778
local udp = socket.udp()
udp:settimeout(0)

local export_interval = 0.25  -- seconds between exports
local last_export = 0

function LuaExportStart()
    -- Called once when mission starts
end

function LuaExportStop()
    udp:close()
end

function LuaExportAfterNextFrame()
    local now = LoGetModelTime()
    if now - last_export < export_interval then
        return
    end
    last_export = now

    local ok, self_data = pcall(LoGetSelfData)
    local ok2, mech_info = pcall(LoGetMechInfo)
    local ok3, nav_info = pcall(LoGetNavigationInfo)
    local ok4, payload = pcall(LoGetPayloadInfo)

    local packet = {
        event = "export",
        ts = now,
        data = {
            SelfData = ok and self_data or {},
            MechInfo = ok2 and mech_info or {},
            NavInfo = ok3 and nav_info or {},
            PayloadInfo = ok4 and payload or {},
        }
    }

    local ok_json, json_str = pcall(JSON.encode, JSON, packet)
    if ok_json then
        udp:sendto(json_str, host, port)
    end
end
```

---

*End of DCS Copilot Build Plan*

*Document version 1.0 | April 2026 | F/A-18C Hornet | DCS World at D:/DCS World*
