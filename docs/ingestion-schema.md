# Ingestion Schema: Structured Knowledge Layer

**Status:** Design proposal — approved for implementation  
**Replaces:** Raw RAG chunking of PDFs as primary retrieval path for procedures and facts  
**Airframe:** F/A-18C Hornet (MVP); schema is airframe-agnostic

---

## 1. Motivation

The RAG pipeline (ChromaDB + BM25 over PDF chunks) is effective for open-ended document search but has three failure modes for a real-time copilot:

1. **Imprecision on step-by-step procedures** — a 300-word chunk may contain part of a checklist, but the model has to reconstruct step order from prose.
2. **Unreliable fact grounding** — critical facts like "TGP must be in TRACK mode to fire laser" are buried in paragraphs and may be split across chunk boundaries.
3. **Diagnostic coupling is impossible** — the diagnostic engine needs machine-readable conditions (`check_field`, `comparator`, `required_value`) not natural-language paragraphs.

The solution is a **structured knowledge layer**: manually curated JSON files that encode procedures, facts, and weapon rules in a queryable, deterministic format. The RAG layer is retained as a fallback for unstructured queries.

---

## 2. File Locations

```
data/fa18c/
├── procedures.json      # Step-by-step checklists
├── system_facts.json    # Discrete facts about systems and limits
└── weapon_rules.json    # Employment conditions per weapon
```

These live alongside the existing `airframe_data/fa18c/switches.json`. A future ingestion script can load all three at startup.

---

## 3. Schema Conventions

All three schemas follow the conventions established in `airframe_data/fa18c/switches.json`:

| Convention | Purpose |
|---|---|
| `"airframe": "FA-18C"` at root | Future multi-airframe filtering |
| `canonical` string | Authoritative name used in LLM output and TTS |
| `alternates` array | Voice recognition variants for STT fuzzy matching |
| Snake\_case keys as IDs | Simple `dict[id]` lookup, no query language needed |
| `source` object on every entry | Provenance — the copilot can cite page/section |
| `check_field` / `expected_value` | Live state coupling for the diagnostic engine |

---

## 4. Schema: `procedures.json`

### Purpose

Encodes ordered, step-by-step checklists. The MCP `get_next_procedure_step` tool walks this structure directly — no embedding search required.

### Top-level structure

```json
{
  "airframe": "FA-18C",
  "procedures": {
    "<procedure_id>": { ... }
  }
}
```

### Procedure entry

```json
{
  "canonical": "TGP Target Designation Sequence",
  "alternates": ["TGP designate", "designate target", "TPOD track sequence"],
  "category": "sensors",
  "source": {
    "doc": "DCS FA-18C Early Access Manual",
    "page": 301,
    "section": "8.1 AN/ASQ-228 ATFLIR"
  },
  "steps": [
    {
      "num": 1,
      "action": "Display TGP page on a DDI",
      "detail": "Press OSB next to FLIR/TPOD on the sensor select page.",
      "check_field": "tgp_powered",
      "expected_value": true
    }
  ]
}
```

### Field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `canonical` | string | yes | Formal procedure name spoken by the copilot |
| `alternates` | string[] | yes | Voice match variants |
| `category` | string | yes | One of: `startup`, `weapons`, `sensors`, `navigation`, `emergency` |
| `source.doc` | string | yes | Source document filename or title |
| `source.page` | int | yes | Page number in source document |
| `source.section` | string | yes | Section heading in source document |
| `steps[].num` | int | yes | 1-based step number |
| `steps[].action` | string | yes | Short imperative — what the pilot does |
| `steps[].detail` | string | yes | Expanded explanation for voice output |
| `steps[].check_field` | string\|null | no | Live state field to verify after this step |
| `steps[].expected_value` | any\|null | no | Value (or array of values) that satisfies the step |

### Procedure IDs (initial set)

| ID | Canonical |
|---|---|
| `master_arm_sequence` | Master Arm Sequence |
| `tgp_designate_sequence` | TGP Target Designation Sequence |
| `lgb_drop_sequence` | LGB Employment Sequence |
| `aim120_bvr_sequence` | AIM-120 BVR Employment Sequence |

---

## 5. Schema: `system_facts.json`

### Purpose

Discrete, citable facts about airframe systems, modes, and limits. Used by the `search_procedures` tool when the query is a "what/why" question rather than a procedure lookup.

### Top-level structure

```json
{
  "airframe": "FA-18C",
  "facts": {
    "<fact_id>": { ... }
  }
}
```

### Fact entry

```json
{
  "canonical": "TGP must be in TRACK mode to fire laser",
  "alternates": ["TGP track for laser", "laser requires track"],
  "category": "sensors",
  "claim": "The AN/ASQ-228 ATFLIR laser will only fire when the pod is actively tracking a designation — POINT TRACK or AREA TRACK. In SEARCH mode the laser trigger is inhibited regardless of Laser Arm state.",
  "source": {
    "doc": "DCS FA-18C Early Access Manual",
    "page": 305,
    "section": "8.1.4 Laser Operation"
  },
  "related_switch": "Laser Arm Switch",
  "check_field": "tgp_tracking",
  "value": true
}
```

### Field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `canonical` | string | yes | The fact stated as a complete sentence |
| `alternates` | string[] | yes | Voice match variants |
| `category` | string | yes | Same categories as procedures |
| `claim` | string | yes | Full factual statement — this is what the copilot reads aloud |
| `source.doc/page/section` | — | yes | Provenance for citation |
| `related_switch` | string\|null | no | `canonical` of a switch from `switches.json` if applicable |
| `check_field` | string\|null | no | Live state field this fact is about |
| `value` | any\|null | no | The state value that makes this fact relevant |

### Fact IDs (initial set)

| ID | Claim summary |
|---|---|
| `tgp_track_required_for_laser` | TGP must be TRACK to fire laser |
| `laser_arm_separate_from_master_arm` | Laser Arm and Master Arm are independent |
| `radar_stby_does_not_emit` | Radar in STBY does not radiate |
| `aim120_pitbull_range` | AIM-120 active seeker activates ~10-15 nm out |
| `master_arm_required_for_all_weapons` | Master Arm required for all weapons including gun |
| `gbu12_min_release_altitude` | GBU-12 minimum release altitude 1500 ft AGL |

---

## 6. Schema: `weapon_rules.json`

### Purpose

Machine-readable employment conditions per weapon — directly consumed by `diagnose_action_blockers`. Each condition maps 1:1 to a `DiagnosticRule` in the diagnostic engine (see `docs/build-plan.md` §11).

### Top-level structure

```json
{
  "airframe": "FA-18C",
  "weapons": {
    "<weapon_id>": { ... }
  }
}
```

### Weapon entry

```json
{
  "canonical": "AIM-120C AMRAAM",
  "alternates": ["AIM-120", "AMRAAM", "slammer", "fox three"],
  "category": "a2a",
  "source": {
    "doc": "DCS FA-18C Early Access Manual",
    "page": 275,
    "section": "7.2 AIM-120C AMRAAM Employment"
  },
  "conditions": [
    {
      "condition_id": "master_arm_armed",
      "description": "Master Arm switch must be ARM",
      "check_field": "master_arm",
      "required_value": "ARM",
      "comparator": "eq",
      "fix_instruction": "Set Master Arm switch to ARM on the instrument panel.",
      "severity": "blocking"
    }
  ]
}
```

### Condition field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `condition_id` | string | yes | Unique ID — matches `DiagnosticRule.condition_id` |
| `description` | string | yes | Human-readable condition description |
| `check_field` | string | yes | Key in `current_state.json` to evaluate |
| `required_value` | any | yes | Scalar or array expected by `comparator` |
| `comparator` | string | yes | One of: `eq`, `ne`, `gte`, `lte`, `in`, `not_in` |
| `fix_instruction` | string | yes | What the copilot tells the pilot to do |
| `severity` | string | yes | `blocking` (weapon cannot fire) or `warning` (suboptimal) |

### Weapon IDs (initial set)

| ID | Canonical |
|---|---|
| `aim120c` | AIM-120C AMRAAM |
| `gbu12` | GBU-12 Paveway II |

---

## 7. How Manual Curation Works

### Who writes the JSON

The JSON files are **pilot-authored** with AI assistance. Workflow:

1. A curator reads the source document (DCS manual, SOP, NATOPS equivalent).
2. They write or review each entry, verifying `canonical`, `claim`, and `steps` against the primary source.
3. `source.page` and `source.section` are required — no entry is merged without a citation.
4. A second curator (or the author after 24 hours) reviews before merging.

### When to add entries

| Trigger | Action |
|---|---|
| New weapon type added | Add entry to `weapon_rules.json` |
| Pilot reports wrong/missing procedure | Add or correct `procedures.json` entry |
| LLM gives wrong fact in testing | Add the correct fact to `system_facts.json` |
| New airframe added | Create `data/<airframe>/` directory, copy schema, populate |

### Entry lifecycle

Entries are **never auto-generated** from PDFs. The RAG pipeline may suggest candidate text for a curator to validate, but structured entries are always human-confirmed. This is the principal difference from the RAG layer.

---

## 8. Relationship to the RAG Layer

The structured layer does **not replace** RAG — it layers on top:

```
Query arrives
    │
    ├─► Match against procedures.json canonical/alternates  ──► Exact procedure hit
    │       (BM25 or substring, no embedding needed)              → walk steps directly
    │
    ├─► Match against system_facts.json canonical/alternates ──► Exact fact hit
    │                                                              → read claim aloud
    │
    ├─► Match against weapon_rules.json alternates           ──► Diagnostic hit
    │                                                              → run diagnose_action_blockers
    │
    └─► No structured match → fall through to ChromaDB RAG   ──► Semantic search result
                                                                   → LLM synthesizes answer
```

**The structured layer wins on precision; RAG wins on coverage.** Free-text questions about topics not yet curated fall through to RAG without degrading structured answers.

---

## 9. MCP Tool Integration

### `search_procedures`

```python
def search_procedures(query: str, airframe: str = "fa18c") -> dict:
    """
    1. Load data/{airframe}/procedures.json and system_facts.json.
    2. Score query against all canonical + alternates using BM25 or
       simple token overlap (no embeddings needed — corpus is tiny).
    3. If best score > threshold:
         - Return matched procedure steps or fact claim with source citation.
    4. Else: fall through to ChromaDB RAG query.
    """
```

**Returns** (procedure hit):
```json
{
  "type": "procedure",
  "id": "tgp_designate_sequence",
  "canonical": "TGP Target Designation Sequence",
  "source": { "doc": "...", "page": 301, "section": "..." },
  "steps": [ ... ],
  "match_type": "structured"
}
```

**Returns** (fact hit):
```json
{
  "type": "fact",
  "id": "tgp_track_required_for_laser",
  "canonical": "TGP must be in TRACK mode to fire laser",
  "claim": "The AN/ASQ-228 ATFLIR laser will only fire ...",
  "source": { "doc": "...", "page": 305, "section": "..." },
  "match_type": "structured"
}
```

### `diagnose_action_blockers`

```python
def diagnose_action_blockers(action: str, live_state: dict) -> dict:
    """
    1. Normalize action string against weapon_rules.json alternates.
    2. Load conditions for matched weapon.
    3. Evaluate each condition against live_state using DiagnosticRule logic.
    4. Return blockers (severity=blocking) and warnings (severity=warning).
    """
```

The weapon entry in `weapon_rules.json` is the source of truth for conditions — the Python `DiagnosticRule` dataclass is constructed by deserializing the JSON, not hardcoded. This means adding a new condition requires only a JSON edit, no Python change.

**Returns**:
```json
{
  "action": "aim120c",
  "can_execute": false,
  "blockers": [
    {
      "condition": "radar_operating",
      "description": "Radar must be powered and operating — not in STBY or OFF",
      "required_value": ["OFF", "STBY"],
      "current_value": "STBY",
      "fix": "Power the radar — select RWS, TWS, or STT mode on the radar switch.",
      "severity": "blocking"
    }
  ],
  "warnings": [],
  "state_age_ms": 87
}
```

---

## 10. Example Voice Interaction

> Pilot: "Why can't I shoot?"

1. Orchestrator routes to `diagnose_action_blockers` with inferred action from `selected_weapon` in live state.
2. Engine evaluates `weapon_rules.json` conditions against `current_state.json`.
3. Finds `master_arm` = SAFE, `tgp_tracking` = false.
4. Copilot responds: **"Two blockers: Master Arm is SAFE — set it to ARM. TGP is not tracking — designate a target first."**

> Pilot: "How do I designate with the TGP?"

1. `search_procedures` matches `tgp_designate_sequence` from `alternates`.
2. Returns step list from `procedures.json`.
3. Copilot reads step 1, then on follow-up reads step 2, etc. via `get_next_procedure_step`.

> Pilot: "Does the radar emit in standby?"

1. `search_procedures` matches `radar_stby_does_not_emit` from `system_facts.json`.
2. Returns `claim` field verbatim.
3. Copilot: **"No. In STBY mode the radar is powered but does not transmit. It is not detectable. Switching to RWS or TWS begins emission."**  
   *(Source: DCS FA-18C Early Access Manual, page 188, §5.3.2)*

---

## 11. Adding a New Airframe

1. Create `data/<airframe_slug>/` directory.
2. Copy `procedures.json`, `system_facts.json`, `weapon_rules.json` from `data/fa18c/` as templates.
3. Change `"airframe"` value at root.
4. Replace all entries with airframe-specific content.
5. No code changes required — loaders parameterize on `airframe`.
