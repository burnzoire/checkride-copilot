from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any
from loguru import logger

_FREQ_INDEX_PATH = Path(__file__).parent.parent / "data" / "airfield_frequencies_by_map.json"
_DCS_ROOT_CANDIDATES = [
    Path("D:/DCS World"),
    Path("/d/DCS World"),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load_frequency_index() -> dict[str, Any]:
    try:
        return json.loads(_FREQ_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"maps": {}}


def detect_theatre() -> str | None:
    # DCS writes current theatre while mission is active.
    p = Path.home() / "AppData" / "Local" / "Temp" / "DCS" / "temp.theatre"
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore").strip().lower()
        return raw or None
    except Exception:
        return None


def _saved_games_log_candidates() -> list[Path]:
    base = Path.home() / "Saved Games"
    return [
        base / "DCS" / "Logs" / "dcs.log",
        base / "DCS.openbeta" / "Logs" / "dcs.log",
    ]


def find_active_miz_path() -> Path | None:
    patterns = [
        re.compile(r"loadMission\s+(.+?\.miz)", re.I),
        re.compile(r'loading mission from:\s+"(.+?\.miz)"', re.I),
    ]
    for log_path in _saved_games_log_candidates():
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hit: str | None = None
        for pat in patterns:
            for m in pat.finditer(text):
                hit = m.group(1)
        if hit:
            p = Path(hit)
            if p.exists():
                return p
    return None


def _map_entries(theatre: str | None) -> list[dict[str, Any]]:
    idx = load_frequency_index().get("maps", {})
    key = (theatre or "").lower()
    if key and key in idx:
        return list(idx.get(key, []))

    # Out-of-mission chat may not have a theatre; search all map entries.
    out: list[dict[str, Any]] = []
    for vals in idx.values():
        out.extend(vals or [])
    return out


def find_airfield_theatre(airfield: str) -> str | None:
    target = _norm(airfield)
    if not target:
        return None
    maps = load_frequency_index().get("maps", {})
    best_theatre = None
    best_len = -1
    for theatre, entries in maps.items():
        for entry in entries or []:
            names = [str(entry.get("canonical", ""))] + [str(a) for a in (entry.get("aliases", []) or [])]
            for name in names:
                key = _norm(name)
                if not key:
                    continue
                if target == key or re.search(rf"\b{re.escape(key)}\b", target):
                    if len(key) > best_len:
                        best_theatre = str(theatre)
                        best_len = len(key)
    return best_theatre


def extract_airfield_from_text(text: str, theatre: str | None = None) -> str | None:
    match = match_airfield_in_text(text, theatre=theatre)
    if not match:
        return None
    return str(match.get("canonical") or "") or None


def match_airfield_in_text(text: str, theatre: str | None = None) -> dict[str, Any] | None:
    target = _norm(text)
    if not target:
        return None
    best_name = None
    best_len = -1
    for entry in _map_entries(theatre):
        names = [str(entry.get("canonical", ""))] + [str(a) for a in (entry.get("aliases", []) or [])]
        canonical = str(entry.get("canonical", "")).strip()
        for name in names:
            key = _norm(name)
            if len(key) < 4:
                continue
            if re.search(rf"\b{re.escape(key)}\b", target) and len(key) > best_len:
                best_name = canonical or name
                best_len = len(key)
    if best_name:
        return {"canonical": best_name, "match_type": "exact", "score": 1.0}

    candidate_map: dict[str, str] = {}
    for entry in _map_entries(theatre):
        canonical = str(entry.get("canonical", "")).strip()
        if not canonical:
            continue
        names = [canonical] + [str(a) for a in (entry.get("aliases", []) or [])]
        for name in names:
            key = _norm(name)
            if len(key) >= 4 and key not in candidate_map:
                candidate_map[key] = canonical

    tokens = target.split()
    suffixes = {"tower", "traffic", "airfield", "airport"}
    for idx, token in enumerate(tokens):
        if token not in suffixes:
            continue
        phrases: list[str] = []
        for width in (3, 2, 1):
            start = max(0, idx - width)
            phrase = " ".join(tokens[start:idx]).strip()
            phrase = re.sub(r"^(?:at|to|near|nearby|the|a|an|for|from|frequency)\s+", "", phrase).strip()
            if phrase:
                phrases.append(phrase)
        for raw in phrases:
            closest = get_close_matches(raw, list(candidate_map.keys()), n=2, cutoff=0.68)
            if not closest:
                continue
            best_key = closest[0]
            best_score = SequenceMatcher(None, raw, best_key).ratio()
            second_score = SequenceMatcher(None, raw, closest[1]).ratio() if len(closest) > 1 else 0.0
            return {
                "canonical": candidate_map[best_key],
                "match_type": "fuzzy",
                "score": best_score,
                "second_score": second_score,
                "raw": raw,
            }
    return None


def airfield_context_terms(theatre: str | None = None, limit: int = 80) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entry in _map_entries(theatre):
        names = [str(entry.get("canonical", ""))] + [str(a) for a in (entry.get("aliases", []) or [])]
        for name in names:
            term = str(name).replace("_", " ").strip()
            if len(term) < 4:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(term)
            if len(out) >= limit:
                return out
    return out


def _canonical_theatre_name(raw: str) -> str:
    v = raw.strip().lower()
    alias = {
        "persiangulf": "persian_gulf",
        "pg": "persian_gulf",
        "sinai_map": "sinai",
        "southatlantic": "south_atlantic",
        "thechannel": "channel",
    }
    return alias.get(v, v)


def _read_radio_lua(dcs_root: Path, theatre: str) -> str | None:
    p = dcs_root / "Mods" / "terrains" / theatre / "Radio.lua"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _parse_radio_entries(radio_lua_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    block_re = re.compile(
        r"\{\s*(?:--\s*(?P<label>[^\n]+)\s*)?.*?"
        r"radioId\s*=\s*'(?P<radioid>[^']+)'.*?"
        r"callsign\s*=\s*(?P<callsign>\{\{.*?\}\}|\{\})\s*;.*?"
        r"frequency\s*=\s*\{(?P<freqs>.*?)\}\s*;",
        re.S,
    )
    for m in block_re.finditer(radio_lua_text):
        label = (m.group("label") or "").strip()
        radioid = m.group("radioid")
        aid_match = re.search(r"(\d+)", radioid)
        if not aid_match:
            continue
        airdrome_id = int(aid_match.group(1))
        freqs_raw = m.group("freqs")
        callsign_raw = m.group("callsign")

        bands: dict[str, float] = {}
        for fm in re.finditer(r"\[(HF|UHF|VHF_HI|VHF_LOW)\]\s*=\s*\{[^,]+,\s*([0-9.]+)\}", freqs_raw):
            bands[fm.group(1)] = float(fm.group(2)) / 1_000_000.0

        call_names = re.findall(r'"([^"]+)"', callsign_raw)
        filtered_call_names = [n for n in call_names if n.lower() not in {"common", "nato", "ussr"}]
        human_callsign = filtered_call_names[-1] if filtered_call_names else (label or radioid)

        if not bands:
            continue

        aliases = sorted({v for v in ([label, human_callsign] + filtered_call_names) if v}, key=lambda s: s.lower())
        entries.append(
            {
                "canonical": human_callsign,
                "aliases": aliases,
                "airdrome_id": airdrome_id,
                "tower_mhz": bands.get("VHF_HI") or bands.get("UHF"),
                "uhf_mhz": bands.get("UHF"),
                "vhf_hi_mhz": bands.get("VHF_HI"),
                "vhf_low_mhz": bands.get("VHF_LOW"),
                "hf_mhz": bands.get("HF"),
            }
        )
    return entries


def scrape_dcs_atc_index(dcs_root: Path | None = None) -> dict[str, Any]:
    roots = [dcs_root] if dcs_root else _DCS_ROOT_CANDIDATES
    root = next((r for r in roots if r and r.exists()), None)
    if root is None:
        return {"version": 2, "maps": {}}

    terrains_root = root / "Mods" / "terrains"
    out: dict[str, Any] = {"version": 2, "maps": {}}
    if not terrains_root.exists():
        return out

    for terrain_dir in terrains_root.iterdir():
        if not terrain_dir.is_dir():
            continue
        radio_text = _read_radio_lua(root, terrain_dir.name)
        if not radio_text:
            continue
        key = _canonical_theatre_name(terrain_dir.name)
        out["maps"][key] = _parse_radio_entries(radio_text)
    return out


def write_scraped_index(dcs_root: Path | None = None) -> dict[str, Any]:
    data = scrape_dcs_atc_index(dcs_root=dcs_root)
    _FREQ_INDEX_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _entry_for_airfield(airfield: str, theatre: str | None) -> dict[str, Any] | None:
    target = _norm(airfield)
    if not target:
        return None
    best = None
    best_len = -1
    for e in _map_entries(theatre):
        names = [str(e.get("canonical", ""))] + [str(a) for a in (e.get("aliases", []) or [])]
        for n in names:
            k = _norm(n)
            if not k:
                continue
            if target == k or re.search(rf"\b{re.escape(k)}\b", target):
                if len(k) > best_len:
                    best, best_len = e, len(k)
    return best


def lookup_local_frequency(airfield: str, theatre: str | None = None, band: str = "tower") -> dict[str, Any] | None:
    entry = _entry_for_airfield(airfield, theatre)
    if not entry:
        return None
    # Map band to frequency key in entry
    band_to_key = {
        "tower": "tower_mhz",
        "UHF": "uhf_mhz",
        "VHF": "vhf_hi_mhz",
        "VHF_HI": "vhf_hi_mhz",
        "VHF_LOW": "vhf_low_mhz",
        "HF": "hf_mhz",
    }
    key = band_to_key.get(band, "tower_mhz")
    mhz = entry.get(key)
    if mhz is None:
        return None
    try:
        mhz = float(mhz)
    except Exception:
        return None
    return {
        "source": "local",
        "role": band if band != "tower" else "tower",
        "mhz": mhz,
        "airfield": entry.get("canonical") or airfield,
    }


def _extract_freq_airdrome_pairs(mission_text: str) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    p1 = re.compile(
        r'\["frequency"\]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,.{0,500}?\["airdromeId"\]\s*=\s*([0-9]+)',
        re.I | re.S,
    )
    p2 = re.compile(
        r'\["airdromeId"\]\s*=\s*([0-9]+)\s*,.{0,500}?\["frequency"\]\s*=\s*([0-9]+(?:\.[0-9]+)?)',
        re.I | re.S,
    )
    for m in p1.finditer(mission_text):
        try:
            freq = float(m.group(1))
            aid = int(m.group(2))
            pairs.append((aid, freq))
        except Exception:
            pass
    for m in p2.finditer(mission_text):
        try:
            aid = int(m.group(1))
            freq = float(m.group(2))
            pairs.append((aid, freq))
        except Exception:
            pass
    return pairs


def _to_mhz(raw: float) -> float | None:
    try:
        v = float(raw)
    except Exception:
        return None
    if v <= 0:
        return None
    # Mission files commonly store radio frequencies in Hz.
    if v >= 1_000_000:
        return v / 1_000_000.0
    if v >= 1_000:
        return v / 1_000.0
    return v


def _read_mission_text_from_miz(miz_path: Path | None = None) -> str | None:
    miz = miz_path or find_active_miz_path()
    if not miz or not miz.exists():
        return None
    try:
        with zipfile.ZipFile(miz, "r") as zf:
            return zf.read("mission").decode("utf-8", errors="ignore")
    except Exception:
        return None


def _classify_contact_kind(name: str, platform_type: str | None = None) -> str | None:
    n = _norm(name)
    pt = _norm(platform_type or "")
    # Surface combatants like TICONDEROG are not carriers.
    if re.search(r"\b(ticonderog|cg\s*\d+)\b", pt):
        return "contact"
    if re.search(r"\b(cvn\s*\d+|carrier|boat|stennis|roosevelt|lincoln|washington|truman|vinson|nimitz|ford)\b", pt):
        return "carrier"
    if re.search(r"\b(kc\s*135|kc\s*130|il\s*78|tanker)\b", pt):
        return "tanker"
    if re.search(r"\b(cvn|carrier|boat|stennis|roosevelt|lincoln|washington|truman|vinson|nimitz|ford)\b", n):
        return "carrier"
    if re.search(r"\b(texaco|arco|shell|tanker|kc 135|kc135|kc 130|kc130|il 78|il78)\b", n):
        return "tanker"
    return "contact"


# Canonical carrier names and nicknames keyed by hull number.
# Add entries as new carriers appear in DCS.
_CVN_NAMES: dict[str, list[str]] = {
    "59": ["forrestal", "uss forrestal"],
    "60": ["saratoga", "sara"],
    "61": ["ranger"],
    "62": ["independence", "indy"],
    "63": ["kitty hawk", "kitty"],
    "64": ["constellation", "connie"],
    "65": ["enterprise", "big e"],
    "66": ["america"],
    "67": ["kennedy", "jfk", "john f kennedy"],
    "68": ["nimitz"],
    "69": ["eisenhower", "ike"],
    "70": ["vinson", "carl vinson"],
    "71": ["roosevelt", "teddy", "fdr", "theodore roosevelt"],
    "72": ["lincoln", "abe", "honest abe", "abraham lincoln"],
    "73": ["washington", "gw", "george washington"],
    "74": ["stennis", "john c stennis"],
    "75": ["truman", "harry truman", "harry s truman"],
    "76": ["reagan", "gipper", "ronald reagan"],
    "77": ["bush", "ghwb", "george h w bush", "george hw bush"],
    "78": ["ford", "gerald ford"],
    "79": ["kennedy", "jfk", "john f kennedy"],
}

# Pre-compiled pattern of all CVN ship names/nicknames for fast routing decisions.
# Sorted longest-first so the alternation prefers more specific matches (e.g.
# "george washington" before "washington").
_CVN_SHIP_NAMES_RE: re.Pattern[str] = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(nick)
        for nick in sorted(
            {n for names in _CVN_NAMES.values() for n in names},
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    re.I,
)


def _contact_aliases(name: str, kind: str, tacan_callsign: str | None = None, platform_type: str | None = None) -> list[str]:
    out = {name}
    hull_from_type = None
    if platform_type:
        m_type = re.search(r"\bcvn[_-]?(\d{1,3})\b", platform_type, re.I)
        if m_type:
            hull_from_type = m_type.group(1)

    if kind == "carrier" or re.search(r"\bcvn\s*-?\s*\d{1,3}\b", name, re.I) or hull_from_type:
        out.update({"carrier", "boat"})
        m = re.search(r"\bcvn\s*-?\s*(\d{1,3})\b", name, re.I)
        hull = m.group(1) if m else hull_from_type
        if hull:
            out.add(f"CVN-{hull}")
            out.add(f"CVN {hull}")
            for nick in _CVN_NAMES.get(hull, []):
                out.add(nick)
    if kind == "tanker":
        for key in ("texaco", "arco", "shell", "tanker"):
            if re.search(rf"\b{key}\b", name, re.I):
                out.add(key)
    if tacan_callsign:
        out.add(tacan_callsign)
    return sorted(out, key=lambda s: s.lower())


def _callsign_display_variants(callsign_name: str | None) -> list[str]:
    raw = str(callsign_name or "").strip()
    if not raw:
        return []
    out: list[str] = [raw]
    m = re.match(r"^([A-Za-z]+)(\d{1,3})$", raw)
    if m:
        base = m.group(1)
        digits = m.group(2)
        out.append(base)
        if len(digits) >= 2:
            out.append(f"{base} {digits[0]}-{digits[1]}")
        else:
            out.append(f"{base} {digits}")
    dedup: list[str] = []
    seen: set[str] = set()
    for v in out:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            dedup.append(v)
    return dedup


def _find_matching_brace(text: str, open_index: int) -> int | None:
    if open_index < 0 or open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(open_index, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_enclosing_unit_chunk(mission_text: str, name_pos: int) -> tuple[int, int] | None:
    # Find the nearest indexed table start before this name token.
    probe_start = max(0, name_pos - 8000)
    prefix = mission_text[probe_start:name_pos + 1]
    starts = list(re.finditer(r"\[\d+\]\s*=\s*\{", prefix))
    for sm in reversed(starts):
        abs_start = probe_start + sm.start()
        brace_idx = mission_text.find("{", abs_start)
        if brace_idx < 0:
            continue
        end_idx = _find_matching_brace(mission_text, brace_idx)
        if end_idx is None:
            continue
        if brace_idx <= name_pos <= end_idx:
            return (brace_idx, end_idx + 1)
    return None


def _extract_miz_named_contacts(mission_text: str) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for m in re.finditer(r'\["name"\]\s*=\s*"([^"]+)"', mission_text):
        name = m.group(1).strip()
        if len(name) < 3:
            continue

        # Restrict extraction to the enclosing unit block so nearby units cannot bleed fields.
        unit_span = _extract_enclosing_unit_chunk(mission_text, m.start())
        if unit_span:
            block_start, block_end = unit_span
        else:
            block_start = max(0, m.start() - 1200)
            block_end = min(len(mission_text), m.start() + 4000)
        chunk = mission_text[block_start:block_end]
        # Select the nearest unit type token to this name occurrence.
        # This avoids pulling unrelated labels from nearby route/waypoint blocks.
        type_matches = list(re.finditer(r'\["type"\]\s*=\s*"([^"]+)"', chunk))
        platform_type = None
        if type_matches:
            anchor = m.start() - block_start
            nearest = min(type_matches, key=lambda tm: abs(tm.start() - anchor))
            candidate = nearest.group(1).strip()
            # Route labels often contain spaces (e.g., "Turning Point"); prefer model-ish tokens.
            if " " not in candidate or re.search(r"\b(stennis|ticonderog|cvn|kc[-_ ]?135|kc[-_ ]?130|il[-_ ]?78)\b", candidate, re.I):
                platform_type = candidate
        if not platform_type:
            continue

        kind = _classify_contact_kind(name, platform_type=platform_type)
        anchor = m.start() - block_start

        def _nearest(pattern: str, flags: int = 0) -> re.Match[str] | None:
            matches = list(re.finditer(pattern, chunk, flags))
            if not matches:
                return None
            return min(matches, key=lambda tm: abs(tm.start() - anchor))

        def _nearest_prefer_forward(matches: list[re.Match[str]], ref: int) -> re.Match[str] | None:
            if not matches:
                return None
            forward = [tm for tm in matches if tm.start() >= ref]
            if forward:
                return min(forward, key=lambda tm: tm.start() - ref)
            return min(matches, key=lambda tm: abs(tm.start() - ref))

        radio_mhz = None
        freq_candidates: list[tuple[int, float]] = []
        for fm in re.finditer(r'\["frequency"\]\s*=\s*([0-9]+(?:\.[0-9]+)?)', chunk):
            mhz = _to_mhz(float(fm.group(1)))
            if mhz is None:
                continue
            if 30.0 <= mhz <= 400.0:
                freq_candidates.append((abs(fm.start() - anchor), mhz))
        if freq_candidates:
            freq_candidates.sort(key=lambda x: x[0])
            radio_mhz = freq_candidates[0][1]

        tacan = None
        callsign_name = None
        ctm = _nearest(r'\["callsign"\]\s*=\s*\{[^\}]*\["name"\]\s*=\s*"([^"]+)"', re.S)
        if ctm:
            callsign_name = ctm.group(1).strip()
        if re.search(r'ActivateBeacon', chunk, re.I):
            beacon_tacan = None
            best_dist: int | None = None
            for bm in re.finditer(r'ActivateBeacon', chunk, re.I):
                # Scope channel/mode/callsign to within the ActivateBeacon ["params"] block
                # to avoid false matches on radio-preset ["channel"] = 1 keys.
                region = chunk[bm.start():bm.start() + 300]
                pm = re.search(r'\["params"\]\s*=\s*(\{)', region)
                if pm is None:
                    continue
                brace_in_chunk = bm.start() + pm.start(1)
                end_in_chunk = _find_matching_brace(chunk, brace_in_chunk)
                params_text = (
                    chunk[brace_in_chunk:end_in_chunk + 1]
                    if end_in_chunk is not None
                    else chunk[brace_in_chunk:brace_in_chunk + 600]
                )
                ch_m = re.search(r'\["channel"\]\s*=\s*([0-9]{1,3})', params_text)
                if ch_m is None:
                    continue
                ch = int(ch_m.group(1))
                if not (1 <= ch <= 126):
                    continue
                mode_m = re.search(r'\["modeChannel"\]\s*=\s*"([XYxy])"', params_text)
                mode = mode_m.group(1).upper() if mode_m else "X"
                cs_m = re.search(r'\["callsign"\]\s*=\s*"([A-Za-z0-9]+)"', params_text)
                dist = abs(bm.start() - anchor)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    beacon_tacan = {
                        "channel": ch,
                        "mode": mode,
                        "callsign": cs_m.group(1) if cs_m else None,
                    }
            if beacon_tacan:
                tacan = beacon_tacan

        # Keep only named contacts that actually expose a usable comm channel.
        if radio_mhz is None and tacan is None:
            continue

        aliases = _contact_aliases(name, kind, tacan_callsign=(tacan or {}).get("callsign"), platform_type=platform_type)
        aliases.extend(_callsign_display_variants(callsign_name))
        aliases = sorted({a for a in aliases if a}, key=lambda s: s.lower())
        contacts.append(
            {
                "name": name,
                "kind": kind,
                "aliases": aliases,
                "platform_type": platform_type,
                "callsign_name": callsign_name,
                "radio_mhz": radio_mhz,
                "tacan": tacan,
            }
        )
        logger.debug(
            "[miz] found contact name='{}' type='{}' kind='{}' radio={} tacan={}",
            name,
            platform_type,
            kind,
            radio_mhz,
            tacan,
        )

    by_name: dict[str, dict[str, Any]] = {}
    for c in contacts:
        key = str(c.get("name", "")).strip().lower()
        if not key:
            continue
        prev = by_name.get(key)
        if not prev:
            by_name[key] = c
            continue
        prev_alias = set(prev.get("aliases", []))
        prev_alias.update(c.get("aliases", []))
        prev["aliases"] = sorted(prev_alias, key=lambda s: s.lower())
        if prev.get("radio_mhz") is None and c.get("radio_mhz") is not None:
            prev["radio_mhz"] = c.get("radio_mhz")
        if prev.get("tacan") is None and c.get("tacan") is not None:
            prev["tacan"] = c.get("tacan")
        if not prev.get("callsign_name") and c.get("callsign_name"):
            prev["callsign_name"] = c.get("callsign_name")

    # Collapse duplicate representations of the same contact (for example
    # group-name and unit-name entries that share platform, radio, and TACAN).
    by_sig: dict[tuple[Any, ...], dict[str, Any]] = {}
    for c in by_name.values():
        t = c.get("tacan") or {}
        sig = (
            str(c.get("kind") or "contact"),
            _norm(str(c.get("platform_type") or "")),
            round(float(c.get("radio_mhz")), 3) if c.get("radio_mhz") is not None else None,
            int(t.get("channel")) if t.get("channel") is not None else None,
            str(t.get("mode") or "").upper() or None,
            str(t.get("callsign") or "").upper() or None,
        )
        prev = by_sig.get(sig)
        if not prev:
            by_sig[sig] = c
            continue
        merged_aliases = set(prev.get("aliases", []))
        merged_aliases.update(c.get("aliases", []))
        merged_aliases.add(str(prev.get("name") or "").strip())
        merged_aliases.add(str(c.get("name") or "").strip())
        prev["aliases"] = sorted({a for a in merged_aliases if a}, key=lambda s: s.lower())
        # Keep the most descriptive display name.
        if len(str(c.get("name") or "")) > len(str(prev.get("name") or "")):
            prev["name"] = c.get("name")

    return list(by_sig.values())


def _query_fuzzy_matches_ship_name(query: str) -> bool:
    """Return True if any word or bigram in query fuzzily matches a known CVN nickname.

    Handles STT transcription errors (e.g. 'washinton' for 'washington') by
    accepting SequenceMatcher ratio ≥ 0.82 against names of 5+ characters.
    """
    from difflib import SequenceMatcher
    q = _norm(query)
    words = [t for t in q.split() if len(t) >= 4]
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    candidates = words + bigrams
    all_names = [
        _norm(n)
        for names in _CVN_NAMES.values()
        for n in names
        if len(_norm(n)) >= 5
    ]
    for cand in candidates:
        for name in all_names:
            if SequenceMatcher(None, cand, name).ratio() >= 0.82:
                return True
    return False


def _query_contact_kind_hint(query: str) -> str | None:
    q = _norm(query)
    if re.search(r"\b(cvn\s*\d+|carrier|boat)\b", q) or _CVN_SHIP_NAMES_RE.search(q):
        return "carrier"
    if re.search(r"\b(texaco|arco|shell|tanker|kc\s*135|kc\s*130|il\s*78)\b", q):
        return "tanker"
    return None


def list_miz_named_contacts(kind: str | None = None, miz_path: Path | None = None) -> list[dict[str, Any]]:
    mission_text = _read_mission_text_from_miz(miz_path=miz_path)
    if not mission_text:
        return []
    contacts = _extract_miz_named_contacts(mission_text)
    if kind:
        k = kind.strip().lower()
        contacts = [c for c in contacts if str(c.get("kind") or "").lower() == k]
    contacts.sort(key=lambda c: str(c.get("name") or "").lower())
    return contacts


def _find_contact_in_miz(query: str, contacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    q = _norm(query)
    if not q:
        return None

    wants_carrier = bool(
        re.search(r"\b(cvn\s*-?\s*\d+|carrier|boat)\b", q)
        or _CVN_SHIP_NAMES_RE.search(q)
    )
    wants_tanker = bool(re.search(r"\b(texaco|arco|shell|tanker|kc\s*-?\s*135|kc\s*-?\s*130|il\s*-?\s*78)\b", q))
    # Specific hull number OR ship name in query — skip the early ambiguity
    # bailout and let alias scoring resolve it precisely below.
    wants_specific_hull = bool(
        re.search(r"\bcvn\s*-?\s*\d+\b", q)
        or _CVN_SHIP_NAMES_RE.search(q)
        or _query_fuzzy_matches_ship_name(q)
    )
    if wants_carrier and not wants_tanker and not wants_specific_hull:
        carrier_hits = [c for c in contacts if str(c.get("kind")) == "carrier"]
        if len(carrier_hits) == 1:
            logger.debug(f"[miz] query hints carrier: single candidate='{carrier_hits[0].get('name')}'")
            return carrier_hits[0]
        if len(carrier_hits) > 1:
            logger.debug(f"[miz] query hints carrier: ambiguous candidates={[c.get('name') for c in carrier_hits]}")
            return {"error": "ambiguous"}
    if wants_tanker and not wants_carrier:
        tanker_hits = [c for c in contacts if str(c.get("kind")) == "tanker"]
        if len(tanker_hits) == 1:
            logger.debug(f"[miz] query hints tanker: single candidate='{tanker_hits[0].get('name')}'")
            return tanker_hits[0]
        if len(tanker_hits) > 1:
            logger.debug(f"[miz] query hints tanker: ambiguous candidates={[c.get('name') for c in tanker_hits]}")
            return {"error": "ambiguous"}

    # If the query names a specific CVN ship, restrict alias scoring to contacts
    # whose platform_type hull number actually matches — prevents callsign fields
    # ("Stennis") from cross-matching the wrong carrier (CVN_73 ≠ CVN_74).
    hull_filters: set[str] = set()
    for hull, nicks in _CVN_NAMES.items():
        for nick in nicks:
            if re.search(rf"\b{re.escape(_norm(nick))}\b", q):
                hull_filters.add(hull)
                break
    # Fuzzy pass: catch STT near-misses (e.g. 'washinton' for 'washington').
    if not hull_filters:
        from difflib import SequenceMatcher
        words = [t for t in q.split() if len(t) >= 4]
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        candidates = words + bigrams
        for hull, nicks in _CVN_NAMES.items():
            if hull in hull_filters:
                continue
            for nick in nicks:
                norm_nick = _norm(nick)
                if len(norm_nick) < 5:
                    continue
                for cand in candidates:
                    if SequenceMatcher(None, cand, norm_nick).ratio() >= 0.82:
                        hull_filters.add(hull)
                        break
                if hull in hull_filters:
                    break

    scoring_contacts = contacts
    if hull_filters:
        scoring_contacts = [
            c for c in contacts
            if any(
                re.search(rf"cvn[_\s-]?0*{re.escape(h)}\b", str(c.get("platform_type") or ""), re.I)
                for h in hull_filters
            )
        ]

    scored: list[tuple[int, dict[str, Any]]] = []
    for c in scoring_contacts:
        aliases = [str(a) for a in c.get("aliases", [])]
        best = 0
        for alias in aliases:
            key = _norm(alias)
            if len(key) < 2:
                continue
            if re.search(rf"\b{re.escape(key)}\b", q):
                best = max(best, len(key))
        if best > 0:
            scored.append((best, c))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[0][0]
        top_hits = [c for s, c in scored if s == top]
        if len(top_hits) > 1:
            # Tiebreak: if all tied candidates share the same platform type they
            # represent the same physical ship (e.g. group-unit vs lead-unit with
            # different TACAN completeness).  Pick the one with the most data.
            platform_types = {str(c.get("platform_type") or "").strip().lower() for c in top_hits}
            if len(platform_types) == 1:
                def _richness(c: dict[str, Any]) -> int:
                    t = c.get("tacan") or {}
                    return (1 if t.get("channel") is not None else 0) + (1 if c.get("radio_mhz") is not None else 0)
                top_hits.sort(key=_richness, reverse=True)
                logger.debug(f"[miz] alias match tiebreak by richness: selected='{top_hits[0].get('name')}' score={top}")
                return top_hits[0]
            logger.debug(f"[miz] alias match ambiguous top_score={top} candidates={[c.get('name') for c in top_hits]}")
            return {"error": "ambiguous"}
        logger.debug(f"[miz] alias match selected='{top_hits[0].get('name')}' score={top}")
        return top_hits[0]

    # Fuzzy fallback for STT drift on callsigns/names.
    tokens = [
        t
        for t in q.split()
        if t not in {"what", "whats", "give", "me", "the", "for", "and", "about", "freq", "frequency", "channel", "radio", "tacan", "uhf", "vhf", "hf"}
        and len(t) >= 2
    ]
    candidate_query = " ".join(tokens).strip() or q

    best_contact: dict[str, Any] | None = None
    best_score = 0.0
    second_score = 0.0
    for c in scoring_contacts:
        aliases = [str(a) for a in c.get("aliases", [])]
        local_best = 0.0
        for alias in aliases:
            key = _norm(alias)
            if len(key) < 3:
                continue
            score = SequenceMatcher(None, candidate_query, key).ratio()
            key_no_digits = re.sub(r"\b\d+\b", " ", key)
            key_no_digits = re.sub(r"\s+", " ", key_no_digits).strip()
            if len(key_no_digits) >= 3:
                score = max(score, SequenceMatcher(None, candidate_query, key_no_digits).ratio())
            local_best = max(local_best, score)
        if local_best > best_score:
            second_score = best_score
            best_score = local_best
            best_contact = c
        elif local_best > second_score:
            second_score = local_best

    if best_contact and best_score >= 0.74 and (best_score - second_score) >= 0.08:
        logger.debug(
            "[miz] fuzzy selected='{}' best={:.3f} second={:.3f}",
            best_contact.get("name"),
            best_score,
            second_score,
        )
        return best_contact
    return None


def resolve_miz_named_contact(query: str, mode: str = "radio", fallback_name: str | None = None, miz_path: Path | None = None) -> dict[str, Any] | None:
    mission_text = _read_mission_text_from_miz(miz_path=miz_path)
    if not mission_text:
        return None
    contacts = _extract_miz_named_contacts(mission_text)
    if not contacts:
        return None

    chosen = _find_contact_in_miz(query, contacts)
    if isinstance(chosen, dict) and chosen.get("error") == "ambiguous":
        hint = _query_contact_kind_hint(query)
        options = contacts
        if hint:
            options = [c for c in contacts if str(c.get("kind") or "").lower() == hint]
        names = [str(c.get("name") or "").strip() for c in options]
        names = [n for n in names if n]
        return {"error": "ambiguous", "kind": hint or "contact", "options": names[:8]}

    if not chosen and fallback_name:
        chosen = _find_contact_in_miz(fallback_name, contacts)
    if not chosen:
        return None

    name = str(chosen.get("name") or "contact")
    if mode == "tacan":
        t = chosen.get("tacan")
        if not isinstance(t, dict) or t.get("channel") is None:
            return {"error": "missing", "contact": name, "mode": "tacan"}
        return {
            "source": "miz",
            "kind": str(chosen.get("kind") or "contact"),
            "contact": name,
            "platform_type": str(chosen.get("platform_type") or "").strip() or None,
            "callsign_name": str(chosen.get("callsign_name") or "").strip() or None,
            "mode": "tacan",
            "channel": int(t.get("channel")),
            "band": str(t.get("mode") or "X").upper(),
            "callsign": t.get("callsign"),
        }

    mhz = chosen.get("radio_mhz")
    if mhz is None:
        return {"error": "missing", "contact": name, "mode": "radio"}
    return {
        "source": "miz",
        "kind": str(chosen.get("kind") or "contact"),
        "contact": name,
        "platform_type": str(chosen.get("platform_type") or "").strip() or None,
        "callsign_name": str(chosen.get("callsign_name") or "").strip() or None,
        "mode": "radio",
        "mhz": float(mhz),
    }


def find_miz_named_contact(query: str, fallback_name: str | None = None, miz_path: Path | None = None) -> dict[str, Any] | None:
    mission_text = _read_mission_text_from_miz(miz_path=miz_path)
    if not mission_text:
        return None
    contacts = _extract_miz_named_contacts(mission_text)
    if not contacts:
        return None

    chosen = _find_contact_in_miz(query, contacts)
    if isinstance(chosen, dict) and chosen.get("error") == "ambiguous":
        hint = _query_contact_kind_hint(query)
        options = contacts
        if hint:
            options = [c for c in contacts if str(c.get("kind") or "").lower() == hint]
        names = [str(c.get("name") or "").strip() for c in options]
        names = [n for n in names if n]
        return {"error": "ambiguous", "kind": hint or "contact", "options": names[:8]}
    if not chosen and fallback_name:
        chosen = _find_contact_in_miz(fallback_name, contacts)
    if not chosen:
        return None

    return {
        "source": "miz",
        "kind": str(chosen.get("kind") or "contact"),
        "contact": str(chosen.get("name") or "contact"),
        "platform_type": str(chosen.get("platform_type") or "").strip() or None,
        "callsign_name": str(chosen.get("callsign_name") or "").strip() or None,
        "has_radio": chosen.get("radio_mhz") is not None,
        "has_tacan": isinstance(chosen.get("tacan"), dict) and chosen.get("tacan", {}).get("channel") is not None,
    }


def lookup_miz_frequency(airfield: str, theatre: str | None = None, miz_path: Path | None = None) -> dict[str, Any] | None:
    entry = _entry_for_airfield(airfield, theatre)
    if not entry:
        return None
    aid = entry.get("airdrome_id")
    if aid is None:
        return None
    try:
        aid = int(aid)
    except Exception:
        return None

    miz = miz_path or find_active_miz_path()
    if not miz or not miz.exists():
        return None

    try:
        with zipfile.ZipFile(miz, "r") as zf:
            mission_text = zf.read("mission").decode("utf-8", errors="ignore")
    except Exception:
        return None

    pairs = _extract_freq_airdrome_pairs(mission_text)
    vals = [freq for a, freq in pairs if a == aid]
    if not vals:
        return None
    mhz = Counter(vals).most_common(1)[0][0]
    return {
        "source": "miz",
        "role": "mission preset",
        "mhz": float(mhz),
        "airfield": entry.get("canonical") or airfield,
    }


def resolve_frequency(airfield: str, in_jet: bool, band: str = "tower") -> dict[str, Any] | None:
    theatre = detect_theatre() if in_jet else None
    if in_jet and band == "tower":
        miz = lookup_miz_frequency(airfield, theatre=theatre)
        if miz:
            return miz
    local = lookup_local_frequency(airfield, theatre=theatre, band=band)
    if local:
        return local
    if in_jet and theatre:
        global_hit = lookup_local_frequency(airfield, theatre=None, band=band)
        if global_hit:
            return {
                "source": "out-of-ao",
                "role": global_hit.get("role", band),
                "airfield": global_hit.get("airfield") or airfield,
                "known_theatre": find_airfield_theatre(airfield),
                "requested_theatre": theatre,
            }
    return lookup_local_frequency(airfield, theatre=None, band=band)
