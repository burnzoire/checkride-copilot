"""
Normalizes raw DCS Export.lua UDP packets into the canonical LiveState dict.

DCS Export packet structure:
  {
    "event": "export",
    "ts": <float>,
    "data": {
      "SelfData":    { Name, Position, Speed, Heading, Mach, AoA, ... },
      "MechInfo":    { gear, flaps, hook, ... },
      "NavInfo":     { ... },
      "PayloadInfo": { Stations: [{CLSID, Count, ...}, ...] }
    }
  }

Fields that require LoGetCockpitParams() (master_arm, radar_mode, TGP state, etc.)
are read from the optional "CockpitParams" key when the Export.lua includes that call.
Until confirmed field indices are mapped, those fields default to "unknown".
"""

from __future__ import annotations

import math
from typing import Any

# Known aircraft name → normalized airframe ID
_AIRFRAME_MAP: dict[str, str] = {
    "FA-18C_hornet": "fa18c",
    "F/A-18C":       "fa18c",
    "fa-18c":        "fa18c",
}

# Known weapon CLSID → display name (subset; extended as needed)
_CLSID_MAP: dict[str, str] = {
    "{40EF17B7-F508-45de-8566-6FFECC0C1AB8}": "AIM-120C",
    "{A1C74F0F-0143-4A03-A6B8-BBDDA69CF5BF}": "AIM-120B",
    "{908E96A0-6D39-11E3-949A-0800200C9A66}": "AIM-9X",
    "{6CEB49FC-DED8-4DED-B053-E1F033FF72D3}": "AIM-9X",
    "{F4W4_AIM-9M}":                          "AIM-9M",
    "{GBU-12}":                               "GBU-12",
    "{GBU-10}":                               "GBU-10",
    "{GBU-16}":                               "GBU-16",
    "{GBU-31}":                               "GBU-31",
    "{GBU-38}":                               "GBU-38",
    "{BRU55_2*GBU-38}":                       "GBU-38",
    "{BRU55_2*GBU-32}":                       "GBU-32",
    "{LAU_61_HYDRA}":                         "Hydra-70",
    "{M151}":                                 "Maverick",
    "{AN_ASQ_228}":                           "LITENING",
    "{AN_AAQ_28}":                            "LITENING",
    "{FPU_8A_FUEL_TANK}":                     "Fuel Tank",
    "{A111-FuelTank-150gal}":                 "Fuel Tank",
}


def _ms_to_kts(ms: float) -> float:
    return ms * 1.94384


def _m_to_ft(m: float) -> float:
    return m * 3.28084


def _rad_to_deg(r: float) -> float:
    return math.degrees(r)


def _gear_value(v: Any) -> str:
    """DCS gear: 0.0 = up, 1.0 = down, in between = in transit."""
    try:
        f = float(v)
        if f <= 0.05:
            return "UP"
        if f >= 0.95:
            return "DOWN"
        return "IN_TRANSIT"
    except (TypeError, ValueError):
        return "unknown"


def _flaps_value(v: Any) -> str:
    try:
        f = float(v)
        if f <= 0.05:
            return "UP"
        if f >= 0.95:
            return "FULL"
        return "HALF"
    except (TypeError, ValueError):
        return "unknown"


def _engine_state(rpm: float) -> str:
    if rpm < 1.0:
        return "off"
    if rpm < 55.0:
        return "starting"
    return "running"


def _parse_stations(payload_info: dict) -> tuple[list[dict], str, int]:
    """Return (weapon_stations, selected_weapon_guess, gun_rounds)."""
    stations = []
    gun_rounds = 0

    raw_stations = payload_info.get("Stations") or []
    for i, station in enumerate(raw_stations):
        if not station:
            continue
        clsid = station.get("CLSID", "")
        count = int(station.get("count", station.get("Count", 0)))
        if count == 0:
            continue

        if "GUN" in clsid.upper() or "M61" in clsid.upper():
            gun_rounds = count
            continue

        store_name = _CLSID_MAP.get(clsid, clsid or "unknown")

        # Skip non-weapon stores
        if store_name in ("Fuel Tank", "LITENING"):
            continue

        category = _infer_category(store_name)
        stations.append({
            "station": i + 1,
            "store_type": store_name,
            "category": category,
            "quantity": count,
            "ready": True,
        })

    return stations, "", gun_rounds


def _infer_category(name: str) -> str:
    n = name.upper()
    if any(x in n for x in ["AIM", "AMRAAM", "SIDEWINDER"]):
        return "AA"
    if any(x in n for x in ["GBU", "MAVERICK", "HARM", "JDAM", "HYDRA"]):
        return "AG"
    return "UNKNOWN"


def normalize(packet: dict) -> dict:
    """
    Convert a raw DCS Export packet to a normalized LiveState dict.

    Fields not derivable from the standard Export API are set to "unknown".
    They will be filled in when CockpitParams indices are confirmed for the
    F/A-18C (see docs/build-plan.md Phase 0 technical risks).
    """
    data = packet.get("data", {})
    self_data = data.get("SelfData") or {}
    mech_info = data.get("MechInfo") or {}
    payload_info = data.get("PayloadInfo") or {}
    cockpit = data.get("CockpitParams") or {}
    flight_data = data.get("FlightData") or {}

    # --- Aircraft identity ---
    raw_name = self_data.get("Name", "unknown")
    airframe_normalized = _AIRFRAME_MAP.get(raw_name, raw_name.lower().replace(" ", ""))

    # --- Position / kinematics ---
    lla = self_data.get("LatLongAlt") or {}
    lat = lla.get("Lat", 0.0)
    lon = lla.get("Long", 0.0)
    alt_m = lla.get("Alt", 0.0)
    altitude_ft = _m_to_ft(alt_m)

    # Prefer FlightData (explicit API calls) over SelfData for kinematics.
    ias_ms = flight_data.get("IAS_ms", self_data.get("IAS", None))
    tas_ms = flight_data.get("TAS_ms", self_data.get("TAS", ias_ms))
    speed_ms = ias_ms if ias_ms is not None else (tas_ms or 0.0)
    airspeed_kts = _ms_to_kts(speed_ms)

    heading_raw = self_data.get("Heading", 0.0)
    heading_deg = _rad_to_deg(heading_raw) % 360.0

    mach = flight_data.get("Mach", self_data.get("Mach", 0.0)) or 0.0
    aoa_deg = flight_data.get("AoA_deg", self_data.get("AoA", 0.0)) or 0.0

    # --- Fuel ---
    # LoGetFuelMass() returns total fuel in kg.
    fuel_kg = flight_data.get("FuelKg")
    if fuel_kg is not None:
        fuel_total_lbs = fuel_kg * 2.20462
        fuel_internal_lbs = fuel_total_lbs  # no separate internal reading available
    else:
        fuel_internal_lbs = self_data.get("FuelInternal", 0.0) * 2.20462
        fuel_total_lbs = self_data.get("FuelTotal", fuel_internal_lbs) * 2.20462

    # --- Mechanical ---
    # Use MechInfo gear.value for physical gear position (0=up, 1=down) — this reflects
    # actual gear extension, not just the lever. The lever can be UP in a DCS hot-start
    # even though the gear is physically down on the ground.
    gear_info = mech_info.get("gear") or {}
    gear_val = gear_info.get("value", None)
    if gear_val is not None:
        gear_position = _gear_value(float(gear_val))
    else:
        # MechInfo not available — fall back to lever position
        gear_lever_str = cockpit.get("gear_lever", "unknown")
        gear_position = gear_lever_str if gear_lever_str in ("UP", "DOWN") else "unknown"

    flaps_sw = cockpit.get("flaps_sw", "unknown")
    if flaps_sw in ("AUTO", "HALF", "FULL", "UP"):
        flaps_position = flaps_sw
    else:
        flaps_raw = (mech_info.get("flaps") or {}).get("value", 0.0)
        flaps_position = _flaps_value(flaps_raw)

    hook_lever_str = cockpit.get("hook_lever", "unknown")
    if hook_lever_str in ("UP", "DOWN"):
        hook_position = hook_lever_str
    else:
        hook_raw = (mech_info.get("hook") or {}).get("value", 0.0)
        hook_position = _gear_value(hook_raw)

    # --- Engines ---
    # Engine data comes from EngineInfo (added to Export.lua via LoGetEngineInfo).
    # LoGetSelfData() does not include engine RPM. Defaults to "unknown" until
    # EngineInfo is present in the packet.
    engine_info = data.get("EngineInfo") or {}
    left_rpm = float(engine_info.get("RPM_left", -1.0))
    right_rpm = float(engine_info.get("RPM_right", -1.0))
    engine_left_state = _engine_state(left_rpm) if left_rpm >= 0 else "unknown"
    engine_right_state = _engine_state(right_rpm) if right_rpm >= 0 else "unknown"

    # --- Weapons / SMS ---
    weapon_stations, _, gun_rounds = _parse_stations(payload_info)
    selected_weapon = cockpit.get("selected_weapon", "unknown")
    gun_selected = cockpit.get("gun_selected", False)

    # --- Cockpit params ---
    # Named fields come from confirmed arg indices in Export.lua.
    # Fields still "unknown" will be resolved during Phase 0 live discovery
    # using the monitor --diff tool.

    master_arm      = cockpit.get("master_arm", "unknown")
    battery_switch  = cockpit.get("battery_switch", "unknown")
    external_power  = cockpit.get("external_power", "unknown")
    autopilot_on    = cockpit.get("autopilot_engaged", False)
    autopilot_mode  = cockpit.get("autopilot_mode", "unknown")

    # Radar: radar_power_sw is the physical switch (OFF/STBY/OPR/EMERG).
    # radar_mode is the sub-mode (RWS/TWS/STT) — still unknown, defaults to
    # radar_power_sw until the arg index is discovered.
    radar_power_sw  = cockpit.get("radar_power_sw", "unknown")
    radar_powered   = radar_power_sw in ("OPR", "EMERG")
    radar_mode_raw  = cockpit.get("radar_mode", "unknown")  # sub-mode, TBD
    # If sub-mode not yet known, infer coarsely from power switch
    if radar_mode_raw == "unknown":
        if radar_power_sw == "OFF":
            radar_mode = "OFF"
        elif radar_power_sw == "STBY":
            radar_mode = "STBY"
        elif radar_power_sw in ("OPR", "EMERG"):
            radar_mode = "RWS"   # conservative default when sub-mode unknown
        else:
            radar_mode = "unknown"
    else:
        radar_mode = radar_mode_raw
    radar_el            = cockpit.get("radar_elevation_deg", 0.0)
    radar_az            = cockpit.get("radar_az_scan", 0.0)
    radar_locked        = cockpit.get("radar_locked_target", False)
    radar_tdc           = cockpit.get("radar_tdc_priority", False)

    # TGP: tgp_power_sw is the physical FLIR switch (ON/STBY/OFF).
    tgp_power_sw    = cockpit.get("tgp_power_sw", "unknown")
    tgp_powered     = tgp_power_sw == "ON"
    tgp_mode        = cockpit.get("tgp_mode", tgp_power_sw)   # fallback to switch state
    tgp_tdc_priority = cockpit.get("tgp_tdc_priority", False)  # TBD
    tgp_tracking    = cockpit.get("tgp_tracking", False)        # TBD
    tgp_lasing      = cockpit.get("tgp_lasing", False)          # TBD

    hmd_enabled     = cockpit.get("hmd_enabled", False)
    jhmcs_mode      = cockpit.get("jhmcs_mode", "unknown")

    laser_armed     = cockpit.get("laser_armed", False)         # TBD
    laser_code      = cockpit.get("laser_code", 1688)

    mfd_left        = cockpit.get("mfd_left_page", "unknown")
    mfd_right       = cockpit.get("mfd_right_page", "unknown")
    mfd_center      = cockpit.get("mfd_center_page", "unknown")

    return {
        # Identity
        "airframe":             raw_name,
        "airframe_normalized":  airframe_normalized,

        # Power
        "battery_switch":       battery_switch,
        "master_arm":           master_arm,
        "external_power":       external_power,

        # Engines
        "engine_left_rpm":      left_rpm,
        "engine_right_rpm":     right_rpm,
        "engine_left_state":    engine_left_state,
        "engine_right_state":   engine_right_state,

        # Navigation
        "altitude_ft":          altitude_ft,
        "airspeed_kts":         airspeed_kts,
        "mach":                 mach,
        "aoa_deg":              aoa_deg,
        "heading_deg":          heading_deg,
        "lat":                  lat,
        "lon":                  lon,

        # Fuel
        "fuel_internal_lbs":    fuel_internal_lbs,
        "fuel_total_lbs":       fuel_total_lbs,

        # Gear / Flaps
        "gear_position":        gear_position,
        "flaps_position":       flaps_position,
        "hook_position":        hook_position,

        # Autopilot
        "autopilot_engaged":    autopilot_on,
        "autopilot_mode":       autopilot_mode,

        # Radar
        "radar_powered":        radar_powered,
        "radar_mode":           radar_mode,
        "radar_elevation_deg":  radar_el,
        "radar_az_scan":        radar_az,
        "radar_locked_target":  radar_locked,
        "radar_tdc_priority":   radar_tdc,

        # TGP
        "tgp_powered":          tgp_powered,
        "tgp_mode":             tgp_mode,
        "tgp_tdc_priority":     tgp_tdc_priority,
        "tgp_tracking":         tgp_tracking,
        "tgp_lasing":           tgp_lasing,

        # JHMCS
        "hmd_enabled":          hmd_enabled,
        "jhmcs_mode":           jhmcs_mode,

        # Weapons
        "selected_weapon":      selected_weapon,
        "weapon_stations":      weapon_stations,
        "gun_rounds":           gun_rounds,
        "gun_selected":         gun_selected,

        # Laser
        "laser_armed":          laser_armed,
        "laser_code":           laser_code,

        # MFDs
        "mfd_left_page":        mfd_left,
        "mfd_right_page":       mfd_right,
        "mfd_center_page":      mfd_center,
    }
