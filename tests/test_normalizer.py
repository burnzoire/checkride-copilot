"""
Tests for the DCS export packet normalizer.

Run:  pytest tests/test_normalizer.py -v
"""

import json
from pathlib import Path

from collector.normalizer import normalize

FIXTURES = Path(__file__).parent / "fixtures"


def load_packet() -> dict:
    return json.loads((FIXTURES / "sample_export_packet.json").read_text())


class TestNormalizer:
    def test_returns_dict(self):
        assert isinstance(normalize(load_packet()), dict)

    def test_airframe_normalized(self):
        state = normalize(load_packet())
        assert state["airframe_normalized"] == "fa18c"

    def test_altitude_converted_to_feet(self):
        state = normalize(load_packet())
        # 6096m → ~20000 ft
        assert 19900 < state["altitude_ft"] < 20100

    def test_airspeed_converted_to_knots(self):
        state = normalize(load_packet())
        # 205.7 m/s → ~400 kts
        assert 390 < state["airspeed_kts"] < 410

    def test_gear_up(self):
        state = normalize(load_packet())
        assert state["gear_position"] == "UP"

    def test_flaps_up(self):
        state = normalize(load_packet())
        # FA-18C flap switch: AUTO = retracted/cruise (no discrete "UP" position)
        assert state["flaps_position"] == "AUTO"

    def test_cockpit_params_passthrough(self):
        state = normalize(load_packet())
        assert state["master_arm"] == "SAFE"
        assert state["radar_mode"] == "RWS"
        assert state["tgp_powered"] is False

    def test_weapon_stations_parsed(self):
        state = normalize(load_packet())
        stations = state["weapon_stations"]
        assert len(stations) >= 2
        types = {s["store_type"] for s in stations}
        assert "AIM-120C" in types
        assert "AIM-9X" in types

    def test_missing_data_key_safe(self):
        """Normalizer should not raise on an empty packet."""
        state = normalize({"event": "export", "ts": 0.0, "data": {}})
        assert isinstance(state, dict)
        assert state["airframe"] == "unknown"

    def test_engine_state_running(self):
        state = normalize(load_packet())
        assert state["engine_left_state"] == "running"
        assert state["engine_right_state"] == "running"
