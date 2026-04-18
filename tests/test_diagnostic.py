"""
Tests for the diagnostic engine and rules.

Run:  pytest tests/test_diagnostic.py -v
"""

import json
from pathlib import Path

import pytest

from diagnostic.engine import diagnose_action, check_condition, DiagnosticRule
from diagnostic.action_map import normalize_action

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# check_condition
# ---------------------------------------------------------------------------

class TestCheckCondition:
    def test_eq_pass(self):
        rule = DiagnosticRule("x", "", "f", "ARM", "eq", "", "blocking")
        assert check_condition(rule, {"f": "ARM"}) is True

    def test_eq_fail(self):
        rule = DiagnosticRule("x", "", "f", "ARM", "eq", "", "blocking")
        assert check_condition(rule, {"f": "SAFE"}) is False

    def test_ne_pass(self):
        rule = DiagnosticRule("x", "", "f", "STBY", "ne", "", "blocking")
        assert check_condition(rule, {"f": "A/G"}) is True

    def test_ne_fail(self):
        rule = DiagnosticRule("x", "", "f", "STBY", "ne", "", "blocking")
        assert check_condition(rule, {"f": "STBY"}) is False

    def test_gte_pass(self):
        rule = DiagnosticRule("x", "", "alt", 1500, "gte", "", "warning")
        assert check_condition(rule, {"alt": 8000.0}) is True

    def test_gte_fail(self):
        rule = DiagnosticRule("x", "", "alt", 1500, "gte", "", "warning")
        assert check_condition(rule, {"alt": 500.0}) is False

    def test_in_pass(self):
        rule = DiagnosticRule("x", "", "mode", ["STT", "TWS"], "in", "", "warning")
        assert check_condition(rule, {"mode": "STT"}) is True

    def test_not_in_pass(self):
        rule = DiagnosticRule("x", "", "mode", ["OFF", "STBY"], "not_in", "", "blocking")
        assert check_condition(rule, {"mode": "RWS"}) is True

    def test_not_in_fail(self):
        rule = DiagnosticRule("x", "", "mode", ["OFF", "STBY"], "not_in", "", "blocking")
        assert check_condition(rule, {"mode": "OFF"}) is False

    def test_missing_field_is_blocking(self):
        rule = DiagnosticRule("x", "", "missing", "ARM", "eq", "", "blocking")
        assert check_condition(rule, {}) is False

    def test_unknown_value_is_blocking(self):
        rule = DiagnosticRule("x", "", "f", "ARM", "eq", "", "blocking")
        assert check_condition(rule, {"f": "unknown"}) is False


# ---------------------------------------------------------------------------
# normalize_action
# ---------------------------------------------------------------------------

class TestNormalizeAction:
    def test_exact_canonical(self):
        assert normalize_action("fire_aim120") == "fire_aim120"

    def test_alias(self):
        assert normalize_action("fire amraam") == "fire_aim120"

    def test_fuzzy(self):
        assert normalize_action("fire amraam missile") == "fire_aim120"

    def test_ambiguous_resolved_by_state(self):
        state = {"selected_weapon": "AIM-120C"}
        assert normalize_action("fire", state) == "fire_aim120"

    def test_ambiguous_gbu_resolved(self):
        state = {"selected_weapon": "GBU-12"}
        assert normalize_action("fire", state) == "release_gbu12"

    def test_ambiguous_no_state_returns_first(self):
        result = normalize_action("fire")
        assert result in ("fire_aim120", "fire_aim9x", "fire_gun")


# ---------------------------------------------------------------------------
# diagnose_action — aim120, three canonical states
# ---------------------------------------------------------------------------

class TestDiagnoseFireAim120:
    """Acceptance criteria from build-plan section 17."""

    def test_master_arm_safe_one_blocker(self):
        state = load_fixture("sample_state_safe.json")
        # Fixture has master_arm=SAFE, radar_mode=RWS
        result = diagnose_action("fire_aim120", state)
        assert result["can_execute"] is False
        assert len(result["blockers"]) >= 1
        assert any(b["condition"] == "master_arm_armed" for b in result["blockers"])

    def test_arm_rws_zero_blockers_one_warning(self):
        state = load_fixture("sample_state_safe.json")
        state["master_arm"] = "ARM"
        state["radar_mode"] = "RWS"
        result = diagnose_action("fire_aim120", state)
        assert result["can_execute"] is True
        assert len(result["blockers"]) == 0
        assert len(result["warnings"]) >= 1

    def test_arm_stt_clear(self):
        state = load_fixture("sample_state_ready.json")
        # Fixture: master_arm=ARM, radar_mode=STT
        result = diagnose_action("fire_aim120", state)
        assert result["can_execute"] is True
        assert len(result["blockers"]) == 0
        assert len(result["warnings"]) == 0

    def test_unknown_master_arm_is_blocking(self):
        state = load_fixture("sample_state_safe.json")
        state["master_arm"] = "unknown"
        result = diagnose_action("fire_aim120", state)
        assert result["can_execute"] is False


# ---------------------------------------------------------------------------
# diagnose_action — TGP
# ---------------------------------------------------------------------------

class TestDiagnoseTGPTrack:
    def test_tgp_off_blocks(self):
        state = load_fixture("sample_state_safe.json")
        state["tgp_powered"] = False
        result = diagnose_action("tgp_track", state)
        assert result["can_execute"] is False
        assert any(b["condition"] == "tgp_powered" for b in result["blockers"])

    def test_tgp_on_no_tdc_blocks(self):
        state = load_fixture("sample_state_safe.json")
        state["tgp_powered"] = True
        state["tgp_tdc_priority"] = False
        state["tgp_mode"] = "A/G"
        result = diagnose_action("tgp_track", state)
        assert result["can_execute"] is False
        assert any(b["condition"] == "tgp_tdc_priority" for b in result["blockers"])

    def test_tgp_stby_blocks(self):
        state = load_fixture("sample_state_safe.json")
        state["tgp_powered"] = True
        state["tgp_tdc_priority"] = True
        state["tgp_mode"] = "STBY"
        result = diagnose_action("tgp_track", state)
        assert result["can_execute"] is False
        assert any(b["condition"] == "tgp_not_stby" for b in result["blockers"])

    def test_tgp_ready_clears(self):
        state = load_fixture("sample_state_safe.json")
        state["tgp_powered"] = True
        state["tgp_tdc_priority"] = True
        state["tgp_mode"] = "A/G"
        result = diagnose_action("tgp_track", state)
        assert result["can_execute"] is True


# ---------------------------------------------------------------------------
# diagnose_action — no rules
# ---------------------------------------------------------------------------

class TestDiagnoseUnknownAction:
    def test_no_rules_returns_can_execute_none(self):
        result = diagnose_action("eject", {"data_age_ms": 50})
        assert result["can_execute"] is None
        assert len(result["warnings"]) == 1
        assert "eject" in result["warnings"][0].lower() or "no diagnostic" in result["warnings"][0].lower()
