"""
Tests for control-binding retrieval and switch lookup behavior.

Run: pytest tests/test_control_bindings.py -v
"""

from orchestrator.session import Session
import orchestrator.tools as tools


def _sample_hit(with_actions: bool = False) -> dict:
    hit = {
        "canonical": "HUD Altitude Switch",
        "area": "center_console",
        "area_label": "Center Console",
        "panel": "hud_control",
        "panel_label": "HUD Control Panel",
        "positions": ["BARO", "RDR"],
    }
    if with_actions:
        hit["dcs_actions"] = {"Radar Altitude": "Select RDR Altitude"}
    return hit


def test_lookup_switch_handles_missing_dcs_actions_without_crashing(monkeypatch):
    monkeypatch.setattr(tools, "cockpit_lookup", lambda _q: _sample_hit(with_actions=False))

    seen = {"arg": None}

    def _describe_switch(dcs_actions, airframe="FA-18C_hornet"):
        seen["arg"] = dcs_actions
        return None

    monkeypatch.setattr(tools, "describe_switch", _describe_switch)

    result = tools.tool_lookup_switch({"query": "where is the hud altitude switch"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["canonical"] == "HUD Altitude Switch"
    assert result["keybinds"] is None
    assert seen["arg"] == {}


def test_lookup_switch_returns_keybinds_when_available(monkeypatch):
    monkeypatch.setattr(tools, "cockpit_lookup", lambda _q: _sample_hit(with_actions=True))
    monkeypatch.setattr(
        tools,
        "describe_switch",
        lambda _actions, airframe="FA-18C_hornet": "You have it bound to VKB Button 11.",
    )

    result = tools.tool_lookup_switch({"query": "hud altitude switch"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["keybinds"] == "You have it bound to VKB Button 11."


def test_lookup_switch_returns_hint_when_no_match(monkeypatch):
    monkeypatch.setattr(tools, "cockpit_lookup", lambda _q: None)
    monkeypatch.setattr(tools, "cockpit_suggest", lambda _q: _sample_hit(with_actions=False))

    result = tools.tool_lookup_switch({"query": "where is altitude mode"}, Session())
    assert result["ok"] is False
    assert result["found"] is False
    assert "HUD Altitude Switch" in result["hint"]
