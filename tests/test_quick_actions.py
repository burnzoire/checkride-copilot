"""
Tests for the quick-action primitive used for short cockpit tasks.

Run: pytest tests/test_quick_actions.py -v
"""

from orchestrator.session import Session
from orchestrator.tools import tool_get_quick_action


def test_turn_on_tgp_returns_power_on_quick_action():
    result = tool_get_quick_action({"query": "How do I turn on the TGP?"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["key"] == "tgp_power_on"
    assert len(result["steps"]) == 3


def test_tgp_zoom_returns_zoom_quick_action():
    result = tool_get_quick_action({"query": "How do I zoom the TGP?"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["key"] == "tgp_zoom"
    joined = " ".join(step["text"] for step in result["steps"]).lower()
    assert "radar antenna elevation" in joined
    assert "raid/fov" in joined
    assert "sensor of interest" in joined
    assert "only for magnification" in joined


def test_wide_narrow_phrase_maps_to_tgp_zoom():
    result = tool_get_quick_action({"query": "How do I swap between wide and narrow?"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["key"] == "tgp_zoom"


def test_cull_the_bull_maps_to_call_the_ball_quick_action():
    result = tool_get_quick_action({"query": "How do I cull the bull on approach?"}, Session())
    assert result["ok"] is True
    assert result["found"] is True
    assert result["key"] == "carrier_call_the_ball"
    assert "Hornet Ball" in (result.get("example") or "")
