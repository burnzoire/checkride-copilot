"""
Tests for optional per-step more_info metadata.

Run: pytest tests/test_procedure_step_more_info.py -v
"""

from orchestrator.session import Session
from orchestrator.tools import tool_get_active_procedure


def test_get_active_procedure_includes_more_info_for_call_the_ball_step():
    session = Session(active_proc="carrier_landing_case_i_recovery", active_step=11)
    result = tool_get_active_procedure({}, session)

    assert result["ok"] is True
    assert result["active"] is True
    assert result["active_step"] == 12
    assert "Hornet Ball" in result["current_step_text"]
    assert result["current_step_more_info"] is not None
    assert "301, Hornet Ball, 3.2" in result["current_step_more_info"]


def test_get_active_procedure_more_info_none_when_not_set():
    session = Session(active_proc="carrier_landing_case_i_recovery", active_step=0)
    result = tool_get_active_procedure({}, session)

    assert result["ok"] is True
    assert result["current_step_more_info"] is None
