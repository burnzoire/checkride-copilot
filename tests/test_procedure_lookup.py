"""
Tests for procedure lookup relevance, especially TGP phrasing.

Run: pytest tests/test_procedure_lookup.py -v
"""

from orchestrator.session import Session
from orchestrator.tools import tool_list_procedures


def _top_keys(query: str, limit: int = 5) -> list[str]:
    result = tool_list_procedures({"query": query, "limit": limit}, Session())
    assert result["ok"] is True
    return [m["key"] for m in result["matches"]]


def test_tgp_point_track_finds_relevant_procedures():
    keys = _top_keys("How do I point track with the TGP?")
    assert "point_track" in keys
