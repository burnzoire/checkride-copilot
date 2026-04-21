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


def test_drop_gbu12_plural_phrase_finds_gbu12_procedure():
    keys = _top_keys("How do I drop GBU-12s?")
    assert "gbu_12_paveway_ii_laser_guided" in keys


def test_drop_compact_gbu12_phrase_finds_gbu12_procedure():
    keys = _top_keys("How do I drop GBU12?")
    assert "gbu_12_paveway_ii_laser_guided" in keys
