from __future__ import annotations

import orchestrator.orchestrator as orch
from mission.frequencies import airfield_context_terms
from orchestrator.session import Session


def test_airfield_context_terms_include_caucasus_airfields():
    terms = airfield_context_terms("caucasus", limit=80)
    lowered = {t.lower() for t in terms}
    assert "sochi" in lowered
    assert "sukhumi" in lowered


def test_build_stt_context_terms_prefers_last_airfield(monkeypatch):
    monkeypatch.setattr(orch, "detect_theatre", lambda: "caucasus")
    session = Session(last_airfield="Sochi")
    terms = orch._build_stt_context_terms(session)
    assert terms is not None
    assert terms[0] == "Sochi"
    assert any(t.lower() == "kobuleti" for t in terms)