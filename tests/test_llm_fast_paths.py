"""
Tests for deterministic LLM fast paths.

Run: pytest tests/test_llm_fast_paths.py -v
"""

from orchestrator.llm import call_ollama_with_tools, _is_closing_statement
from orchestrator.session import Session
import orchestrator.llm as llm_module


def test_call_the_ball_query_uses_fast_path_and_returns_example_phraseology():
    reply = call_ollama_with_tools(
        "How do I cull the bull on a carrier approach?",
        Session(),
        model="unused-for-fast-path",
    )

    assert "Hornet Ball" in reply
    assert "fuel" in reply.lower() or "thousands" in reply.lower()


def test_question_form_is_not_treated_as_closing_statement():
    assert _is_closing_statement("Okay, what is it?") is False


def test_airfield_tower_frequency_query_uses_fast_path(monkeypatch):
    def _quick_action(_args, _session):
        return {
            "ok": True,
            "found": True,
            "key": "radio_airfield_inbound",
            "example": "Sochi-Adler traffic, Enfield 1-1, inbound full stop. Tune 127.000 MHz.",
            "steps": [{"num": 1, "text": "Frequency: 127.000 MHz (local)."}],
        }

    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "get_quick_action", _quick_action)

    def _should_not_call_ollama(*_args, **_kwargs):
        raise AssertionError("Ollama should not be called for tower frequency fast path")

    monkeypatch.setattr(llm_module.httpx, "post", _should_not_call_ollama)

    out = call_ollama_with_tools(
        "I'm inbound, Sochi. What's the tower frequency?",
        Session(),
        model="unused",
    )
    assert out == "Tower frequency is 127.000 MHz."


def test_band_specific_frequency_follow_up(monkeypatch):
    """Follow-up 'How about UHF?' should use session airfield context."""
    from orchestrator.llm import _is_airfield_frequency_query, _extract_band_from_query
    import mission.frequencies as freq_module
    
    # Test band detection
    assert _is_airfield_frequency_query("How about UHF?") is True
    assert _is_airfield_frequency_query("What's the VHF frequency?") is True
    assert _extract_band_from_query("How about UHF?") == "UHF"
    assert _extract_band_from_query("VHF frequency?") == "VHF_HI"
    assert _extract_band_from_query("Give me HF.") == "HF"
    
    # Test with mocked resolver
    def _mock_resolve(airfield: str, in_jet: bool, band: str = "tower"):
        if band == "UHF" and airfield == "Sochi":
            return {"source": "local", "role": "UHF", "mhz": 131.5, "airfield": "Sochi"}
        return None
    
    monkeypatch.setattr(freq_module, "resolve_frequency", _mock_resolve)
    
    session = Session()
    session.last_airfield = "Sochi"
    
    out = call_ollama_with_tools(
        "How about UHF?",
        session,
        model="unused",
    )
    # Should extract band from query and resolve with session airfield
    assert "131.5" in out


def test_direct_airfield_extraction_for_frequency_query():
    """Frequency query with airfield name (not via quick-action) should extract and resolve."""
    session = Session()
    
    # "What's the frequency for Sochi?" should:
    # 1. Detect as frequency query
    # 2. Extract "Sochi" directly (not via quick-action)
    # 3. Resolve frequency from data
    # 4. Return tower frequency without calling LLM
    
    out = call_ollama_with_tools(
        "What's the frequency for Sochi?",
        session,
        model="unused",
    )
    
    # Should be deterministic frequency response (not a radio template)
    assert "frequency is" in out.lower()
    assert ("mhz" in out.lower() or "decimal" in out.lower())
    # Should NOT be the radio_airfield_inbound template
    assert "traffic" not in out.lower()
    assert "inbound" not in out.lower()
    # Should have set the session airfield for follow-ups
    assert session.last_airfield is not None


def test_frequency_query_for_known_other_theatre_returns_outside_ao(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(llm_module, "fetch_state", lambda: {"ok": True})
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: "caucasus")
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for outside-AO frequency reply")))
    out = call_ollama_with_tools(
        "Give me the frequency for Cologne Tower.",
        Session(),
        model="unused",
    )
    assert out == "Cologne is outside the current AO."


def test_sakumi_tower_frequency_query_returns_only_frequency(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for Sakumi frequency reply")))
    out = call_ollama_with_tools(
        "Give me the frequency for Sakumi Tower.",
        Session(),
        model="unused",
    )
    assert out == "I couldn't confidently identify the airfield. Say the field again."


def test_sukumi_tower_frequency_query_returns_only_frequency(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for Sukumi frequency reply")))
    out = call_ollama_with_tools(
        "Give me the frequency for Sukumi Tower.",
        Session(),
        model="unused",
    )
    assert out == "I couldn't confidently identify the airfield. Say the field again."


def test_giving_frequency_for_sukumi_tower_returns_only_frequency(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for spoken-variant frequency reply")))
    out = call_ollama_with_tools(
        "Giving the frequency for Sukumi Tower.",
        Session(),
        model="unused",
    )
    assert out == "I couldn't confidently identify the airfield. Say the field again."


def test_jaira_in_syria_resolves_to_jirah_frequency(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: "syria")
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for Jaira frequency reply")))
    out = call_ollama_with_tools(
        "No, the tower frequency for Jaira.",
        Session(),
        model="unused",
    )
    assert out == "Tower frequency is 118.100 MHz."


def test_jira_in_syria_resolves_to_jirah_frequency(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: "syria")
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for Jira frequency reply")))
    out = call_ollama_with_tools(
        "Give me the tower frequency for Jira.",
        Session(),
        model="unused",
    )
    assert out == "Tower frequency is 118.100 MHz."


def test_jiara_in_syria_resolves_to_jirah_frequency(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: "syria")
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for Jiara frequency reply")))
    out = call_ollama_with_tools(
        "Give me the tower frequency for Jiara.",
        Session(),
        model="unused",
    )
    assert out == "Tower frequency is 118.100 MHz."


def test_missing_uhf_band_returns_deterministic_cant_provide(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: "syria")
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for missing UHF band reply")))

    session = Session()
    session.last_airfield = "Jirah"

    out = call_ollama_with_tools(
        "And give me the UHF.",
        session,
        model="unused",
    )
    assert out == "I can't provide a UHF frequency for Jirah."
