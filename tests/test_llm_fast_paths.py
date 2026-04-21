"""
Tests for deterministic LLM fast paths.

Run: pytest tests/test_llm_fast_paths.py -v
"""

from orchestrator.llm import (
    call_ollama_with_tools,
    _full_procedure_fast_reply,
    _is_full_procedure_query,
    _is_closing_statement,
    _is_cockpit_state_query,
    _cockpit_state_fast_reply,
)
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


def test_explanatory_acknowledgment_is_treated_as_closing_statement():
    assert _is_closing_statement("Yeah, that's because I am stationary on the carrier CVN-71.") is True


# ── Cockpit state fast path ───────────────────────────────────────────────────

def test_heading_query_is_detected_as_state_query():
    assert _is_cockpit_state_query("What's my current heading?") is True
    assert _is_cockpit_state_query("Hey, tell me what heading I'm facing.") is True
    assert _is_cockpit_state_query("What direction am I heading?") is True


def test_airspeed_query_is_detected_as_state_query():
    assert _is_cockpit_state_query("What is my current speed?") is True
    assert _is_cockpit_state_query("Give me my airspeed.") is True


def test_altitude_query_is_detected_as_state_query():
    assert _is_cockpit_state_query("What's my altitude?") is True
    assert _is_cockpit_state_query("Tell me my current elevation.") is True


def test_procedural_how_to_is_not_state_query():
    assert _is_cockpit_state_query("How do I read my heading on the HUD?") is False
    assert _is_cockpit_state_query("How do I set my airspeed?") is False


def test_cockpit_state_fast_reply_heading(monkeypatch):
    monkeypatch.setattr(
        "orchestrator.tools.fetch_state",
        lambda: {"heading_deg": 19.0, "altitude_ft": 65.0, "airspeed_kts": 48.0},
    )
    reply = _cockpit_state_fast_reply("What's my heading?")
    assert reply is not None
    assert "19" in reply
    assert "magnetic" in reply.lower()


def test_cockpit_state_fast_reply_airspeed(monkeypatch):
    monkeypatch.setattr(
        "orchestrator.tools.fetch_state",
        lambda: {"heading_deg": 19.0, "altitude_ft": 65.0, "airspeed_kts": 48.0},
    )
    reply = _cockpit_state_fast_reply("What is my current speed?")
    assert reply is not None
    assert "48" in reply


def test_cockpit_state_fast_reply_no_state(monkeypatch):
    monkeypatch.setattr("orchestrator.tools.fetch_state", lambda: None)
    reply = _cockpit_state_fast_reply("What's my heading?")
    assert reply is not None
    assert "DCS" in reply or "available" in reply.lower()


def test_full_procedure_query_detection_for_gbu_drop():
    assert _is_full_procedure_query("How do I drop GBU-12s?") is True
    assert _is_full_procedure_query("Walk me through dropping GBU12") is True
    assert _is_full_procedure_query("GBU-12") is False


def test_full_procedure_fast_reply_starts_clear_winner(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "tool_list_procedures",
        lambda _args, _session: {
            "ok": True,
            "matches": [
                {
                    "score": 6,
                    "key": "gbu_12_paveway_ii_laser_guided",
                    "canonical": "GBU-12 Paveway II (Laser-Guided)",
                },
                {"score": 1, "key": "other_proc", "canonical": "Other"},
            ],
        },
    )
    monkeypatch.setattr(
        llm_module,
        "tool_start_procedure",
        lambda _args, _session: {
            "ok": True,
            "active_proc": "gbu_12_paveway_ii_laser_guided",
            "intro": "GBU-12 Paveway II (Laser-Guided). 35 steps total. First: Set Master Arm switch to ARM.",
        },
    )

    out = _full_procedure_fast_reply("How do I drop GBU-12s?", Session())
    assert out is not None
    assert "GBU-12" in out


def test_full_procedure_fast_reply_rejects_designator_mismatch(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "tool_list_procedures",
        lambda _args, _session: {
            "ok": True,
            "matches": [
                {
                    "score": 7,
                    "key": "gbu_12_paveway_ii_laser_guided",
                    "canonical": "GBU-12 Paveway II (Laser-Guided)",
                },
                {"score": 1, "key": "other_proc", "canonical": "Other"},
            ],
        },
    )
    out = _full_procedure_fast_reply("How do I drop MK82?", Session())
    assert out is None


def test_call_ollama_uses_full_procedure_fast_path(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_full_procedure_fast_reply",
        lambda _t, _s: "GBU-12 Paveway II (Laser-Guided). 35 steps total. First: Set Master Arm switch to ARM.",
    )

    def _should_not_call_ollama(*_args, **_kwargs):
        raise AssertionError("Ollama should not be called for full procedure fast path")

    monkeypatch.setattr(llm_module.httpx, "post", _should_not_call_ollama)

    out = call_ollama_with_tools("How do I drop GBU-12s?", Session(), model="unused")
    assert "GBU-12" in out


def test_grounding_fallback_prefers_quick_action_then_procedures_before_natops(monkeypatch):
    call_order: list[str] = []
    monkeypatch.setattr(llm_module, "_is_full_procedure_query", lambda _t: False)

    def _quick_action(_args, _session):
        call_order.append("get_quick_action")
        return {"ok": True, "found": False}

    def _list_procedures(_args, _session):
        call_order.append("list_procedures")
        return {"ok": True, "matches": [{"key": "mk82_ccip", "canonical": "MK-82 CCIP"}]}

    def _search_natops(_args, _session):
        call_order.append("search_natops")
        return {"ok": True, "hits": []}

    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "get_quick_action", _quick_action)
    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "list_procedures", _list_procedures)
    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "search_natops", _search_natops)

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = {"n": 0}

    def _fake_post(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Model does not tool-call on first pass; fallback chain should run.
            return _Resp({"message": {"content": "", "tool_calls": []}, "done_reason": "stop"})
        # Second pass returns final content.
        return _Resp({"message": {"content": "Use MK-82 CCIP profile."}, "done_reason": "stop"})

    monkeypatch.setattr(llm_module.httpx, "post", _fake_post)

    out = call_ollama_with_tools("How do I drop MK82?", Session(), model="unused")
    assert "MK-82" in out
    assert call_order[:2] == ["get_quick_action", "list_procedures"]


def test_quick_action_miss_falls_back_to_procedures_before_natops(monkeypatch):
    call_order: list[str] = []
    monkeypatch.setattr(llm_module, "_is_full_procedure_query", lambda _t: False)

    def _quick_action(_args, _session):
        call_order.append("get_quick_action")
        return {"ok": True, "found": False}

    def _list_procedures(_args, _session):
        call_order.append("list_procedures")
        return {"ok": True, "matches": []}

    def _search_natops(_args, _session):
        call_order.append("search_natops")
        return {"ok": True, "hits": []}

    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "get_quick_action", _quick_action)
    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "list_procedures", _list_procedures)
    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "search_natops", _search_natops)

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_post(*_args, **_kwargs):
        return _Resp({"message": {"content": "Grounded answer.", "tool_calls": []}, "done_reason": "stop"})

    monkeypatch.setattr(llm_module.httpx, "post", _fake_post)

    call_ollama_with_tools("How do I drop GBU-12?", Session(), model="unused")
    # With MAX_TOOL_CALLS=2, quick-action miss should still spend remaining budget
    # on procedures before any docs fallback.
    assert call_order[:2] == ["get_quick_action", "list_procedures"]


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


def test_airfield_frequency_query_uses_mission_context_when_state_polling_unavailable(monkeypatch):
    import mission.frequencies as freq_module

    monkeypatch.setattr(llm_module, "fetch_state", lambda: None)
    monkeypatch.setattr(freq_module, "detect_theatre", lambda: None)
    monkeypatch.setattr(freq_module, "find_active_miz_path", lambda: object())

    seen = {}

    def _mock_resolve(airfield: str, in_jet: bool, band: str = "tower"):
        seen["in_jet"] = in_jet
        return {"source": "local", "role": "tower", "mhz": 127.0, "airfield": "Sochi"}

    monkeypatch.setattr(freq_module, "resolve_frequency", _mock_resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for mission-context fallback")))

    out = call_ollama_with_tools(
        "What's the frequency for Sochi?",
        Session(),
        model="unused",
    )

    assert seen.get("in_jet") is True
    assert out == "Tower frequency is 127.000 MHz."


def test_cvn_radio_frequency_query_uses_miz_contact_fast_path(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(llm_module, "fetch_state", lambda: {"ok": True})
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "carrier",
        "contact": "CVN-71",
        "mode": "radio",
        "mhz": 127.5,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for CVN contact frequency")))

    out = call_ollama_with_tools(
        "What's the carrier frequency for CVN-71?",
        Session(),
        model="unused",
    )
    assert out == "CVN-71 frequency is 127.500 MHz."


def test_cvn_frequency_query_does_not_require_fetch_state(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(llm_module, "fetch_state", lambda: None)
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "carrier",
        "contact": "CVN-71",
        "mode": "radio",
        "mhz": 127.5,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for CVN contact frequency without fetch_state")))

    out = call_ollama_with_tools(
        "Can you give me the frequency to the carrier?",
        Session(),
        model="unused",
    )
    assert out == "CVN-71 frequency is 127.500 MHz."


def test_texaco_tacan_query_uses_miz_contact_fast_path(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(llm_module, "fetch_state", lambda: {"ok": True})
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "tanker",
        "contact": "Texaco 1-1",
        "mode": "tacan",
        "channel": 31,
        "band": "Y",
        "callsign": "TKR",
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for tanker TACAN")))

    out = call_ollama_with_tools(
        "Texaco TACAN?",
        Session(),
        model="unused",
    )
    assert out == "Texaco TACAN is 31Y."


def test_contact_followup_missing_tacan_is_deterministic(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(llm_module, "fetch_state", lambda: {"ok": True})
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "error": "missing",
        "contact": "Texaco 1-1",
        "mode": "tacan",
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for missing contact TACAN")))

    session = Session()
    session.last_contact = "Texaco 1-1"

    out = call_ollama_with_tools(
        "And TACAN?",
        session,
        model="unused",
    )
    assert out == "I can't provide TACAN for Texaco 1-1."


def test_cvn_presence_query_uses_deterministic_contact_visibility(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "find_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "carrier",
        "contact": "CVN-71",
        "has_radio": True,
        "has_tacan": True,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for CVN visibility")))

    out = call_ollama_with_tools(
        "Why not? Can you see CVN 71?",
        Session(),
        model="unused",
    )
    assert out == "I can see CVN-71 in the current mission."


def test_ambiguous_carrier_query_lists_options(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "error": "ambiguous",
        "kind": "carrier",
        "options": ["CVN-71", "CVN-74"],
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for ambiguous carrier list prompt")))

    out = call_ollama_with_tools(
        "What's the carrier frequency?",
        Session(),
        model="unused",
    )
    assert out == "I found multiple carriers: CVN-71, CVN-74. Which one are you after?"


def test_ship_type_query_is_deterministic(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "find_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "carrier",
        "contact": "Naval-1-1",
        "platform_type": "CVN_71",
        "has_radio": True,
        "has_tacan": True,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for ship type query")))

    out = call_ollama_with_tools(
        "What type of ship is Naval 1-1?",
        Session(),
        model="unused",
    )
    assert out == "CVN-71 is a CVN-71 carrier."


def test_carrier_frequency_uses_cvn_label_not_unit_name(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "carrier",
        "contact": "Naval-1-1",
        "platform_type": "CVN_71",
        "mode": "radio",
        "mhz": 127.5,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for carrier CVN label")))

    out = call_ollama_with_tools(
        "What's the carrier frequency?",
        Session(),
        model="unused",
    )
    assert out == "CVN-71 frequency is 127.500 MHz."


def test_tanker_frequency_uses_callsign_not_unit_name(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "tanker",
        "contact": "Texaco 1-1",
        "callsign_name": "Texaco11",
        "platform_type": "KC-135MPRS",
        "mode": "radio",
        "mhz": 251.0,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for tanker callsign label")))

    out = call_ollama_with_tools(
        "Give me Texaco frequency",
        Session(),
        model="unused",
    )
    assert out == "Texaco frequency is 251.000 MHz."


def test_how_about_the_tanker_followup_is_deterministic(monkeypatch):
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "source": "miz",
        "kind": "tanker",
        "contact": "Aerial-9",
        "callsign_name": "Texaco11",
        "platform_type": "KC130",
        "mode": "radio",
        "mhz": 254.0,
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for tanker followup")))

    out = call_ollama_with_tools(
        "How about the tanker?",
        Session(),
        model="unused",
    )
    assert out == "Texaco frequency is 254.000 MHz."
