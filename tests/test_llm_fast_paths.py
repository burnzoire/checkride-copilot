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
    assert _is_airfield_frequency_query("Switch to uniform") is True
    assert _extract_band_from_query("How about UHF?") == "UHF"
    assert _extract_band_from_query("switch to uniform") == "UHF"
    assert _extract_band_from_query("switch to victor") == "VHF_HI"
    assert _extract_band_from_query("switch to hotel") == "HF"
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


def test_followup_band_query_uses_previous_user_turn_for_airfield_context(monkeypatch):
    import mission.frequencies as freq_module

    def _mock_resolve(airfield: str, in_jet: bool, band: str = "tower"):
        if airfield == "Kobuleti" and band == "UHF":
            return {"source": "local", "role": "UHF", "mhz": 262.0, "airfield": "Kobuleti"}
        if airfield == "Kobuleti" and band == "tower":
            return {"source": "local", "role": "tower", "mhz": 133.0, "airfield": "Kobuleti"}
        return None

    monkeypatch.setattr(freq_module, "resolve_frequency", _mock_resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for contextual frequency follow-up")))

    session = Session()
    # Simulate prior turn already present in conversation history.
    session.add_user_turn("how do i contact Kobuleti tower?", ts=1.0)
    session.add_assistant_turn("Tower frequency is 133.000 MHz.", ts=1.0)

    out = call_ollama_with_tools(
        "switch to uniform",
        session,
        model="unused",
    )
    assert out == "UHF frequency is 262.000 MHz."


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
    assert out == "CVN-71 carrier frequency is 127.500 MHz."


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
    assert out == "CVN-71 carrier frequency is 127.500 MHz."


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
    assert out == "tanker TACAN is 31Y."


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
    assert out == "I can't provide TACAN for contact."


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
    assert out == "I can see CVN-71 carrier in the current mission."


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
    assert "multiple carriers" in out
    assert "CVN-71" in out
    assert "CVN-74" in out


def test_ambiguous_carrier_query_hides_unit_aliases_when_hulls_present(monkeypatch):
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *_args, **_kwargs: {
        "error": "ambiguous",
        "kind": "carrier",
        "options": ["CVN-71", "CVN-73", "Naval-1-1", "Naval-2-1"],
    })
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Naval-1-1", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None},
        {"name": "Naval-2-1", "kind": "carrier", "platform_type": "CVN_73", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for ambiguous carrier list prompt")))

    out = call_ollama_with_tools(
        "What's the carrier frequency?",
        Session(),
        model="unused",
    )
    assert "CVN-71" in out
    assert "CVN-73" in out
    assert "Naval-1-1" not in out
    assert "Naval-2-1" not in out


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
    assert out == "CVN-71 carrier is a CVN-71 carrier."


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
    assert out == "CVN-71 carrier frequency is 127.500 MHz."


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
    assert out == "Texaco11 frequency is 251.000 MHz."


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
    assert out == "Texaco11 frequency is 254.000 MHz."


def test_ambiguous_contact_prompt_consumes_followup_answer(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "carrier frequency" in q:
            return {
                "error": "ambiguous",
                "kind": "carrier",
                "options": ["Stennis", "Unit #001"],
            }
        if "cvn-74" in q:
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Stennis",
                "platform_type": "CVN_74",
                "mode": mode,
                "mhz": 127.5,
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Stennis", "kind": "carrier", "platform_type": "CVN_74", "callsign_name": None},
        {"name": "Carrier", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for pending contact follow-up")))

    session = Session()
    first = call_ollama_with_tools("what is the carrier frequency?", session, model="unused")
    assert "multiple carriers" in first.lower()
    assert "CVN-74" in first
    assert "CVN-71" in first

    second = call_ollama_with_tools("CVN-74", session, model="unused")
    assert second == "CVN-74 carrier frequency is 127.500 MHz."


def test_ambiguous_carrier_alias_options_accept_hull_followup(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "carrier frequency" in q:
            return {
                "error": "ambiguous",
                "kind": "carrier",
                "options": ["Naval-1-1", "Naval-2-1"],
            }
        if "naval-1-1" in q:
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Naval-1-1",
                "platform_type": "CVN_71",
                "mode": mode,
                "mhz": 127.5,
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Naval-1-1", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None},
        {"name": "Naval-2-1", "kind": "carrier", "platform_type": "CVN_73", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for pending carrier hull follow-up")))

    session = Session()
    first = call_ollama_with_tools("what is the carrier frequency?", session, model="unused")
    assert "multiple carriers" in first.lower()
    assert "CVN-71" in first
    assert "CVN-73" in first

    second = call_ollama_with_tools("CVN-71", session, model="unused")
    assert second == "CVN-71 carrier frequency is 127.500 MHz."


def test_named_unit_followup_routes_to_contact_fast_path(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "carrier frequency" in q:
            return {
                "error": "ambiguous",
                "kind": "carrier",
                "options": ["Unit #001", "Stennis"],
            }
        if "unit #001" in q or "unit 001" in q:
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Unit #001",
                "platform_type": "CVN_71",
                "mode": mode,
                "mhz": 127.5,
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Unit #001", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None},
        {"name": "Stennis", "kind": "carrier", "platform_type": "CVN_74", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for unit follow-up")))

    session = Session()
    first = call_ollama_with_tools("what is the carrier frequency?", session, model="unused")
    assert "multiple carriers" in first.lower()

    second = call_ollama_with_tools("Unit #001", session, model="unused")
    assert second == "CVN-71 carrier frequency is 127.500 MHz."


def test_tacan_followup_uses_last_contact_context_without_repeating_name(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "carrier" in q and "frequency" in q:
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Stennis",
                "platform_type": "CVN_74",
                "mode": "radio",
                "mhz": 127.5,
            }
        if "tacan" in q and fallback_name == "Stennis":
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Stennis",
                "platform_type": "CVN_74",
                "mode": "tacan",
                "channel": 74,
                "band": "X",
                "callsign": "STN",
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for contact follow-up TACAN")))

    session = Session()
    first = call_ollama_with_tools("what is the carrier frequency?", session, model="unused")
    assert first == "CVN-74 carrier frequency is 127.500 MHz."

    second = call_ollama_with_tools("what's the TACAN?", session, model="unused")
    assert second == "CVN-74 carrier TACAN is 74X."


def test_frequency_followup_uses_last_contact_context_without_repeating_name(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "tanker" in q and "frequency" in q:
            return {
                "source": "miz",
                "kind": "tanker",
                "contact": "Texaco 1-1",
                "platform_type": "KC-135MPRS",
                "mode": "radio",
                "mhz": 251.0,
                "callsign_name": "Texaco11",
            }
        if "what about uhf" in q and fallback_name == "Texaco 1-1":
            return {
                "source": "miz",
                "kind": "tanker",
                "contact": "Texaco 1-1",
                "platform_type": "KC-135MPRS",
                "mode": "radio",
                "mhz": 252.0,
                "callsign_name": "Texaco11",
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for contact follow-up frequency")))

    session = Session()
    first = call_ollama_with_tools("what is the tanker frequency?", session, model="unused")
    assert first == "Texaco11 frequency is 251.000 MHz."

    second = call_ollama_with_tools("what about UHF?", session, model="unused")
    assert second == "Texaco11 frequency is 252.000 MHz."


def test_ambiguous_carrier_aliases_auto_resolve_when_only_one_distinct_carrier(monkeypatch):
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda query, mode="radio", fallback_name=None: (
        {
            "error": "ambiguous",
            "kind": "carrier",
            "options": ["carrier", "Stennis", "Unit #001"],
        }
        if "carrier frequency" in str(query).lower()
        else {
            "source": "miz",
            "kind": "carrier",
            "contact": "Stennis",
            "platform_type": "CVN_74",
            "mode": "radio",
            "mhz": 127.5,
        }
    ))
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Stennis", "kind": "carrier", "platform_type": "CVN_74", "callsign_name": None}
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for single-carrier alias collapse")))

    out = call_ollama_with_tools("what's the carrier frequency?", Session(), model="unused")
    assert out == "CVN-74 carrier frequency is 127.500 MHz."


def test_ambiguous_carrier_aliases_same_hull_auto_resolve(monkeypatch):
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda query, mode="radio", fallback_name=None: (
        {
            "error": "ambiguous",
            "kind": "carrier",
            "options": ["carrier", "Stennis", "Unit #001"],
        }
        if "carrier frequency" in str(query).lower()
        else {
            "source": "miz",
            "kind": "carrier",
            "contact": "Stennis",
            "platform_type": "CVN_74",
            "mode": "radio",
            "mhz": 127.5,
        }
    ))
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Stennis", "kind": "carrier", "platform_type": "CVN_74", "callsign_name": None},
        {"name": "Unit #001", "kind": "carrier", "platform_type": "CVN_74", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called when one carrier hull is present")))

    out = call_ollama_with_tools("what is the carrier frequency?", Session(), model="unused")
    assert out == "CVN-74 carrier frequency is 127.500 MHz."


def test_ambiguous_carrier_noisy_metadata_uses_known_name_to_hull(monkeypatch):
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda query, mode="radio", fallback_name=None: (
        {
            "error": "ambiguous",
            "kind": "carrier",
            "options": ["Carrier", "Stennis", "Unit #001"],
        }
        if "carrier" in str(query).lower() and "freq" in str(query).lower()
        else {
            "source": "miz",
            "kind": "carrier",
            "contact": "Stennis",
            "platform_type": "Turning Point",
            "mode": "radio",
            "mhz": 127.5,
        }
        if str(query).strip().lower() == "stennis"
        else None
    ))
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Carrier", "kind": "carrier", "platform_type": "TICONDEROG", "callsign_name": None},
        {"name": "Stennis", "kind": "carrier", "platform_type": "Turning Point", "callsign_name": None},
        {"name": "Unit #001", "kind": "carrier", "platform_type": "Stennis", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for noisy carrier metadata case")))

    out = call_ollama_with_tools("what is the carrier freq?", Session(), model="unused")
    assert out == "CVN-74 carrier frequency is 127.500 MHz."


def test_tacan_followup_single_hull_missing_tacan_returns_missing_not_ambiguous(monkeypatch):
    import mission.frequencies as freq_module

    def _resolve(query, mode="radio", fallback_name=None):
        q = str(query).lower()
        if "carrier" in q and "freq" in q:
            return {
                "source": "miz",
                "kind": "carrier",
                "contact": "Stennis",
                "platform_type": "CVN_74",
                "mode": "radio",
                "mhz": 127.5,
            }
        if "tacan" in q:
            return {
                "error": "ambiguous",
                "kind": "carrier",
                "options": ["Carrier", "Stennis", "Unit #001"],
            }
        if str(query).strip().lower() == "stennis" and mode == "tacan":
            return {
                "error": "missing",
                "kind": "carrier",
                "contact": "Stennis",
                "platform_type": "CVN_74",
                "mode": "tacan",
            }
        return None

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _resolve)
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Carrier", "kind": "carrier", "platform_type": "TICONDEROG", "callsign_name": None},
        {"name": "Stennis", "kind": "carrier", "platform_type": "Turning Point", "callsign_name": None},
        {"name": "Unit #001", "kind": "carrier", "platform_type": "Stennis", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not be called for TACAN missing follow-up")))

    session = Session()
    first = call_ollama_with_tools("what is the carrier freq?", session, model="unused")
    assert first == "CVN-74 carrier frequency is 127.500 MHz."

    second = call_ollama_with_tools("what about tacan?", session, model="unused")
    assert second == "I can't provide TACAN for CVN-74 carrier."


def test_list_carriers_query_returns_deduplicated_hull_list(monkeypatch):
    """'what carriers are in this mission?' must return unique hulls with ship names."""
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Naval-1-1", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None, "radio_mhz": 127.5, "tacan": None},
        {"name": "Naval-1",   "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None, "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"}},
        {"name": "Naval-2-1", "kind": "carrier", "platform_type": "CVN_73", "callsign_name": None, "radio_mhz": 132.0, "tacan": None},
        {"name": "Naval-2",   "kind": "carrier", "platform_type": "CVN_73", "callsign_name": None, "radio_mhz": 132.0, "tacan": {"channel": 73, "mode": "X", "callsign": "GW"}},
    ] if kind == "carrier" else [])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama must not be called for list query")))

    out = call_ollama_with_tools("what carriers are in this mission?", Session(), model="unused")
    assert "CVN-71" in out
    assert "CVN-73" in out
    # Must not list the same hull twice
    assert out.count("CVN-71") == 1
    assert out.count("CVN-73") == 1
    # Should include ship nicknames
    assert "Theodore Roosevelt" in out or "Roosevelt" in out
    assert "George Washington" in out or "Washington" in out


def test_list_tankers_query_returns_types(monkeypatch):
    """'what tankers are available?' must return type labels."""
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Aerial-9", "kind": "tanker", "platform_type": "KC130", "callsign_name": "Texaco1", "radio_mhz": 254.0, "tacan": {"channel": 91, "mode": "X"}},
        {"name": "Aerial-8", "kind": "tanker", "platform_type": "KC135MPRS", "callsign_name": "Arco1", "radio_mhz": 251.0, "tacan": {"channel": 81, "mode": "X"}},
    ] if kind == "tanker" else [])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama must not be called for list query")))

    out = call_ollama_with_tools("what tankers are available?", Session(), model="unused")
    assert "Texaco1" in out
    assert "Arco1" in out


def test_list_carriers_query_does_not_fire_for_freq_query(monkeypatch):
    """'what's the carrier frequency?' must NOT trigger the list path."""
    import mission.frequencies as freq_module

    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda *a, **kw: {
        "source": "miz", "kind": "carrier", "contact": "Naval-1",
        "platform_type": "CVN_71", "mode": "radio", "mhz": 127.5,
    })
    monkeypatch.setattr(freq_module, "list_miz_named_contacts", lambda kind=None: [
        {"name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71", "callsign_name": None},
    ])
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Ollama must not be called")))

    out = call_ollama_with_tools("what's the carrier frequency?", Session(), model="unused")
    assert "frequency" in out.lower()
    assert "MHz" in out


def _arco_resolve(query, mode="radio", fallback_name=None):
    return {
        "source": "miz",
        "kind": "tanker",
        "contact": "Aerial-8",
        "callsign_name": "Arco11",
        "platform_type": "KC135MPRS",
        "mode": mode,
        "mhz": 251.0,
        "tacan": {"channel": 51, "mode": "Y"},
    } if mode == "radio" else {
        "source": "miz",
        "kind": "tanker",
        "contact": "Aerial-8",
        "callsign_name": "Arco11",
        "platform_type": "KC135MPRS",
        "mode": "tacan",
        "channel": 51,
        "band": "Y",
    }


def test_arco_frequency_query_uses_callsign_label(monkeypatch):
    """'what is the freq for Arco?' should return callsign label."""
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _arco_resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Ollama must not be called")))

    out = call_ollama_with_tools("what is the freq for Arco?", Session(), model="unused")
    assert out == "Arco11 frequency is 251.000 MHz."


def test_texaco_tacan_query_uses_callsign_label(monkeypatch):
    """'what is the TACAN for Texaco?' should return callsign label."""
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", lambda q, mode="radio", fallback_name=None: {
        "source": "miz",
        "kind": "tanker",
        "contact": "Aerial-9",
        "callsign_name": "Texaco11",
        "platform_type": "KC130",
        "mode": "tacan",
        "channel": 31,
        "band": "Y",
    })
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Ollama must not be called")))

    out = call_ollama_with_tools("what is the TACAN for Texaco?", Session(), model="unused")
    assert out == "Texaco11 TACAN is 31Y."


def test_arco_one_one_frequency_query_routes_to_tanker(monkeypatch):
    """'give me the frequency for Arco 1 1' should resolve via callsign alias."""
    import mission.frequencies as freq_module
    monkeypatch.setattr(freq_module, "resolve_miz_named_contact", _arco_resolve)
    monkeypatch.setattr(llm_module.httpx, "post", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Ollama must not be called")))

    out = call_ollama_with_tools("give me the frequency for Arco 1 1", Session(), model="unused")
    assert out == "Arco11 frequency is 251.000 MHz."
