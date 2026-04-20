"""
Tests for automatic docs fallback when quick actions do not match.

Run: pytest tests/test_llm_docs_fallback.py -v
"""

from __future__ import annotations

import types

from orchestrator.llm import call_ollama_with_tools, clean_reply
from orchestrator.session import Session
import orchestrator.llm as llm_module


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_quick_action_miss_triggers_docs_fallback(monkeypatch):
    calls = {"quick": 0, "docs": 0, "post": 0}

    def _quick_action(_args, _session):
        calls["quick"] += 1
        return {"ok": True, "found": False}

    def _search_docs(_args, _session):
        calls["docs"] += 1
        return {
            "ok": True,
            "query": "How do I do this unknown cockpit action?",
            "confidence": "MEDIUM",
            "top_score": 15.2,
            "hits": [{"section": "Test", "page": 1, "score": 15.2, "text": "Doc hit"}],
        }

    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "get_quick_action", _quick_action)
    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "search_natops", _search_docs)

    def _fake_post(*_args, **_kwargs):
        calls["post"] += 1
        if calls["post"] == 1:
            return _FakeResponse(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_quick_action",
                                    "arguments": {"query": "How do I do this unknown cockpit action?"},
                                }
                            }
                        ],
                    },
                    "done_reason": "stop",
                }
            )

        return _FakeResponse(
            {
                "message": {"content": "Use the documented procedure.", "tool_calls": []},
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr(llm_module.httpx, "post", _fake_post)

    out = call_ollama_with_tools("How do I do this unknown cockpit action?", Session(), "dummy")
    assert out == "Use the documented procedure."
    assert calls["quick"] == 1
    assert calls["docs"] == 1


def test_no_tool_first_response_forces_docs_grounding(monkeypatch):
    calls = {"docs": 0, "post": 0}

    def _search_docs(_args, _session):
        calls["docs"] += 1
        return {
            "ok": True,
            "query": "How do I set this switch?",
            "confidence": "MEDIUM",
            "top_score": 14.8,
            "hits": [{"section": "Test", "page": 2, "score": 14.8, "text": "Grounded hit"}],
        }

    monkeypatch.setitem(llm_module.TOOL_REGISTRY, "search_natops", _search_docs)

    def _fake_post(*_args, **_kwargs):
        calls["post"] += 1
        if calls["post"] == 1:
            return _FakeResponse(
                {
                    "message": {"content": "Flip the center-console mode switch.", "tool_calls": []},
                    "done_reason": "stop",
                }
            )

        return _FakeResponse(
            {
                "message": {"content": "Use the mapped switch from the grounded procedure.", "tool_calls": []},
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr(llm_module.httpx, "post", _fake_post)

    out = call_ollama_with_tools("How do I set this switch?", Session(), "dummy")
    assert out == "Use the mapped switch from the grounded procedure."
    assert calls["docs"] == 1
    assert calls["post"] == 2


def test_clean_reply_removes_duplicate_sentence():
    text = (
        "Use HOTAS Radar Antenna Elevation control on the throttle to zoom in and out. "
        "Use HOTAS Radar Antenna Elevation control on the throttle to zoom in and out."
    )
    assert clean_reply(text) == "Use HOTAS Radar Antenna Elevation control on the throttle to zoom in and out."
