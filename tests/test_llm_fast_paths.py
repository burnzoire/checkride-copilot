"""
Tests for deterministic LLM fast paths.

Run: pytest tests/test_llm_fast_paths.py -v
"""

from orchestrator.llm import call_ollama_with_tools
from orchestrator.session import Session


def test_call_the_ball_query_uses_fast_path_and_returns_example_phraseology():
    reply = call_ollama_with_tools(
        "How do I cull the bull on a carrier approach?",
        Session(),
        model="unused-for-fast-path",
    )

    assert "Hornet Ball" in reply
    assert "fuel" in reply.lower() or "thousands" in reply.lower()
