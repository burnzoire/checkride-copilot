"""Tests for TTS text normalization rules."""

from voice.tts import _normalize


def test_units_expand_for_tts():
    text = "Set index to 370 ft and fly 250 kts at 3/4 nm."
    out = _normalize(text)
    assert "370 feet" in out
    assert "250 knots" in out
    assert "three-quarter nautical mile" in out


def test_compound_units_expand_for_tts():
    text = "Descend at 4000 ft/min until 10 nm."
    out = _normalize(text)
    assert "4000 feet per minute" in out
    assert "10 nautical miles" in out


def test_ranges_expand_for_tts():
    text = "Maintain 140-150 kts and 1000-3000 ft."
    out = _normalize(text)
    assert "140 to 150 knots" in out
    assert "1000 to 3000 feet" in out
