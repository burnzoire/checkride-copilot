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


def test_tanker_callsign_digits_spoken_individually():
    assert _normalize("Texaco11 frequency is 251.000 MHz.").startswith("Texaco 1 1 ")
    assert _normalize("Arco11 TACAN is 51X.") .startswith("Arco 1 1 ")
    assert _normalize("Shell2 is on frequency.")  .startswith("Shell 2 ")


def test_ranges_expand_for_tts():
    text = "Maintain 140-150 kts and 1000-3000 ft."
    out = _normalize(text)
    assert "140 to 150 knots" in out
    assert "1000 to 3000 feet" in out


def test_frequency_brevity_whole_decimal():
    # 127.000 MHz → one two seven decimal zero
    out = _normalize("Tune 127.000 MHz.")
    assert out == "Tune one two seven decimal zero."


def test_frequency_brevity_trailing_nonzero():
    # 133.350 MHz → one three three decimal three five
    out = _normalize("Contact on 133.350 MHz.")
    assert "one three three decimal three five" in out


def test_frequency_brevity_guard():
    # 243.0 MHz → two four three decimal zero
    out = _normalize("Guard is 243.0 MHz.")
    assert "two four three decimal zero" in out


def test_frequency_brevity_no_decimal():
    # 126 MHz → one two six decimal zero (assume zero fraction)
    out = _normalize("Tower 126 MHz.")
    assert "one two six decimal zero" in out


def test_frequency_brevity_case_insensitive():
    out = _normalize("Tune 127.000 mhz.")
    assert "one two seven decimal zero" in out
