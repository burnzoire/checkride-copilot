"""
Tests for STT post-correction mappings.

Run: pytest tests/test_voice_corrections.py -v
"""

from voice.corrections import correct


def test_cull_the_bull_normalizes_to_call_the_ball():
    assert correct("How do I cull the bull on a carrier approach?") == (
        "How do I call the ball on a carrier approach?"
    )


def test_sounded_out_hotas_normalizes():
    assert correct("Set h o t a s controls for radar") == "Set HOTAS controls for radar"


def test_phonetic_hotas_normalizes():
    assert correct("Use ho tas for sensor select") == "Use HOTAS for sensor select"


def test_hotels_phrase_normalizes_to_hotas():
    assert correct("Can I zoom the TGP on my hotels?") == "Can I zoom the TGP on my HOTAS?"


def test_hotez_normalizes_to_hotas():
    assert correct("I said Hotez, not Hotel.") == "I said HOTAS, not Hotel."


def test_cobaltia_airfield_normalizes_to_kobuleti():
    assert correct("I need to land at nearby Cobaltia airfield.") == (
        "I need to land at nearby Kobuleti airfield."
    )


def test_senakee_traffic_normalizes_to_senaki_kolkhi():
    assert correct("Senakee traffic, Enfield 1-1 inbound full stop.") == (
        "Senaki-Kolkhi traffic, Enfield 1-1 inbound full stop."
    )


def test_uniform_phonetic_to_uhf():
    assert correct("Give me the Uniform frequency for Sochi.") == (
        "Give me the UHF frequency for Sochi."
    )


def test_uniform_freak_to_uhf():
    assert correct("Give me the Uniform Freak for Sochi Airfield.") == (
        "Give me the UHF frequency for Sochi Airfield."
    )


def test_victor_phonetic_to_vhf():
    assert correct("What's the Victor frequency?") == (
        "What's the VHF frequency?"
    )


def test_victor_high_to_vhf_hi():
    assert correct("Switch to Victor High.") == (
        "Switch to VHF_HI."
    )


def test_victor_low_to_vhf_low():
    assert correct("Try Victor Low on approach.") == (
        "Try VHF_LOW on approach."
    )


def test_flare_switch_disambiguates_to_flir_switch():
    assert correct("Where is the flare switch?") == "Where is the FLIR switch?"


def test_flay_switch_disambiguates_to_flir_switch():
    assert correct("Where is the flay switch?") == "Where is the FLIR switch?"


def test_felia_switch_disambiguates_to_flir_switch():
    assert correct("Where is the felia switch?") == "Where is the FLIR switch?"


def test_flare_pod_disambiguates_to_flir_pod():
    assert correct("How do I zoom the flare pod?") == "How do I zoom the FLIR pod?"


def test_flare_standalone_not_corrected():
    """'flare' without a sensor-context noun stays as-is (countermeasures)."""
    assert correct("How do I dispense flares?") == "How do I dispense flares?"
    assert correct("Drop flares and chaff.") == "Drop flares and chaff."


def test_flay_standalone_corrects_to_flir():
    assert correct("Tell me about the flay.") == "Tell me about the FLIR."


