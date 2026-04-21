from retrieval.cockpit_lookup import lookup


def test_lookup_flir_sensor_switch_returns_flir_switch():
    result = lookup("Where is the FLIR sensor switch?")
    assert result is not None
    assert result["canonical"] == "FLIR Switch"
    assert result["panel"] == "radar_sensor"


def test_lookup_tgp_sensor_switch_returns_flir_switch():
    result = lookup("Where is the targeting pod sensor switch?")
    assert result is not None
    assert result["canonical"] == "FLIR Switch"