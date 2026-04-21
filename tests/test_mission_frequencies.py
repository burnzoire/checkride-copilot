from __future__ import annotations

import mission.frequencies as freq_module
from mission.frequencies import _parse_radio_entries, extract_airfield_from_text, lookup_local_frequency, resolve_frequency


def test_parse_radio_entries_extracts_vhf_and_airdrome_id():
    sample = """
    radio = {
      {
        -- Sochi
        radioId = 'airfield18_0';
        role = {\"ground\", \"tower\", \"approach\"};
        callsign = {{[\"common\"] = {_(\"Sochi\"), \"Sochi\"}}};
        frequency = {[HF] = {MODULATIONTYPE_AM, 4050000.000000}, [UHF] = {MODULATIONTYPE_AM, 256000000.000000}, [VHF_HI] = {MODULATIONTYPE_AM, 127000000.000000}, [VHF_LOW] = {MODULATIONTYPE_AM, 39600000.000000}};
      };
    }
    """
    rows = _parse_radio_entries(sample)
    assert rows
    row = rows[0]
    assert row["airdrome_id"] == 18
    assert row["canonical"] == "Sochi"
    assert row["tower_mhz"] == 127.0


def test_parse_radio_entries_handles_syria_style_radio_blocks():
    sample = """
    radio = {
      {
        radioId = 'As_Suwayda1';
        role = {"ground", "tower", "approach"};
        callsign = {{["common"] = {_("As_Suwayda"), "As_Suwayda"}}};
        frequency = {[VHF_HI] = {MODULATIONTYPE_AM, 122400000.000000}};
        sceneObjects = {'t:7823385'};
      };
    }
    """
    rows = _parse_radio_entries(sample)
    assert rows
    row = rows[0]
    assert row["airdrome_id"] == 1
    assert row["canonical"] == "As_Suwayda"
    assert row["tower_mhz"] == 122.4


def test_lookup_local_frequency_sochi_from_scraped_index():
    out = lookup_local_frequency("Sochi", theatre="caucasus")
    assert out is not None
    assert out["source"] == "local"
    assert out["mhz"] == 127.0


def test_extract_airfield_from_text_uses_frequency_index():
  assert extract_airfield_from_text("Give me the frequency for Cologne Tower.") == "Cologne"


def test_extract_airfield_from_text_fuzzy_matches_sukumi_tower():
  assert extract_airfield_from_text("Give me the frequency for Sukumi Tower.") == "Uklad"


def test_resolve_frequency_reports_out_of_ao_for_known_other_theatre(monkeypatch):
  monkeypatch.setattr(freq_module, "detect_theatre", lambda: "caucasus")
  out = resolve_frequency("Cologne", in_jet=True)
  assert out is not None
  assert out["source"] == "out-of-ao"
  assert out["airfield"] == "Cologne"
