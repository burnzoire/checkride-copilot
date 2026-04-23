from __future__ import annotations

import mission.frequencies as freq_module
from mission.frequencies import _find_contact_in_miz, _parse_radio_entries, _extract_miz_named_contacts, extract_airfield_from_text, find_miz_named_contact, lookup_local_frequency, resolve_frequency


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


def test_extract_miz_named_contacts_includes_non_carrier_named_contact():
  mission_text = '''
  ["units"] = {
    [1] = {
      ["type"] = "E-2D",
  ["name"] = "Wizard 1-1",
  ["frequency"] = 251000000,
  ["route"] = {
    ["points"] = {
      {
        ["task"] = {
          ["id"] = "ComboTask",
          ["params"] = {
            ["tasks"] = {
              {
                ["id"] = "WrappedAction",
                ["params"] = {
                  ["action"] = {
                    ["id"] = "ActivateBeacon",
                    ["params"] = {
                      ["AA"] = true,
                      ["channel"] = 31,
                      ["modeChannel"] = "Y",
                      ["callsign"] = "WZD"
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
    },
  }
  '''
  rows = _extract_miz_named_contacts(mission_text)
  assert rows
  r = rows[0]
  assert r["name"] == "Wizard 1-1"
  assert r["kind"] == "contact"
  assert r["radio_mhz"] == 251.0
  assert r["tacan"]["channel"] == 31


def test_extract_miz_named_contacts_uses_platform_type_for_carrier_aliases():
  mission_text = '''
  ["units"] = {
    [1] = {
      ["type"] = "CVN_71",
      ["name"] = "Naval-1-1",
      ["frequency"] = 127500000,
    },
  }
  '''
  rows = _extract_miz_named_contacts(mission_text)
  assert rows
  r = rows[0]
  assert r["name"] == "Naval-1-1"
  assert r["kind"] == "carrier"
  assert "CVN-71" in r["aliases"]
  assert r["radio_mhz"] == 127.5


def test_find_contact_in_miz_fuzzy_matches_callsign_name():
  contacts = [{
      "name": "Wizard 1-1",
      "kind": "contact",
      "aliases": ["Wizard 1-1", "WZD"],
      "radio_mhz": 251.0,
      "tacan": {"channel": 31, "mode": "Y", "callsign": "WZD"},
  }]
  out = _find_contact_in_miz("what's the frequency for wizurd", contacts)
  assert out is not None
  assert out.get("name") == "Wizard 1-1"


  def test_find_contact_in_miz_prefers_single_carrier_for_generic_carrier_query():
    contacts = [
      {
        "name": "CVN-71",
        "kind": "carrier",
        "aliases": ["CVN-71", "CVN 71", "carrier", "boat"],
        "radio_mhz": 127.5,
        "tacan": {"channel": 71, "mode": "X", "callsign": "CVN"},
      },
      {
        "name": "Wizard 1-1",
        "kind": "contact",
        "aliases": ["Wizard 1-1"],
        "radio_mhz": 251.0,
        "tacan": None,
      },
    ]
    out = _find_contact_in_miz("can you give me the carrier frequency", contacts)
    assert out is not None
    assert out.get("name") == "CVN-71"


    def test_find_contact_in_miz_marks_ambiguous_when_multiple_carriers_exist():
      contacts = [
        {
          "name": "CVN-71",
          "kind": "carrier",
          "aliases": ["CVN-71", "carrier", "boat"],
          "radio_mhz": 127.5,
          "tacan": None,
        },
        {
          "name": "CVN-74",
          "kind": "carrier",
          "aliases": ["CVN-74", "carrier", "boat"],
          "radio_mhz": 128.5,
          "tacan": None,
        },
      ]
      out = _find_contact_in_miz("what's the carrier frequency", contacts)
      assert out is not None
      assert out.get("error") == "ambiguous"


    def test_find_miz_named_contact_returns_platform_type(monkeypatch):
      mission_text = '''
      ["units"] = {
        [1] = {
          ["type"] = "CVN_71",
          ["name"] = "Naval-1-1",
          ["frequency"] = 127500000,
        },
      }
      '''
      monkeypatch.setattr(freq_module, "_read_mission_text_from_miz", lambda miz_path=None: mission_text)
      out = find_miz_named_contact("what type of ship is naval 1-1")
      assert out is not None
      assert out.get("contact") == "Naval-1-1"
      assert out.get("platform_type") == "CVN_71"


    def test_extract_miz_named_contacts_parses_tanker_callsign_table_name():
      mission_text = '''
      ["units"] = {
        [1] = {
          ["type"] = "KC135MPRS",
          ["name"] = "Aerial-8-1",
          ["frequency"] = 251,
          ["callsign"] = {
            ["name"] = "Texaco11",
          },
        },
      }
      '''
      rows = _extract_miz_named_contacts(mission_text)
      assert rows
      r = rows[0]
      assert r["kind"] == "tanker"
      assert r["callsign_name"] == "Texaco11"
      assert "Texaco" in r["aliases"]


def test_extract_miz_named_contacts_discards_non_carrier_alias_noise():
  mission_text = '''
  ["units"] = {
    [1] = {
      ["name"] = "Carrier",
      ["type"] = "TICONDEROG",
      ["frequency"] = 127500000,
    },
    [2] = {
      ["name"] = "Stennis",
      ["type"] = "Turning Point",
      ["frequency"] = 127500000,
    },
    [3] = {
      ["name"] = "Unit #001",
      ["type"] = "Stennis",
      ["frequency"] = 127500000,
    },
  }
  '''
  rows = _extract_miz_named_contacts(mission_text)
  names = {row["name"]: row for row in rows}
  assert "Unit #001" in names
  assert names["Unit #001"]["kind"] == "carrier"
  assert "Carrier" in names
  assert names["Carrier"]["kind"] == "contact"
  assert "Stennis" not in names


def test_extract_miz_named_contacts_keeps_tacan_bound_to_nearest_carrier_block():
  mission_text = '''
  ["units"] = {
    [1] = {
      ["type"] = "CVN_73",
      ["name"] = "Naval-2-1",
      ["frequency"] = 132000000,
      ["route"] = {
        ["points"] = {
          {
            ["task"] = {
              ["id"] = "ComboTask",
              ["params"] = {
                ["tasks"] = {
                  {
                    ["id"] = "WrappedAction",
                    ["params"] = {
                      ["action"] = {
                        ["id"] = "ActivateBeacon",
                        ["params"] = {
                          ["AA"] = true,
                          ["channel"] = 73,
                          ["modeChannel"] = "X",
                          ["callsign"] = "GW"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    [2] = {
      ["type"] = "CVN_71",
      ["name"] = "Naval-1-1",
      ["frequency"] = 127500000,
      ["route"] = {
        ["points"] = {
          {
            ["task"] = {
              ["id"] = "ComboTask",
              ["params"] = {
                ["tasks"] = {
                  {
                    ["id"] = "WrappedAction",
                    ["params"] = {
                      ["action"] = {
                        ["id"] = "ActivateBeacon",
                        ["params"] = {
                          ["AA"] = true,
                          ["channel"] = 71,
                          ["modeChannel"] = "X",
                          ["callsign"] = "GW"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
  }
  '''
  rows = _extract_miz_named_contacts(mission_text)
  by_name = {str(r.get("name")): r for r in rows}
  assert "Naval-1-1" in by_name
  assert "Naval-2-1" in by_name
  assert by_name["Naval-1-1"]["tacan"]["channel"] == 71
  assert by_name["Naval-2-1"]["tacan"]["channel"] == 73


def test_find_contact_resolves_specific_hull_without_disambig():
  """Querying 'CVN-73 freq' with two carriers in the list must resolve directly
  to CVN_73 without triggering the ambiguous-carrier path."""
  contacts = [
    {
      "name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71",
      "aliases": ["carrier", "CVN-71", "CVN 71", "Naval-1", "TR"],
      "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"},
    },
    {
      "name": "Naval-2", "kind": "carrier", "platform_type": "CVN_73",
      "aliases": ["carrier", "CVN-73", "CVN 73", "Naval-2", "GW"],
      "radio_mhz": 132.0, "tacan": {"channel": 73, "mode": "X", "callsign": "GW"},
    },
  ]
  result = _find_contact_in_miz("what is the freq for CVN-73", contacts)
  assert result is not None
  assert result.get("error") is None, f"Got ambiguous error unexpectedly: {result}"
  assert result["platform_type"] == "CVN_73"


def test_find_contact_resolves_carrier_by_ship_name():
  """'what is the freq for the stennis' must resolve to CVN_74 without ambiguity."""
  contacts = [
    {
      "name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71",
      "aliases": ["carrier", "CVN-71", "CVN 71", "Naval-1", "TR",
                  "roosevelt", "teddy", "fdr", "theodore roosevelt"],
      "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"},
    },
    {
      "name": "Naval-2", "kind": "carrier", "platform_type": "CVN_74",
      "aliases": ["carrier", "CVN-74", "CVN 74", "Naval-2", "GW",
                  "stennis", "john c stennis"],
      "radio_mhz": 132.0, "tacan": {"channel": 74, "mode": "X", "callsign": "GW"},
    },
  ]
  result = _find_contact_in_miz("what is the freq for the stennis", contacts)
  assert result is not None
  assert result.get("error") is None, f"Got error: {result}"
  assert result["platform_type"] == "CVN_74"


def test_find_contact_rejects_ship_name_when_hull_not_in_mission():
  """'stennis' must return None (not a false positive on GW/CVN-73) when the
  Stennis hull (CVN_74) is absent from the mission — even if a contact's
  callsign/alias field happens to contain the word 'Stennis'."""
  contacts = [
    {
      "name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71",
      "aliases": ["carrier", "CVN-71", "CVN 71", "Naval-1", "TR",
                  "roosevelt", "teddy", "fdr", "theodore roosevelt"],
      "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"},
    },
    {
      # CVN_73 contact whose callsign field happens to say "Stennis" (DCS quirk)
      "name": "Naval-2", "kind": "carrier", "platform_type": "CVN_73",
      "aliases": ["carrier", "CVN-73", "CVN 73", "Naval-2", "GW",
                  "washington", "george washington", "Stennis"],
      "radio_mhz": 132.0, "tacan": {"channel": 73, "mode": "X", "callsign": "GW"},
    },
  ]
  result = _find_contact_in_miz("what is the freq for the stennis", contacts)
  assert result is None, f"Expected None but got: {result}"


def test_find_contact_resolves_two_char_callsign_alias():
  """'what is the freq for the gw?' — 2-char TACAN callsign alias must match."""
  contacts = [
    {
      "name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71",
      "aliases": ["carrier", "CVN-71", "CVN 71", "Naval-1", "TR",
                  "roosevelt", "teddy", "fdr", "theodore roosevelt"],
      "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"},
    },
    {
      "name": "Naval-2", "kind": "carrier", "platform_type": "CVN_73",
      "aliases": ["carrier", "CVN-73", "CVN 73", "Naval-2", "GW",
                  "washington", "george washington"],
      "radio_mhz": 132.0, "tacan": {"channel": 73, "mode": "X", "callsign": "GW"},
    },
  ]
  result = _find_contact_in_miz("what is the freq for the gw", contacts)
  assert result is not None
  assert result.get("error") is None, f"Got error: {result}"
  assert result["platform_type"] == "CVN_73"


def test_find_contact_resolves_stt_typo_in_ship_name():
  """'george washinton' (STT typo, missing h) must still resolve to CVN_73."""
  contacts = [
    {
      "name": "Naval-1", "kind": "carrier", "platform_type": "CVN_71",
      "aliases": ["carrier", "CVN-71", "CVN 71", "Naval-1", "TR",
                  "roosevelt", "teddy", "fdr", "theodore roosevelt"],
      "radio_mhz": 127.5, "tacan": {"channel": 71, "mode": "X", "callsign": "TR"},
    },
    {
      "name": "Naval-2", "kind": "carrier", "platform_type": "CVN_73",
      "aliases": ["carrier", "CVN-73", "CVN 73", "Naval-2", "GW",
                  "washington", "george washington"],
      "radio_mhz": 132.0, "tacan": {"channel": 73, "mode": "X", "callsign": "GW"},
    },
  ]
  result = _find_contact_in_miz("what is the freq for the george washinton", contacts)
  assert result is not None
  assert result.get("error") is None, f"Got error: {result}"
  assert result["platform_type"] == "CVN_73"


def test_extract_miz_named_contacts_ignores_radio_preset_channel_before_beacon():
  """A radio-preset task with ["channel"] = 1 before the ActivateBeacon params
  must not pollute the TACAN channel extraction."""
  mission_text = '''
  ["units"] = {
    [1] = {
      ["type"] = "CVN_71",
      ["name"] = "Naval-1-1",
      ["frequency"] = 127500000,
      ["route"] = {
        ["points"] = {
          [1] = {
            ["task"] = {
              ["id"] = "ComboTask",
              ["params"] = {
                ["tasks"] = {
                  [1] = {
                    ["id"] = "ActivateACLS",
                    ["params"] = {
                      ["channel"] = 1,
                    }
                  },
                  [2] = {
                    ["id"] = "WrappedAction",
                    ["params"] = {
                      ["action"] = {
                        ["id"] = "ActivateBeacon",
                        ["params"] = {
                          ["AA"] = false,
                          ["channel"] = 71,
                          ["modeChannel"] = "X",
                          ["callsign"] = "TR"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
  }
  '''
  rows = _extract_miz_named_contacts(mission_text)
  assert rows
  r = rows[0]
  assert r["tacan"]["channel"] == 71, f"Expected 71, got {r['tacan']['channel']}"
