from diagnostic.rules.fa18c_weapons import FIRE_AIM120_RULES, RELEASE_GBU12_RULES, FIRE_AIM9X_RULES
from diagnostic.rules.fa18c_sensors import TGP_TRACK_RULES, LASE_TARGET_RULES
from diagnostic.rules.fa18c_startup import START_ENGINE_RULES
from diagnostic.engine import DiagnosticRule

_ACTION_RULE_MAP: dict[str, list[DiagnosticRule]] = {
    "fire_aim120":   FIRE_AIM120_RULES,
    "fire_aim9x":    FIRE_AIM9X_RULES,
    "release_gbu12": RELEASE_GBU12_RULES,
    "tgp_track":     TGP_TRACK_RULES,
    "lase_target":   LASE_TARGET_RULES,
    "start_engine":  START_ENGINE_RULES,
}


def get_rules_for_action(action_id: str) -> list[DiagnosticRule]:
    return _ACTION_RULE_MAP.get(action_id, [])
