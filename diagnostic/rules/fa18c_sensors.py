"""F/A-18C sensor employment diagnostic rules (TGP, laser, radar)."""

from diagnostic.engine import DiagnosticRule

TGP_TRACK_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="tgp_powered",
        description="TGP must be powered",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP on the right DDI.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="tgp_tdc_priority",
        description="TGP DDI must have TDC priority",
        check_field="tgp_tdc_priority",
        required_value=True,
        comparator="eq",
        fix_instruction="Slew TDC to the TGP DDI to give it priority.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="tgp_not_stby",
        description="TGP must not be in STBY",
        check_field="tgp_mode",
        required_value="STBY",
        comparator="ne",
        fix_instruction="TGP is in STBY — switch to A/G mode on the TGP DDI.",
        severity="blocking",
    ),
]

LASE_TARGET_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="tgp_powered_lase",
        description="TGP must be powered to lase",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="laser_armed",
        description="Laser must be armed",
        check_field="laser_armed",
        required_value=True,
        comparator="eq",
        fix_instruction="Arm the laser on the TGP DDI.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="tgp_tracking_lase",
        description="TGP must be tracking a target to lase it",
        check_field="tgp_tracking",
        required_value=True,
        comparator="eq",
        fix_instruction="Designate a target in the TGP before lasing.",
        severity="blocking",
    ),
]
