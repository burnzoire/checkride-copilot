"""F/A-18C weapon employment diagnostic rules."""

from diagnostic.engine import DiagnosticRule

FIRE_AIM120_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm switch must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm switch to ARM on the left console.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="aim120_selected",
        description="AIM-120C must be the selected weapon on SMS",
        check_field="selected_weapon",
        required_value="AIM-120C",
        comparator="eq",
        fix_instruction="Select AIM-120C on the SMS DDI page.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="radar_on",
        description="Radar must be powered and not in STBY or OFF",
        check_field="radar_mode",
        required_value=["OFF", "STBY"],
        comparator="not_in",
        fix_instruction="Power the radar — select RWS, TWS, or STT mode.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="radar_tracking_mode",
        description="Radar should be in STT or TWS for active AMRAAM guidance",
        check_field="radar_mode",
        required_value=["STT", "TWS"],
        comparator="in",
        fix_instruction="Radar is in RWS — lock a target in STT or use TWS for pitbull AMRAAM.",
        severity="warning",
    ),
]

FIRE_AIM9X_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm switch must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm switch to ARM on the left console.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="aim9x_selected",
        description="AIM-9X must be the selected weapon",
        check_field="selected_weapon",
        required_value="AIM-9X",
        comparator="eq",
        fix_instruction="Select AIM-9X on the SMS DDI page.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="hmd_for_off_boresight",
        description="JHMCS should be enabled for off-boresight AIM-9X shots",
        check_field="hmd_enabled",
        required_value=True,
        comparator="eq",
        fix_instruction="Enable JHMCS for off-boresight cuing — or use bore for within-gimbal shots.",
        severity="warning",
    ),
]

RELEASE_GBU12_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="master_arm_armed",
        description="Master Arm must be ARM",
        check_field="master_arm",
        required_value="ARM",
        comparator="eq",
        fix_instruction="Set Master Arm to ARM.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="gbu12_selected",
        description="GBU-12 must be selected on SMS",
        check_field="selected_weapon",
        required_value="GBU-12",
        comparator="eq",
        fix_instruction="Select GBU-12 on the SMS DDI page.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="tgp_powered_lgb",
        description="TGP must be powered for laser guidance",
        check_field="tgp_powered",
        required_value=True,
        comparator="eq",
        fix_instruction="Power the TGP on the right DDI.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="tgp_tracking_lgb",
        description="TGP must have a tracked designation",
        check_field="tgp_tracking",
        required_value=True,
        comparator="eq",
        fix_instruction="Designate a target in the TGP DDI before releasing.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="laser_armed_lgb",
        description="Laser should be armed before the release point",
        check_field="laser_armed",
        required_value=True,
        comparator="eq",
        fix_instruction="Arm the laser on the TGP DDI — fire laser before impact.",
        severity="warning",
    ),
    DiagnosticRule(
        condition_id="safe_release_altitude",
        description="Altitude should be above minimum safe release altitude",
        check_field="altitude_ft",
        required_value=1500,
        comparator="gte",
        fix_instruction="You may be below safe release altitude for GBU-12 fuze arming.",
        severity="warning",
    ),
]
