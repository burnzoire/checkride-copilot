"""F/A-18C engine start diagnostic rules."""

from diagnostic.engine import DiagnosticRule

START_ENGINE_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        condition_id="battery_on",
        description="Battery switch must be ON before engine start",
        check_field="battery_switch",
        required_value="ON",
        comparator="eq",
        fix_instruction="Set battery switch to ON on the left console.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="apu_or_ext_power",
        description="APU or external power required for JFS engagement",
        check_field="apu_running",
        required_value=True,
        comparator="eq",
        fix_instruction="Start the APU — press APU START on the left console and wait for stabilization.",
        severity="blocking",
    ),
    DiagnosticRule(
        condition_id="engine_not_already_running",
        description="Engine is already running",
        check_field="engine_left_state",
        required_value="off",
        comparator="eq",
        fix_instruction="Left engine is already running — no start needed.",
        severity="warning",
    ),
]
