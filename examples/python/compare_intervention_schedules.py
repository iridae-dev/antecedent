"""Choose two-period schedules when an action's effect and its costs are uncertain.

Pass a filename to save the bundle. Retain the printed identity separately and
replay with `--read FILE IDENTITY` in either the Rust or Python example.
"""

import sys
from dataclasses import replace
from pathlib import Path

from antecedent.intervention_schedule_decision import (
    ScheduleDecisionDeclaration,
    ScheduleDecisionRefusal,
    ScheduleSourceSubject,
    compile_schedule_decision_bundle,
    consume_schedule_decision_bundle,
)


def read(raw: bytes, identity: str) -> None:
    replay = consume_schedule_decision_bundle(raw, expected_identity=identity)
    assert replay.optimum_set == (0, 2, 3)
    print("Compatible schedules:", [format(code, "02b") for code in replay.optimum_set])
    print("Coverage scope:", replay.coverage_scope)
    print("Calibration:", replay.calibration.status)


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--read":
        read(Path(sys.argv[2]).read_bytes(), sys.argv[3])
        return
    if len(sys.argv) > 2:
        raise SystemExit("expected FILE or --read FILE IDENTITY")
    # In each source stratum the late action depends on the first action. Some
    # endpoints are missing. The target has a different mix of baseline strata.
    endpoints = ((0.125, 0.25, 0.5, 0.875), (0.25, 0.375, 0.625, 1.0))
    early = (0.5, 0.75)
    late = ((0.25, 0.75), (0.5, 0.25))
    retention = ((0.875, 0.75, 0.75, 0.875), (0.75, 0.875, 0.875, 0.75))
    source = []
    for baseline, outcomes in enumerate(endpoints):
        for schedule, outcome in enumerate(outcomes):
            first, second = divmod(schedule, 2)
            p = early[baseline]
            q = late[baseline][first]
            assigned = int(8192 * (p if first else 1 - p) * (q if second else 1 - q))
            observed = int(assigned * retention[baseline][schedule])
            source.extend(
                ScheduleSourceSubject(
                    bool(baseline),
                    (bool(first), bool(second)),
                    outcome if i < observed else None,
                )
                for i in range(assigned)
            )
    target = [False, True, True, True] * 256
    declaration = ScheduleDecisionDeclaration(
        schedules=(0, 1, 2, 3),
        structures=("both_actions_causal", "first_action_only"),
        preference_costs=(
            (0.0, 0.0625, 0.125, 0.1875),
            (0.0, 0.1875, 0.4609375, 0.4609375),
        ),
    )
    run = compile_schedule_decision_bundle(source, target, declaration)
    portable = run.export()
    read(bytes(portable), run.identity)
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_bytes(bytes(portable))
        print("Retain this identity:", run.identity)
    try:
        compile_schedule_decision_bundle(source, target, replace(declaration, preference_costs=()))
    except ScheduleDecisionRefusal as error:
        assert error.detail == "integrated_schedule.missing_joint_evidence"
        print("With preference evidence removed:", error)
    else:
        raise AssertionError("a schedule decision requires the joint preference costs")


if __name__ == "__main__":
    main()
