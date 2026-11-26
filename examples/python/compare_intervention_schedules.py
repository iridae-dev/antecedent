"""Choose two-period schedules when an action's effect and its costs are uncertain."""

from dataclasses import replace

from antecedent.intervention_schedule_decision import (
    ScheduleDecisionDeclaration,
    ScheduleDecisionRefusal,
    ScheduleSourceSubject,
    compile_schedule_decision_bundle,
    consume_schedule_decision_bundle,
)


def main() -> None:
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
                    bool(baseline), (bool(first), bool(second)), outcome if i < observed else None
                )
                for i in range(assigned)
            )
    target = [False, True, True, True] * 256
    declaration = ScheduleDecisionDeclaration(
        schedules=(0, 1, 2, 3),
        structures=("both_actions_causal", "first_action_only"),
        preference_costs=((0.0, 0.0625, 0.125, 0.1875), (0.0, 0.1875, 0.4609375, 0.4609375)),
    )
    run = compile_schedule_decision_bundle(source, target, declaration)
    portable = run.export()
    assert portable.expectation is not None
    replay = consume_schedule_decision_bundle(portable, expected_identity=portable.expectation)
    assert replay.optimum_set == (0, 2, 3)
    print("Compatible schedules:", [format(code, "02b") for code in replay.optimum_set])
    print("Coverage scope:", replay.coverage_scope)
    print("Calibration:", replay.calibration.status)
    try:
        compile_schedule_decision_bundle(source, target, replace(declaration, preference_costs=()))
    except ScheduleDecisionRefusal as error:
        assert error.detail == "integrated_schedule.missing_joint_evidence"
        print("With preference evidence removed:", error)
    else:
        raise AssertionError("a schedule decision requires the joint preference costs")


if __name__ == "__main__":
    main()
