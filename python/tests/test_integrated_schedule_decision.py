"""Exercise the integrated decision across a fresh Python process."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent import load
from antecedent.artifacts import artifact_identity
from antecedent.execution import Execution, ResourceLimits
from antecedent.intervention_schedule_decision import (
    ScheduleDecisionDeclaration,
    ScheduleDecisionRefusal,
    ScheduleSourceSubject,
    compile_integrated_schedule_decision,
    compile_schedule_decision_bundle,
    consume_schedule_decision_bundle,
)
from antecedent.outcomes import require_success


def test_schedule_decision_joint_scope_and_portable_replay(tmp_path: Path) -> None:
    endpoints = ((0.125, 0.25, 0.5, 0.875), (0.25, 0.375, 0.625, 1.0))
    early = (0.5, 0.75)
    late = ((0.25, 0.75), (0.5, 0.25))
    retention = ((0.875, 0.75, 0.75, 0.875), (0.75, 0.875, 0.875, 0.75))
    source = []
    for x, strata in enumerate(endpoints):
        for code, value in enumerate(strata):
            first = early[x] if code >= 2 else 1 - early[x]
            second = late[x][code // 2] if code % 2 else 1 - late[x][code // 2]
            assigned = int(8192 * first * second)
            observed = int(assigned * retention[x][code])
            source.extend(
                ScheduleSourceSubject(
                    bool(x), (code >= 2, bool(code % 2)), value if j < observed else None
                )
                for j in range(assigned)
            )
    target = [False, True, True, True] * 256
    declaration = ScheduleDecisionDeclaration(
        (0, 1, 2, 3),
        ("both_actions_causal", "first_action_only"),
        ((0.0, 0.0625, 0.125, 0.1875), (0.0, 0.1875, 0.460_937_5, 0.460_937_5)),
    )
    compiled = require_success(compile_integrated_schedule_decision(source, target, declaration))
    assert compiled.plan_view()["route"] == "IntegratedScheduleDecision"
    source_program, target_program, declaration_program = compiled.child_program_identities
    assert len({source_program, target_program, declaration_program}) == 3
    assert all(len(identity) == 64 for identity in compiled.child_program_identities)
    limited_execution = Execution(limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(ScheduleDecisionRefusal) as stopped:
        require_success(compiled.execute(execution=limited_execution))
    assert stopped.value.detail == "integrated_schedule.resource_refused"
    built = require_success(compiled.execute())
    assert built.child_program_identities == compiled.child_program_identities
    assert built.decision.minimum_coverage == 0.95
    assert built.decision.target_prevalence == 0.75
    assert built.plan["route"] == "IntegratedScheduleDecision"
    assert built.plan["depth"] == "complete"
    assert built.decision.optimum_set == (0, 2, 3)
    assert built.decision.first_assignment == early
    assert built.decision.second_assignment == late
    assert built.decision.observation_survival == retention
    assert built.decision.calibration.status == "not_measured"
    assert "entire optimum set" in built.decision.coverage_scope
    exported = built.export()
    assert exported.expectation is not None
    assert exported.expectation.identity == built.identity
    report = require_success(load(bytes(exported), expected=exported.expectation))
    assert report.identity == built.identity
    assert report.all_replayed
    terminal = next(node for node in report.nodes if node.kind == "integrated_schedule_decision")
    assert terminal.facts["optimum_set"] == "0,2,3"
    assert terminal.facts["coverage_scope"] == built.decision.coverage_scope
    assert terminal.facts["calibration_status"] == built.decision.calibration.status
    assert (
        consume_schedule_decision_bundle(
            exported,
            expected_identity=exported.expectation,
            retained_identity=artifact_identity(exported),
        )
        == built.decision
    )
    artifact = tmp_path / "integrated-schedule.aio"
    artifact.write_bytes(built.data)
    checked = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
import sys
from antecedent.intervention_schedule_decision import consume_schedule_decision_bundle
result = consume_schedule_decision_bundle(Path(sys.argv[1]).read_bytes(), expected_identity=sys.argv[2])
print(json.dumps([result.optimum_set, result.minimum_coverage]))
""",
            str(artifact),
            built.identity,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(checked.stdout) == [[0, 2, 3], 0.95]

    changed = compile_schedule_decision_bundle(source, [False] * len(target), declaration)
    assert changed.child_program_identities[0] == source_program
    assert changed.child_program_identities[1] != target_program
    assert changed.child_program_identities[2] == declaration_program
    assert changed.identity != built.identity
    assert (
        changed.decision.scenarios[0].point_utilities != built.decision.scenarios[0].point_utilities
    )
    with pytest.raises(ScheduleDecisionRefusal, match="retained identity") as mismatch:
        consume_schedule_decision_bundle(changed.data, expected_identity=built.identity)
    assert mismatch.value.detail == "integrated_schedule.artifact_refused"
    assert load(changed.data, expected=exported.expectation).refusal is not None
    with pytest.raises(ScheduleDecisionRefusal) as limited:
        consume_schedule_decision_bundle(
            built.data,
            expected_identity=built.identity,
            execution=Execution(limits=ResourceLimits(memory_bytes=1)),
        )
    assert limited.value.detail == "integrated_schedule.resource_refused"


def test_missing_joint_evidence_explains_what_to_supply() -> None:
    incomplete = ScheduleDecisionDeclaration(
        (0, 1),
        ("both_actions_causal",),
        (),
    )
    with pytest.raises(
        ScheduleDecisionRefusal,
        match="both structural alternatives and finite joint preference costs",
    ) as missing:
        compile_schedule_decision_bundle([], [], incomplete)
    assert missing.value.detail == "integrated_schedule.missing_joint_evidence"
