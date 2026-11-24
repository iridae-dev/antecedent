"""A finite transport scenario set through its compiled program.

``compile_transport_scenarios`` decides the set exactly as ``prepare_transport_scenarios`` does
and binds it into the ``TransportScenarioSet`` program; executing the program reports the same
scenarios, statuses, points, envelope and masses as the prepared stage's ``estimate()``, before
and after a refresh, and exports the same artifact. The independent check is the hand-computed
standardized and direct transport formulas of ``test_transport_scenarios`` (0.35 and 0.56).
"""

from __future__ import annotations

import json
import re

import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport
from antecedent.transport._scenarios import (
    CompiledTransportScenarios,
    compile_transport_scenarios,
)

from test_transport_scenarios import (
    DIRECT,
    STANDARDIZED,
    by_name,
    catalog,
    laws,
    mean,
    prepare,
    scenarios,
)

REFRESHED_SOURCE = (0.1, 0.3, 0.3, 0.3)


def compile_(sets=None, data=None) -> CompiledTransportScenarios:
    return compile_transport_scenarios(
        sets or scenarios(),
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=catalog(),
        laws=data or laws(),
        at={"x": 1.0},
    )


def test_the_compiled_set_reports_what_the_prepared_stage_estimates():
    compiled = compile_()
    report = compiled.execute()
    assert report == json.loads(require_success(prepare().estimate()))
    assert compiled.route == "TransportScenarioSet"
    assert re.fullmatch(r"[0-9a-f]{64}", compiled.program_identity)
    s = by_name(report)
    assert s["standardize"]["status"] == s["direct"]["status"] == "identified"
    assert mean(s["standardize"]) == pytest.approx(STANDARDIZED, abs=1e-12)
    assert mean(s["direct"]) == pytest.approx(DIRECT, abs=1e-12)
    assert s["outcome_shift"]["status"] == "structurally_unidentified"
    assert compiled.plan_summary() == [
        (str(name), str(plan)) for name, plan in prepare().plan_summary()
    ]


def test_a_refresh_compiles_a_new_program_with_the_stages_refreshed_points():
    compiled = compile_()
    before = compiled.execute()
    refreshed = compiled.refresh(laws(source=REFRESHED_SOURCE))
    assert refreshed.program_identity != compiled.program_identity
    after = refreshed.execute()
    stage = prepare()
    require_success(stage.refresh(laws(source=REFRESHED_SOURCE)))
    assert after == json.loads(require_success(stage.estimate()))
    assert [s["status"] for s in after["scenarios"]] == [s["status"] for s in before["scenarios"]]
    assert mean(by_name(after)["direct"]) == pytest.approx(0.6)
    # The original program is unchanged by the refresh.
    assert compiled.execute() == before


def test_the_program_identity_repeats_and_binds_the_declared_weights():
    first = compile_()
    assert compile_().program_identity == first.program_identity
    weighted = compile_(scenarios((0.3, 0.2, 0.4)))
    assert weighted.program_identity != first.program_identity


def test_the_compiled_export_is_the_stages_artifact_and_replays():
    compiled = compile_(scenarios((0.3, 0.2, 0.4)))
    report = compiled.execute()
    artifact = compiled.export()
    stage = prepare(scenarios((0.3, 0.2, 0.4)))
    require_success(stage.estimate())
    assert artifact == bytes(stage.export())
    consumed = transport.consume_transport_scenarios_artifact(artifact).to_dict()
    assert consumed["scenarios"] == report["scenarios"]


def test_export_before_execute_and_an_aggregate_interval_are_refused():
    compiled = compile_()
    with pytest.raises(Exception, match="no_execution_claim"):
        compiled.export()
    with pytest.raises(CausalUnsupportedError) as refused:
        compiled.aggregate_interval()
    with pytest.raises(CausalUnsupportedError) as stage_refused:
        prepare().aggregate_interval()
    assert type(refused.value) is type(stage_refused.value)
    assert str(refused.value) == str(stage_refused.value)
    assert refused.value.reason_code == "scenario_aggregate_not_supported"
