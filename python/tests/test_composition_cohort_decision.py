"""The cohort-to-affine bundle on the Rust reader's inherited compact fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent import cohort_decision as cd
from antecedent import composition_bundle as cb
from antecedent import decision
from antecedent.compact_export import CompactExport
from antecedent.execution import Execution
from antecedent.outcomes import require_success

FIXTURES = (
    Path(__file__).resolve().parents[2] / "crates" / "antecedent-io-core" / "tests" / "fixtures"
)
SOURCE_IDENTITY = "6fd024b0c15c6416187582f952b3c6c0cdb1ed8ebca01b8851209a50eb82140d"
MODEL_IDENTITY = "83c829a5e3751dbc49b6f7d29098d9a4445a797c9902560307f70c8cfa7de29f"


def _source() -> cd.CohortDecisionSource:
    return cd.consume_cohort_decision_source(
        (FIXTURES / "cohort_decision_source_v1.bin").read_bytes(),
        expected_identity=SOURCE_IDENTITY,
    )


def _reweighted() -> cd.CohortDecisionSource:
    model = require_success(
        CompactExport.consume(
            (FIXTURES / "compact_runtime_export_v1.bin").read_bytes(),
            expected_identity=MODEL_IDENTITY,
        )
    )
    return cd.decision_source_from_compact_export(
        model,
        cohort=cd.CohortBinding(
            snapshot_id="clinic-2026-q1",
            population="target",
            row_ids=["r0", "r1", "r2"],
            columns={"z": [-1.0, 0.5, 1.0]},
            weights=[0.8, 0.1, 0.1],
        ),
        outcome_units="mmHg",
        artifact_id="reweighted-cohort",
    )


def _contract(source: cd.CohortDecisionSource) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("treat", inputs=(source.quantity,), utility=decision.x(0) * 4.0 - 8.0),
            decision.Action("wait", inputs=(source.quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _bundle(
    source: cd.CohortDecisionSource, contract: decision.Contract, result: bytes, *, connect=True
) -> cb.Bundle:
    builder = (
        cb.Bundle.builder()
        .add_artifact(source, node_id="cohort")
        .add_artifact(contract.export(artifact_id="contract"), node_id="contract")
        .add_artifact(result, node_id="decision")
    )
    if connect:
        builder.connect("cohort", "decision").connect("contract", "decision")
    return builder.build()


def test_affine_cohort_bundle_replays_in_a_fresh_process_and_reweights(tmp_path: Path) -> None:
    receipts = []
    for source, mean, utility, selected in [
        (_source(), 2.225, 0.9, ("treat",)),
        (_reweighted(), 1.675, -1.3, ("wait",)),
    ]:
        contract = _contract(source)
        direct = require_success(contract.evaluate(source))
        assert direct.selected == selected
        assert direct.actions[0].expected_utility == pytest.approx(utility)
        assert direct.actions[0].standard_error is None and direct.evpi is None
        bundle = _bundle(
            source, contract, cb.cohort_decision(contract, source, artifact_id="decision")
        )
        raw = require_success(
            cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
        )
        assert raw.all_replayed
        assert raw.value("cohort", "mean_contrast") == pytest.approx(mean)
        assert raw.value("decision", "treat.expected_utility") == pytest.approx(utility)
        plan = require_success(cb.compile_cohort_decision_bundle_recomputation(bundle, "decision"))
        assert plan.route == "cohort_decision"
        run = require_success(plan.execute(execution=Execution(seed=7)))
        assert len(run.dependencies) == 3
        portable = tmp_path / f"{selected[0]}-bundle.bin"
        portable.write_bytes(bytes(run.export()))
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json, pathlib, sys
from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success
run = require_success(cb.consume_bundle_recomputation(
    pathlib.Path(sys.argv[1]).read_bytes(), route="cohort_decision",
    expected_bundle=sys.argv[2], expected_program=sys.argv[3],
))
print(json.dumps([run.replayed, run.decision_fingerprint, run.program_identity]))
""",
                str(portable),
                bundle.identity,
                plan.program_identity,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(done.stdout) == [True, run.decision_fingerprint, plan.program_identity]
        receipts.append(run)
    assert receipts[0].program_identity != receipts[1].program_identity
    assert receipts[0].decision_fingerprint != receipts[1].decision_fingerprint


def test_cohort_bundle_refuses_swapped_source_and_nonaffine_contract() -> None:
    source = _source()
    contract = _contract(source)
    result = cb.cohort_decision(contract, source)
    original = _bundle(source, contract, result)
    changed = _bundle(_reweighted(), contract, result)
    with pytest.raises(cb.CompositionBundleRefusal):
        require_success(
            cb.consume_bundle(bytes(changed.export()), expected_identity=original.identity)
        )
    plan = require_success(cb.compile_cohort_decision_bundle_recomputation(changed, "decision"))
    with pytest.raises(cb.TamperedQuantityRefusal):
        require_success(plan.execute())
    detached = _bundle(source, contract, result, connect=False)
    with pytest.raises(cb.CompositionBundleRefusal):
        require_success(cb.compile_cohort_decision_bundle_recomputation(detached, "decision"))
    nonlinear = replace(
        contract,
        actions=(
            decision.Action(
                "treat", inputs=(source.quantity,), utility=decision.x(0) * decision.x(0)
            ),
            contract.actions[1],
        ),
    )
    with pytest.raises(cb.UnsupportedLawRefusal):
        cb.cohort_decision(nonlinear, source)


def test_compact_model_refits_original_inputs_from_standalone_and_bundle(tmp_path: Path) -> None:
    features = {"z": [-1.0, 0.0, 1.0] * 4}
    treatment = [0.0] * 6 + [1.0] * 6
    outcome = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.5, 5.0, 4.0, 5.5, 7.0]
    fit = cd.fit_native_cohort_effect(
        outcome=outcome,
        treatment=treatment,
        features=features,
        outcome_name="y",
        treatment_name="t",
        outcome_units="mmHg",
        population="original",
    )
    learner = "ols_hc3_treatment_by_basis"
    refitted = cb.refit_compact_model(
        fit, expected_identity=fit.identity, learner=learner,
        outcome=outcome, treatment=treatment, features=features,
    )
    assert refitted.identity == fit.identity
    assert refitted.evaluate({"z": 0.5}).point == pytest.approx(2.75)

    bundle = cb.BundleBuilder().add_artifact(fit, node_id="native-model").build()
    checked = cb.refit_compact_model(
        bundle, expected_identity=bundle.identity, learner=learner,
        outcome=outcome, treatment=treatment, features=features, model_node="native-model",
    )
    assert checked.identity == fit.identity

    path = tmp_path / "compact-model-bundle.ante"
    path.write_bytes(bytes(bundle.export()))
    child = subprocess.run(
        [
            sys.executable, "-c",
            """import json, sys
from pathlib import Path
from antecedent.composition_bundle import refit_compact_model
inputs = json.loads(sys.stdin.read())
checked = refit_compact_model(
    Path(sys.argv[1]).read_bytes(), expected_identity=sys.argv[2],
    model_node="native-model", **inputs,
)
print(json.dumps([checked.identity, checked.evaluate({"z": 0.5}).point]))""",
            str(path), bundle.identity,
        ],
        input=json.dumps({
            "learner": learner, "outcome": outcome, "treatment": treatment, "features": features,
        }),
        text=True, capture_output=True, check=True,
    )
    identity, point = json.loads(child.stdout)
    assert identity == fit.identity
    assert point == pytest.approx(2.75)

    revised = list(outcome)
    revised[0], revised[3] = revised[3], revised[0]
    with pytest.raises(cb.CompositionBundleRefusal) as error:
        cb.refit_compact_model(
            fit, expected_identity=fit.identity, learner=learner,
            outcome=revised, treatment=treatment, features=features,
        )
    assert error.value.detail == "composition_bundle.graph_or_snapshot_mismatch"
    with pytest.raises(cb.CompositionBundleRefusal) as error:
        cb.refit_compact_model(
            fit, expected_identity=fit.identity, learner="another_learner",
            outcome=outcome, treatment=treatment, features=features,
        )
    assert error.value.detail == "composition_bundle.provider_request_changed"

    legacy = CompactExport.consume(
        (FIXTURES / "compact_runtime_export_v1.bin").read_bytes(),
        expected_identity=MODEL_IDENTITY,
    )
    with pytest.raises(cb.CompositionBundleRefusal) as error:
        cb.refit_compact_model(
            legacy, expected_identity=legacy.identity, learner=learner,
            outcome=outcome, treatment=treatment, features=features,
        )
    assert error.value.detail == "composition_bundle.callback_unavailable"
