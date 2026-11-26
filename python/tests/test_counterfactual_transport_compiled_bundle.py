"""Exact and sampled target-event points through the public bundle compiler."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent import Execution, ResourceLimits, decision
from antecedent import _reason_details as reason_details
from antecedent import composition_bundle as cb
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

from test_producer_consumer_matrix import _transported


@pytest.mark.parametrize("sampled", [False, True])
def test_transport_bundle_compiles_and_replays_affine_decision(
    sampled: bool, tmp_path: Path
) -> None:
    original = _transported()
    replacement = _transported("revised-target")
    source = original.sampled_treated if sampled else original.treated
    revised = replacement.sampled_treated if sampled else replacement.treated
    facts = cb.describe_artifact("auto", source).facts
    quantity = ScientificQuantity.from_wire(json.loads(facts["quantity"]))
    contract = decision.Contract(
        actions=(
            decision.Action("act", inputs=(quantity,), utility=decision.x(0) * 4.0 - 1.0),
            decision.Action("hold", inputs=(quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population=quantity.population_id,
    )
    terminal = cb.mean_decision(contract, source)

    def build(parent: object, *, connect_contract: bool = True) -> cb.Bundle:
        builder = (
            cb.Bundle.builder()
            .add_artifact(parent, node_id="transport")
            .add_artifact(contract, node_id="contract")
            .add_artifact(terminal, node_id="decision")
            .connect("transport", "decision")
        )
        if connect_contract:
            builder.connect("contract", "decision")
        return builder.build()

    bundle = build(source)
    plan = require_success(
        cb.compile_counterfactual_transport_decision_bundle_recomputation(bundle, "decision")
    )
    assert plan.plan_view()["route"] == "CounterfactualTransportDecisionBundleRecomputation"
    result = require_success(plan.execute())
    assert result.program_identity == plan.program_identity
    assert result.bundle_identity == bundle.identity
    assert {node.node_id for node in result.dependencies} == {"transport", "contract", "decision"}
    graph = require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )
    assert graph.value("decision", "act.expected_utility") == pytest.approx(2.0, abs=1e-12, rel=0.0)

    portable = bytes(result.export())

    def replay(*, execution: Execution | None = None) -> cb.BundleRecomputation:
        return require_success(
            cb.consume_bundle_recomputation(
                portable,
                expected_bundle=bundle.identity,
                expected_program=plan.program_identity,
                route=result.route,
                execution=execution,
            )
        )

    replayed = replay()
    assert replayed.replayed
    assert replayed.decision_fingerprint == result.decision_fingerprint
    assert replayed.dependencies == result.dependencies

    path = tmp_path / "transport-affine-receipt.bin"
    path.write_bytes(portable)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
import sys

from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success

result = require_success(cb.consume_bundle_recomputation(
    Path(sys.argv[1]).read_bytes(),
    expected_bundle=sys.argv[2],
    expected_program=sys.argv[3],
    route="counterfactual_transport_decision",
))
print(json.dumps([result.replayed, result.program_identity, result.decision_fingerprint]))
""",
            str(path),
            bundle.identity,
            plan.program_identity,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(child.stdout) == [True, plan.program_identity, result.decision_fingerprint]

    limited = Execution(limits=ResourceLimits(memory_bytes=1))
    for run in (
        lambda: require_success(plan.execute(execution=limited)),
        lambda: replay(execution=limited),
    ):
        with pytest.raises(cb.CompositionBundleRefusal) as refused:
            run()
        assert refused.value.detail == reason_details.composition_bundle.RESOURCE_REFUSED

    changed = require_success(
        cb.compile_counterfactual_transport_decision_bundle_recomputation(
            build(revised), "decision"
        )
    )
    assert changed.program_identity != plan.program_identity
    with pytest.raises(cb.CompositionBundleRefusal):
        require_success(changed.execute())

    with pytest.raises(cb.CompositionBundleRefusal) as missing:
        require_success(
            cb.compile_counterfactual_transport_decision_bundle_recomputation(
                build(source, connect_contract=False), "decision"
            )
        )
    assert missing.value.detail == reason_details.composition_bundle.GRAPH_OR_SNAPSHOT_MISMATCH

    with pytest.raises(cb.CompositionBundleRefusal) as refusal:
        cb.mean_decision(replace(contract, criterion=decision.Criterion.quantile(0.5)), source)
    assert refusal.value.detail == reason_details.composition_bundle.UNSUPPORTED_LAW
