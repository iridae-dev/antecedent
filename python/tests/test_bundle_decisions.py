"""Live recomputation of a composition bundle's terminal, through the Python surface.

The fixtures are the Python spelling of the Rust tests that compile each terminal:

* a propagation-chain terminal ``2 * source`` over ``1, 2, 3``
  (``compiler_recomputation_tests`` in ``crates/antecedent/src/analysis/composition.rs``);
* a predictive decision terminal, the enumerated ``P * Q`` decision of
  ``crates/antecedent/tests/composition_facade.rs`` (contract, law and result nodes);
* a parameter-posterior decision terminal over coefficient draws ``1, 2, 3``
  (``crates/antecedent/tests/composition_posterior_decision_bundle.rs``);
* the producer -> chain -> decision path of ``composition_predictive_bundle_tests.rs``, whose
  chain-decision terminal is the paired action-world decision of ``test_chain_decisions.py``.

Each live receipt is exported as a portable artifact and consumed under the bundle and program
identities the consumer retained; another identity is refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent import Execution, decision
from antecedent import composition_bundle as cb
from antecedent import propagation_chain as pc
from antecedent.errors import CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

WRONG = "0" * 64


def _replay(
    receipt: cb.BundleRecomputation, bundle: cb.Bundle, *, expected_program: str | None = None
) -> cb.BundleRecomputation:
    data = receipt.export()
    assert data.expectation is not None
    assert data.expectation.identity == receipt.program_identity
    replayed = require_success(
        cb.consume_bundle_recomputation(
            bytes(data),
            route=receipt.route,
            expected_bundle=bundle.identity,
            expected_program=expected_program or receipt.program_identity,
        )
    )
    assert replayed.replayed and not receipt.replayed
    assert replayed.route == receipt.route
    assert replayed.program_identity == receipt.program_identity
    assert replayed.dependencies == receipt.dependencies
    assert replayed.terminal_identity == receipt.terminal_identity
    assert replayed.decision_fingerprint == receipt.decision_fingerprint
    for wrong in (
        {"expected_bundle": "wrong", "expected_program": receipt.program_identity},
        {"expected_bundle": bundle.identity, "expected_program": WRONG},
    ):
        with pytest.raises(cb.CompositionBundleRefusal) as refused:
            require_success(
                cb.consume_bundle_recomputation(bytes(data), route=receipt.route, **wrong)
            )
        assert refused.value.code == "external_binding_mismatch"
        assert refused.value.detail == "bundle_recomputation.replay_refused"
        assert_registered_refusal(refused.value)
    return replayed


# ----------------------------------------------------------------------------- chain terminal


def _chain() -> pc.PropagationChain:
    source = pc.draws("source", [1.0, 2.0, 3.0], meaning="sampling", alignment="aligned-bundle")
    stage = pc.affine("output", "derived_quantity", ["source"], scale=2.0, shift=0.0)
    return require_success(pc.propagate([source], [stage], output="output"))


def test_a_chain_terminal_is_recomputed_by_its_compiler_and_replayed_portably() -> None:
    chain = _chain()
    bundle = (
        cb.Bundle.builder()
        .add_artifact("propagation_chain", chain.export(), node_id="terminal")
        .build()
    )
    plan = require_success(cb.compile_chain_bundle_recomputation(bundle, "terminal"))
    assert plan.route == "chain"
    assert plan.bundle_identity == bundle.identity
    assert plan.plan_view()["route"] == "BundleRecomputation"
    assert plan.unit_support()["unit"] == "composition.bundle"
    assert plan.unit_support()["supported"] is True
    declared = require_success(
        cb.compile_chain_bundle_recomputation(bundle, "terminal", chain=chain)
    )
    assert declared.program_identity == plan.program_identity

    receipt = require_success(plan.execute(execution=Execution(seed=23)))
    assert receipt.route == "chain"
    assert [dependency.node_id for dependency in receipt.dependencies] == ["terminal"]
    assert receipt.terminal_node == "terminal"
    assert receipt.terminal_identity == chain.identity
    assert receipt.program_identity == plan.program_identity
    assert receipt.bundle_identity == bundle.identity
    _replay(receipt, bundle)

    with pytest.raises(cb.CompositionBundleRefusal) as absent:
        require_success(cb.compile_chain_bundle_recomputation(bundle, "missing"))
    assert absent.value.code == "invalid_argument"
    assert isinstance(absent.value, CausalValueError)


# -------------------------------------------------------------------------- decision terminals


def _decision_bundle(contract: decision.Contract, law: JointDistributionArtifact) -> cb.Bundle:
    result = require_success(contract.evaluate(law))
    return (
        cb.Bundle.builder()
        .add_artifact(
            "decision_contract", contract.export(artifact_id="contract"), node_id="contract"
        )
        .add_artifact("distribution", law.export("law"), node_id="law")
        .add_artifact("decision_result", result.export(artifact_id="result"), node_id="result")
        .connect("contract", "result")
        .connect("law", "result")
        .build()
    )


def test_a_predictive_decision_terminal_is_rescored_live_and_replayed_portably() -> None:
    contract, law = gen.decision_contract(), gen.decision_source()
    bundle = _decision_bundle(contract, law)
    plan = require_success(
        cb.compile_decision_bundle_recomputation(bundle, "result", contract=contract, source=law)
    )
    assert plan.route == "predictive_decision"
    receipt = require_success(plan.execute(execution=Execution(seed=77)))
    assert len(receipt.dependencies) == 3
    assert receipt.terminal_node == "result"
    assert receipt.decision_fingerprint
    _replay(receipt, bundle)
    data = receipt.export()
    with pytest.raises(cb.CompositionBundleRefusal) as other_route:
        require_success(
            cb.consume_bundle_recomputation(
                bytes(data),
                route="posterior_decision",
                expected_bundle=bundle.identity,
                expected_program=receipt.program_identity,
            )
        )
    assert other_route.value.code == "external_binding_mismatch"


def _coefficient(functional: str = "coefficient") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="coefficient",
        variable_name="coefficient",
        role="outcome",
        units="units",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id=functional,
    )


def _posterior_law() -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="parameter_posterior",
        quantities=(_coefficient(),),
        alignment="joint",
        source_id="declared",
        provider_id="fixture",
        rng_id="fixture",
        snapshot_id="one",
        causal_contract_id="model",
    )
    return JointDistributionArtifact(
        identity, np.array([[1.0], [2.0], [3.0]]), calibration="point_only"
    )


def _posterior_contract(functional: str = "coefficient") -> decision.Contract:
    quantity = _coefficient(functional)
    return decision.Contract(
        actions=(
            decision.Action("use", inputs=(quantity,), utility=decision.x(0), kind="policy"),
            decision.Action(
                "baseline", inputs=(quantity,), utility=decision.x(0) * 0.0, kind="policy"
            ),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy="report_only",
    )


def test_a_parameter_posterior_decision_replays_without_becoming_predictive() -> None:
    contract, law = _posterior_contract(), _posterior_law()
    bundle = _decision_bundle(contract, law)
    plan = require_success(
        cb.compile_decision_bundle_recomputation(
            bundle, "result", contract=contract, source=law, carrier="parameter_posterior"
        )
    )
    assert plan.route == "posterior_decision"
    assert plan.plan_view()["route"] == "PosteriorDecisionBundleRecomputation"
    receipt = require_success(plan.execute(execution=Execution(seed=77)))
    assert receipt.route == "posterior_decision"
    _replay(receipt, bundle)

    with pytest.raises(cb.CompositionBundleRefusal) as outcome_risk:
        require_success(
            cb.compile_decision_bundle_recomputation(
                bundle,
                "result",
                contract=_posterior_contract("outcome"),
                source=law,
                carrier="parameter_posterior",
            )
        )
    assert outcome_risk.value.code == "decision_contract_unsatisfied"
    with pytest.raises(cb.CompositionBundleRefusal) as not_predictive:
        require_success(
            cb.compile_decision_bundle_recomputation(
                bundle, "result", contract=contract, source=law
            )
        )
    assert not_predictive.value.code == "decision_contract_unsatisfied"
    assert "predictive outcome law" in str(not_predictive.value)
    with pytest.raises(CausalValueError):
        require_success(
            cb.compile_decision_bundle_recomputation(
                bundle, "result", contract=contract, source=law, carrier="interval"
            )
        )


# ------------------------------------------------------------------ predictive producer paths


def _world(variable: str, name: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=name,
        role="outcome",
        units="count",
        population_id="game-a",
        regime_id=regime,
        horizon=7,
        functional_id="outcome",
    )


def _world_decision() -> pc.ChainDecision:
    worlds = (
        _world("steam:wishlists", "wishlists", "declared-traffic"),
        _world("steam:sales", "sales", "alternative-traffic"),
    )
    shock = [("disturbance", "traffic-world-shock")]
    sources = [
        pc.draws(column, values, meaning="predictive", alignment="joint:game-a", factors=shock)
        for column, values in (("traffic", [1.0, 2.0, 3.0, 4.0]), ("conversion", [2.0] * 4))
    ]
    stages = [
        pc.affine("wishlists", "derived_quantity", ["traffic"]),
        pc.affine("sales", "derived_quantity", ["conversion"]),
    ]
    chain = require_success(
        pc.propagate(
            sources,
            stages,
            output="wishlists",
            reading="predictive",
            output_quantity=worlds[0],
            joint_outputs=[("wishlists", worlds[0]), ("sales", worlds[1])],
        )
    )
    contract = decision.Contract(
        actions=tuple(
            decision.Action(action, inputs=(world,), utility=decision.x(0), kind="intervention")
            for action, world in zip(("promote", "hold"), worlds, strict=True)
        ),
        utility_units="score",
        criterion=decision.Criterion.expected_regret(),
        target_population="game-a",
        horizon=7,
        structural_policy="require_invariant_best_action",
    )
    compiled = require_success(pc.compile_chain_decision(chain, contract, action_worlds=True))
    return require_success(compiled.execute(execution=Execution(seed=31)))


def test_a_chain_decision_terminal_is_embedded_and_a_producer_must_replay_as_its_kind() -> None:
    decided = _world_decision()
    execution = Execution(seed=31)
    builder = cb.Bundle.builder()
    builder.add_predictive_chain_decision(decided, node_id="decision", execution=execution)
    builder.add_artifact("propagation_chain", decided.chain.export(), node_id="chain")
    bundle = builder.connect("chain", "decision").build()
    assert {node.id: node.kind for node in bundle.nodes} == {
        "decision": "predictive_chain_decision",
        "chain": "propagation_chain",
    }

    for kind in cb.PREDICTIVE_FIT_KINDS:
        with pytest.raises(cb.CompositionBundleRefusal) as foreign:
            cb.Bundle.builder().add_predictive_fit(
                kind,
                decided,
                program=decided.program_identity,
                node_id="fit",
                execution=execution,
            )
        assert_registered_refusal(foreign.value)
    not_a_producer_kind: str = "propagation_chain"
    with pytest.raises(CausalValueError):
        cb.Bundle.builder().add_predictive_fit(
            not_a_producer_kind, decided, program=decided.program_identity, node_id="fit"
        )

    for producer in ("fit", "chain"):
        with pytest.raises(cb.CompositionBundleRefusal) as not_a_producer:
            require_success(
                cb.compile_predictive_fit_decision_bundle(
                    bundle, producer=producer, chain="chain", terminal="decision"
                )
            )
        assert not_a_producer.value.code == "invalid_argument"
    with pytest.raises(cb.CompositionBundleRefusal) as undecodable:
        require_success(
            cb.consume_bundle_recomputation(
                b"\xa0",
                route="predictive_fit_decision",
                expected_bundle=bundle.identity,
                expected_program=WRONG,
            )
        )
    assert_registered_refusal(undecodable.value)
