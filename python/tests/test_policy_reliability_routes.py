"""Compiled policy programs and their independent replay.

* An observational policy value is sealed with its complete request and replayed by
  recompilation under its split seed, with and without a declared causal graph.
* A graph-checked cross-fitted policy learner acting on a strict subset of its adjustment
  columns is refitted on 100 subsamples of 150 of its 300 rows, sealed and replayed.
* Two fixed one-step sequential policies are evaluated by common-stream rollouts and by a
  whole-trajectory bootstrap of a logged behaviour policy, and the decision among them is
  assessed and replayed from the raw declaration.
* The same producer, its decision draws and its diagnostic are embedded in a composition bundle,
  recomputed, sealed and replayed under the retained bundle and program identities.

The data are a normal confounder ``z`` driving treatment ``a`` and outcome ``y``, with a noise
column ``w`` that affects only the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution, observational_policy
from antecedent import composition_bundle as cb
from antecedent import decision_reliability as dr
from antecedent import sequential_evaluation as se
from antecedent.errors import CausalError, CausalValueError
from antecedent.outcomes import require_success

WRONG_IDENTITY = "0" * 64
GRAPH = [("z", "a"), ("z", "y"), ("a", "y")]
NARROW_GRAPH = [("z", "a"), ("z", "y"), ("w", "y"), ("a", "y")]


def _draw(n: int, seed: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    # Propensities stay inside [0.1, 0.9] so every scored row clears the declared positivity floor.
    a = (rng.uniform(size=n) < 0.1 + 0.8 / (1 + np.exp(-z))).astype(float)
    y = 1.0 + z + 1.5 * z * a + 0.5 * rng.normal(size=n)
    return y, a, z


def _held_out(**graph: object) -> observational_policy.ObservationalPolicyValue:
    y, a, z = _draw(1200)
    return observational_policy.policy_value(
        y,
        a,
        support=[0.0, 1.0],
        objective="maximize",
        evaluation="held_out",
        evaluation_fraction=0.4,
        adjust={"z": z},
        execution=Execution(seed=2),
        **graph,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- policy value


@pytest.mark.parametrize("checked", [False, True])
def test_a_policy_value_is_sealed_and_replayed_by_recompilation(checked: bool) -> None:
    graph = {"graph": GRAPH, "treatment_name": "a", "outcome_name": "y"} if checked else {}
    value = _held_out(**graph)
    assert value.calibration == "not_measured"
    assert any("declared causally sufficient DAG" in text for text in value.assumptions) is checked

    artifact = value.export()
    assert artifact.expectation is not None
    identity = artifact.expectation.identity
    replayed = observational_policy.consume_compiled_observational_policy(
        artifact, expected_identity=identity, execution=Execution(seed=2)
    )
    assert replayed.to_dict() == value.to_dict()
    unbound = observational_policy.consume_compiled_observational_policy(
        bytes(artifact), execution=Execution(seed=2)
    )
    assert unbound.to_dict() == value.to_dict()
    with pytest.raises(ValueError, match="replayed policy value"):
        replayed.export()

    for expected, seed in ((identity, 3), (WRONG_IDENTITY, 2)):
        with pytest.raises(CausalError) as refused:
            observational_policy.consume_compiled_observational_policy(
                artifact, expected_identity=expected, execution=Execution(seed=seed)
            )
        assert refused.value.reason_code == "invalid_argument"


def test_a_graph_with_an_open_backdoor_path_is_refused() -> None:
    y, a, _ = _draw(400)
    with pytest.raises(CausalError) as refused:
        observational_policy.policy_value(
            y,
            a,
            support=[0.0, 1.0],
            objective="maximize",
            evaluation="held_out",
            evaluation_fraction=0.4,
            graph=GRAPH,
            treatment_name="a",
            outcome_name="y",
            execution=Execution(seed=2),
        )
    assert refused.value.reason_code == "invalid_argument"
    with pytest.raises(CausalValueError, match="treatment_name and outcome_name"):
        _held_out(graph=GRAPH)


# --------------------------------------------------------------------------- subsampling


def _subsampling(replicates: int = 100) -> observational_policy.CrossFittedPolicyValueCandidate:
    y, a, z = _draw(300)
    w = np.sin(np.arange(300) * 0.31)
    return observational_policy.compile_policy_subsampling(
        y,
        a,
        support=[0.0, 1.0],
        objective="maximize",
        adjust={"z": z, "w": w},
        graph=NARROW_GRAPH,
        treatment_name="a",
        outcome_name="y",
        information=["z"],
        subsample_rows=150,
        replicates=replicates,
        folds=3,
        execution=Execution(seed=2),
    )


def test_policy_subsampling_compiles_its_schedule_and_replays_every_refit() -> None:
    candidate = _subsampling()
    assert candidate.value.evaluation == "cross_fitted_learning"
    assert candidate.value.evaluation_rows == 300
    assert candidate.lower <= candidate.upper
    assert candidate.level == 0.95
    assert candidate.unit_support() == {
        "unit": "estimate.observational_policy_value",
        "status": "supported",
        "supported": True,
        "missing_evidence": [],
    }
    assert candidate.calibration == candidate.slot["status"]
    assert candidate.basis["key"]["interval_method"] == "cross_fitted_policy_subsampling_basic_v2"

    artifact = candidate.export()
    assert artifact.expectation is not None
    assert artifact.expectation.identity == candidate.program_identity
    replayed = observational_policy.consume_policy_subsampling(
        artifact, expected_identity=candidate.program_identity, execution=Execution(seed=2)
    )
    assert replayed.value.to_dict() == candidate.value.to_dict()
    assert (replayed.lower, replayed.upper, replayed.calibration) == (
        candidate.lower,
        candidate.upper,
        candidate.calibration,
    )
    assert replayed.program_identity == candidate.program_identity
    assert bytes(replayed.export()) == bytes(artifact)

    for expected, seed in ((candidate.program_identity, 3), (WRONG_IDENTITY, 2)):
        with pytest.raises(CausalError) as refused:
            observational_policy.consume_policy_subsampling(
                artifact, expected_identity=expected, execution=Execution(seed=seed)
            )
        assert refused.value.reason_code == "invalid_argument"


def test_policy_subsampling_refuses_a_schedule_below_its_replicate_floor() -> None:
    with pytest.raises(CausalError) as refused:
        _subsampling(replicates=99)
    assert refused.value.reason_code == "invalid_argument"


# --------------------------------------------------------------------------- sequential policies


def _model() -> se.FiniteModel:
    def commit(label: str, utility: tuple[float, float]) -> se.Action:
        return se.Action(
            label, "commit", transition=((1.0, 0.0), (0.0, 1.0)), immediate_utility=utility
        )

    return se.FiniteModel(
        population="p",
        regime="r",
        period="t",
        state_variable="state",
        states=("a", "b"),
        initial=(0.5, 0.5),
        horizon=1,
        actions=(commit("a", (2.0, 0.0)), commit("b", (0.0, 1.0))),
        terminal_utility=(0.0, 0.0),
        label="test",
    )


POLICIES = [se.Policy.deterministic("a", {(): 0}), se.Policy.deterministic("b", {(): 1})]


def _rollouts() -> dr.SequentialPolicyReliability:
    return require_success(
        dr.sequential_policy_reliability(
            _model(), POLICIES, unit="utility", seed=5, rollouts=128, execution=Execution(seed=1)
        )
    )


def test_rollout_reliability_compiles_and_replays_the_raw_declaration() -> None:
    result = _rollouts()
    assert result.route == "SequentialPolicyReliability"
    assert result.unit_support() == {
        "unit": "decision_reliability.sampled_decision",
        "status": "supported",
        "supported": True,
        "missing_evidence": [],
    }
    assert len(result.draws.candidate_ids) == 2
    assert result.report.decided_id in result.draws.candidate_ids

    artifact = result.export()
    assert artifact.expectation is not None
    assert artifact.expectation.identity == result.program_identity
    replayed = require_success(
        dr.consume_sequential_policy_reliability(
            artifact, expected_identity=result.program_identity, execution=Execution(seed=1)
        )
    )
    assert replayed.to_dict() == result.to_dict()
    assert replayed.acceptance is not None and replayed.acceptance.replayed

    with pytest.raises(dr.DecisionReliabilityRefusal) as refused:
        require_success(
            dr.consume_sequential_policy_reliability(
                artifact, expected_identity=WRONG_IDENTITY, execution=Execution(seed=1)
            )
        )
    assert refused.value.reason_code == "external_binding_mismatch"


def test_logged_trajectories_are_bootstrapped_whole_and_replayed() -> None:
    log = [
        se.LoggedTrajectory(
            id=str(i),
            steps=(se.LoggedStep(action=i % 2, observation=None, probability=0.5),),
            outcome_utility=float(i % 4),
        )
        for i in range(16)
    ]
    result = require_success(
        dr.sequential_policy_reliability(
            _model(),
            POLICIES,
            unit="utility",
            seed=3,
            behaviour=se.Policy.stochastic("logging", {(): {0: 0.5, 1: 0.5}}),
            log=log,
            replicates=64,
            minimum_effective_sample_size=2.0,
            execution=Execution(seed=1),
        )
    )
    replayed = require_success(
        dr.consume_sequential_policy_reliability(result.export(), execution=Execution(seed=1))
    )
    assert replayed.to_dict() == result.to_dict()


def test_a_learned_policy_and_an_undeclared_mode_are_refused() -> None:
    with pytest.raises(dr.DecisionReliabilityRefusal) as refused:
        require_success(
            dr.sequential_policy_reliability(
                _model(),
                [*POLICIES, se.LearnedFromEvaluationData(rule="same log")],
                unit="utility",
                seed=1,
                rollouts=32,
            )
        )
    assert refused.value.reason_code == "route_not_supported"
    with pytest.raises(CausalValueError, match="declare rollouts"):
        require_success(dr.sequential_policy_reliability(_model(), POLICIES, unit="u", seed=1))


# --------------------------------------------------------------------------- bundle


def test_the_producer_draws_and_diagnostic_recompute_as_one_bundle() -> None:
    execution = Execution(seed=1)
    reliability = _rollouts()
    bundle = (
        cb.Bundle.builder().add_sequential_policy_reliability(reliability, execution=execution)
    ).build()
    assert {node.kind for node in bundle.nodes} == {
        "sequential_policy_reliability_fit",
        "sequential_decision_draws",
        "sequential_reliability_diagnostic",
    }

    receipt = require_success(
        cb.compile_sequential_reliability_node_bundle(bundle, execution=execution)
    )
    assert receipt.route == "SequentialPolicyReliabilityNodeBundleRecomputation"
    assert receipt.unit_support() == {
        "unit": "composition.bundle",
        "status": "supported",
        "supported": True,
        "missing_evidence": [],
    }
    assert receipt.bundle_identity == bundle.identity
    assert receipt.producer_program_identity == reliability.program_identity
    assert receipt.draws_digest == reliability.draws.digests()[0]
    assert {dependency["node_id"] for dependency in receipt.dependencies} == {
        "sequential_producer",
        "sequential_draws",
        "sequential_diagnostic",
    }

    artifact = receipt.export(execution=execution)
    replayed = require_success(
        cb.consume_portable_sequential_reliability_node_bundle(
            artifact,
            expected_bundle=bundle.identity,
            expected_identity=receipt.program_identity,
            execution=execution,
        )
    )
    assert replayed == receipt
    assert replayed.acceptance is not None and replayed.acceptance.identity_bound

    for expected_bundle, expected_identity in (
        ("wrong", receipt.program_identity),
        (bundle.identity, WRONG_IDENTITY),
    ):
        with pytest.raises(cb.ExpectedIdentityMismatchRefusal):
            require_success(
                cb.consume_portable_sequential_reliability_node_bundle(
                    artifact,
                    expected_bundle=expected_bundle,
                    expected_identity=expected_identity,
                    execution=execution,
                )
            )


def test_a_bundle_without_its_direct_edges_is_refused() -> None:
    execution = Execution(seed=1)
    bundle = (
        cb.Bundle.builder().add_sequential_policy_reliability(reliability=_rollouts())
    ).build()
    with pytest.raises(cb.CompositionBundleRefusal):
        require_success(
            cb.compile_sequential_reliability_node_bundle(
                bundle,
                draws="sequential_diagnostic",
                diagnostic="sequential_draws",
                execution=execution,
            )
        )
