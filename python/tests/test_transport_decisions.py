"""Transported decisions from Python: compile, fit, decide, seal and replay.

Each fixture is the smallest one the Rust route tests use, on the graph
``a -> y`` with an isolated selection target ``s`` and a binary response query
transported from ``trial`` to ``target``:

* trial-to-target IPW on 80 independent units (40 trial, 40 target), selection and
  treatment probabilities one half, 100 posterior draws under seed 31. The effect
  fixture has ``Y = 1`` for treated trial units and ``0`` otherwise, so every effect
  draw is positive and ``treat`` (utility: the effect) beats ``hold`` (utility 0). The
  predictive fixture has trial outcomes ``i mod 4``, so untreated draws lie in
  ``{1, 3}`` and treated draws in ``{0, 2}``;
* the joint Gaussian transport of one 40-unit trial whose effect is exactly 2 at every
  unit, 1024 conjugate draws: the target effect mean is 2, and with a cross-world
  disturbance correlation of one the paired outcome difference is the effect draw.

Every decision is exported, consumed from its bytes under the retained program
identity, and refused under a foreign identity. The effect decisions also enter a
composition bundle as fit nodes whose chain is the live terminal.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Admg, Execution, ResponseCurve, decision
from antecedent.composition_bundle import Bundle
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced, decisions

from _refusal import assert_registered_refusal

FOREIGN = "0" * 64


def structure() -> tuple[Admg, advanced.TransportQuery]:
    graph = Admg.from_edges(["a", "s", "y"], [("a", "y")])
    query = advanced.TransportQuery(
        ResponseCurve("a", "y", grid=[0.0, 1.0]),
        advanced.SelectionDiagram("trial", "target", ["s"]),
        source_experiments=["a"],
    )
    return graph, query


def trial_rows(outcome) -> decisions.TrialIpwRows:
    n = 80
    return decisions.TrialIpwRows(
        unit_ids=[f"unit-{i}" for i in range(n)],
        outcome=[outcome(i) for i in range(n)],
        treatment=[i % 2 == 0 for i in range(n)],
        trial=[i < 40 for i in range(n)],
        selection_probability=[0.5] * n,
        treatment_probability=[0.5] * n,
    )


def effect_contract(effect, treat) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("treat", (effect,), treat, kind="external"),
            decision.Action("hold", (effect,), decision.x(0) * 0.0, kind="external"),
        ),
        utility_units="score",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy="require_invariant_best_action",
    )


def worlds_contract(untreated, treated, treat) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("hold", (untreated,), decision.x(0)),
            decision.Action("treat", (treated,), treat),
        ),
        utility_units="score",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        structural_policy="require_invariant_best_action",
    )


def trial_effect(execution: Execution | None = None):
    graph, query = structure()
    effect = decisions.effect_quantity(graph=graph, query=query, units="score", name="effect")
    compiled = decisions.compile_trial_ipw_decision(
        graph=graph,
        query=query,
        rows=trial_rows(lambda i: 1.0 if i < 40 and i % 2 == 0 else 0.0),
        contract=effect_contract(effect, decision.x(0)),
        effect=effect,
        draws=100,
        execution=execution or Execution(seed=31),
    )
    return compiled, compiled.run()


def joint_inputs(rows: int = 40):
    source = advanced.JointTransportSource(
        "trial-a",
        "trial",
        advanced.JointTransportIdentity("trial-snapshot", tuple(f"trial-{i}" for i in range(rows))),
        {
            "a": np.array([1.0 if i % 2 == 0 else 0.0 for i in range(rows)]),
            "y": np.array([5.0 if i % 2 == 0 else 3.0 for i in range(rows)]),
        },
        0.1,
    )
    target = advanced.JointTransportTarget(
        "target",
        advanced.JointTransportIdentity(
            "target-snapshot", tuple(f"target-{i}" for i in range(rows))
        ),
        {},
    )
    prior = advanced.GaussianTransportPrior((0.0,), ((10.0,),))
    return [source], target, advanced.JointTransportPriors(prior, prior)


def joint_effect():
    graph, query = structure()
    sources, target, priors = joint_inputs()
    effect = decisions.effect_quantity(graph=graph, query=query, units="score")
    compiled = decisions.compile_joint_transport_decision(
        graph=graph,
        query=query,
        sources=sources,
        target=target,
        priors=priors,
        contract=effect_contract(effect, decision.x(0) - 1.0),
        effect=effect,
        draws=1024,
        execution=Execution(seed=91),
    )
    return compiled, compiled.run()


def refused_foreign(consume) -> None:
    with pytest.raises(CausalUnsupportedError) as refused:
        consume(FOREIGN)
    assert refused.value.reason_code == "external_binding_mismatch"
    assert_registered_refusal(refused.value)


# ---------------------------------------------------------------------------
# Trial-to-target IPW.
# ---------------------------------------------------------------------------


def test_trial_ipw_effect_decision_refits_seals_and_replays():
    compiled, result = trial_effect()
    assert compiled.route == "trial_ipw_effect"
    assert len(compiled.identity) == 64
    assert compiled.plan
    assert result.program_identity == compiled.identity
    assert result.calibration == "not_measured"
    assert result.effect_draws.shape == (100,)
    assert (result.effect_draws > 0).all()
    assert result.verdict.kind == "uniquely_optimal"
    assert result.selected == ("treat",)
    assert result.chain.identity
    data = result.export()
    assert isinstance(data, bytes)
    replay = decisions.consume_transport_decision(
        data, route=compiled.route, expected_identity=compiled.identity
    )
    assert replay.independently_verified
    assert replay.acceptance.replayed
    np.testing.assert_array_equal(replay.effect_draws, result.effect_draws)
    assert replay.verdict == result.verdict
    assert replay.export() == data
    refused_foreign(
        lambda identity: decisions.consume_transport_decision(
            data, route=compiled.route, expected_identity=identity
        )
    )


def test_trial_ipw_predictive_decision_reads_both_outcome_worlds_and_replays():
    graph, query = structure()
    untreated, treated = decisions.outcome_worlds(graph=graph, query=query, units="score")
    declaration = {
        "graph": graph,
        "query": query,
        "rows": trial_rows(lambda i: float(i % 4) if i < 40 else 0.0),
        "contract": worlds_contract(untreated, treated, decision.x(0) * decision.x(0)),
        "worlds": (untreated, treated),
        "draws": 100,
        "execution": Execution(seed=31),
    }
    with pytest.raises(CausalValueError) as undeclared:
        decisions.compile_trial_ipw_decision(**declaration)
    assert undeclared.value.reason_code == "invalid_argument"
    compiled = decisions.compile_trial_ipw_decision(
        **declaration, independent_potential_outcomes=True
    )
    assert compiled.route == "trial_ipw_predictive"
    result = compiled.run()
    assert result.effect_draws is None
    untreated_draws, treated_draws = result.predictive_draws
    assert set(untreated_draws.tolist()) <= {1.0, 3.0}
    assert set(treated_draws.tolist()) <= {0.0, 2.0}
    data = result.export()
    replay = decisions.consume_transport_decision(
        data, route=compiled.route, expected_identity=compiled.identity
    )
    np.testing.assert_array_equal(replay.predictive_draws[1], treated_draws)
    assert replay.verdict == result.verdict
    refused_foreign(
        lambda identity: decisions.consume_transport_decision(
            data, route=compiled.route, expected_identity=identity
        )
    )


def test_a_decision_declares_exactly_one_coordinate():
    graph, query = structure()
    effect = decisions.effect_quantity(graph=graph, query=query, units="score")
    with pytest.raises(CausalValueError) as neither:
        decisions.compile_trial_ipw_decision(
            graph=graph,
            query=query,
            rows=trial_rows(lambda i: 0.0),
            contract=effect_contract(effect, decision.x(0)),
        )
    assert neither.value.reason_code == "invalid_argument"


# ---------------------------------------------------------------------------
# Joint source-target transport.
# ---------------------------------------------------------------------------


def test_joint_transport_effect_decision_recovers_the_known_effect_and_replays():
    compiled, result = joint_effect()
    assert compiled.route == "joint_transport_effect"
    assert result.target_effect_mean == pytest.approx(2.0, abs=0.05)
    assert result.effect_draws.shape == (1024,)
    assert result.calibration == "not_measured"
    assert result.selected == ("treat",)
    data = result.export()
    replay = decisions.consume_transport_decision(
        data,
        route=compiled.route,
        expected_identity=compiled.identity,
        execution=Execution(seed=91),
    )
    assert replay.target_effect_mean == result.target_effect_mean
    np.testing.assert_array_equal(replay.effect_draws, result.effect_draws)
    assert replay.verdict == result.verdict
    refused_foreign(
        lambda identity: decisions.consume_transport_decision(
            data, route=compiled.route, expected_identity=identity
        )
    )


def test_joint_transport_predictive_decision_pairs_the_outcome_worlds_and_replays():
    graph, query = structure()
    sources, target, priors = joint_inputs()
    untreated, treated = decisions.outcome_worlds(graph=graph, query=query, units="score")
    execution = Execution(seed=92)
    compiled = decisions.compile_joint_transport_decision(
        graph=graph,
        query=query,
        sources=sources,
        target=target,
        priors=priors,
        contract=worlds_contract(untreated, treated, decision.x(0) - 1.0),
        worlds=(untreated, treated),
        target_noise_variance=0.1,
        target_noise_correlation=1.0,
        sharing="shared_varying_block",
        draws=1024,
        execution=execution,
    )
    assert compiled.route == "joint_transport_predictive"
    result = compiled.run()
    untreated_draws, treated_draws = result.predictive_draws
    assert float(np.mean(treated_draws - untreated_draws)) == pytest.approx(2.0, abs=0.05)
    assert result.selected == ("treat",)
    data = result.export()
    replay = decisions.consume_transport_decision(
        data, route=compiled.route, expected_identity=compiled.identity, execution=execution
    )
    np.testing.assert_array_equal(replay.predictive_draws[0], untreated_draws)
    assert replay.verdict == result.verdict
    refused_foreign(
        lambda identity: decisions.consume_transport_decision(
            data, route=compiled.route, expected_identity=identity, execution=execution
        )
    )


# ---------------------------------------------------------------------------
# Bundles whose fit node is a transported decision.
# ---------------------------------------------------------------------------


def fit_bundle(result, execution: Execution) -> Bundle:
    builder = Bundle.builder()
    decisions.add_fit_node(builder, result, node_id="fit", execution=execution)
    builder.add_artifact("propagation_chain", result.chain, node_id="terminal")
    builder.connect("fit", "terminal")
    return builder.build()


@pytest.mark.parametrize(
    ("produce", "route", "seed"),
    [(trial_effect, "trial_ipw", 31), (joint_effect, "joint_transport_effect", 91)],
)
def test_a_transported_fit_node_bundle_refits_and_replays(produce, route, seed):
    compiled, result = produce()
    execution = Execution(seed=seed)
    bundle = fit_bundle(result, execution)
    plan = decisions.compile_node_bundle(
        bundle, producer_node="fit", terminal_node="terminal", execution=execution
    )
    assert plan.route == route
    receipt = plan.run()
    assert receipt.bundle_identity == bundle.identity
    assert receipt.producer_program_identity == compiled.identity
    assert receipt.program_identity == plan.identity
    assert receipt.terminal_identity == result.chain.identity
    assert {dependency["node_id"] for dependency in receipt.dependencies} >= {"fit", "terminal"}
    data = plan.export(receipt)
    replay = decisions.consume_node_bundle(
        data,
        route=route,
        expected_bundle=bundle.identity,
        expected_identity=plan.identity,
        execution=execution,
    )
    assert replay.independently_verified
    assert replay.terminal_identity == receipt.terminal_identity
    assert replay.dependencies == receipt.dependencies
    refused_foreign(
        lambda identity: decisions.consume_node_bundle(
            data,
            route=route,
            expected_bundle=identity,
            expected_identity=plan.identity,
            execution=execution,
        )
    )
    consumed = decisions.consume_fit_bundle(
        bytes(bundle.export()), route=route, expected_identity=bundle.identity, execution=execution
    )
    assert consumed.all_verified
    assert consumed.node("fit").replayed


def test_only_an_effect_decision_is_a_fit_node():
    graph, query = structure()
    untreated, treated = decisions.outcome_worlds(graph=graph, query=query, units="score")
    result = decisions.compile_trial_ipw_decision(
        graph=graph,
        query=query,
        rows=trial_rows(lambda i: float(i % 4) if i < 40 else 0.0),
        contract=worlds_contract(untreated, treated, decision.x(0)),
        worlds=(untreated, treated),
        independent_potential_outcomes=True,
        draws=100,
    ).run()
    with pytest.raises(CausalUnsupportedError) as refused:
        decisions.add_fit_node(Bundle.builder(), result, node_id="fit")
    assert refused.value.reason_code == "route_not_supported"
