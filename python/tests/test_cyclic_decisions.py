# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cyclic decisions: feedback components, paired action worlds, fitted posteriors and bundles.

Each fixture is the smallest one the matching Rust route test uses; the fitted schedules are the
ones whose R-hat and ESS gates those tests rely on.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from antecedent import decision
from antecedent.composition_bundle import Bundle
from antecedent.cyclic import (
    DecisionOutput,
    Equation,
    EquilibriumModel,
    FeedbackComponent,
    FeedbackConnection,
    FeedbackOutput,
    LogisticOutputs,
    MultiOutputComponent,
    SolvePolicy,
    TargetFrame,
    add_posterior_fit,
    compile_action_world_decision,
    compile_bundle_recomputation,
    compile_component_decision,
    compile_posterior_decision,
    consume_bundle_recomputation,
    consume_bundle_refitting,
    consume_decision,
    prepare_component_feedback,
)
from antecedent.errors import CausalSerializationError, CausalValueError
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration, rows
from antecedent.execution import Execution
from antecedent.interface_contract import (
    ConsumedPort,
    InterfaceContract,
    ProducedPort,
    SupportDomain,
)
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

EXECUTION = Execution(seed=47)
REGIMES = ("ordinary", "promoted")


def outcome(
    variable: str = "wishlists",
    *,
    regime: str,
    name: str | None = None,
    population: str = "game",
    horizon: int = 0,
    functional: str = "outcome",
    units: str = "count",
    transform: str = "identity",
) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=name or variable,
        role="outcome",
        units=units,
        population_id=population,
        regime_id=regime,
        horizon=horizon,
        functional_id=functional,
        transform_id=transform,
    )


def contract(
    quantities: dict[str, ScientificQuantity], *, units: str, population: str, horizon: int
) -> decision.Contract:
    return decision.Contract(
        actions=tuple(
            decision.Action(regime, inputs=(quantity,), utility=decision.x(0))
            for regime, quantity in quantities.items()
        ),
        utility_units=units,
        criterion=decision.Criterion.expected_regret(),
        target_population=population,
        horizon=horizon,
        structural_policy="require_invariant_best_action",
    )


def paired(quantities: dict[str, ScientificQuantity], equation: int) -> list[DecisionOutput]:
    return [
        DecisionOutput(world, equation, quantities[regime]) for world, regime in enumerate(REGIMES)
    ]


# --------------------------------------------------------------------------- components


def port(variable: str = "game:wishlists", name: str = "wishlists", **fields: Any):
    return outcome(
        variable,
        name=name,
        population="game-a",
        horizon=1,
        functional="equilibrium_value",
        **fields,
    )


def produced(port_name: str, quantity: ScientificQuantity, evidence=None) -> ProducedPort:
    return ProducedPort(
        port_name,
        quantity,
        "model:specified",
        SupportDomain.real(),
        operations=("mean",),
        evidence=evidence,
    )


def consumed(
    port_name: str, quantity: ScientificQuantity, dependence: str = "unconstrained"
) -> ConsumedPort:
    return ConsumedPort(
        port_name,
        quantity,
        "model:specified",
        SupportDomain.real(),
        operations=("mean",),
        dependence=dependence,
    )


def component(
    name: str, self_weight: float, *, consumes=(("in", "unconstrained"),), evidence=None
) -> FeedbackComponent:
    quantity = port(regime="specified-equilibrium")
    return FeedbackComponent(
        InterfaceContract(
            name,
            produces=[produced("out", quantity, evidence)],
            consumes=[consumed(item, quantity, dependence) for item, dependence in consumes],
        ),
        Equation(f"{name}.mechanism", 1.0, (self_weight,), "identity"),
    )


def scalar_connections() -> list[FeedbackConnection]:
    return [
        FeedbackConnection("traffic", "out", "wishlists", "in", 0.3),
        FeedbackConnection("wishlists", "out", "traffic", "in", 0.2),
    ]


def game_component() -> MultiOutputComponent:
    wishlists = port(regime="specified-equilibrium")
    traffic = port("game:traffic", "traffic", regime="specified-equilibrium")
    return MultiOutputComponent(
        InterfaceContract(
            "game",
            produces=[produced("traffic", traffic), produced("wishlists", wishlists)],
            consumes=[consumed("from_wishlists", wishlists), consumed("from_traffic", traffic)],
        ),
        (
            FeedbackOutput(
                "traffic",
                Equation("traffic.mechanism", 1.0, (0.1,), "identity"),
                ("from_wishlists",),
            ),
            FeedbackOutput(
                "wishlists",
                Equation("wishlists.mechanism", 1.0, (0.2,), "identity"),
                ("from_traffic",),
            ),
        ),
    )


def game_connections() -> list[FeedbackConnection]:
    return [
        FeedbackConnection(
            "game", "traffic", "game", "from_traffic", 0.3, consumer_output="wishlists"
        ),
        FeedbackConnection(
            "game", "wishlists", "game", "from_wishlists", 0.2, consumer_output="traffic"
        ),
    ]


def component_training() -> list[dict[str, Any]]:
    training = []
    for i in range(260):
        traffic = (i % 5) * 0.4
        disturbance = math.sin(i * 0.19) * 0.2
        training.append(
            {
                "values": [traffic, (1.0 + 0.3 * traffic + disturbance) / 0.8],
                "clamps": [{"variable": 0, "value": traffic}],
            }
        )
    return training


COMPONENT_SCHEDULE: dict[str, Any] = {
    "parameters": [{"kind": "weight", "equation": 1, "parent": 0}],
    "prior_sd": 1.0,
    "chains": 4,
    "warmup": 500,
    "draws": 1500,
    "proposal_scale": 0.03,
}


def component_quantities(**fields: Any) -> dict[str, ScientificQuantity]:
    return {regime: port(regime=regime, **fields) for regime in REGIMES}


def test_scalar_components_join_one_cyclic_program() -> None:
    feedback = prepare_component_feedback(
        [component("wishlists", 0.2), component("traffic", 0.1)],
        scalar_connections(),
        noise_covariance=(1.0, 0.0, 0.0, 1.0),
        execution=EXECUTION,
    )
    assert feedback.component_order == ("traffic", "wishlists")
    assert not feedback.multi_output
    request = json.loads(feedback.system.request_json)
    assert [equation["weights"] for equation in request["model"]["equations"]] == [
        [0.1, 0.2],
        [0.3, 0.2],
    ]
    # The cross-component cycle is solved as one system: (I - W) x = 1.
    solved = require_success(feedback.system.execute()).to_dict()["worlds"][0][0]["values"]
    assert solved == pytest.approx([1.0 / 0.66, 1.2 / 0.66])
    with pytest.raises(CausalValueError, match="contraction"):
        prepare_component_feedback(
            [component("wishlists", 0.2), component("traffic", 0.1)],
            scalar_connections(),
            noise_covariance=(1.0, 0.0, 0.0, 1.0),
            policy=SolvePolicy(require_contraction=False),
            execution=EXECUTION,
        )


def test_ledger_proves_disjoint_inputs_and_binds_the_program_identity() -> None:
    ledger = EvidenceLedger()
    first = ledger.add_source(
        SourceDeclaration("ads", "observation_set", observations=rows("campaign", [b"ads"]))
    )
    second = ledger.add_source(
        SourceDeclaration(
            "community", "observation_set", observations=rows("campaign", [b"community"])
        )
    )

    def declare(community_evidence: str):
        return (
            [
                component(
                    "wishlists",
                    0.2,
                    consumes=(("in", "disjoint_evidence"), ("community_in", "disjoint_evidence")),
                ),
                component("traffic", 0.1, evidence=(first,)),
                component("community", 0.1, evidence=(community_evidence,)),
            ],
            [
                *scalar_connections(),
                FeedbackConnection("community", "out", "wishlists", "community_in", 0.1),
                FeedbackConnection("wishlists", "out", "community", "in", 0.1),
            ],
        )

    components, connections = declare(second)
    covariance = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(CausalValueError, match="ledger"):
        prepare_component_feedback(
            components, connections, noise_covariance=covariance, execution=EXECUTION
        )
    prepared = prepare_component_feedback(
        components, connections, noise_covariance=covariance, ledger=ledger, execution=EXECUTION
    )
    assert len(prepared.component_order) == 3
    ledger.add_source(
        SourceDeclaration("later", "observation_set", observations=rows("campaign", [b"later"]))
    )
    changed = prepare_component_feedback(
        components, connections, noise_covariance=covariance, ledger=ledger, execution=EXECUTION
    )
    assert changed.identity != prepared.identity
    shared_components, shared_connections = declare(first)
    with pytest.raises(CausalValueError, match="disjoint"):
        prepare_component_feedback(
            shared_components,
            shared_connections,
            noise_covariance=covariance,
            ledger=ledger,
            execution=EXECUTION,
        )


def test_multi_output_component_lowers_one_connected_program_with_and_without_ledger() -> None:
    feedback = prepare_component_feedback(
        [game_component()],
        game_connections(),
        noise_covariance=(1.0, 0.0, 0.0, 1.0),
        execution=EXECUTION,
    )
    assert feedback.multi_output
    assert feedback.component_order == ("game::traffic", "game::wishlists")
    request = json.loads(feedback.system.request_json)
    assert [equation["weights"] for equation in request["model"]["equations"]] == [
        [0.1, 0.2],
        [0.3, 0.2],
    ]
    with_ledger = prepare_component_feedback(
        [game_component()],
        game_connections(),
        noise_covariance=(1.0, 0.0, 0.0, 1.0),
        ledger=EvidenceLedger(),
        execution=EXECUTION,
    )
    assert with_ledger.component_order == feedback.component_order
    assert with_ledger.composition_identity != feedback.composition_identity
    assert with_ledger.identity != feedback.identity


def test_joint_component_posterior_decision_refits_and_replays() -> None:
    feedback = prepare_component_feedback(
        [component("wishlists", 0.2), component("traffic", 0.1)],
        scalar_connections(),
        noise_covariance=(0.04, 0.0, 0.0, 0.04),
        worlds=[{"traffic": 0.0}, {"traffic": 1.0}],
        execution=EXECUTION,
    )
    quantities = component_quantities()
    decided = contract(quantities, units="count", population="game-a", horizon=1)
    plan = compile_component_decision(
        feedback,
        paired(quantities, 1),
        decided,
        rows=component_training(),
        execution=EXECUTION,
        **COMPONENT_SCHEDULE,
    )
    assert plan.route == "JointComponentPosteriorDecision"
    assert plan.kind == "joint_component_posterior_decision_v1"
    result = plan.execute()
    assert result.posterior is not None and result.posterior["diagnostics_pass"]
    assert result.calibration == "not_measured"
    artifact = result.export()
    replay = consume_decision(artifact, expected_identity=plan.identity, execution=EXECUTION)
    assert replay.identity == plan.identity
    assert replay.decision == result.decision
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity="0" * 64, execution=EXECUTION)
    different = component_quantities()
    different["ordinary"] = port("different-port", regime="ordinary")
    with pytest.raises(CausalValueError, match="coordinate"):
        compile_component_decision(
            feedback,
            paired(different, 1),
            decided,
            rows=component_training(),
            execution=EXECUTION,
            **COMPONENT_SCHEDULE,
        )


def test_multi_output_component_decision_and_its_logistic_outputs_replay() -> None:
    feedback = prepare_component_feedback(
        [game_component()],
        game_connections(),
        noise_covariance=(0.04, 0.0, 0.0, 0.04),
        worlds=[{"game::traffic": 0.0}, {"game::traffic": 1.0}],
        execution=EXECUTION,
    )
    quantities = component_quantities()
    real = contract(quantities, units="count", population="game-a", horizon=1)
    plan = compile_component_decision(
        feedback,
        paired(quantities, 1),
        real,
        rows=component_training(),
        execution=EXECUTION,
        **COMPONENT_SCHEDULE,
    )
    assert plan.kind == "joint_vector_component_posterior_decision_v1"
    result = plan.execute()
    replay = consume_decision(result.export(), expected_identity=plan.identity, execution=EXECUTION)
    assert replay.decision == result.decision

    bounded_quantities = component_quantities(units="dimensionless", transform="logistic")
    logistic = LogisticOutputs(
        tuple(bounded_quantities.values()),
        contract(bounded_quantities, units="dimensionless", population="game-a", horizon=1),
    )
    bounded = compile_component_decision(
        feedback,
        paired(quantities, 1),
        real,
        rows=component_training(),
        logistic=logistic,
        execution=EXECUTION,
        **COMPONENT_SCHEDULE,
    )
    assert bounded.route == "BoundedComponentPosteriorDecision"
    assert bounded.identity != plan.identity
    outcome_law = bounded.execute()
    assert all(0.0 <= value <= 1.0 for column in outcome_law.draws() for value in column)
    assert outcome_law.calibration == "not_measured"
    assert outcome_law.real_valued_decision is not None
    artifact = outcome_law.export()
    consumed_law = consume_decision(
        artifact, expected_identity=bounded.identity, execution=EXECUTION
    )
    assert consumed_law.decision == outcome_law.decision
    assert consumed_law.real_valued_decision == outcome_law.real_valued_decision
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity=plan.identity, execution=EXECUTION)


# --------------------------------------------------------------------------- action worlds


def test_specified_action_worlds_are_paired_and_replayed() -> None:
    model = EquilibriumModel(
        (
            Equation("wishlists", 0.0, (0.1, 0.3), "identity"),
            Equation("traffic", 0.0, (0.2, 0.0), "identity"),
        ),
        (1.0, 0.0, 0.0, 1.0),
    )
    noise = [[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]
    system = model.prepare(
        [[{"variable": 1, "value": 1.0}], [{"variable": 1, "value": 2.0}]],
        noise,
        execution=EXECUTION,
    )
    quantities = {regime: outcome(regime=regime) for regime in REGIMES}
    decided = contract(quantities, units="wishlist_count", population="game", horizon=0)
    plan = compile_action_world_decision(
        system, paired(quantities, 0), decided, execution=EXECUTION
    )
    assert plan.route == "CyclicActionWorldDecision"
    result = plan.execute()
    assert result.decision.draws == 4
    assert result.posterior is None and result.posterior_samples() is None
    ordinary, promoted = result.draws()
    assert all(after > before for before, after in zip(ordinary, promoted, strict=True))
    assert result.decision.verdict == decision.Verdict("uniquely_optimal", ("promoted",))
    artifact = result.export()
    replay = consume_decision(artifact, expected_identity=plan.identity, execution=EXECUTION)
    assert replay.decision == result.decision
    assert replay.kind == "cyclic_derived_action_world_decision_v1"
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity="0" * 64, execution=EXECUTION)
    with pytest.raises(CausalValueError, match="identical clamps|world"):
        repeated = model.prepare(
            [[{"variable": 1, "value": 1.0}], [{"variable": 1, "value": 1.0}]],
            noise,
            execution=EXECUTION,
        )
        compile_action_world_decision(repeated, paired(quantities, 0), decided, execution=EXECUTION)


def posterior_training() -> list[dict[str, Any]]:
    training = []
    for i in range(260):
        ad = (i % 5) * 0.4
        ew = math.sin(i * 0.19) * 0.2
        et = math.cos(i * 0.23) * 0.2
        wishlists = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06)
        training.append(
            {
                "values": [wishlists, 0.2 * wishlists + et, ad],
                "clamps": [{"variable": 2, "value": ad}],
            }
        )
    return training


@pytest.fixture(scope="module")
def posterior() -> tuple[Any, Any, Any]:
    """The fitted posterior producer, its result and its exported artifact."""
    model = EquilibriumModel(
        (
            Equation("wishlists", 0.0, (0.0, 0.3, 0.2), "identity"),
            Equation("traffic", 0.0, (0.2, 0.0, 0.0), "identity"),
            Equation("ads", 0.0, (0.0, 0.0, 0.0), "identity"),
        ),
        (0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04),
    )
    system = model.prepare(
        [[{"variable": 2, "value": 0.0}], [{"variable": 2, "value": 1.0}]],
        [[0.0, 0.0, 0.0]],
        parameters=[{"kind": "weight", "equation": 0, "parent": 1}],
        rows=posterior_training(),
        execution=EXECUTION,
    )
    quantities = {regime: outcome(regime=regime) for regime in REGIMES}
    plan = compile_posterior_decision(
        system,
        paired(quantities, 0),
        contract(quantities, units="wishlist_count", population="game", horizon=0),
        prior_sd=1.0,
        chains=4,
        warmup=500,
        draws=1500,
        proposal_scale=0.18,
        execution=EXECUTION,
    )
    result = plan.execute()
    return plan, result, result.export()


def test_fitted_posterior_action_worlds_replay(posterior: tuple[Any, Any, Any]) -> None:
    plan, result, artifact = posterior
    assert plan.route == "CyclicPosteriorActionDecision"
    assert result.posterior["diagnostics_pass"]
    assert result.decision.draws == 6000
    assert len(result.posterior_samples()) == 6000
    replay = consume_decision(artifact, expected_identity=plan.identity, execution=EXECUTION)
    assert replay.decision == result.decision
    assert replay.posterior_samples() == result.posterior_samples()
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity="0" * 64, execution=EXECUTION)
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity=plan.identity, execution=Execution(seed=48))


def test_target_feedback_worlds_standardize_refit_and_replay() -> None:
    training = []
    for i in range(260):
        ad = (i % 5) * 0.4
        baseline = (i % 9) * 0.2
        ew = math.sin(i * 0.19) * 0.2
        et = math.cos(i * 0.23) * 0.2
        wishlists = (0.2 * ad + 0.4 * baseline + ew + 0.3 * et) / (1.0 - 0.06)
        training.append(
            {
                "values": [wishlists, 0.2 * wishlists + et, ad, baseline],
                "clamps": [{"variable": 2, "value": ad}, {"variable": 3, "value": baseline}],
            }
        )
    covariance = tuple(0.04 if row == column else 0.0 for row in range(4) for column in range(4))
    model = EquilibriumModel(
        (
            Equation("wishlists", 0.0, (0.0, 0.3, 0.2, 0.4), "identity"),
            Equation("traffic", 0.0, (0.2, 0.0, 0.0, 0.0), "identity"),
            Equation("ads", 0.0, (0.0,) * 4, "identity"),
            Equation("baseline", 0.0, (0.0,) * 4, "identity"),
        ),
        covariance,
    )
    system = model.prepare(
        [[{"variable": 2, "value": 0.0}], [{"variable": 2, "value": 1.0}]],
        [[0.0] * 4],
        parameters=[{"kind": "weight", "equation": 0, "parent": 1}],
        rows=training,
        execution=EXECUTION,
    )
    quantities = {regime: outcome(regime=regime) for regime in REGIMES}
    decided = contract(quantities, units="wishlist_count", population="game", horizon=0)
    frame = TargetFrame("trial", "target-frame", "game", (3,), ("t0", "t1"), ((0.2,), (1.4,)))
    schedule: dict[str, Any] = {
        "prior_sd": 1.0,
        "chains": 4,
        "warmup": 500,
        "draws": 1500,
        "proposal_scale": 0.16,
    }
    with pytest.raises(CausalValueError, match="range"):
        compile_posterior_decision(
            system,
            paired(quantities, 0),
            decided,
            target=TargetFrame(
                "trial", "target-frame", "game", (3,), ("t0", "t1"), ((2.0,), (1.4,))
            ),
            execution=EXECUTION,
            **schedule,
        )
    plan = compile_posterior_decision(
        system, paired(quantities, 0), decided, target=frame, execution=EXECUTION, **schedule
    )
    assert plan.route == "CyclicTargetFeedbackDecision"
    result = plan.execute()
    assert result.calibration == "not_measured"
    ordinary, promoted = result.draws()
    samples = result.posterior_samples()
    assert samples is not None
    # Shared target rows and disturbances cancel: each paired effect is fixed by its own draw.
    for before, after, parameters in zip(ordinary, promoted, samples, strict=True):
        assert after - before == pytest.approx(0.2 / (1.0 - 0.2 * parameters[0]), abs=1e-7)
    artifact = result.export()
    replay = consume_decision(artifact, expected_identity=plan.identity, execution=EXECUTION)
    assert replay.posterior_samples() == samples
    assert replay.decision == result.decision
    with pytest.raises(CausalSerializationError):
        consume_decision(artifact, expected_identity="0" * 64, execution=EXECUTION)


# --------------------------------------------------------------------------- bundles


def test_bundle_terminal_is_regenerated_from_the_producer_artifact(
    posterior: tuple[Any, Any, Any],
) -> None:
    producer, result, artifact = posterior
    builder = Bundle.builder()
    builder.add_artifact("propagation_chain", result.chain_artifact(), node_id="terminal")
    bundle = builder.build()
    plan = compile_bundle_recomputation(bundle, "terminal", producer=artifact, execution=EXECUTION)
    receipt = plan.execute()
    assert receipt.identity == plan.identity
    assert receipt.producer_program_identity == producer.identity
    assert receipt.bundle_identity == bundle.identity
    assert receipt.terminal_identity == result.decision.chain_identity
    portable = receipt.export()
    replay = consume_bundle_recomputation(
        portable,
        expected_bundle=bundle.identity,
        expected_identity=plan.identity,
        execution=EXECUTION,
    )
    assert replay == receipt
    with pytest.raises(CausalSerializationError):
        consume_bundle_recomputation(
            portable,
            expected_bundle="wrong",
            expected_identity=plan.identity,
            execution=EXECUTION,
        )
    with pytest.raises(CausalSerializationError, match="cyclic decision"):
        consume_decision(portable, expected_identity=plan.identity, execution=EXECUTION)


def test_embedded_fit_node_is_refitted_by_the_bundle_consumer(
    posterior: tuple[Any, Any, Any],
) -> None:
    producer, result, artifact = posterior
    corrupted = bytearray(bytes(artifact))
    corrupted[0] ^= 1
    with pytest.raises(CausalValueError):
        add_posterior_fit(
            Bundle.builder(),
            "fit",
            bytes(corrupted),
            producer_identity=producer.identity,
            execution=EXECUTION,
        )
    builder = Bundle.builder()
    add_posterior_fit(builder, "fit", artifact, execution=EXECUTION)
    builder.add_artifact("propagation_chain", result.chain_artifact(), node_id="terminal")
    with pytest.raises(CausalValueError, match="dependency"):
        compile_bundle_recomputation(
            builder.build(), "terminal", producer_node="fit", execution=EXECUTION
        )
    builder.connect("fit", "terminal")
    bundle = builder.build()
    for route in ("cyclic_posterior", "native_fit"):
        consumed = consume_bundle_refitting(
            bundle.export(), expected_identity=bundle.identity, route=route, execution=EXECUTION
        )
        assert consumed.node("fit").replayed
        assert consumed.all_replayed
    plan = compile_bundle_recomputation(
        bundle, "terminal", producer_node="fit", execution=EXECUTION
    )
    receipt = plan.execute()
    assert any(dependency.node_id == "fit" for dependency in receipt.dependencies)
    portable = receipt.export()
    replay = consume_bundle_recomputation(
        portable,
        expected_bundle=bundle.identity,
        expected_identity=plan.identity,
        execution=EXECUTION,
    )
    assert replay == receipt
    with pytest.raises(CausalSerializationError):
        consume_bundle_recomputation(
            portable,
            expected_bundle="wrong",
            expected_identity=plan.identity,
            execution=EXECUTION,
        )
