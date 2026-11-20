"""Decisions on a compiled graph-imputation mixture: effect, worlds and transport, end to end.

Each route compiles from a prepared graph-imputation handle, executes, exports its portable
bytes and is replayed by the independent consumer under the retained program identity; a
wrong identity refuses. The fixtures are the smallest ones of the native route tests. Nothing
here asserts a calibration: every result says ``not_measured``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from antecedent import Execution
from antecedent.composition_bundle import Bundle
from antecedent.decision import Action, Contract, Criterion, x
from antecedent.graph_imputation_decision import (
    CompiledGraphImputationDecision,
    GaussianResidual,
    GraphImputationDecisionRefusal,
    GraphImputationDecisionResult,
    MissingAtRandomTarget,
    TargetBootstrap,
    TransportTarget,
    compile_graph_imputation_decision,
    compile_graph_imputation_transport_bundle,
    consume_graph_imputation_decision,
    consume_graph_imputation_fit_bundle,
    consume_graph_imputation_transport_bundle,
)
from antecedent.joint_distribution import ScientificQuantity
from antecedent.missing import PreparedGraphImputation

from _refusal import assert_registered_refusal

ROWS = 80
ROOT = "treatment_root_pre_treatment_covariates"
CONFOUNDED = "pre_treatment_covariates"
WRONG_PROGRAM = "0" * 64


def _execution() -> Execution:
    return Execution(seed=19)


def _producer(
    columns: dict[str, list[float | None]], domains: dict[str, str], support: str
) -> PreparedGraphImputation:
    return PreparedGraphImputation(
        columns,
        treatment="t",
        outcome="y",
        graph_support=support,  # type: ignore[arg-type]
        domains=domains,
        imputations=2,
        iterations=2,
        n_draws=100,
        prior_scale=1.0,
        execution=_execution(),
    )


def _treated(row: int) -> float:
    return float(row % 4 in (0, 3))


def _binary_source() -> dict[str, list[float | None]]:
    """A binary covariate that modifies the effect; every 13th outcome is missing."""
    covariate = [float(row % 2 == 0) for row in range(ROWS)]
    return {
        "t": [_treated(row) for row in range(ROWS)],
        "y": [
            None
            if row % 13 == 0
            else 0.2 * covariate[row]
            + _treated(row) * (0.5 + 0.5 * covariate[row])
            + 0.1 * (row % 5)
            for row in range(ROWS)
        ],
        "x": covariate,
    }


def _continuous_source() -> dict[str, list[float | None]]:
    """A continuous effect modifier on a grid; every 13th outcome is missing."""
    covariate = [(row % 20) / 19.0 for row in range(ROWS)]
    return {
        "t": [_treated(row) for row in range(ROWS)],
        "y": [
            None
            if row % 13 == 0
            else 0.2 * covariate[row]
            + _treated(row) * (0.5 + 0.5 * covariate[row])
            + 0.1 * (row % 5)
            for row in range(ROWS)
        ],
        "x": covariate,
    }


def _confounded_mixed_source() -> dict[str, list[float | None]]:
    """A binary confounder of treatment and a continuous modifier."""
    binary = [float(row % 2 == 1) for row in range(ROWS)]
    continuous = [(row % 20) / 19.0 for row in range(ROWS)]
    treated = [float(row % 10 < 2 if binary[row] == 0.0 else row % 10 < 8) for row in range(ROWS)]
    return {
        "t": treated,
        "y": [
            None
            if row % 13 == 0
            else 0.2 * binary[row]
            + 0.1 * continuous[row]
            + treated[row] * (0.5 + 0.3 * continuous[row])
            + 0.03 * (row % 7)
            for row in range(ROWS)
        ],
        "x": continuous,
        "z": binary,
    }


def _binary_outcome_source() -> dict[str, list[float | None]]:
    """A binary outcome for Bernoulli predictive worlds (96 rows)."""
    treated = [float(row % 2 == 0) for row in range(96)]
    return {
        "t": treated,
        "y": [
            None
            if row % 13 == 0
            else float((row * 7 + row // 3) % 11 < 4 + int(treated[row] > 0.5))
            for row in range(96)
        ],
        "x": [float(row % 4 < 2) for row in range(96)],
    }


def _missing_at_random_source(kernel: bool) -> dict[str, list[float | None]]:
    covariate = [(row % 8) / 8.0 if kernel else float(row % 2) for row in range(ROWS)]
    treated = [float((row // 2) % 2) for row in range(ROWS)]
    return {
        "t": treated,
        "y": [
            None
            if row % 13 == 0
            else treated[row] * (0.5 + covariate[row]) + 0.2 * covariate[row] + 0.1 * (row % 5)
            for row in range(ROWS)
        ],
        "x": covariate,
    }


def _clusters(rows: int = ROWS, size: int = 2) -> list[str]:
    return [f"cluster-{row // size}" for row in range(rows)]


def _binary_target(*, clusters: bool = False, rows: int = ROWS) -> TransportTarget:
    return TransportTarget(
        columns={"x": [float(row % 5 != 0) for row in range(rows)]},
        selection=("x",),
        target_population="target",
        target_sample="independent-target-sample",
        clusters=_clusters(rows) if clusters else None,
    )


def _kernel_target(*, clusters: bool = False) -> TransportTarget:
    return TransportTarget(
        columns={"x": [0.2 + 0.8 * (row % 20) / 19.0 for row in range(ROWS)]},
        selection=("x",),
        target_population="target",
        target_sample="independent-continuous-target",
        clusters=_clusters() if clusters else None,
        bandwidths={"x": 0.3},
        min_kernel_overlap=0.01,
    )


def _confounded_kernel_target(*, clusters: bool = False) -> TransportTarget:
    return TransportTarget(
        columns={
            "x": [0.1 + 0.8 * (row % 20) / 19.0 for row in range(ROWS)],
            "z": [float(row % 3 != 0) for row in range(ROWS)],
        },
        selection=("x", "z"),
        target_population="target",
        target_sample="independent-target",
        clusters=_clusters() if clusters else None,
        bandwidths={"x": 0.3},
        min_kernel_overlap=0.01,
    )


def _missing_at_random_target(kernel: bool) -> TransportTarget:
    covariate = _missing_at_random_source(kernel)["x"]
    return TransportTarget(
        columns={"x": [None if row % 7 == 0 else covariate[row] for row in range(ROWS)]},
        selection=("x",),
        target_population="target",
        target_sample="target-sample",
        clusters=_clusters(size=4),
        bandwidths={"x": 0.5} if kernel else {},
        min_kernel_overlap=0.001 if kernel else 0.0,
        missing_at_random=MissingAtRandomTarget(
            observation_mechanism="mar-v1",
            sampling_design="cluster-design-v1",
            observation_probabilities=[0.8] * ROWS,
            weights=[2.0 if row % 2 == 0 else 1.0 for row in range(ROWS)],
        ),
    )


def _effect_contract(population: str, threshold: float = 0.2) -> Contract:
    effect = ScientificQuantity(
        "y",
        "target effect",
        "outcome",
        "score",
        population,
        "do(t=1)-do(t=0)",
        0,
        "average_effect",
    )
    return Contract(
        actions=(
            Action("treat", (effect,), x(0) - threshold, kind="external"),
            Action("hold", (effect,), x(0) * 0.0, kind="external"),
        ),
        utility_units="score",
        criterion=Criterion.expected_utility(),
        target_population=population,
        structural_policy="require_invariant_best_action",
    )


def _world_contract(population: str, functional: str) -> Contract:
    worlds = [
        ScientificQuantity(
            "y", "outcome", "outcome", "score", population, f"do(t={arm})", 0, functional
        )
        for arm in (0, 1)
    ]
    return Contract(
        actions=tuple(Action(f"action-{arm}", (world,), x(0)) for arm, world in enumerate(worlds)),
        utility_units="score",
        criterion=Criterion.expected_utility(),
        target_population=population,
        structural_policy="require_invariant_best_action",
    )


@dataclass(frozen=True)
class Case:
    """One route: its producer, contract and declaration, and what the result must show."""

    route: str
    producer: Callable[[], PreparedGraphImputation]
    contract: Contract
    declaration: dict[str, Any] = field(default_factory=dict)
    replicates: int | None = None


def _binary_producer(support: str = ROOT) -> Callable[[], PreparedGraphImputation]:
    domains = {"t": "binary", "y": "continuous", "x": "binary"}
    return lambda: _producer(_binary_source(), domains, support)


def _kernel_producer() -> PreparedGraphImputation:
    return _producer(
        _continuous_source(), {"t": "binary", "y": "continuous", "x": "continuous"}, ROOT
    )


def _confounded_kernel_producer() -> PreparedGraphImputation:
    domains = {"t": "binary", "y": "continuous", "x": "continuous", "z": "binary"}
    return _producer(_confounded_mixed_source(), domains, CONFOUNDED)


def _binary_outcome_producer() -> PreparedGraphImputation:
    domains = {"t": "binary", "y": "binary", "x": "binary"}
    return _producer(_binary_outcome_source(), domains, ROOT)


def _missing_at_random_producer(
    kernel: bool, support: str
) -> Callable[[], PreparedGraphImputation]:
    domains = {"t": "binary", "y": "continuous", "x": "continuous" if kernel else "binary"}
    return lambda: _producer(_missing_at_random_source(kernel), domains, support)


def _bernoulli_target(*, clusters: bool = False) -> TransportTarget:
    return TransportTarget(
        columns={"x": [float(row % 4 < 2) for row in range(96)]},
        selection=("x",),
        target_population="target",
        target_sample="target-sample",
        clusters=_clusters(96) if clusters else None,
    )


ROW_BOOTSTRAP = TargetBootstrap(replicates=40, seed=7)
CLUSTER_BOOTSTRAP = TargetBootstrap(replicates=40, seed=29)
RESIDUAL = GaussianResidual(variance=0.4, cross_world_correlation=0.3, seed=5)

CASES: dict[str, Case] = {
    "effect": Case("effect", _binary_producer(), _effect_contract("source", 0.1)),
    "world_means": Case("world", _binary_producer(), _world_contract("source", "mean")),
    "transport": Case(
        "transport", _binary_producer(), _effect_contract("target"), {"target": _binary_target()}
    ),
    "transport_rows": Case(
        "transport",
        _binary_producer(),
        _effect_contract("target"),
        {"target": _binary_target(), "bootstrap": ROW_BOOTSTRAP},
        40,
    ),
    "transport_clusters": Case(
        "transport",
        _binary_producer(),
        _effect_contract("target"),
        {"target": _binary_target(clusters=True), "bootstrap": CLUSTER_BOOTSTRAP},
        40,
    ),
    "confounded_transport": Case(
        "transport",
        _binary_producer(CONFOUNDED),
        _effect_contract("target"),
        {"target": _binary_target()},
    ),
    "confounded_transport_rows": Case(
        "transport",
        _binary_producer(CONFOUNDED),
        _effect_contract("target"),
        {"target": _binary_target(), "bootstrap": TargetBootstrap(replicates=40, seed=11)},
        40,
    ),
    "confounded_transport_clusters": Case(
        "transport",
        _binary_producer(CONFOUNDED),
        _effect_contract("target"),
        {"target": _binary_target(clusters=True), "bootstrap": TargetBootstrap(40, 19)},
        40,
    ),
    "kernel_transport": Case(
        "kernel_transport",
        _kernel_producer,
        _effect_contract("target"),
        {"target": _kernel_target()},
    ),
    "kernel_transport_rows": Case(
        "kernel_transport",
        _kernel_producer,
        _effect_contract("target"),
        {"target": _kernel_target(), "bootstrap": ROW_BOOTSTRAP},
        40,
    ),
    "kernel_transport_clusters": Case(
        "kernel_transport",
        _kernel_producer,
        _effect_contract("target"),
        {"target": _kernel_target(clusters=True), "bootstrap": CLUSTER_BOOTSTRAP},
        40,
    ),
    "confounded_kernel_transport": Case(
        "kernel_transport",
        _confounded_kernel_producer,
        _effect_contract("target", 0.0),
        {"target": _confounded_kernel_target()},
    ),
    "confounded_kernel_transport_rows": Case(
        "kernel_transport",
        _confounded_kernel_producer,
        _effect_contract("target", 0.0),
        {"target": _confounded_kernel_target(), "bootstrap": TargetBootstrap(40, 13)},
        40,
    ),
    "confounded_kernel_transport_clusters": Case(
        "kernel_transport",
        _confounded_kernel_producer,
        _effect_contract("target", 0.0),
        {"target": _confounded_kernel_target(clusters=True), "bootstrap": TargetBootstrap(40, 19)},
        40,
    ),
    **{
        f"mar_transport_{'kernel' if kernel else 'binary'}_{'confounded' if confounded else 'root'}": Case(
            "mar_transport",
            _missing_at_random_producer(kernel, CONFOUNDED if confounded else ROOT),
            _effect_contract("target", 0.0),
            {"target": _missing_at_random_target(kernel)},
        )
        for kernel in (False, True)
        for confounded in (False, True)
    },
    "mar_transport_clusters": Case(
        "mar_transport",
        _missing_at_random_producer(False, ROOT),
        _effect_contract("target", 0.0),
        {"target": _missing_at_random_target(False), "bootstrap": ROW_BOOTSTRAP},
        40,
    ),
    "transport_world_means": Case(
        "world",
        _binary_producer(),
        _world_contract("target", "mean"),
        {"target": _binary_target()},
    ),
    "transport_world_means_rows": Case(
        "world",
        _binary_producer(),
        _world_contract("target", "mean"),
        {"target": _binary_target(), "bootstrap": TargetBootstrap(40, 37)},
        40,
    ),
    "bernoulli_predictive_worlds": Case(
        "world",
        _binary_outcome_producer,
        _world_contract("target", "outcome"),
        {"target": _bernoulli_target()},
    ),
    "bernoulli_predictive_worlds_clusters": Case(
        "world",
        _binary_outcome_producer,
        _world_contract("target", "outcome"),
        {"target": _bernoulli_target(clusters=True), "bootstrap": TargetBootstrap(40, 17)},
        40,
    ),
    "gaussian_predictive_worlds": Case(
        "world",
        _binary_producer(),
        _world_contract("target", "outcome"),
        {"target": _binary_target(), "residual": RESIDUAL},
    ),
    "gaussian_predictive_worlds_clusters": Case(
        "world",
        _binary_producer(),
        _world_contract("target", "outcome"),
        {
            "target": _binary_target(clusters=True),
            "bootstrap": TargetBootstrap(40, 17),
            "residual": RESIDUAL,
        },
        40,
    ),
}


def _compile(case: Case) -> CompiledGraphImputationDecision:
    return compile_graph_imputation_decision(case.producer(), case.contract, **case.declaration)


def _same_decision(
    left: GraphImputationDecisionResult, right: GraphImputationDecisionResult
) -> None:
    assert left.receipt == right.receipt
    assert left.verdict == right.verdict
    assert left.draws == right.draws
    assert left.worlds == right.worlds
    assert [a.expected_utility for a in left.actions] == [a.expected_utility for a in right.actions]
    assert left.chain_identity == right.chain_identity


@pytest.mark.parametrize("name", sorted(CASES))
def test_decision_executes_exports_and_replays_under_its_program(name: str) -> None:
    case = CASES[name]
    plan = _compile(case)
    assert plan.route == case.route
    result = plan.execute()
    assert result.route == case.route and result.program == plan.program
    assert result.calibration == "not_measured"
    assert {action.id for action in result.actions} <= {"treat", "hold", "action-0", "action-1"}
    if case.route == "world":
        assert result.worlds and result.mean is None and not result.draws
        assert all(math.isfinite(value) for pair in result.worlds for value in pair)
        if case.replicates is not None:
            assert len(result.worlds) == case.replicates
    else:
        assert result.draws and result.mean is not None and math.isfinite(result.mean)
        if case.replicates is not None:
            assert len(result.draws) == case.replicates
    artifact = plan.export()
    assert artifact.program == plan.program and artifact.route == case.route
    replay = consume_graph_imputation_decision(
        artifact.data, program=artifact.program, execution=_execution()
    )
    _same_decision(result, replay)
    assert replay.route == case.route and replay.program == plan.program
    with pytest.raises(GraphImputationDecisionRefusal) as refused:
        consume_graph_imputation_decision(artifact, program=WRONG_PROGRAM, execution=_execution())
    assert_registered_refusal(refused.value)


def test_predictive_bernoulli_worlds_are_binary_outcomes() -> None:
    result = _compile(CASES["bernoulli_predictive_worlds"]).execute()
    assert all(value in (0.0, 1.0) for pair in result.worlds for value in pair)


def test_the_resampling_policy_is_part_of_the_program() -> None:
    fixed = _compile(CASES["transport"]).program
    rows = _compile(CASES["transport_rows"]).program
    clusters = _compile(CASES["transport_clusters"]).program
    confounded = _compile(CASES["confounded_transport"]).program
    assert len({fixed, rows, clusters, confounded}) == 4


def test_export_before_execution_and_changed_bytes_refuse() -> None:
    plan = _compile(CASES["transport"])
    with pytest.raises(GraphImputationDecisionRefusal):
        plan.export()
    with pytest.raises(GraphImputationDecisionRefusal):
        plan.chain_artifact()
    plan.execute()
    artifact = plan.export()
    changed = bytearray(artifact.data)
    changed[len(changed) // 2] ^= 1
    with pytest.raises(GraphImputationDecisionRefusal):
        consume_graph_imputation_decision(
            bytes(changed), program=artifact.program, execution=_execution()
        )
    with pytest.raises(GraphImputationDecisionRefusal):
        consume_graph_imputation_decision(b"candidate", program=artifact.program)


@pytest.mark.parametrize(
    ("contract", "declaration"),
    [
        # Resampling and a residual law need a target frame.
        (_effect_contract("source"), {"bootstrap": ROW_BOOTSTRAP}),
        (_world_contract("source", "mean"), {"residual": RESIDUAL}),
        # A residual law belongs to predictive worlds, not an average effect.
        (_effect_contract("target"), {"target": _binary_target(), "residual": RESIDUAL}),
        # Predictive worlds resample whole clusters, world means resample rows.
        (
            _world_contract("target", "outcome"),
            {"target": _binary_target(), "bootstrap": ROW_BOOTSTRAP, "residual": RESIDUAL},
        ),
        (
            _world_contract("target", "mean"),
            {"target": _binary_target(clusters=True), "bootstrap": CLUSTER_BOOTSTRAP},
        ),
        # Treatment worlds take a binary target frame.
        (_world_contract("target", "mean"), {"target": _kernel_target()}),
        # A complete-frame route does not accept a missing target cell.
        (
            _effect_contract("target"),
            {
                "target": TransportTarget(
                    columns={"x": [None] + [1.0] * (ROWS - 1)},
                    selection=("x",),
                    target_population="target",
                    target_sample="t",
                )
            },
        ),
        # A target column must be a source column.
        (
            _effect_contract("target"),
            {
                "target": TransportTarget(
                    columns={"w": [1.0] * ROWS},
                    selection=("w",),
                    target_population="target",
                    target_sample="t",
                )
            },
        ),
        # The effect coordinate names another population than the target.
        (_effect_contract("source"), {"target": _binary_target()}),
    ],
)
def test_declarations_that_do_not_fit_the_mixture_refuse(
    contract: Contract, declaration: dict[str, Any]
) -> None:
    with pytest.raises(GraphImputationDecisionRefusal) as refused:
        compile_graph_imputation_decision(_binary_producer()(), contract, **declaration)
    assert_registered_refusal(refused.value)


def test_arguments_of_the_wrong_type_raise_type_errors() -> None:
    with pytest.raises(TypeError, match="PreparedGraphImputation"):
        compile_graph_imputation_decision(object(), _effect_contract("source"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Contract"):
        compile_graph_imputation_decision(_binary_producer()(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TransportTarget"):
        compile_graph_imputation_decision(
            _binary_producer()(),
            _effect_contract("target"),
            target={"x": [1.0]},  # type: ignore[arg-type]
        )


def test_transported_fit_composes_into_a_bundle_and_replays() -> None:
    plan = _compile(CASES["transport"])
    plan.execute()
    artifact = plan.export()
    builder = Bundle.builder()
    builder.add_graph_imputation_transport_fit(artifact, node_id="fit", execution=_execution())
    builder.add_artifact("propagation_chain", plan.chain_artifact(), node_id="terminal")
    builder.connect("fit", "terminal")
    bundle = builder.build()
    consumed = consume_graph_imputation_fit_bundle(
        bundle.export(), expected_identity=bundle.identity, execution=_execution()
    )
    # The transported fit node is a supported route in release and development builds alike.
    assert consumed.all_replayed and consumed.node("fit").verified
    recomputation = compile_graph_imputation_transport_bundle(
        bundle, fit="fit", terminal="terminal", execution=_execution()
    )
    receipt = recomputation.execute()
    assert receipt.bundle_identity == bundle.identity
    assert receipt.producer_program == plan.program
    assert receipt.program == recomputation.program
    assert {dependency.node_id for dependency in receipt.dependencies} >= {"fit", "terminal"}
    sealed = recomputation.export()
    replayed = consume_graph_imputation_transport_bundle(
        sealed.data, bundle=sealed.bundle, program=sealed.program, execution=_execution()
    )
    assert replayed == receipt
    with pytest.raises(GraphImputationDecisionRefusal):
        consume_graph_imputation_transport_bundle(
            sealed, bundle="wrong", program=sealed.program, execution=_execution()
        )
    with pytest.raises(GraphImputationDecisionRefusal):
        consume_graph_imputation_transport_bundle(
            sealed, bundle=sealed.bundle, program=WRONG_PROGRAM, execution=_execution()
        )


def test_a_bundle_terminal_must_depend_on_the_fit() -> None:
    plan = _compile(CASES["transport"])
    plan.execute()
    builder = Bundle.builder()
    builder.add_graph_imputation_transport_fit(plan.export(), node_id="fit", execution=_execution())
    builder.add_artifact("propagation_chain", plan.chain_artifact(), node_id="terminal")
    with pytest.raises(GraphImputationDecisionRefusal):
        compile_graph_imputation_transport_bundle(
            builder.build(), fit="fit", terminal="terminal", execution=_execution()
        )
