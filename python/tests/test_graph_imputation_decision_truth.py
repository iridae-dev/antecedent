"""Known-truth checks of every graph-imputation decision route against its data-generating truth.

Each source is generated from a declared outcome mechanism whose causal effect is known in
closed form, and the checks compare the posterior-mean decision quantities with that truth. The
truth is written down here from the generating coefficients and the declared target frame, and
nothing is read from the library:

* the outcome is ``1 + 2 t + x + 1.5 t x`` plus a small deterministic residual, so the effect of
  the treatment at covariate value ``x`` is ``2 + 1.5 x``;
* the effect transported to a target frame is therefore ``2 + 1.5 * (target mean of x)``, which
  differs from the effect in the source (mean of ``x`` one half) by more than the tolerance, so a
  route that ignored its target would fail;
* the mean outcome under ``do(t = a)`` is ``1 + 2 a + (1 + 1.5 a) * (mean of x)``;
* a binary outcome has declared cell probabilities, and the predictive probability of success
  under ``do(t = a)`` is the target mix of them;
* two Gaussian predictive worlds that share a declared residual law and one target row have a
  difference whose variance is ``2 variance (1 - correlation)`` plus the variance of the effect
  over the target rows plus the posterior variance of the mean difference.

A weak prior (scale 10) keeps the ridge shrinkage far below every tolerance, and the residuals
sum to zero within each cell of the binary designs, so the least-squares coefficients are the
cell means. Nothing here asserts a calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from antecedent import Execution
from antecedent.decision import Action, Contract, Criterion, x
from antecedent.graph_imputation_decision import (
    GaussianResidual,
    GraphImputationDecisionResult,
    MissingAtRandomTarget,
    TargetBootstrap,
    TransportTarget,
    compile_graph_imputation_decision,
)
from antecedent.joint_distribution import ScientificQuantity
from antecedent.missing import PreparedGraphImputation

ROOT = "treatment_root_pre_treatment_covariates"
CONFOUNDED = "pre_treatment_covariates"
PRIOR_SCALE = 10.0
# Residuals that sum to zero over ten consecutive cells of a design.
NOISE = (0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.15, -0.15, 0.0, 0.0)


def _execution() -> Execution:
    return Execution(seed=23)


def _producer(
    columns: dict[str, list[float | None]],
    domains: dict[str, str],
    support: str,
) -> PreparedGraphImputation:
    return PreparedGraphImputation(
        columns,
        treatment="t",
        outcome="y",
        graph_support=support,  # type: ignore[arg-type]
        domains=domains,
        imputations=2,
        iterations=2,
        n_draws=400,
        prior_scale=PRIOR_SCALE,
        execution=_execution(),
    )


def _outcome(treated: float, modifier: float, noise: float) -> float:
    return 1.0 + 2.0 * treated + modifier + 1.5 * treated * modifier + noise


def _effect_at(modifier: float) -> float:
    return 2.0 + 1.5 * modifier


def _world_at(arm: int, modifier: float) -> float:
    return 1.0 + 2.0 * arm + (1.0 + 1.5 * arm) * modifier


def _factorial() -> dict[str, list[float | None]]:
    """A balanced binary design: every (treatment, modifier) cell holds 40 rows."""
    rows = 160
    modifier = [float(row % 2) for row in range(rows)]
    treated = [float((row // 2) % 2) for row in range(rows)]
    return {
        "t": treated,
        "y": [
            _outcome(treated[row], modifier[row], NOISE[(row // 4) % len(NOISE)])
            for row in range(rows)
        ],
        "x": modifier,
    }


def _confounded() -> dict[str, list[float | None]]:
    """A binary confounder of treatment: 40 and 10 rows in the cells that agree and differ."""
    treated: list[float] = []
    modifier: list[float] = []
    noise: list[float] = []
    for cell, (arm, level, count) in enumerate(
        [(0.0, 0.0, 40), (1.0, 0.0, 10), (0.0, 1.0, 10), (1.0, 1.0, 40)]
    ):
        for index in range(count):
            treated.append(arm)
            modifier.append(level)
            noise.append(NOISE[(index + cell) % len(NOISE)])
    return {
        "t": treated,
        "y": [_outcome(treated[row], modifier[row], noise[row]) for row in range(len(treated))],
        "x": modifier,
    }


def _continuous() -> dict[str, list[float | None]]:
    """A modifier on a twenty-point grid in both arms; the residual is a small fixed pattern."""
    rows = 160
    modifier = [(row % 20) / 19.0 for row in range(rows)]
    treated = [float((row // 20) % 2) for row in range(rows)]
    return {
        "t": treated,
        "y": [
            _outcome(treated[row], modifier[row], 0.05 * (((row * 37) % 11) - 5) / 5.0)
            for row in range(rows)
        ],
        "x": modifier,
    }


def _confounded_continuous() -> dict[str, list[float | None]]:
    """A continuous modifier and a binary confounder of treatment; the effect ignores the latter."""
    rows = 160
    modifier = [(row % 20) / 19.0 for row in range(rows)]
    confounder = [float((row // 20) % 2) for row in range(rows)]
    treated = [
        float((row // 2) % 2) if confounder[row] == 0.0 else float(row % 4 != 0)
        for row in range(rows)
    ]
    return {
        "t": treated,
        "y": [
            _outcome(treated[row], modifier[row], 0.05 * (((row * 37) % 11) - 5) / 5.0)
            + 0.5 * confounder[row]
            for row in range(rows)
        ],
        "x": modifier,
        "z": confounder,
    }


def _binary_outcome() -> dict[str, list[float | None]]:
    """Cell success rates: 0.2, 0.4 (x = 0) and 0.4, 0.8 (x = 1) for control and treated."""
    treated: list[float] = []
    modifier: list[float] = []
    outcome: list[float | None] = []
    for arm, level, successes in [(0.0, 0.0, 8), (1.0, 0.0, 16), (0.0, 1.0, 16), (1.0, 1.0, 32)]:
        for index in range(40):
            treated.append(arm)
            modifier.append(level)
            outcome.append(float(index < successes))
    return {"t": treated, "y": outcome, "x": modifier}


def _clusters(rows: int, size: int = 2) -> list[str]:
    return [f"cluster-{row // size}" for row in range(rows)]


def _binary_target(
    share: float = 0.8, rows: int = 100, *, clusters: bool = False
) -> TransportTarget:
    ones = round(share * rows)
    return TransportTarget(
        columns={"x": [float(row % rows < ones) for row in range(rows)]},
        selection=("x",),
        target_population="target",
        target_sample="independent-target-sample",
        clusters=_clusters(rows) if clusters else None,
    )


def _kernel_target(*, clusters: bool = False) -> TransportTarget:
    rows = 80
    return TransportTarget(
        columns={"x": [0.6 + 0.4 * (row % 20) / 19.0 for row in range(rows)]},
        selection=("x",),
        target_population="target",
        target_sample="independent-continuous-target",
        clusters=_clusters(rows) if clusters else None,
        bandwidths={"x": 0.3},
        min_kernel_overlap=0.01,
    )


def _confounded_kernel_target(*, clusters: bool = False) -> TransportTarget:
    rows = 80
    return TransportTarget(
        columns={
            "x": [0.6 + 0.4 * (row % 20) / 19.0 for row in range(rows)],
            "z": [float(row % 10 < 7) for row in range(rows)],
        },
        selection=("x", "z"),
        target_population="target",
        target_sample="independent-target",
        clusters=_clusters(rows) if clusters else None,
        bandwidths={"x": 0.3},
        min_kernel_overlap=0.01,
    )


def _mar_target(*, kernel: bool) -> tuple[TransportTarget, float]:
    """A raw target with missing cells and declared weights, and its weighted covariate mean."""
    rows = 100
    values = [
        (0.6 + 0.4 * (row % 20) / 19.0) if kernel else float(row % 5 != 0) for row in range(rows)
    ]
    cells = [None if row % 7 == 0 else values[row] for row in range(rows)]
    weights = [1.0 + 2.0 * float(values[row] >= 0.8) for row in range(rows)]
    complete = [row for row in range(rows) if cells[row] is not None]
    mean = sum(weights[row] * values[row] for row in complete) / sum(
        weights[row] for row in complete
    )
    target = TransportTarget(
        columns={"x": cells},
        selection=("x",),
        target_population="target",
        target_sample="target-sample",
        clusters=_clusters(rows, 4),
        bandwidths={"x": 0.3} if kernel else {},
        min_kernel_overlap=0.001 if kernel else 0.0,
        missing_at_random=MissingAtRandomTarget(
            observation_mechanism="mar-v1",
            sampling_design="cluster-design-v1",
            observation_probabilities=[0.8] * rows,
            weights=weights,
        ),
    )
    return target, mean


def _effect_contract(population: str, threshold: float) -> Contract:
    effect = ScientificQuantity(
        "y", "effect", "outcome", "score", population, "do(t=1)-do(t=0)", 0, "average_effect"
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


def _utilities(result: GraphImputationDecisionResult) -> dict[str, float]:
    return {action.id: action.expected_utility for action in result.actions}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    centre = _mean(values)
    return sum((value - centre) ** 2 for value in values) / (len(values) - 1)


@dataclass(frozen=True)
class EffectCase:
    """An average-effect route: its producer, target, resampling and the true effect."""

    producer: Callable[[], PreparedGraphImputation]
    population: str
    truth: float
    tolerance: float
    declaration: dict[str, Any] = field(default_factory=dict)
    replicates: int | None = None


def _binary_producer(support: str = ROOT) -> Callable[[], PreparedGraphImputation]:
    domains = {"t": "binary", "y": "continuous", "x": "binary"}
    return lambda: _producer(_factorial(), domains, support)


def _confounded_producer() -> PreparedGraphImputation:
    domains = {"t": "binary", "y": "continuous", "x": "binary"}
    return _producer(_confounded(), domains, CONFOUNDED)


def _kernel_producer() -> PreparedGraphImputation:
    domains = {"t": "binary", "y": "continuous", "x": "continuous"}
    return _producer(_continuous(), domains, ROOT)


def _confounded_kernel_producer() -> PreparedGraphImputation:
    domains = {"t": "binary", "y": "continuous", "x": "continuous", "z": "binary"}
    return _producer(_confounded_continuous(), domains, CONFOUNDED)


def _mar_producer(kernel: bool) -> Callable[[], PreparedGraphImputation]:
    if kernel:
        return _kernel_producer
    return _binary_producer()


_MAR_BINARY_TARGET, _MAR_BINARY_MEAN = _mar_target(kernel=False)
_MAR_KERNEL_TARGET, _MAR_KERNEL_MEAN = _mar_target(kernel=True)

# The true effect at the declared target frames: 2 + 1.5 * (target mean of x).
EFFECT_CASES: dict[str, EffectCase] = {
    "effect": EffectCase(_binary_producer(), "source", _effect_at(0.5), 0.05),
    "transport": EffectCase(
        _binary_producer(), "target", _effect_at(0.8), 0.05, {"target": _binary_target()}
    ),
    "transport_rows": EffectCase(
        _binary_producer(),
        "target",
        _effect_at(0.8),
        0.15,
        {"target": _binary_target(), "bootstrap": TargetBootstrap(40, 7)},
        40,
    ),
    "transport_clusters": EffectCase(
        _binary_producer(),
        "target",
        _effect_at(0.8),
        0.2,
        {"target": _binary_target(clusters=True), "bootstrap": TargetBootstrap(40, 29)},
        40,
    ),
    "confounded_transport": EffectCase(
        _confounded_producer, "target", _effect_at(0.8), 0.05, {"target": _binary_target()}
    ),
    "confounded_transport_clusters": EffectCase(
        _confounded_producer,
        "target",
        _effect_at(0.8),
        0.2,
        {"target": _binary_target(clusters=True), "bootstrap": TargetBootstrap(40, 19)},
        40,
    ),
    "kernel_transport": EffectCase(
        _kernel_producer, "target", _effect_at(0.8), 0.1, {"target": _kernel_target()}
    ),
    "kernel_transport_clusters": EffectCase(
        _kernel_producer,
        "target",
        _effect_at(0.8),
        0.2,
        {"target": _kernel_target(clusters=True), "bootstrap": TargetBootstrap(40, 29)},
        40,
    ),
    "confounded_kernel_transport": EffectCase(
        _confounded_kernel_producer,
        "target",
        _effect_at(0.8),
        0.1,
        {"target": _confounded_kernel_target()},
    ),
    "confounded_kernel_transport_clusters": EffectCase(
        _confounded_kernel_producer,
        "target",
        _effect_at(0.8),
        0.2,
        {"target": _confounded_kernel_target(clusters=True), "bootstrap": TargetBootstrap(40, 19)},
        40,
    ),
    "mar_transport_binary": EffectCase(
        _mar_producer(False),
        "target",
        _effect_at(_MAR_BINARY_MEAN),
        0.08,
        {"target": _MAR_BINARY_TARGET},
    ),
    "mar_transport_kernel": EffectCase(
        _mar_producer(True),
        "target",
        _effect_at(_MAR_KERNEL_MEAN),
        0.12,
        {"target": _MAR_KERNEL_TARGET},
    ),
    "mar_transport_clusters": EffectCase(
        _mar_producer(False),
        "target",
        _effect_at(_MAR_BINARY_MEAN),
        0.2,
        {"target": _MAR_BINARY_TARGET, "bootstrap": TargetBootstrap(40, 7)},
        40,
    ),
}


@pytest.mark.parametrize("name", sorted(EFFECT_CASES))
def test_an_effect_decision_matches_the_true_effect_of_its_population(name: str) -> None:
    case = EFFECT_CASES[name]
    # A threshold between the source effect and the target effect where they differ, so the
    # chosen action is also decided by the truth.
    threshold = 3.0 if case.population == "target" else 2.0
    plan = compile_graph_imputation_decision(
        case.producer(), _effect_contract(case.population, threshold), **case.declaration
    )
    result = plan.execute()
    assert result.mean is not None and abs(result.mean - case.truth) < case.tolerance, (
        result.mean,
        case.truth,
    )
    if case.replicates is not None:
        assert len(result.draws) == case.replicates
    utilities = _utilities(result)
    assert abs(utilities["treat"] - (case.truth - threshold)) < case.tolerance
    assert abs(utilities["hold"]) < 1e-9
    expected = "treat" if case.truth > threshold else "hold"
    assert result.chosen[0] == expected


def test_the_target_frame_decides_the_transported_action_and_the_source_does_not() -> None:
    threshold = 3.0
    source = compile_graph_imputation_decision(
        _binary_producer()(), _effect_contract("source", threshold)
    ).execute()
    target = compile_graph_imputation_decision(
        _binary_producer()(),
        _effect_contract("target", threshold),
        target=_binary_target(),
    ).execute()
    # The effect is 2.75 in the source and 3.2 in a target where 80% of the units carry x = 1.
    assert source.chosen[0] == "hold" and target.chosen[0] == "treat"
    assert abs(_utilities(target)["treat"] - (_effect_at(0.8) - threshold)) < 0.05


def test_source_treatment_worlds_have_the_true_mean_outcomes() -> None:
    result = compile_graph_imputation_decision(
        _binary_producer()(), _world_contract("source", "mean")
    ).execute()
    truth = [_world_at(arm, 0.5) for arm in (0, 1)]
    utilities = _utilities(result)
    assert abs(utilities["action-0"] - truth[0]) < 0.05
    assert abs(utilities["action-1"] - truth[1]) < 0.05
    assert result.chosen[0] == "action-1"
    assert abs(_mean([pair[1] - pair[0] for pair in result.worlds]) - _effect_at(0.5)) < 0.05


@pytest.mark.parametrize("rows", [False, True])
def test_target_treatment_worlds_have_the_true_mean_outcomes(rows: bool) -> None:
    declaration: dict[str, Any] = {"target": _binary_target()}
    tolerance = 0.05
    if rows:
        declaration["bootstrap"] = TargetBootstrap(40, 37)
        tolerance = 0.2
    result = compile_graph_imputation_decision(
        _binary_producer()(), _world_contract("target", "mean"), **declaration
    ).execute()
    utilities = _utilities(result)
    assert abs(utilities["action-0"] - _world_at(0, 0.8)) < tolerance
    assert abs(utilities["action-1"] - _world_at(1, 0.8)) < tolerance
    assert result.chosen[0] == "action-1"


@pytest.mark.parametrize("clusters", [False, True])
def test_gaussian_predictive_worlds_have_the_true_mean_and_declared_residual_spread(
    clusters: bool,
) -> None:
    residual = GaussianResidual(variance=0.4, cross_world_correlation=0.3, seed=5)
    declaration: dict[str, Any] = {
        "target": _binary_target(clusters=clusters),
        "residual": residual,
    }
    tolerance = 0.1
    if clusters:
        declaration["bootstrap"] = TargetBootstrap(40, 17)
        tolerance = 0.4
    result = compile_graph_imputation_decision(
        _binary_producer()(), _world_contract("target", "outcome"), **declaration
    ).execute()
    utilities = _utilities(result)
    assert abs(utilities["action-0"] - _world_at(0, 0.8)) < tolerance
    assert abs(utilities["action-1"] - _world_at(1, 0.8)) < tolerance
    assert result.chosen[0] == "action-1"
    if not clusters:
        # Each world pair is drawn at one target row, so the difference is the row's effect
        # 2 + 1.5 x plus the residual difference. The residuals have correlation 0.3, so their
        # difference has variance 2 * 0.4 * (1 - 0.3) = 0.56; the effect varies over the target
        # mix (80% x = 1) with variance 1.5^2 * 0.8 * 0.2 = 0.36; the posterior variance of the
        # mean difference is negligible.
        spread = _variance([pair[1] - pair[0] for pair in result.worlds])
        assert abs(spread - (2 * 0.4 * (1 - 0.3) + 1.5**2 * 0.8 * 0.2)) < 0.15, spread


@pytest.mark.parametrize("clusters", [False, True])
def test_bernoulli_predictive_worlds_have_the_true_success_probabilities(clusters: bool) -> None:
    rows = 160
    target = TransportTarget(
        columns={"x": [float(row % 5 != 0) for row in range(rows)]},
        selection=("x",),
        target_population="target",
        target_sample="target-sample",
        clusters=_clusters(rows) if clusters else None,
    )
    declaration: dict[str, Any] = {"target": target}
    tolerance = 0.06
    if clusters:
        declaration["bootstrap"] = TargetBootstrap(40, 17)
        tolerance = 0.3
    producer = _producer(_binary_outcome(), {"t": "binary", "y": "binary", "x": "binary"}, ROOT)
    result = compile_graph_imputation_decision(
        producer, _world_contract("target", "outcome"), **declaration
    ).execute()
    # The target mix is 80% x = 1: success under control is 0.2 * 0.2 + 0.8 * 0.4 = 0.36 and
    # under treatment 0.2 * 0.4 + 0.8 * 0.8 = 0.72.
    utilities = _utilities(result)
    assert abs(utilities["action-0"] - 0.36) < tolerance, utilities
    assert abs(utilities["action-1"] - 0.72) < tolerance, utilities
    assert result.chosen[0] == "action-1"
    assert all(value in (0.0, 1.0) for pair in result.worlds for value in pair) or clusters
