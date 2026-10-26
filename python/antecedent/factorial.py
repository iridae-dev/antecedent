"""Native-backed utilities for two-factor randomized 2×2 experiments.

This direct estimator is not a licensed ``analyze`` workflow. Its design
assumptions are stated in each result and must be justified by the caller.

The estimate is compiled, not computed beside the compiler: independent Bernoulli
assignment of two factors with known row probabilities is exactly independent
assignment among the four cells with product probabilities, so ``estimate`` builds
a multi-arm randomized-effect study over the cells, runs it through ``analyze``
(``Study.compile`` and the retained checked route), and reads the cell means and
variance contributions from the executed result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._data import as_columns
from .errors import CausalTypeError, CausalValueError


@dataclass(frozen=True, slots=True)
class FactorialDesign:
    """Independent Bernoulli probabilities for two binary randomized factors."""

    probability_a: float | Sequence[float]
    probability_b: float | Sequence[float]

    def probability_vectors(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            _probability_vector(self.probability_a, n, "probability_a"),
            _probability_vector(self.probability_b, n, "probability_b"),
        )


@dataclass(frozen=True, slots=True)
class FactorialEstimate:
    """Point contrasts and conservative variance bounds from the native kernel."""

    cell_means: dict[str, float]
    cell_support: dict[str, int]
    factor_a_effect: float
    factor_b_effect: float
    interaction_effect: float
    factor_a_variance_bound: float
    factor_b_variance_bound: float
    interaction_variance_bound: float
    assumptions: tuple[str, ...] = (
        "Both factors are independently Bernoulli randomized with the supplied known probabilities.",
        "Consistency, no interference, and a fixed finite population of input rows.",
        "The two factors have no assignment dependence within or across rows.",
    )
    uncertainty_semantics: str = (
        "Point estimates use Horvitz-Thompson cell means. Reported variances are "
        "covariance-free Young upper bounds; no confidence interval or coverage claim is made."
    )


def _probability_vector(value: float | Sequence[float], n: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim == 0:
        vector = np.full(n, float(vector), dtype=np.float64)
    if vector.ndim != 1 or len(vector) != n:
        raise CausalValueError(f"{name} must be scalar or have one value per row")
    if not np.isfinite(vector).all() or ((vector <= 0.0) | (vector >= 1.0)).any():
        raise CausalValueError(f"{name} values must be strictly between zero and one")
    return vector


def _assignment(values: Sequence[bool], n: int, name: str) -> list[bool]:
    result = list(values)
    if len(result) != n:
        raise CausalValueError(f"{name} must have one value per data row")
    if any(not isinstance(value, (bool, np.bool_)) for value in result):
        raise CausalTypeError(f"{name} must contain only booleans")
    return [bool(value) for value in result]


def estimate(
    data: Any,
    *,
    factor_a: Sequence[bool],
    factor_b: Sequence[bool],
    design: FactorialDesign,
    outcome: str,
) -> FactorialEstimate:
    """Estimate marginal main effects and the 2×2 interaction.

    Main effects average over the other factor's two levels. The interaction is
    the difference-in-differences ``mu11 - mu10 - mu01 + mu00``. Every cell must
    have observed rows, and every row must have strictly positive probability
    of assignment to all four cells. This remains a point utility outside the
    support matrix: the retained route publishes cell means and variance
    contributions, and the factorial contrasts below are formed from them.
    """

    if not isinstance(design, FactorialDesign):
        raise CausalTypeError("design must be a FactorialDesign")
    names, columns = as_columns(data)
    try:
        y = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from data") from error
    n = len(y)
    a = _assignment(factor_a, n, "factor_a")
    b = _assignment(factor_b, n, "factor_b")
    pa, pb = design.probability_vectors(n)
    labels = ("00", "01", "10", "11")
    cell = [2 * int(x) + int(z) for x, z in zip(a, b, strict=True)]
    # Independent factors give each row the product cell probabilities, so the design
    # is independent assignment among four named cells.
    probabilities = [
        [
            (pa[i] if u else 1.0 - pa[i]) * (pb[i] if v else 1.0 - pb[i])
            for u in (False, True)
            for v in (False, True)
        ]
        for i in range(n)
    ]
    from ._analyze import analyze
    from .experiment import MultiArmExperimentDesign, RandomizedEffect
    from .results import AnalysisResult

    cell_design = MultiArmExperimentDesign(
        [labels[index] for index in cell],
        labels,
        probabilities,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    result = analyze({outcome: y}, query=RandomizedEffect(outcome, cell_design), refute="none")
    if not isinstance(result, AnalysisResult):
        raise CausalValueError("the compiled randomized route returned an unexpected result type")
    fit = result.randomized_effect
    if fit is None or len(fit.multi_arm_values) != 4:
        raise CausalValueError("the compiled randomized route returned no four-cell result")
    by_label = {
        label: (value, variance, support)
        for label, value, variance, support in fit.multi_arm_values
    }
    means = [by_label[label][0] for label in labels]
    variances = [by_label[label][1] for label in labels]

    def contrast(weights: Sequence[float]) -> float:
        return float(sum(m * w for m, w in zip(means, weights, strict=True)))

    def bound(weights: Sequence[float]) -> float:
        # Young's inequality: Var(sum w_j X_j) <= 4 sum w_j^2 Var(X_j).
        return float(4.0 * sum(v * w * w for v, w in zip(variances, weights, strict=True)))

    weights_a = (-0.5, -0.5, 0.5, 0.5)
    weights_b = (-0.5, 0.5, -0.5, 0.5)
    weights_ab = (1.0, -1.0, -1.0, 1.0)
    return FactorialEstimate(
        cell_means=dict(zip(labels, means, strict=True)),
        cell_support={label: by_label[label][2] for label in labels},
        factor_a_effect=contrast(weights_a),
        factor_b_effect=contrast(weights_b),
        interaction_effect=contrast(weights_ab),
        factor_a_variance_bound=bound(weights_a),
        factor_b_variance_bound=bound(weights_b),
        interaction_variance_bound=bound(weights_ab),
    )


__all__ = ["FactorialDesign", "FactorialEstimate", "estimate"]
