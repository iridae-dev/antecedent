"""Native-backed utilities for two-factor randomized 2×2 experiments.

This direct estimator is not a licensed ``analyze`` workflow. Its design
assumptions are stated in each result and must be justified by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._data import as_columns
from ._native import estimate_factorial_2x2 as _estimate_factorial_2x2
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
    support matrix.
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
    raw = _estimate_factorial_2x2(y, a, b, pa, pb)
    labels = ("00", "01", "10", "11")
    return FactorialEstimate(
        cell_means=dict(zip(labels, raw.cell_means, strict=True)),
        cell_support=dict(zip(labels, raw.cell_support, strict=True)),
        factor_a_effect=raw.factor_a_effect,
        factor_b_effect=raw.factor_b_effect,
        interaction_effect=raw.interaction_effect,
        factor_a_variance_bound=raw.factor_a_variance_bound,
        factor_b_variance_bound=raw.factor_b_variance_bound,
        interaction_variance_bound=raw.interaction_variance_bound,
    )


__all__ = ["FactorialDesign", "FactorialEstimate", "estimate"]
