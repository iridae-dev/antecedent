"""The adjustment contrast of a finite-discrete frequency table with its posterior draws.

A binary treatment ``t``, a binary adjuster ``z`` and a binary outcome ``y`` give an eight-cell
frequency table; cell ``4 * t + 2 * z + y`` counts the rows with that combination. The adjustment
formula ``sum_z P(z) * P(y | t, z)`` is planned exactly over the table, and the same plan is
evaluated at weighted posterior draws of the joint, so the plug-in contrast and the posterior
mean and equal-tailed interval come from one exact plan of one expression.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from .errors import CausalTypeError, CausalValueError

__all__ = ["FiniteDiscreteContrast", "finite_discrete_contrast"]


@dataclass(frozen=True)
class FiniteDiscreteContrast:
    """Active mean minus control mean, as the plug-in table and the posterior report it."""

    plug_in_effect: float
    """Empirical plug-in contrast of the supplied table."""
    posterior_mean: float
    """Weighted posterior mean of the contrast over the draws."""
    posterior_lower: float
    """Equal-tailed lower posterior quantile."""
    posterior_upper: float
    """Equal-tailed upper posterior quantile."""
    sample_size: int
    """Rows of the frequency table."""
    plan_compilations: int
    """Exact plans compiled for the four cells."""


def _cells(values: Sequence[int], what: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of eight counts")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CausalValueError(f"{what} must be non-negative integers")
        out.append(int(value))
    if len(out) != 8:
        raise CausalValueError(f"{what} must have exactly 8 cells (index 4t + 2z + y)")
    return out


def finite_discrete_contrast(
    counts: Sequence[int],
    draws: Sequence[tuple[Sequence[int], float]],
    *,
    cancel: Any = None,
) -> FiniteDiscreteContrast:
    """Plan the adjustment over ``counts`` and evaluate it at weighted posterior ``draws``.

    ``counts`` is the eight-cell frequency table; ``draws`` is a sequence of
    ``(counts, weight)`` pairs, each an eight-cell table of the same joint, whose weights sum to
    one. A table with no mass, a table too large to read back, a weight that is negative or not
    finite, or weights that do not sum to one are refused.
    """
    table = _cells(counts, "counts")
    weighted = []
    for item in draws:
        cells, weight = item
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise CausalTypeError("each posterior draw weight must be a real number")
        if not math.isfinite(weight):
            raise CausalValueError("each posterior draw weight must be finite")
        weighted.append((_cells(cells, "each posterior draw"), float(weight)))
    effect, mean, lower, upper, rows, plans = _native.finite_discrete_adjustment_contrast(
        table, weighted, cancel
    )
    return FiniteDiscreteContrast(
        plug_in_effect=effect,
        posterior_mean=mean,
        posterior_lower=lower,
        posterior_upper=upper,
        sample_size=rows,
        plan_compilations=plans,
    )
