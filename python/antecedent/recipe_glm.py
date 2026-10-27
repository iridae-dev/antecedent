"""A logistic g-computation run through admitted recipes.

The outcome model is a logistic regression on ``[1 | T | Z...]``. It is fitted by the
generalized-linear recipe: the provider must declare a differentiable log density and estimating
equations, its score is checked against the gradient of its log density, and the Newton solve is
the recipe's. The standardized means at ``T = 1`` and ``T = 0`` are the frequentist
g-computation recipe's plug-in means, evaluated by a prediction-only boundary provider at the
fitted coefficients. The effect is their difference.

A provider that lacks a capability, or whose body does not answer the log density, is refused
before any row is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution

__all__ = ["LogisticFit", "LogisticProvider", "logistic_provider", "logistic_standardized_effect"]

LogisticProvider = _native.LogisticProvider
"""The provider declaration the logistic fit requires, with its log density answered."""


def logistic_provider(id: int, name: str) -> LogisticProvider:  # noqa: A002
    """Declare the logistic outcome-model provider ``id`` named ``name``.

    It declares a differentiable log density and estimating equations and answers its log
    density. A blank name is refused.
    """
    if isinstance(id, bool) or not isinstance(id, int) or id < 0:
        raise CausalTypeError("id must be a non-negative integer")
    if not isinstance(name, str):
        raise CausalTypeError("name must be a string")
    return _native.LogisticProvider(id, name)


@dataclass(frozen=True)
class LogisticFit:
    """A standardized logistic effect and the fit behind it."""

    effect: float
    """Standardized mean outcome at ``T = 1`` minus at ``T = 0``."""
    treated_mean: float
    control_mean: float
    coefficients: NDArray[np.float64]
    """Fitted coefficients on ``[1 | T | Z...]``."""
    covariance: NDArray[np.float64]
    """Sandwich covariance of the coefficients, square in the coefficients."""
    iterations: int
    rows: int


def logistic_standardized_effect(
    data: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str] = (),
    provider: LogisticProvider,
    execution: Execution | None = None,
) -> LogisticFit:
    """Fit the logistic outcome model and average its predictions at ``T = 1`` and ``T = 0``.

    ``data`` maps column names to complete numeric arrays; ``outcome`` is coded 0/1. ``provider``
    is bound to both recipes, so it must be :func:`logistic_provider` (or declare the same
    capabilities and answer its log density).

    Raises:
        CausalValueError: a column is missing, not numeric or has a missing value, the response
            is not 0/1, the score disagrees with the log-density gradient, the solve does not
            converge or the provider lacks a declared or answered capability.
    """
    execution = resolve_execution(execution)
    if not isinstance(data, Mapping):
        raise CausalTypeError("data must be a mapping of column name to array")
    if not isinstance(provider, _native.LogisticProvider):
        raise CausalTypeError("provider must come from logistic_provider")
    needed = list(dict.fromkeys([treatment, outcome, *adjustment]))
    columns = []
    for name in needed:
        if name not in data:
            raise CausalValueError(f"data has no column {name!r}")
        values = np.asarray(data[name], dtype=np.float64)
        if values.ndim != 1:
            raise CausalValueError(f"column {name!r} must be one-dimensional")
        columns.append((name, [float(v) for v in values]))
    effect, treated, control, coefficients, covariance, iterations, rows = (
        _native.recipe_logistic_standardized_effect(
            columns, treatment, outcome, list(adjustment), provider, execution.cancel
        )
    )
    width = len(coefficients)
    return LogisticFit(
        effect=effect,
        treated_mean=treated,
        control_mean=control,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64).reshape(width, width),
        iterations=iterations,
        rows=rows,
    )
