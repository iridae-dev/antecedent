"""Plug-in additive-GAM derivatives against an analytic quadratic dose-response.

The outcome is ``y = 1 + a^2 + 3 b + e`` over a 21 x 21 lattice of ``(a, b)`` on [-1, 1]^2 with
seeded noise of standard deviation 0.02. A cubic B-spline basis contains every quadratic and the
treatment smooths are unpenalized, so the fitted smooth in ``a`` carries no approximation bias
and the plug-in gradient at an interior point is the analytic one: ``dy/da = 2 a`` and
``dy/db = 3``. At ``(0.3, 0.5)`` the Jacobian row is ``(0.6, 3.0)`` and the derivative along
``(-2, 1)`` is ``-2 * 0.6 + 3 = 1.8``. The lattice makes the design non-collinear. The remaining
error is sampling noise on 441 rows, whose derivative standard error is about 0.005, so the
tolerance of 0.05 is about ten standard errors and still far below the 0.6 slope being recovered.
The single-treatment case checks a negative direction on the nonlinear coordinate alone:
``-2 * (2 * 0.3) = -1.2`` on ``y = 1 + a^2 + e`` (same noise, same tolerance).
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

TOLERANCE = 0.05


def _lattice() -> dict[str, np.ndarray]:
    grid = np.linspace(-1.0, 1.0, 21)
    a, b = (v.ravel() for v in np.meshgrid(grid, grid))
    noise = np.random.default_rng(17).normal(scale=0.02, size=a.size)
    return {"a": a, "b": b, "y": 1.0 + a**2 + 3.0 * b + noise}


def test_the_jacobian_recovers_the_analytic_gradient_of_a_quadratic_response() -> None:
    result = require_success(
        antecedent.analyze(
            _lattice(),
            query=antecedent.ResponseJacobian(["a", "b"], ["y"], at=[0.3, 0.5]),
            graph=[("a", "y"), ("b", "y")],
        )
    )
    assert np.asarray(result.estimate).ravel() == pytest.approx([0.6, 3.0], abs=TOLERANCE)


def test_the_directional_derivative_is_the_gradient_dotted_with_the_direction() -> None:
    result = require_success(
        antecedent.analyze(
            _lattice(),
            query=antecedent.DirectionalDerivative(
                ["a", "b"], ["y"], at=[0.3, 0.5], direction=[-2.0, 1.0]
            ),
            graph=[("a", "y"), ("b", "y")],
        )
    )
    assert np.asarray(result.estimate).ravel() == pytest.approx([1.8], abs=TOLERANCE)


def test_a_single_treatment_direction_scales_the_analytic_slope() -> None:
    a = _lattice()["a"]
    noise = np.random.default_rng(23).normal(scale=0.02, size=a.size)
    result = require_success(
        antecedent.analyze(
            {"a": a, "y": 1.0 + a**2 + noise},
            query=antecedent.DirectionalDerivative(["a"], ["y"], at=[0.3], direction=[-2.0]),
            graph=[("a", "y")],
        )
    )
    assert np.asarray(result.estimate).ravel() == pytest.approx([-1.2], abs=TOLERANCE)
