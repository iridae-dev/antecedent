"""Frequentist randomized quantile treatment effect through the package.

The design has forty rows: the control arm carries the outcomes 1, ..., 20 and the treated arm
3, ..., 22 under ``t -> y``. The treated law is the control law shifted by two, so the median
effect is exactly 2. On the pooled-outcome grid (the integers 3, ..., 20) the arm distribution
functions are ``F0(c) = c / 20`` and ``F1(c) = (c - 2) / 20``, the piecewise-linear inversion at
1/2 lands on a grid point and gives the quantiles 10 and 12, and the influence of each arm's
distribution function at its quantile is ``(1{Y <= q} - 1/2) / (1/2) = +-1`` on the arm's rows.

The density dividing that influence is each arm's Siddiqui sparsity estimate
``2h / (Q(1/2 + h) - Q(1/2 - h))`` over the arm's type-7 empirical quantile function, with the
Hall-Sheather bandwidth ``h = (z^2 * 1.5 * phi(0)^2 / 20)^(1/3) = 0.3579...`` (``z`` the 97.5%
normal point, ``phi(0)^2 = 1 / (2 pi)``), below the edge cap ``0.9 * 1/2``. Each arm is twenty
equally spaced integers, so its quantile function is linear, ``Q0(p) = 1 + 19 p`` and
``Q1(p) = 3 + 19 p``; the spread is ``38 h`` and the density ``2h / (38 h) = 1/19`` whatever
``h`` is. Every row influence of the difference is therefore ``+-19``, with mean zero, so the
standard error is ``sqrt(40 * 361 / (39 * 40)) = 19 / sqrt(39)``. Both numbers are derived on
paper (see ``crates/antecedent-estimate/tests/randomized_quantile_reference.rs``), not read
from the library; the tolerance is 1e-9, the double-precision error of the bracket weight and
of the cancellation of ``h``.
"""

from __future__ import annotations

import math

import antecedent
import pytest
from antecedent.errors import CausalError
from antecedent.outcomes import require_success
from antecedent.query import Quantile


def _data() -> dict[str, list[float]]:
    return {
        "t": [0.0] * 20 + [1.0] * 20,
        "y": [float(v) for v in range(1, 21)] + [float(v) for v in range(3, 23)],
    }


def _run(graph: antecedent.Dag, data: dict[str, list[float]], tau: float = 0.5):
    return antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y", outcome_functional=Quantile(tau)),
        estimator=antecedent.Estimator.RANDOMIZED_QUANTILE,
        refute="none",
        execution=antecedent.Execution(seed=3),
    )


def test_randomized_quantile_recovers_the_hand_derived_median_effect() -> None:
    graph = antecedent.Dag.from_edges(["t", "y"], [("t", "y")])
    result = require_success(_run(graph, _data()))
    assert result.estimate.estimator_id == "randomized.quantile"
    assert result.estimate.ate == pytest.approx(2.0, abs=1e-9)
    assert result.estimate.se_analytic == pytest.approx(19.0 / math.sqrt(39.0), abs=1e-9)


def test_a_confounded_graph_is_refused_with_the_adjusting_route_named() -> None:
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    data = _data() | {"z": [float(i % 2) for i in range(40)]}
    outcome = _run(graph, data)
    with pytest.raises(CausalError, match="adjustment set"):
        require_success(outcome)
