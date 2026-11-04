"""Frequentist randomized quantile treatment effect through the package.

The design has forty rows: the control arm carries the outcomes 1, ..., 20 and the treated arm
3, ..., 22 under ``t -> y``. The treated law is the control law shifted by two, so the median
effect is exactly 2. On the pooled-outcome grid (the integers 3, ..., 20) the arm distribution
functions are ``F0(c) = c / 20`` and ``F1(c) = (c - 2) / 20``, the piecewise-linear inversion at
1/2 gives the quantiles 10 and 12, and every row influence of the difference is +-20, so the
standard error is ``20 / sqrt(39)``. Both numbers are derived on paper (see
``crates/antecedent-estimate/tests/randomized_quantile_reference.rs``), not read from the
library; the tolerance is 1e-9, the double-precision error of the bracket weight.
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
    assert result.estimate.se_analytic == pytest.approx(20.0 / math.sqrt(39.0), abs=1e-9)


def test_a_confounded_graph_is_refused_with_the_adjusting_route_named() -> None:
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    data = _data() | {"z": [float(i % 2) for i in range(40)]}
    outcome = _run(graph, data)
    with pytest.raises(CausalError, match="adjustment set"):
        require_success(outcome)
