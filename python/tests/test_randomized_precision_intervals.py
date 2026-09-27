from __future__ import annotations

import antecedent as ant
import numpy as np
from antecedent import interference


def _data(n: int) -> tuple[dict[str, np.ndarray], ant.ExperimentDesign]:
    assigned = [i % 2 == 0 for i in range(n)]
    x1 = np.sin(0.17 * np.arange(n))
    x2 = np.cos(0.11 * np.arange(n))
    outcome = (
        2.0 + 1.2 * x1 - 0.7 * x2 + 0.5 * np.sin(0.29 * np.arange(n))
        + 1.4 * np.asarray(assigned, dtype=float)
    )
    design = ant.ExperimentDesign(
        interference.BernoulliAssignment(0.5), assigned,
        [f"unit-{i}" for i in range(n)], [f"row-{i}" for i in range(n)],
    )
    return {"outcome": outcome, "baseline_a": x1, "baseline_b": x2}, design


def test_precision_adjusted_intervals_use_retained_public_flow():
    data, design = _data(400)
    for query in (
        ant.RandomizedEffect("outcome", design, ancova_covariates=("baseline_a", "baseline_b")),
        ant.RandomizedEffect("outcome", design, cuped=ant.FixedCUPED("baseline_a", 1.2)),
    ):
        result = ant.analyze(data, query=query, refute="none")
        fit = result.randomized_effect
        assert fit is not None
        assert fit.interval_95 is not None
        assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
        assert fit.standard_error > 0
        assert fit.support_status == "licensed"
        assert result.evidence_status == "licensed"
        assert ant.prepare(data, query=query, refute="none").estimate(data).randomized_effect.interval_95 == fit.interval_95


def test_sparse_precision_adjustment_reports_point_only():
    data, design = _data(40)
    for query in (
        ant.RandomizedEffect("outcome", design, ancova_covariates=("baseline_a", "baseline_b")),
        ant.RandomizedEffect("outcome", design, cuped=ant.FixedCUPED("baseline_a", 1.2)),
    ):
        result = ant.analyze(data, query=query, refute="none")
        assert result.randomized_effect.interval_95 is None
        assert result.randomized_effect.standard_error is None
        assert result.evidence_status == "off_axis"
