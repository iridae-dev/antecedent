from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.outcomes import require_success


def _data(n: int) -> tuple[dict[str, np.ndarray], ant.experiment.ExperimentDesign]:
    assigned = [i % 2 == 0 for i in range(n)]
    x1 = np.sin(0.17 * np.arange(n))
    x2 = np.cos(0.11 * np.arange(n))
    outcome = (
        2.0
        + 1.2 * x1
        - 0.7 * x2
        + 0.5 * np.sin(0.29 * np.arange(n))
        + 1.4 * np.asarray(assigned, dtype=float)
    )
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assigned,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    return {"outcome": outcome, "baseline_a": x1, "baseline_b": x2}, design


def _references(data, n):
    """Independent point estimates and the ANCOVA robust (HC0) standard error, from numpy only."""
    arm = np.array([i % 2 == 0 for i in range(n)], dtype=float)
    y, x1, x2 = data["outcome"], data["baseline_a"], data["baseline_b"]

    def difference(values):
        return values[arm == 1].mean() - values[arm == 0].mean()

    design = np.column_stack([np.ones(n), arm, x1 - x1.mean(), x2 - x2.mean()])
    gram_inverse = np.linalg.inv(design.T @ design)
    coefficients = gram_inverse @ design.T @ y
    residual = y - design @ coefficients
    covariance = gram_inverse @ (design.T * residual**2) @ design @ gram_inverse
    return {
        "plain": difference(y),
        "cuped": difference(y - 1.2 * (x1 - x1.mean())),
        "ancova": coefficients[1],
        "ancova_se": float(np.sqrt(covariance[1, 1])),
    }


def test_precision_adjusted_intervals_use_retained_public_flow():
    data, design = _data(400)
    reference = _references(data, 400)
    ancova_query = ant.experiment.RandomizedEffect(
        "outcome", design, ancova_covariates=("baseline_a", "baseline_b")
    )
    plain = require_success(
        ant.analyze(data, query=ant.experiment.RandomizedEffect("outcome", design), refute="none")
    ).randomized_effect
    ancova = require_success(ant.analyze(data, query=ancova_query, refute="none")).randomized_effect
    # The point estimates equal the independent least-squares and centred-difference references,
    # the ANCOVA standard error is the robust (HC0) one, and it is far below the unadjusted one.
    assert plain.effect == pytest.approx(reference["plain"], abs=1e-9)
    assert ancova.effect == pytest.approx(reference["ancova"], abs=1e-9)
    assert ancova.standard_error == pytest.approx(reference["ancova_se"], rel=1e-6)
    assert ancova.standard_error < 0.2 * plain.standard_error
    for query in (
        ant.experiment.RandomizedEffect(
            "outcome", design, ancova_covariates=("baseline_a", "baseline_b")
        ),
        ant.experiment.RandomizedEffect(
            "outcome", design, cuped=ant.experiment.FixedCUPED("baseline_a", 1.2)
        ),
    ):
        result = require_success(ant.analyze(data, query=query, refute="none"))
        fit = result.randomized_effect
        assert fit is not None
        assert fit.interval_95 is not None
        assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
        assert fit.standard_error > 0
        # The interval is the normal-theory one: effect +/- 1.96 standard errors.
        assert fit.interval_95[1] - fit.interval_95[0] == pytest.approx(
            2 * 1.959964 * fit.standard_error, rel=1e-4
        )
        # The design's true effect is 1.4 and the baselines absorb all but a small trend.
        assert fit.effect == pytest.approx(1.4, abs=0.02)
        assert fit.support_status == "licensed"
        if query.cuped is not None:
            assert fit.effect == pytest.approx(reference["cuped"], abs=1e-9)
        assert result.evidence_status == "licensed"
        assert (
            require_success(
                ant.estimate(ant.prepare(data, query=query, refute="none"))
            ).randomized_effect.interval_95
            == fit.interval_95
        )


def test_sparse_precision_adjustment_reports_point_only():
    data, design = _data(40)
    for query in (
        ant.experiment.RandomizedEffect(
            "outcome", design, ancova_covariates=("baseline_a", "baseline_b")
        ),
        ant.experiment.RandomizedEffect(
            "outcome", design, cuped=ant.experiment.FixedCUPED("baseline_a", 1.2)
        ),
    ):
        result = require_success(ant.analyze(data, query=query, refute="none"))
        assert result.randomized_effect.interval_95 is None
        assert result.randomized_effect.standard_error is None
        assert result.evidence_status == "off_axis"


def test_three_hundred_balanced_rows_publish_each_bernoulli_interval():
    data, design = _data(300)
    for query in (
        ant.experiment.RandomizedEffect("outcome", design),
        ant.experiment.RandomizedEffect(
            "outcome", design, ancova_covariates=("baseline_a", "baseline_b")
        ),
        ant.experiment.RandomizedEffect(
            "outcome", design, cuped=ant.experiment.FixedCUPED("baseline_a", 1.2)
        ),
    ):
        result = require_success(ant.analyze(data, query=query, refute="none"))
        fit = result.randomized_effect
        assert fit.interval_95 is not None
        assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
        assert fit.interval_95[1] - fit.interval_95[0] == pytest.approx(
            2 * 1.959964 * fit.standard_error, rel=1e-4
        )
        assert fit.effect == pytest.approx(1.4, abs=0.05)
        assert fit.support_status == "licensed"


def test_weak_arm_support_reports_point_only_and_names_the_arm_rule():
    n = 300
    assigned = [i % 15 == 0 for i in range(n)]
    data, _ = _data(n)
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assigned,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    result = require_success(
        ant.analyze(data, query=ant.experiment.RandomizedEffect("outcome", design), refute="none")
    )
    assert result.randomized_effect.interval_95 is None
    assert result.evidence_status == "off_axis"
