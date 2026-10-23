"""Sequentially randomized unit-period designs end to end through Python.

A two-arm micro-randomized panel with a known excursion effect, its Bayesian counterpart, a
dose grid with a declared curvature bound, and the typed refusals of options a design does
not read. Every result is unlicensed and uncalibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from antecedent.errors import CausalError
from antecedent.temporal_designs import estimate_temporal_design


def _two_arm(units: int = 200, periods: int = 8, effect: float = 1.2, seed: int = 3):
    rng = np.random.default_rng(seed)
    unit, period, assignment, probability, outcome = [], [], [], [], []
    for i in range(units):
        level = rng.uniform(-0.5, 0.5)
        previous_y, previous_a = 0.0, 0.0
        for t in range(periods):
            p = 0.65 if previous_y > 0 else 0.35
            a = bool(rng.uniform() < p)
            y = level + effect * previous_a + rng.uniform(-0.5, 0.5)
            unit.append(f"u{i}")
            period.append(t)
            assignment.append(a)
            probability.append(p)
            outcome.append(y)
            previous_y, previous_a = y, float(a)
    return unit, period, assignment, probability, np.asarray(outcome)


def test_the_two_arm_excursion_recovers_the_effect_in_both_modes() -> None:
    unit, period, assignment, probability, outcome = _two_arm()
    design = {"design": "two_arm", "assignment": assignment, "probability": probability}
    fit = estimate_temporal_design(
        {"y": outcome}, outcome="y", units=unit, periods=period, design=design, lag=1
    )
    assert fit.estimator == "temporal_design.excursion"
    assert not fit.licensed and fit.calibration == "unmeasured"
    estimate, lower, upper = fit.coordinate("excursion_effect")
    assert lower < estimate < upper
    assert abs(estimate - 1.2) < 4 * fit.standard_error[0]
    posterior = estimate_temporal_design(
        {"y": outcome},
        outcome="y",
        units=unit,
        periods=period,
        design=design,
        lag=1,
        inference="bayesian",
        n_draws=300,
    )
    assert posterior.estimator == "bayesian.temporal_design.excursion"
    assert posterior.draws == 300
    assert abs(posterior.estimate[0] - estimate) < 0.5 * fit.standard_error[0] + 0.02


def test_the_local_quadratic_dose_grid_needs_its_curvature_bound() -> None:
    rng = np.random.default_rng(5)
    unit, period, dose, outcome = [], [], [], []
    for i in range(200):
        for t in range(6):
            d = rng.uniform(0.0, 2.0)
            unit.append(f"u{i}")
            period.append(t)
            dose.append(d)
            outcome.append(1.0 + d + 0.5 * d * d + rng.uniform(-0.5, 0.5))
    design = {
        "design": "dose",
        "dose": dose,
        "density": [0.5] * len(dose),
        "bandwidth": 0.3,
        "target": {"kind": "local_quadratic", "grid": [0.5, 1.0, 1.5], "curvature_bound": 0.0},
    }
    fit = estimate_temporal_design(
        {"y": np.asarray(outcome)}, outcome="y", units=unit, periods=period, design=design
    )
    assert fit.band_coordinates == (0, 1, 2)
    assert fit.smoothing_bias_bound is not None
    for index, d in enumerate((0.5, 1.0, 1.5)):
        assert abs(fit.estimate[index] - (1.0 + d + 0.5 * d * d)) < 4 * fit.standard_error[index]
    design["target"] = {"kind": "local_quadratic", "grid": [0.5, 1.0]}
    with pytest.raises(CausalError, match="curvature_bound"):
        estimate_temporal_design(
            {"y": np.asarray(outcome)}, outcome="y", units=unit, periods=period, design=design
        )


def test_bootstrap_replicates_are_refused_where_the_interval_is_analytic() -> None:
    unit, period, assignment, probability, outcome = _two_arm(units=40, periods=4)
    design = {"design": "two_arm", "assignment": assignment, "probability": probability}
    with pytest.raises(CausalError, match="Next attempt"):
        estimate_temporal_design(
            {"y": outcome},
            outcome="y",
            units=unit,
            periods=period,
            design=design,
            bootstrap=99,
        )
