"""Continuous classical measurement error through the Python facade.

Oracles are computed here, never by native code: the data-generating coefficient of a simulated
DGP and the closed-form attenuation factor ``Var(X) / (Var(X) + sigma_u^2) = 0.8``; for outcome
error, the least-squares slope from ``numpy.linalg.lstsq``. Simulations use fixed seeds and the
absolute bands are written in each test.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import measurement_models as mm
from antecedent.errors import CausalValueError
from antecedent.measurement_models import MeasurementModelRefusal

from _refusal import assert_registered_refusal

SCALAR = mm.RepresentationVersion("scalar", 1)


def analysis() -> mm.AnalysisBinding:
    return mm.AnalysisBinding(
        construct_id="construct:latent", population_id="cohort", regime_id="observational"
    )


def model(parameters) -> mm.MeasurementModel:
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable("schema:indicator", "noisy_misclassified", "unit"),
        construct=mm.MeasurementVariable("construct:latent", "latent_scientific", "unit"),
        parameters=parameters,
        population_id="cohort",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("test"),
        representation=SCALAR,
    )


def simulate(n: int, seed: int, logistic: bool):
    """``X ~ N(0,1)``, ``W = X + N(0, 0.25)``; linear ``Y = 1 + 2X + N(0, 0.25)`` or logistic
    ``P(Y=1) = sigmoid(-0.5 + X)``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    w = x + 0.5 * rng.standard_normal(n)
    if logistic:
        p = 1.0 / (1.0 + np.exp(0.5 - x))
        y = (rng.uniform(size=n) < p).astype(float)
    else:
        y = 1.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
    return y, w


def prepare(parameters, y, w) -> mm.PreparedMeasurementModel:
    return mm.prepare_measurement_model(
        model(parameters), analysis(), mm.ClassicalLinearData(outcome=y, regressor=w)
    )


CLASSICAL = mm.ClassicalErrorParameters(0.25, 0.0, min_reliability=0.1)


def test_linear_calibration_and_simex_recover_the_slope():
    y, w = simulate(2000, 1, logistic=False)
    prepared = prepare(CLASSICAL, y, w)
    rc = prepared.estimate_continuous(mm.ContinuousErrorData(outcome=y, regressor=w))
    assert rc.naive_point == pytest.approx(1.6, abs=0.1)
    assert rc.point == pytest.approx(2.0, abs=0.15)
    assert abs(rc.point - 2.0) < abs(rc.naive_point - 2.0)
    assert rc.functional_id == "linear_coefficient.regression_calibration"
    assert rc.reliability == pytest.approx(0.8, abs=0.05)
    assert rc.bootstrap_replicates == 0

    sx = prepared.estimate_continuous(
        mm.ContinuousErrorData(
            outcome=y,
            regressor=w,
            method="simex",
            simex_replicates=10,
            bootstrap_replicates=50,
            seed=3,
        )
    )
    assert sx.point == pytest.approx(2.0, abs=0.15)
    assert sx.functional_id == "linear_coefficient.simex"
    assert sx.simex_lambdas == (0.0, 0.5, 1.0, 1.5, 2.0)
    slopes = [row[1] for row in sx.simex_means]
    assert all(b < a for a, b in zip(slopes, slopes[1:], strict=False))
    assert sx.bootstrap_replicates + sx.failed_bootstrap_replicates == 50
    assert sx.standard_error > 0.0


def test_logistic_calibration_recovers_the_slope_reproducibly():
    y, w = simulate(4000, 2, logistic=True)
    prepared = prepare(CLASSICAL, y, w)
    data = mm.ContinuousErrorData(
        outcome=y, regressor=w, link="logit", bootstrap_replicates=50, seed=11
    )
    first = prepared.estimate_continuous(data)
    assert first.point == pytest.approx(1.0, abs=0.15)
    assert abs(first.point - 1.0) < abs(first.naive_point - 1.0)
    assert first.functional_id == "logit_coefficient.regression_calibration"
    assert first.inference == "uncalibrated_standard_error"
    again = prepared.estimate_continuous(data)
    assert again == first
    with pytest.raises(MeasurementModelRefusal) as raised:
        first.interval()
    assert raised.value.reason_code == "cell_not_licensed"


def test_outcome_error_under_an_identity_link_needs_no_correction():
    rng = np.random.default_rng(4)
    t = rng.standard_normal(1500)
    y = 1.0 + 1.5 * t + rng.standard_normal(1500) + rng.standard_normal(1500)
    prepared = prepare(CLASSICAL, y, t)
    result = prepared.estimate_continuous(
        mm.ContinuousErrorData(outcome=y, regressor=t, location="outcome")
    )
    design = np.column_stack([np.ones_like(t), t])
    slope = np.linalg.lstsq(design, y, rcond=None)[0][1]
    assert result.point == pytest.approx(slope, abs=1e-10)
    assert result.method == "outcome_error_no_correction"
    assert result.error_variance is None


def test_unidentified_error_variance_and_binary_outcome_error_are_refused():
    y, w = simulate(300, 5, logistic=True)
    repeated = mm.prepare_measurement_model(
        model(mm.RepeatedParameters(min_replicated_units=2)),
        analysis(),
        mm.RepeatedLinearData(outcome=y, replicate_values=w, replicate_counts=np.ones(300, int)),
    )
    with pytest.raises(MeasurementModelRefusal) as raised:
        repeated.estimate_continuous(mm.ContinuousErrorData(outcome=y, regressor=w))
    assert_registered_refusal(raised.value)
    assert raised.value.reason_code == "measurement_not_identified"
    assert "supply replicates or a validation sample" in str(raised.value)

    prepared = prepare(CLASSICAL, y, w)
    with pytest.raises(MeasurementModelRefusal) as raised:
        prepared.estimate_continuous(
            mm.ContinuousErrorData(outcome=y, regressor=w, location="outcome", link="logit")
        )
    assert raised.value.reason_code == "route_not_supported"
    assert "binary misclassification" in str(raised.value)

    # The nonlinear operation itself stays refused and names this entry point.
    with pytest.raises(MeasurementModelRefusal) as raised:
        prepared.check_operation("nonlinear_outcome_model")
    assert raised.value.reason_code == "cell_not_licensed"


def test_declaration_errors_are_input_errors():
    with pytest.raises(CausalValueError):
        mm.ContinuousErrorData(outcome=[0.0], regressor=[0.0], location="covariate")
    with pytest.raises(CausalValueError):
        mm.ContinuousErrorData(outcome=[0.0], regressor=[0.0], method="moments")
