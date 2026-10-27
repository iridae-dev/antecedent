"""The treatment-propensity nuisance product of a prepared study, and its weighting boundary.

The fixture is the two-cell propensity: 100 rows at ``z = 0`` with 30 treated and 100 rows at
``z = 1`` with 60 treated. References are Newton modes of the logistic model under the stated
Gaussian prior and the score moments of a large Laplace sample drawn here with numpy. The product
is a posterior of a nuisance, never of an effect; it is unlicensed with calibration unmeasured,
and its artifact is refit on load in this process and in a fresh one.
"""

from __future__ import annotations

import subprocess
import sys

import antecedent
import numpy as np
import pytest
from antecedent import nuisance_scores
from antecedent.errors import CausalSerializationError
from antecedent.execution import Execution
from antecedent.outcomes import require_success

from _refusal import REGISTERED


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def newton_mode(x, y, precision):
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = sigmoid(x @ beta)
        gradient = x.T @ (y - p) - precision @ beta
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + precision
        beta = beta + np.linalg.solve(hessian, gradient)
    p = sigmoid(x @ beta)
    return beta, np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]) + precision)


def study_data():
    z = np.r_[np.zeros(100), np.ones(100)]
    t = np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)]
    y = 1.0 + 2.0 * t + 0.1 * (np.arange(200) % 7)
    return {"z": z, "t": t, "y": y}


GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def prepared():
    return antecedent.prepare(
        study_data(),
        graph=GRAPH,
        query=antecedent.AverageEffect("t", "y"),
        inference=antecedent.Frequentist(bootstrap=0),
        execution=Execution(seed=0),
        refute=False,
    )


def test_the_product_matches_newton_and_a_laplace_sample() -> None:
    data = study_data()
    design = np.column_stack([np.ones(200), data["z"]])
    for prior_sd in (2.0, 0.1):
        product = require_success(
            prepared().treatment_score_posterior(
                prior_sd=prior_sd,
                draws=4000,
                execution=Execution(seed=13),
                clip=0.35,
                summary_rows=[0, 150],
            )
        )
        mode, covariance = newton_mode(design, data["t"], np.eye(2) / prior_sd**2)
        sample = np.random.default_rng(0).multivariate_normal(mode, covariance, 400_000)
        for index, row in enumerate(([1.0, 0.0], [1.0, 1.0])):
            scores = sigmoid(sample @ np.array(row))
            assert product[f"row{index}.at_mode"] == pytest.approx(
                float(sigmoid(mode @ np.array(row))), abs=1e-5
            )
            assert product[f"row{index}.mean"] == pytest.approx(float(scores.mean()), abs=0.004)
            assert product[f"row{index}.sd"] == pytest.approx(float(scores.std()), rel=0.06)
            outside = float(((scores < 0.35) | (scores > 0.65)).mean())
            assert product[f"row{index}.outside_band"] == pytest.approx(outside, abs=0.03)
    assert product.estimator_id == "bayesian.propensity.coefficient_posterior"
    assert "not_an_effect_posterior" in product.uncertainty
    assert product.licensing == "unlicensed" and product.calibration == "unmeasured"
    assert not product.licensed and not product.calibration_measured
    assert "nuisance" in product.explain()


def test_the_artifact_is_refit_on_load_here_and_in_a_fresh_process() -> None:
    product = require_success(
        prepared().treatment_score_posterior(summary_rows=[0, 150], draws=500)
    )
    loaded = nuisance_scores.load_artifact(bytes(product.export()))
    assert dict(loaded.values) == dict(product.values)
    assert loaded.request_digest == product.request_digest
    code = (
        "import sys, json\n"
        "from antecedent import nuisance_scores\n"
        "p = nuisance_scores.load_artifact(sys.stdin.buffer.read())\n"
        "print(json.dumps(dict(p.values)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        input=bytes(product.export()),
        capture_output=True,
        check=True,
    )
    import json

    assert json.loads(done.stdout) == dict(product.values)


def test_an_edited_or_truncated_artifact_is_refused() -> None:
    product = require_success(prepared().treatment_score_posterior(summary_rows=[0], draws=500))
    artifact = bytearray(bytes(product.export()))
    artifact[len(artifact) // 2] ^= 0x01
    with pytest.raises(CausalSerializationError):
        nuisance_scores.load_artifact(bytes(artifact))
    with pytest.raises(CausalSerializationError):
        nuisance_scores.load_artifact(bytes(product.export())[:-5])


def test_a_bayesian_weighting_estimator_is_the_typed_boundary() -> None:
    result = antecedent.analyze(
        study_data(),
        graph=GRAPH,
        query=antecedent.AverageEffect("t", "y"),
        estimator=antecedent.Estimator.PROPENSITY_WEIGHTING,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=2000),
        execution=Execution(seed=0),
        refute=False,
    )
    assert result.value is None and result.refusal is not None
    assert result.refusal.code == "route_not_supported"
    text = result.refusal.message
    assert "penalized_propensity.effect_posterior" in text
    assert "ancillary" in text and "Next attempt" in text
    assert "Bayesian g-computation" in text
    assert result.refusal.code in REGISTERED


def test_the_product_is_never_read_as_an_effect_score_artifact() -> None:
    """In-sample posterior summaries are not out-of-fold effect scores: a retarget refuses them."""
    from antecedent import artifacts, recalc_cell
    from antecedent.recalc import Utility

    product = require_success(
        prepared().treatment_score_posterior(summary_rows=[0, 150], draws=500)
    )
    names = ["z", "t", "y"]
    with pytest.raises(CausalSerializationError) as retarget:
        recalc_cell.resume_from_scores(
            bytes(product.export()),
            variables=names,
            edges=GRAPH,
            utility=Utility(benefit_per_unit=1.0),
        )
    assert not isinstance(retarget.value, AttributeError)
    with pytest.raises(CausalSerializationError):
        artifacts.loads(bytes(product.export()))


# ----- temporal treatment score on the independent-subject rows of a panel --------------------

TEMPORAL_GRAPH = [("c", 1, "x", 1), ("c", 1, "y", 0), ("x", 1, "y", 0)]


def panel_units(seed: int = 9, units: int = 300, steps: int = 3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(units):
        c = rng.standard_normal(steps)
        x = (rng.random(steps) < sigmoid(0.5 + 0.9 * c)).astype(float)
        y = np.zeros(steps)
        for t in range(1, steps):
            y[t] = float(rng.random() < sigmoid(-0.5 + x[t - 1] + 0.8 * c[t - 1]))
        out.append({"x": x, "c": c, "y": y})
    return out


def prepared_panel(units):
    from antecedent.data import panel

    return antecedent.prepare(
        panel(units),
        graph=TEMPORAL_GRAPH,
        query=antecedent.PulseEffect(
            treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
        ),
        inference=antecedent.Frequentist(bootstrap=0),
        execution=Execution(seed=0),
        refute=False,
    )


def test_the_temporal_penalized_score_is_the_ridge_mode_and_the_posterior_is_a_laplace_mean() -> (
    None
):
    units = panel_units()
    rows = np.array([[1.0, u["c"][0]] for u in units])
    treated = np.array([u["x"][0] for u in units])
    prepared_study = prepared_panel(units)
    histories = [[0.0], [1.5]]
    ridge = require_success(prepared_study.temporal_treatment_score(histories, penalty=2.0))
    mode, _ = newton_mode(rows, treated, np.diag([0.0, 2.0]))
    assert ridge["coefficient.0"] == pytest.approx(mode[0], abs=1e-6)
    assert ridge["coefficient.1"] == pytest.approx(mode[1], abs=1e-6)
    assert ridge["score1.at_mode"] == pytest.approx(
        float(sigmoid(mode[0] + 1.5 * mode[1])), abs=1e-6
    )
    assert ridge.estimator_id == "temporal.penalized_propensity"
    assert ridge.licensing == "unlicensed" and ridge.calibration == "unmeasured"

    posterior = require_success(
        prepared_study.temporal_treatment_score(
            histories, intercept_sd=1.0, slope_sd=0.5, draws=4000, execution=Execution(seed=3)
        )
    )
    mode, hessian_inverse = newton_mode(rows, treated, np.diag([1.0, 4.0]))
    sample = np.random.default_rng(2).multivariate_normal(mode, hessian_inverse, 400_000)
    assert posterior["score0.at_mode"] == pytest.approx(float(sigmoid(mode[0])), abs=1e-6)
    assert posterior["score0.mean"] == pytest.approx(float(sigmoid(sample[:, 0]).mean()), abs=0.01)
    assert posterior["score1.mean"] == pytest.approx(
        float(sigmoid(sample[:, 0] + 1.5 * sample[:, 1]).mean()), abs=0.01
    )
    assert posterior.estimator_id == "bayesian.temporal.propensity_posterior"
    loaded = nuisance_scores.load_artifact(bytes(posterior.export()))
    assert dict(loaded.values) == dict(posterior.values)


def test_a_single_dependent_series_has_no_temporal_treatment_score() -> None:
    rng = np.random.default_rng(1)
    series = {
        "x": (rng.random(80) < 0.5).astype(float),
        "c": rng.standard_normal(80),
        "y": (rng.random(80) < 0.5).astype(float),
    }
    prepared_series = antecedent.prepare(
        series,
        graph=TEMPORAL_GRAPH,
        query=antecedent.PulseEffect(
            treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
        ),
        inference=antecedent.Frequentist(bootstrap=0),
        execution=Execution(seed=0),
        refute=False,
    )
    result = prepared_series.temporal_treatment_score([[0.0]], penalty=1.0)
    assert result.value is None and result.refusal is not None
    assert result.refusal.code == "route_not_supported"
    assert "single dependent series" in result.refusal.message
    assert "Next attempt" in result.refusal.message
    assert result.refusal.code in REGISTERED
