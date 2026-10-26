"""Multiple imputation beyond the average-effect route: conditional effects, response curves,
interventional distributions, randomized designs and a single series. Each pooled answer carries
the joint pooling of its coordinates and the digest of the assumptions it rests on."""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalError
from antecedent.missing import MissingPolicy
from antecedent.outcomes import require_success
from antecedent.results import ImputationVectorView, ImputationWaldView


def _policy(*, imputations: int = 6, seed: int = 11, iterations: int = 4) -> MissingPolicy:
    return MissingPolicy.multiple_imputation(
        imputations=imputations, iterations=iterations, seed=seed
    )


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _assert_rubin_vector(vector: ImputationVectorView, imputations: int) -> None:
    """The covariances are Rubin's: ``T = U + (1 + 1/m) B`` entrywise, symmetric, and the
    scalar-pooled coordinates agree with the diagonal."""
    p = len(vector.labels)
    assert p >= 1
    assert len(vector.point) == p
    assert len(vector.points) == imputations
    assert all(len(row) == p for row in vector.points)
    for matrix in (vector.within, vector.between, vector.total):
        assert len(matrix) == p * p
    for k in range(p * p):
        expected = vector.within[k] + (1.0 + 1.0 / imputations) * vector.between[k]
        assert vector.total[k] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    total = vector.covariance("total")
    assert len(total) == p and all(len(row) == p for row in total)
    for i in range(p):
        for j in range(p):
            assert total[i][j] == pytest.approx(total[j][i], rel=1e-9, abs=1e-12)
    mean = [sum(row[i] for row in vector.points) / imputations for i in range(p)]
    assert vector.point == pytest.approx(mean, rel=1e-9, abs=1e-12)
    assert len(vector.coordinate_df) == p
    assert len(vector.coordinate_fraction_missing_information) == p
    if vector.wald is not None:
        assert isinstance(vector.wald, ImputationWaldView)
        assert vector.wald.df_numerator == p
        assert vector.wald.statistic >= 0.0
        assert vector.wald.null == (0.0,) * p
    with pytest.raises(ValueError, match="which must be"):
        vector.covariance("posterior")


def _assert_digest(result) -> str:
    digest = result.missing_data.imputation.assumptions_digest
    assert isinstance(digest, str) and len(digest) == 64
    int(digest, 16)
    return digest


# ------------------------------------------------------------------ conditional effect


def _conditional_data(n: int, seed: int) -> dict[str, np.ndarray]:
    """``y = 1 + 2 t + 0.5 t w + 0.8 w + e``; ``y`` is missing with probability rising in the
    modifier ``w`` and ``w`` for one row in ten."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * t * w + 0.8 * w + rng.normal(size=n)
    y = np.where(rng.random(n) < _logistic(1.2 * w - 0.4), np.nan, y)
    w = np.where(np.arange(n) % 10 == 3, np.nan, w)
    return {"t": t, "y": y, "w": w}


def _conditional(data, missing, **extra):
    return require_success(
        ant.analyze(
            data,
            graph=[("t", "y"), ("w", "y")],
            query=ant.ConditionalEffect("t", "y", "w"),
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=1),
            missing=missing,
            **extra,
        )
    )


def test_a_conditional_effect_reports_its_vector_pooling_and_assumption_digest():
    data = _conditional_data(1_500, 21)
    result = _conditional(data, _policy())
    report = result.missing_data.imputation
    assert report is not None and report.pooling == "rubin"
    vector = report.vector
    assert vector is not None
    _assert_rubin_vector(vector, 6)
    assert vector.labels == ("effect",)
    assert vector.covariance_basis == "published_variance"
    assert vector.wald is not None
    # The scalar fields describe the first coordinate; the vector agrees with them.
    assert report.total == pytest.approx(vector.total[0])
    assert report.within == pytest.approx(vector.within[0])
    assert report.between == pytest.approx(vector.between[0])
    assert result.estimate.ate == pytest.approx(vector.point[0])
    # The marginal conditional effect is 2 + 0.5 E[w] = 2.
    assert result.estimate.ate == pytest.approx(2.0, abs=0.3)

    digest = _assert_digest(result)
    assert _assert_digest(_conditional(data, _policy())) == digest
    assert _assert_digest(_conditional(data, _policy(seed=12))) != digest
    assert _assert_digest(_conditional(data, _policy(imputations=7))) != digest
    assert _assert_digest(_conditional(data, _policy(iterations=5))) != digest


def test_a_bayesian_conditional_effect_stacks_the_joint_draws():
    data = _conditional_data(600, 23)
    result = _conditional(data, _policy(), inference=ant.Bayesian(backend="conjugate", n_draws=200))
    report = result.missing_data.imputation
    assert report.pooling == "stacked_posterior"
    assert report.vector is not None
    assert report.vector.covariance_basis == "posterior_draws"
    assert report.vector.wald is None
    assert all(df is None for df in report.vector.coordinate_df)
    assert report.vector.total[0] >= report.vector.within[0]
    assert report.vector.between[0] > 0.0
    _assert_digest(result)


# ------------------------------------------------------------------ response curve


def _response_data(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    t = 0.5 * w + rng.normal(size=n)
    y = 1.7 * t + 0.8 * w + 0.5 * rng.normal(size=n)
    y = np.where(rng.random(n) < _logistic(w + 0.5 * t - 0.6), np.nan, y)
    w = np.where(np.arange(n) % 12 == 5, np.nan, w)
    return {"t": t, "y": y, "w": w}


def test_a_response_curve_pools_every_grid_point_jointly():
    data = _response_data(1_600, 31)
    grid = [-0.5, 0.0, 0.5]
    result = require_success(
        ant.analyze(
            data,
            graph=[("w", "t"), ("w", "y"), ("t", "y")],
            query=ant.ResponseCurve("t", "y", grid=grid),
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=2),
            missing=_policy(),
        )
    )
    report = result.missing_data.imputation
    assert report.pooling == "rubin"
    vector = report.vector
    assert vector is not None
    _assert_rubin_vector(vector, 6)
    assert len(vector.labels) == len(grid)
    assert vector.covariance_basis in ("joint_covariance", "marginal_variances")
    assert vector.wald is not None
    for dose, value in zip(grid, vector.point, strict=True):
        assert value == pytest.approx(1.7 * dose, abs=0.35)
    # Neighbouring grid points share rows: their pooled covariance is not zero.
    if vector.covariance_basis == "joint_covariance":
        assert abs(vector.covariance("total")[0][1]) > 1e-12
    _assert_digest(result)


# ------------------------------------------------------------------ interventional distribution


def _distribution_data(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 1_000
    z = (rng.random(n) < 0.5).astype(float)
    t = (rng.random(n) < 0.3 + 0.4 * z).astype(float)
    y = (rng.random(n) < 0.2 + 0.4 * t + 0.1 * z).astype(float)
    y = np.where(rng.random(n) < 0.15, np.nan, y)
    z = np.where(rng.random(n) < 0.1, np.nan, z)
    return {"t": t, "y": y, "z": z}


def test_an_interventional_distribution_pools_atoms_and_needs_a_variance_to_pool():
    data = _distribution_data(41)
    kwargs = {
        "graph": [("z", "t"), ("t", "y"), ("z", "y")],
        "query": ant.InterventionalDistribution("y", interventions={"t": 1.0}),
        "refute": "none",
        "missing": _policy(),
    }
    result = require_success(ant.analyze(data, inference=ant.Frequentist(bootstrap=60), **kwargs))
    vector = result.missing_data.imputation.vector
    assert vector is not None
    _assert_rubin_vector(vector, 6)
    assert vector.covariance_basis == "marginal_variances"
    assert vector.labels[-1] == "mean"
    # P(y = 1 | do(t = 1)) = 0.6 + 0.1 * 0.5.
    assert vector.point[-1] == pytest.approx(0.65, abs=0.08)
    atoms = vector.point[:-1]
    assert sum(atoms) == pytest.approx(1.0, abs=1e-9)
    # Without bootstrap replicates the plug-in publishes no variance to pool: refused.
    with pytest.raises(CausalError) as refused:
        require_success(ant.analyze(data, inference=ant.Frequentist(bootstrap=0), **kwargs))
    assert getattr(refused.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "standard error" in str(refused.value)


# ------------------------------------------------------------------ randomized trial

TRIAL_EFFECT = 0.3


def _trial(n: int, seed: int):
    rng = np.random.default_rng(seed)
    arm = np.arange(n) % 2 == 1
    baseline = np.clip(rng.normal(size=n), -2.0, 2.0)
    outcome = 1.0 + TRIAL_EFFECT * arm + 2.0 * baseline + rng.normal(size=n)
    eta = 0.8 - (0.8 + 0.8 * arm) * baseline
    responded = rng.random(n) < _logistic(eta)
    return arm, baseline, np.where(responded, outcome, np.nan)


def _trial_query(arm: np.ndarray) -> ant.experiment.RandomizedEffect:
    n = len(arm)
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        [bool(a) for a in arm],
        [f"participant-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    return ant.experiment.RandomizedEffect("outcome", design)


def test_a_randomized_trial_imputes_with_its_arm_and_pools_the_design_variance():
    arm, baseline, observed = _trial(6_000, 61)
    data = {"outcome": observed, "baseline": baseline}
    result = require_success(
        ant.analyze(data, query=_trial_query(arm), missing=_policy(), refute="none")
    )
    report = result.missing_data.imputation
    assert report is not None and report.pooling == "rubin"
    # The arm each participant was assigned is in the imputation model, complete and never drawn.
    assert report.imputed_cells["__assignment"] == 0
    assert report.imputed_cells["outcome"] > 500
    vector = report.vector
    assert vector is not None
    _assert_rubin_vector(vector, 6)
    assert vector.labels == ("effect",)
    assert result.missing_data.accounting.rows_dropped == 0
    complete = require_success(
        ant.analyze(data, query=_trial_query(arm), missing="complete_case", refute="none")
    ).estimate.ate
    assert abs(complete - TRIAL_EFFECT) > 0.2
    assert result.estimate.ate == pytest.approx(TRIAL_EFFECT, abs=0.25)
    assert result.estimate.ate == pytest.approx(vector.point[0])
    _assert_digest(result)


# ------------------------------------------------------------------ a single series

_SERIES_DAG = [("t", 1, "t", 0), ("z", 1, "t", 0), ("z", 1, "y", 0), ("t", 1, "y", 0)]


def _series(seed: int, n: int = 600, share: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.6 * t[i - 1] + 0.8 * z[i - 1] + 0.5 * rng.normal()
        y[i] = 0.8 * t[i - 1] + 0.9 * z[i - 1] + 0.5 * rng.normal()
    t = np.where(rng.random(n) < share, np.nan, t)
    return {"t": t, "y": y, "z": z}


def test_a_series_is_imputed_by_a_lagged_model_over_contiguous_blocks():
    data = _series(5)
    query = ant.PulseEffect("t", "y", treatment_lag=1)
    result = require_success(
        ant.analyze(
            data,
            graph=_SERIES_DAG,
            query=query,
            refute="none",
            inference=ant.Frequentist(bootstrap=60),
            missing=_policy(),
        )
    )
    report = result.missing_data.imputation
    assert report is not None
    assert report.lags >= 1
    assert report.unit == "group"
    vector = report.vector
    assert vector is not None
    _assert_rubin_vector(vector, 6)
    assert result.estimate.ate == pytest.approx(0.8, abs=0.3)
    digest = _assert_digest(result)
    again = require_success(
        ant.analyze(
            data,
            graph=_SERIES_DAG,
            query=query,
            refute="none",
            inference=ant.Frequentist(bootstrap=60),
            missing=_policy(),
        )
    )
    assert _assert_digest(again) == digest
    assert again.missing_data.imputation.vector.point == vector.point


def test_the_joint_normal_model_has_no_lagged_form_on_a_series():
    data = _series(6)
    joint = MissingPolicy.multiple_imputation(model="joint_normal", imputations=4, seed=3)
    with pytest.raises(CausalError) as refused:
        require_success(
            ant.analyze(
                data,
                graph=_SERIES_DAG,
                query=ant.PulseEffect("t", "y", treatment_lag=1),
                refute="none",
                inference=ant.Frequentist(bootstrap=0),
                missing=joint,
            )
        )
    assert getattr(refused.value, "reason_code", None) == "missing_policy_not_applicable"
    assert "chained-equations" in str(refused.value)


def test_the_wald_view_treats_a_missing_denominator_as_the_chi_square_reference():
    wald = ImputationWaldView(null=(0.0,), statistic=1.0, df_numerator=1.0, df_denominator=math.inf)
    assert math.isinf(wald.df_denominator)
