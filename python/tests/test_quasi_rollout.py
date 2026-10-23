"""Heterogeneity-robust rollouts, covariate-adjusted DiD, conformal synthetic control and
optioned local ratios through the installed ``antecedent`` package, on simulated truths."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.quasi import (
    CovariateAdjustedDiD,
    DesignEffect,
    FuzzyRegressionDiscontinuity,
    QuasiRolloutEstimate,
    StaggeredRollout,
    SyntheticControl,
)


def _staggered_panel(seed: int = 3, per_cohort: int = 60) -> dict[str, list[object]]:
    """Cohorts 0 (never treated), 2 and 4 over periods 1..5; the effect is 3 x event time,
    so the cohort-size-weighted average over the six post-adoption cells is 3.5."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[object]] = {"y": [], "unit": [], "period": [], "cohort": []}
    for cohort in (0, 2, 4):
        for index in range(per_cohort):
            level = rng.normal()
            for period in range(1, 6):
                effect = 3.0 * (period - cohort) if cohort and period >= cohort else 0.0
                rows["y"].append(level + 0.5 * period + effect + 0.5 * rng.normal())
                rows["unit"].append(f"c{cohort}-{index}")
                rows["period"].append(period)
                rows["cohort"].append(cohort)
    return rows


def _rollout(data: dict[str, list[object]], query: StaggeredRollout) -> QuasiRolloutEstimate:
    result = antecedent.analyze(data, query=query).quasi_rollout
    assert isinstance(result, QuasiRolloutEstimate)
    assert result.licensed is False
    assert result.calibration == "unmeasured"
    return result


def test_callaway_santanna_recovers_the_simple_aggregate_the_two_way_coefficient_misses():
    result = _rollout(_staggered_panel(), StaggeredRollout("y", "unit", "period", "cohort"))
    assert result.design == "staggered_did"
    overall = result.overall
    assert isinstance(overall, DesignEffect)
    assert overall.kind == "overall"
    assert overall.estimate == pytest.approx(3.5, abs=0.4)
    twfe = result.diagnostic("twfe_static_coefficient")
    assert twfe is not None
    assert abs(twfe - 3.5) > 1.0
    assert result.uncertainty == "point_only_no_resample_requested"


def test_cluster_multiplier_refit_reports_intervals_and_simultaneous_bands():
    query = StaggeredRollout(
        "y",
        "unit",
        "period",
        "cohort",
        aggregation="dynamic",
        resampling="cluster_multiplier",
        replicates=199,
    )
    result = _rollout(_staggered_panel(), query)
    assert result.replicates_requested == 199
    assert result.replicates_ok + result.replicates_failed == 199
    assert result.overall is not None and result.overall.interval is not None
    lower, upper = result.overall.interval
    assert lower < result.overall.estimate < upper
    dynamic = [
        effect
        for effect in result.effects_of("event_time")
        if effect.event_time is not None and effect.event_time >= 0
    ]
    assert dynamic
    for effect in dynamic:
        assert effect.interval is not None and effect.band is not None
        assert effect.band[0] <= effect.interval[0] <= effect.interval[1] <= effect.band[1]
        assert effect.estimate == pytest.approx(3.0 * effect.event_time, abs=0.6)


def test_sun_abraham_refuses_an_unbalanced_panel_naming_the_next_attempt():
    data = _staggered_panel()
    keep = [index for index in range(len(data["y"])) if index not in (7, 31, 402)]
    unbalanced = {name: [values[index] for index in keep] for name, values in data.items()}
    query = StaggeredRollout("y", "unit", "period", "cohort", estimator="sun_abraham")
    with pytest.raises(CausalError, match="(?i)next attempt"):
        antecedent.analyze(unbalanced, query=query)


def test_doubly_robust_covariate_adjusted_did_recovers_the_effect():
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.normal(size=n)
    treated = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-x))
    level = rng.normal(size=n)
    rows: dict[str, list[object]] = {"y": [], "id": [], "group": [], "after": [], "x": []}
    for subject in range(n):
        for post in (0, 1):
            trend = (1.0 + 1.5 * x[subject]) * post
            effect = 2.0 * post * treated[subject]
            rows["y"].append(level[subject] + trend + effect + 0.5 * rng.normal())
            rows["id"].append(f"s{subject}")
            rows["group"].append(bool(treated[subject]))
            rows["after"].append(bool(post))
            rows["x"].append(float(x[subject]))
    query = CovariateAdjustedDiD("y", "id", "group", "after", covariates=("x",))
    result = antecedent.analyze(rows, query=query).quasi_rollout
    assert isinstance(result, QuasiRolloutEstimate)
    assert result.design == "covariate_adjusted_did"
    assert result.overall is not None
    assert result.overall.estimate == pytest.approx(2.0, abs=0.25)
    assert result.licensed is False


def test_conformal_synthetic_control_recovers_a_known_effect_with_an_interval():
    rng = np.random.default_rng(7)
    rows: dict[str, list[object]] = {"y": [], "unit": [], "period": []}
    for period in range(1, 46):
        donors = [float(j) + rng.normal() for j in range(10)]
        for j, value in enumerate(donors):
            rows["y"].append(value)
            rows["unit"].append(f"donor{j:02d}")
            rows["period"].append(period)
        treated = 0.5 * donors[3] + 0.3 * donors[4] + 0.2 * donors[5] + 0.5 * rng.normal()
        rows["y"].append(treated + (3.0 if period >= 41 else 0.0))
        rows["unit"].append("treated")
        rows["period"].append(period)
    query = SyntheticControl("y", "unit", "period", "treated", 41, conformal=True)
    result = antecedent.analyze(rows, query=query).quasi_rollout
    assert isinstance(result, QuasiRolloutEstimate)
    assert result.calibration == "unmeasured"
    overall = result.overall
    assert overall is not None and overall.interval is not None
    assert overall.estimate == pytest.approx(3.0, abs=0.8)
    assert overall.interval[0] < overall.estimate < overall.interval[1]
    assert overall.interval[0] > 0.0


def test_mse_optimal_fuzzy_rd_with_a_density_test_recovers_the_effect():
    rng = np.random.default_rng(5)
    n = 4000
    running = rng.uniform(-1.0, 1.0, size=n)
    uptake = rng.uniform(size=n) < 0.2 + 0.6 * (running >= 0.0)
    outcome = 1.0 + running + 0.5 * running**2 + 2.0 * uptake + 0.5 * rng.normal(size=n)
    query = FuzzyRegressionDiscontinuity(
        "y", "t", "x", 0.0, 1.0, bandwidth_rule="mse_optimal", density_test=True
    )
    result = antecedent.analyze(
        {"y": outcome, "t": uptake.astype(float), "x": running}, query=query
    ).quasi_rollout
    assert isinstance(result, QuasiRolloutEstimate)
    overall = result.overall
    assert overall is not None and overall.standard_error is not None
    assert abs(overall.estimate - 2.0) <= 4.0 * overall.standard_error
    assert result.diagnostic("density_p_value") is not None
    bandwidth = result.diagnostic("bandwidth")
    assert bandwidth is not None and 0.0 < bandwidth <= 1.0


def test_a_bayesian_rollout_request_is_refused_naming_the_next_attempt():
    with pytest.raises(CausalError, match="Next attempt"):
        antecedent.analyze(
            _staggered_panel(),
            query=StaggeredRollout("y", "unit", "period", "cohort"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=2000),
        )


def test_rollout_queries_validate_their_options():
    with pytest.raises(CausalValueError, match="estimator"):
        StaggeredRollout("y", "unit", "period", "cohort", estimator="twfe")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="resampling"):
        StaggeredRollout("y", "unit", "period", "cohort", resampling="rows")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="at least one covariate"):
        CovariateAdjustedDiD("y", "id", "group", "after", covariates=())
    with pytest.raises(CausalValueError, match="treated_unit, intervention_period"):
        SyntheticControl("y", "unit", "period", "a", 5, adoptions=[("b", 5)])
    with pytest.raises(CausalValueError, match="one or the other"):
        SyntheticControl("y", "unit", "period", "a", 5, conformal=True, adoptions=[("a", 5)])
    with pytest.raises(CausalValueError, match="bandwidth_rule"):
        FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0, bandwidth_rule="cv")  # type: ignore[arg-type]


def _resampled_dynamic_rollout() -> object:
    query = StaggeredRollout(
        "y",
        "unit",
        "period",
        "cohort",
        aggregation="dynamic",
        resampling="cluster_multiplier",
        replicates=199,
    )
    return antecedent.analyze(_staggered_panel(), query=query)


def test_rollout_export_load_round_trip_reproduces_the_live_result():
    result = _resampled_dynamic_rollout()
    artifact = result.export()
    loaded = antecedent.load(artifact)
    assert loaded.export() == artifact
    assert loaded.quasi_rollout == result.quasi_rollout
    assert loaded.answer == result.answer


def test_a_fresh_consumer_rederives_the_dynamic_summary_from_the_loaded_effects():
    loaded = antecedent.load(_resampled_dynamic_rollout().export())
    rollout = loaded.quasi_rollout
    assert isinstance(rollout, QuasiRolloutEstimate)
    post = [
        effect.estimate
        for effect in rollout.effects_of("event_time")
        if effect.event_time is not None and effect.event_time >= 0
    ]
    assert post
    assert rollout.overall is not None
    assert rollout.overall.estimate == pytest.approx(sum(post) / len(post), rel=1e-12, abs=1e-12)


def test_rollout_answer_is_structured_unlicensed_and_unmeasured():
    result = _resampled_dynamic_rollout()
    answer = result.answer
    assert answer.kind == "structured"
    assert answer.detail is not None and answer.detail.startswith("unlicensed_quasi_rollout_")
    structured = answer.structured
    assert structured is not None
    assert structured["calibration"] == "unmeasured"
    assert structured["licensed"] is False
    assert structured["design"] == "staggered_did"
    assert structured["effects"][0]["kind"] == "overall"
    assert structured["effects"][0]["estimate"] == result.quasi_rollout.overall.estimate


def test_resampled_augmented_panel_publishes_its_standard_error():
    from antecedent.quasi import AugmentedPanelDiD

    rng = np.random.default_rng(17)
    n = 400
    x = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-0.5 * x))
    treated = rng.uniform(size=n) < propensity
    level = rng.normal(size=n)
    change = 1.0 + 0.5 * x
    data = {
        "id": [f"s{i}" for i in range(n)],
        "pre": level,
        "post": level + change + 2.0 * treated + 0.3 * rng.normal(size=n),
        "treated": treated.tolist(),
        "p": propensity,
        "m0": change,
    }
    query = AugmentedPanelDiD(
        "pre",
        "post",
        "id",
        "treated",
        "p",
        "m0",
        True,
        resampling="cluster_bootstrap",
        replicates=199,
    )
    section = antecedent.analyze(data, query=query).panel_did
    assert section.standard_error is not None and section.standard_error > 0.0
    assert section.interval_95 is not None
    half = 1.959963984540054 * section.standard_error
    assert section.interval_95[0] == pytest.approx(section.estimate - half, rel=1e-9, abs=1e-12)
    assert section.interval_95[1] == pytest.approx(section.estimate + half, rel=1e-9, abs=1e-12)
    assert section.estimate == pytest.approx(2.0, abs=0.3)
