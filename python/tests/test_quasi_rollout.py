"""Heterogeneity-robust rollouts, covariate-adjusted DiD, conformal synthetic control and
optioned local ratios through the installed ``antecedent`` package, on simulated truths."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.outcomes import require_success
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
    result = require_success(antecedent.analyze(data, query=query)).quasi_rollout
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


def _unbalanced(data: dict[str, list[object]], seed: int = 8, keep: float = 0.85):
    """Drop unit-periods at random, keeping at least three rows of every unit."""
    rng = np.random.default_rng(seed)
    rows = len(data["y"])
    wanted = rng.uniform(size=rows) < keep
    seen: dict[object, list[int]] = {}
    for index, unit in enumerate(data["unit"]):
        seen.setdefault(unit, []).append(index)
    for indices in seen.values():
        short = 3 - int(sum(wanted[i] for i in indices))
        for index in [i for i in indices if not wanted[i]][: max(short, 0)]:
            wanted[index] = True
    keep_rows = [index for index in range(rows) if wanted[index]]
    return {name: [values[index] for index in keep_rows] for name, values in data.items()}


def _interacted_reference(
    data: dict[str, list[object]],
) -> tuple[float, dict[tuple[int, int], float]]:
    """Full-dummy saturated regression by numpy lstsq: a dummy per unit, period and
    cohort-period cell other than the cohort's base period, on the observed rows."""
    units = sorted(set(data["unit"]))
    periods = sorted(set(data["period"]))
    cohorts = sorted(set(data["cohort"]) - {0})
    unit_of = np.array([units.index(u) for u in data["unit"]])
    period = np.array(data["period"])
    cohort = np.array(data["cohort"])
    y = np.array(data["y"], dtype=float)
    columns = [(unit_of == i).astype(float) for i in range(len(units))]
    columns += [(period == t).astype(float) for t in periods[1:]]
    cells = [(g, t) for g in cohorts for t in periods if t != g - 1]
    columns += [((cohort == g) & (period == t)).astype(float) for g, t in cells]
    design = np.column_stack(columns)
    assert np.linalg.matrix_rank(design) == design.shape[1]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    delta = dict(zip(cells, beta[len(units) + len(periods) - 1 :], strict=True))
    sizes = {
        g: len({u for u, c in zip(data["unit"], data["cohort"], strict=True) if c == g})
        for g in cohorts
    }
    post = [(g, t) for g, t in cells if t >= g]
    simple = sum(sizes[g] * delta[(g, t)] for g, t in post) / sum(sizes[g] for g, _ in post)
    return simple, {key: float(delta[key]) for key in post}


def test_sun_abraham_reads_an_unbalanced_panel_and_matches_the_dummy_regression_reference():
    data = _unbalanced(_staggered_panel(seed=5, per_cohort=100))
    assert len(data["y"]) < 300 * 5
    query = StaggeredRollout("y", "unit", "period", "cohort", estimator="sun_abraham")
    result = _rollout(data, query)
    assert result.diagnostic("balanced_panel") == 0.0
    simple, cells = _interacted_reference(data)
    overall = result.overall
    assert isinstance(overall, DesignEffect)
    assert overall.estimate == pytest.approx(simple, abs=1e-8)
    assert overall.estimate == pytest.approx(3.5, abs=0.4)
    group_time = {
        (effect.group, effect.period): effect.estimate for effect in result.effects_of("group_time")
    }
    assert set(group_time) == set(cells)
    for key, value in cells.items():
        assert group_time[key] == pytest.approx(value, abs=1e-8)
        # Each cohort keeps its own exposure path, not the pooled one.
        assert value == pytest.approx(3.0 * (key[1] - key[0]), abs=0.9)
    assert result.diagnostic("cohort_2_units") == 100.0
    assert result.diagnostic("cohort_4_units") == 100.0
    assert 0.0 < result.diagnostic("cohort_2_rows") < 500.0
    assert result.diagnostic("dropped_rows") == 0.0


def test_a_declared_attrition_bound_reports_an_assumption_dependent_range():
    data = _unbalanced(_staggered_panel(seed=6, per_cohort=100))
    query = StaggeredRollout(
        "y",
        "unit",
        "period",
        "cohort",
        estimator="sun_abraham",
        observation="informative",
        attrition_shift_bound=0.5,
    )
    result = _rollout(data, query)
    overall = result.overall
    assert isinstance(overall, DesignEffect)
    slope = result.diagnostic("attrition_slope")
    assert slope is not None and 0.0 < slope < 1.0
    assert result.diagnostic("assumption_range_lower") == pytest.approx(
        overall.estimate - 0.5 * slope
    )
    assert result.diagnostic("assumption_range_upper") == pytest.approx(
        overall.estimate + 0.5 * slope
    )
    with pytest.raises(CausalValueError):
        StaggeredRollout("y", "unit", "period", "cohort", attrition_shift_bound=0.5)
    with pytest.raises(CausalValueError):
        StaggeredRollout(
            "y", "unit", "period", "cohort", observation="informative", attrition_shift_bound=-1.0
        )


def test_sun_abraham_refuses_a_declared_informative_observation_process_naming_the_next_attempt():
    data = _unbalanced(_staggered_panel())
    query = StaggeredRollout(
        "y", "unit", "period", "cohort", estimator="sun_abraham", observation="informative"
    )
    with pytest.raises(CausalError, match="(?i)next attempt"):
        require_success(antecedent.analyze(data, query=query))
    with pytest.raises(CausalValueError):
        StaggeredRollout("y", "unit", "period", "cohort", observation="missing_at_random")


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
    result = require_success(antecedent.analyze(rows, query=query)).quasi_rollout
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
    result = require_success(antecedent.analyze(rows, query=query)).quasi_rollout
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
    result = require_success(
        antecedent.analyze({"y": outcome, "t": uptake.astype(float), "x": running}, query=query)
    ).quasi_rollout
    assert isinstance(result, QuasiRolloutEstimate)
    overall = result.overall
    assert overall is not None and overall.standard_error is not None
    assert abs(overall.estimate - 2.0) <= 4.0 * overall.standard_error
    assert result.diagnostic("density_p_value") is not None
    bandwidth = result.diagnostic("bandwidth")
    assert bandwidth is not None and 0.0 < bandwidth <= 1.0


def _bayes(n_draws: int = 400) -> object:
    return antecedent.Bayesian(backend="conjugate", n_draws=n_draws)


def test_bayesian_rollout_is_a_whole_unit_bayesian_bootstrap_beside_the_posterior():
    query = StaggeredRollout("y", "unit", "period", "cohort", aggregation="dynamic")
    result = require_success(
        antecedent.analyze(_staggered_panel(), query=query, inference=_bayes())
    )
    assert result.posterior is not None
    rollout = result.quasi_rollout
    assert isinstance(rollout, QuasiRolloutEstimate)
    assert rollout.uncertainty.startswith("bayesian_")
    assert rollout.licensed is False and rollout.calibration == "unmeasured"
    overall = rollout.overall
    assert overall is not None and overall.interval is not None
    # Dynamic summary: the mean of the post event-time effects 0, 3, 6, 9.
    assert overall.estimate == pytest.approx(4.5, abs=0.5)
    assert overall.interval[0] < overall.estimate < overall.interval[1]
    banded = [effect for effect in rollout.effects if effect.band is not None]
    assert banded
    for effect in banded:
        assert effect.interval is not None
        assert effect.band[0] <= effect.interval[0] <= effect.interval[1] <= effect.band[1]
    assert result.answer.kind == "structured"
    loaded = require_success(antecedent.load(bytes(result.export())))
    assert loaded.quasi_rollout == rollout
    assert loaded.answer == result.answer


def test_bayesian_rollout_refuses_a_frequentist_resample_naming_the_next_attempt():
    query = StaggeredRollout(
        "y", "unit", "period", "cohort", resampling="cluster_bootstrap", replicates=99
    )
    with pytest.raises(CausalError, match="Next attempt"):
        require_success(antecedent.analyze(_staggered_panel(), query=query, inference=_bayes()))


def test_bayesian_conformal_synthetic_control_reports_posterior_predictive_periods():
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
    result = require_success(antecedent.analyze(rows, query=query, inference=_bayes()))
    assert result.posterior is not None
    rollout = result.quasi_rollout
    assert isinstance(rollout, QuasiRolloutEstimate)
    assert rollout.design == "posterior_predictive_synthetic_control"
    periods = rollout.effects_of("period")
    assert len(periods) == 5
    assert all(effect.interval is not None for effect in periods)


def _ar1_running_time_rd(seed: int = 21, n: int = 600) -> dict[str, np.ndarray]:
    """Calendar time is the running variable; errors are AR(1) with coefficient 0.8."""
    rng = np.random.default_rng(seed)
    running = np.arange(n) / (n / 2) - 1.0
    uptake = (rng.uniform(size=n) < 0.2 + 0.6 * (running >= 0.0)).astype(float)
    noise = np.empty(n)
    noise[0] = rng.normal() / np.sqrt(1.0 - 0.8**2)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal()
    outcome = 1.0 + 0.5 * running + 2.0 * uptake + noise
    return {"y": outcome, "t": uptake, "x": running}


def test_bayesian_temporal_fuzzy_rd_widens_beyond_the_independent_row_error():
    data = _ar1_running_time_rd()
    query = FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0, temporal_replicates=199)
    frequentist = require_success(antecedent.analyze(data, query=query)).quasi_rollout
    bayesian = require_success(antecedent.analyze(data, query=query, inference=_bayes()))
    assert bayesian.posterior is not None
    rollout = bayesian.quasi_rollout
    assert isinstance(rollout, QuasiRolloutEstimate) and isinstance(
        frequentist, QuasiRolloutEstimate
    )
    assert rollout.design == "temporal_fuzzy_rd"
    assert rollout.uncertainty.startswith("bayesian_")
    assert rollout.diagnostic("block_length") is not None
    posterior_sd = rollout.diagnostic("posterior_sd")
    independent = frequentist.diagnostic("hc0_ratio_standard_error")
    assert posterior_sd is not None and independent is not None
    assert posterior_sd > 1.3 * independent


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
    return require_success(antecedent.analyze(_staggered_panel(), query=query))


def test_rollout_export_load_round_trip_reproduces_the_live_result():
    result = _resampled_dynamic_rollout()
    artifact = bytes(result.export())
    loaded = require_success(antecedent.load(artifact))
    assert bytes(loaded.export()) == artifact
    assert loaded.quasi_rollout == result.quasi_rollout
    assert loaded.answer == result.answer


def test_a_fresh_consumer_rederives_the_dynamic_summary_from_the_loaded_effects():
    loaded = require_success(antecedent.load(bytes(_resampled_dynamic_rollout().export())))
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
    section = require_success(antecedent.analyze(data, query=query)).panel_did
    assert section.standard_error is not None and section.standard_error > 0.0
    assert section.interval_95 is not None
    half = 1.959963984540054 * section.standard_error
    assert section.interval_95[0] == pytest.approx(section.estimate - half, rel=1e-9, abs=1e-12)
    assert section.interval_95[1] == pytest.approx(section.estimate + half, rel=1e-9, abs=1e-12)
    assert section.estimate == pytest.approx(2.0, abs=0.3)
