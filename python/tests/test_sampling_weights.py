"""Sampling weights: the ``weights=`` column role of the observational estimators.

``weights="w"`` names a column of sampling weights (or gives one weight per row). The estimate
speaks for the weighted population: it is checked against weighted least squares and a
weighted Hajek estimator written here in numpy, a rescaling of the weights changes nothing, a
zero-weight row is a deleted row that is counted, and an invalid weight is refused naming its
row. The effective sample size comes back on ``result.estimate.sampling_weights``.
"""

from __future__ import annotations

import dataclasses

import antecedent as ant
import numpy as np
import pandas as pd
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimators import (
    DML,
    Aipw,
    CausalForest,
    ClusterDml,
    DistanceMatching,
    DRLearner,
    GlmAdjustment,
    LinearAdjustment,
    Overlap,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
)
from antecedent.outcomes import require_success

EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect(treatment="t", outcome="y")
ONE_SHOT = {
    "inference": ant.Frequentist(bootstrap=0),
    "refute": False,
    "execution": ant.Execution(seed=1),
}


def _survey(n: int = 160, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    y = 0.5 * z + (1.0 + 0.4 * z) * t + 0.6 * rng.normal(size=n)
    w = 0.4 + 4.0 * rng.uniform(size=n) * rng.uniform(size=n)
    w[::7] += 3.0
    return {"z": z, "t": t, "y": y, "w": w}


def _run(data, estimator, *, bootstrap=0, **options):
    declared = {**ONE_SHOT, "inference": ant.Frequentist(bootstrap=bootstrap), **options}
    return require_success(
        ant.analyze(data, graph=EDGES, query=QUERY, estimator=estimator, **declared)
    )


def _wls_hc1(data) -> tuple[float, float]:
    """Weighted least squares of y on [1, t, z] and the HC1 sandwich of the t coefficient."""
    keep = data["w"] > 0
    w = data["w"][keep]
    w = w * len(w) / w.sum()
    x = np.column_stack([np.ones(keep.sum()), data["t"][keep], data["z"][keep]])
    y = data["y"][keep]
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    beta = bread @ (x.T @ (w * y))
    residual = y - x @ beta
    scores = (w * residual)[:, None] * x
    n, p = x.shape
    cov = bread @ (scores.T @ scores) @ bread * n / (n - p)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def _weighted_logit(z, t, m):
    beta = np.zeros(2)
    x = np.column_stack([np.ones_like(z), z])
    for _ in range(100):
        e = 1 / (1 + np.exp(-(x @ beta)))
        step = np.linalg.solve(x.T @ ((m * e * (1 - e))[:, None] * x), x.T @ (m * (t - e)))
        beta += step
        if np.abs(step).max() < 1e-13:
            break
    return 1 / (1 + np.exp(-(x @ beta)))


def _hajek(data) -> float:
    w = data["w"] * len(data["w"]) / data["w"].sum()
    e = _weighted_logit(data["z"], data["t"], w)
    ipw = np.where(data["t"] > 0.5, 1 / e, 1 / (1 - e)) * w
    treated = data["t"] > 0.5
    return float(
        (ipw[treated] * data["y"][treated]).sum() / ipw[treated].sum()
        - (ipw[~treated] * data["y"][~treated]).sum() / ipw[~treated].sum()
    )


def _with_zero_weight_rows(data):
    pad = {
        key: np.concatenate([values[:3], [999.0, -5.0], values[3:]]) for key, values in data.items()
    }
    pad["w"] = np.concatenate([data["w"][:3], [0.0, 0.0], data["w"][3:]])
    pad["t"] = np.concatenate([data["t"][:3], [1.0, 0.0], data["t"][3:]])
    return pad


def _hajek_of_mass(data, mass) -> float:
    """The Hajek effect of the weighted-logistic inverse-probability weights at case masses."""
    e = _weighted_logit(data["z"], data["t"], mass)
    treated = data["t"] > 0.5
    ipw = np.where(treated, 1 / e, 1 / (1 - e)) * mass
    return float(
        (ipw[treated] * data["y"][treated]).sum() / ipw[treated].sum()
        - (ipw[~treated] * data["y"][~treated]).sum() / ipw[~treated].sum()
    )


def _case_mass_scores(effect, mass, step=1e-5) -> np.ndarray:
    """``m_i dT/dm_i`` of an estimator ``T(mass)`` by central differences: the with-replacement
    linearisation of an estimator of the weighted measure sum_i m_i delta_i has these per-unit
    contributions, and their squares sum to its variance."""
    scores = np.empty(len(mass))
    for i in range(len(mass)):
        plus, minus = mass.copy(), mass.copy()
        plus[i] *= 1 + step
        minus[i] *= 1 - step
        scores[i] = (effect(plus) - effect(minus)) / (2 * step)
    return scores


def _cluster_sandwich(scores, labels) -> float:
    """sqrt(G / (G - 1) * sum_g (sum_{i in g} score_i)^2) for scores that sum to the influence."""
    totals = pd.Series(scores).groupby(np.asarray(labels)).sum().to_numpy()
    groups = len(totals)
    return float(np.sqrt(groups / (groups - 1) * (totals**2).sum()))


def _stratified_effect(data, strata, mass, k) -> float:
    """Pooled weighted within-stratum contrast, strata fixed, pooled by weighted stratum mass."""
    total = pooled = 0.0
    for s in range(k):
        in_stratum = strata == s
        arms = []
        for arm in (1.0, 0.0):
            rows = in_stratum & (data["t"] == arm)
            arms.append((mass[rows].sum(), (mass[rows] * data["y"][rows]).sum()))
        if all(weight > 0 for weight, _ in arms):
            share = arms[0][0] + arms[1][0]
            pooled += share * (arms[0][1] / arms[0][0] - arms[1][1] / arms[1][0])
            total += share
    return pooled / total


def _strata_of(data, mass, k) -> np.ndarray:
    """Cut the weighted-logistic propensity at its sample quantiles; assign by ``score <= cut``."""
    n = len(mass)
    score = _weighted_logit(data["z"], data["t"], mass)
    order = np.lexsort((np.arange(n), score))
    cuts = [score[order[min(-(-s * n // k) - 1, n - 1)]] for s in range(1, k)]
    return np.array(
        [min(next((j for j, c in enumerate(cuts) if v <= c), len(cuts)), k - 1) for v in score]
    )


def _score_table_sandwich(estimate, labels) -> tuple[float, float]:
    """Mean and cluster-sandwich SE of the contrast scores ``b - a`` of a cross-fitted score
    table (two arm columns, column-major), from the definition."""
    table = estimate.score_table
    scores = np.asarray(table.scores).reshape(2, table.n_rows)
    contrast = scores[1] - scores[0]
    centred = contrast - contrast.mean()
    totals = pd.Series(centred).groupby(np.asarray(labels)).sum().to_numpy()
    groups = len(totals)
    se = np.sqrt(groups / (groups - 1) * (totals**2).sum()) / table.n_rows
    return float(contrast.mean()), float(se)


def test_linear_adjustment_is_weighted_least_squares_with_the_sampling_sandwich():
    data = _survey()
    ate, se = _wls_hc1(data)
    result = _run(data, LinearAdjustment(weights="w"))
    assert result.estimate.ate == pytest.approx(ate, rel=1e-9)
    assert result.estimate.se_analytic == pytest.approx(se, rel=1e-9)
    unweighted = _run({k: v for k, v in data.items() if k != "w"}, LinearAdjustment())
    assert abs(unweighted.estimate.ate - result.estimate.ate) > 1e-4


def test_propensity_weighting_is_the_hajek_effect_of_the_weighted_logistic_fit():
    data = _survey()
    result = _run(data, PropensityWeighting(weights="w", bootstrap=0))
    assert result.estimate.ate == pytest.approx(_hajek(data), rel=1e-7)


def test_the_effective_sample_size_and_row_accounting_are_reported():
    data = _with_zero_weight_rows(_survey())
    result = _run(data, LinearAdjustment(weights="w"))
    record = result.estimate.sampling_weights
    positive = data["w"][data["w"] > 0]
    assert record.rows_used == len(positive)
    assert record.zero_weight_rows == 2
    assert record.effective_sample_size == pytest.approx(positive.sum() ** 2 / (positive**2).sum())
    assert record.design_effect == pytest.approx(len(positive) / record.effective_sample_size)
    assert record.weight_sum == pytest.approx(positive.sum())
    assert _run(_survey(), LinearAdjustment()).estimate.sampling_weights is None


@pytest.mark.parametrize(
    "estimator",
    [
        LinearAdjustment(weights="w"),
        PropensityWeighting(weights="w", bootstrap=0),
        PropensityStratification(weights="w", bootstrap=0, n_strata=4),
        PropensityMatching(weights="w", bootstrap=0),
        DistanceMatching(weights="w", bootstrap=0),
        GlmAdjustment(weights="w", bootstrap=0, family="gaussian_identity"),
    ],
)
def test_rescaling_the_weights_and_zero_weight_rows_change_nothing(estimator):
    data = _survey()
    base = _run(data, estimator)
    for factor in (0.001, 1000.0):
        scaled = _run({**data, "w": data["w"] * factor}, estimator)
        assert scaled.estimate.ate == pytest.approx(base.estimate.ate, rel=1e-9)
        assert scaled.estimate.se_analytic == pytest.approx(base.estimate.se_analytic, rel=1e-9)
        assert scaled.estimate.sampling_weights.effective_sample_size == pytest.approx(
            base.estimate.sampling_weights.effective_sample_size
        )
    padded = _run(_with_zero_weight_rows(data), estimator)
    assert padded.estimate.ate == pytest.approx(base.estimate.ate, rel=1e-9)
    assert padded.estimate.sampling_weights.zero_weight_rows == 2


def test_invalid_weights_are_refused_naming_the_first_offender():
    data = _survey(60)
    for bad, text in ((-1.0, "-1"), (float("nan"), "NaN")):
        weights = data["w"].copy()
        weights[9] = bad
        with pytest.raises(
            CausalValueError,
            match=rf"sampling_weights\.invalid: sampling weight {text} at source row 9",
        ) as raised:
            _run({**data, "w": weights}, LinearAdjustment(weights="w"))
        assert "source row 9" in str(raised.value)
    with pytest.raises(CausalValueError, match="every sampling weight is zero"):
        _run({**data, "w": np.zeros(60)}, LinearAdjustment(weights="w"))


def test_weights_are_read_by_name_or_as_one_weight_per_row():
    data = _survey()
    named = _run(data, LinearAdjustment(weights="w"))
    frame = pd.DataFrame(data)
    from_frame = _run(frame, LinearAdjustment(weights="w"))
    by_array = _run(frame.drop(columns="w"), LinearAdjustment(weights=list(data["w"])))
    assert from_frame.estimate.ate == pytest.approx(named.estimate.ate, rel=1e-12)
    assert by_array.estimate.ate == pytest.approx(named.estimate.ate, rel=1e-12)


def test_weights_follow_their_rows_through_the_complete_case_mask():
    data = _survey()
    holed = {k: v.copy() for k, v in data.items()}
    holed["y"][[4, 11]] = np.nan
    keep = ~np.isnan(holed["y"])
    expected, _ = _wls_hc1({k: v[keep] for k, v in data.items()})
    result = _run(holed, LinearAdjustment(weights="w"))
    assert result.estimate.ate == pytest.approx(expected, rel=1e-9)


def test_an_estimator_that_reads_no_weights_refuses_them():
    data = _survey()
    with pytest.raises(CausalUnsupportedError, match="does not read"):
        _run(data, "iv.2sls", estimator_config={"weights": "w"})
    with pytest.raises(CausalValueError, match="ordinary least-squares"):
        LinearAdjustment(weights="w", fit="ridge", fit_lambda=1.0)


def test_aipw_weights_move_the_estimate_and_report_the_record():
    data = _survey(300)
    weighted = _run(data, Aipw(weights="w", bootstrap=0))
    plain = _run({k: v for k, v in data.items() if k != "w"}, Aipw(bootstrap=0))
    assert abs(weighted.estimate.ate - plain.estimate.ate) > 1e-4
    assert weighted.estimate.sampling_weights.effective_sample_size < 300


def test_a_refreshed_table_with_other_weights_is_refused():
    data = _survey()
    study = ant.prepare(
        data, graph=EDGES, query=QUERY, estimator=LinearAdjustment(weights="w"), **ONE_SHOT
    )
    other = {**data, "w": data["w"] * np.linspace(0.5, 1.5, len(data["w"]))}
    with pytest.raises(CausalValueError, match="weights column 'w' changed"):
        ant.rebind(study, data=other)


def test_clusters_on_propensity_weighting_and_stratification_are_valid():
    data = _survey(240)
    data["site"] = np.array([f"s{i // 8}" for i in range(240)])
    mass = data["w"] * len(data["w"]) / data["w"].sum()
    # Weighting: the cluster sandwich of the case-mass scores of the whole estimator (the
    # propensity refit included), computed here from the definition.
    weighting_scores = _case_mass_scores(lambda m: _hajek_of_mass(data, m), mass)
    # Stratification: the same sandwich of the pooled stratified point's scores, strata fixed.
    strata = _strata_of(data, mass, 5)
    strata_scores = _case_mass_scores(lambda m: _stratified_effect(data, strata, m, 5), mass)
    cases = (
        (PropensityWeighting, weighting_scores, _hajek(data)),
        (
            PropensityStratification,
            strata_scores,
            _stratified_effect(data, strata, mass, 5),
        ),
    )
    for estimator, scores, point in cases:
        clustered = _run(data, estimator(cluster="site", weights="w", bootstrap=0))
        flat = _run(data, estimator(weights="w", bootstrap=0))
        assert clustered.estimate.ate == pytest.approx(flat.estimate.ate, rel=1e-12)
        assert clustered.estimate.ate == pytest.approx(point, rel=1e-7)
        expected = _cluster_sandwich(scores, data["site"])
        assert clustered.estimate.se_analytic == pytest.approx(expected, rel=1e-5), estimator
        assert clustered.estimate.se_analytic != pytest.approx(flat.estimate.se_analytic, rel=1e-9)


def test_the_cluster_bootstrap_resamples_whole_clusters():
    data = _survey(240)
    data["site"] = np.array([f"s{i // 8}" for i in range(240)])
    result = _run(data, PropensityWeighting(cluster="site", bootstrap=200))
    assert result.estimate.se_bootstrap is not None
    # Independent whole-cluster bootstrap: draw the 30 clusters with replacement, refit the
    # logistic propensity and the Hajek contrast on the drawn rows, take the SD over replicates.
    rng = np.random.default_rng(2024)
    members = [np.flatnonzero(data["site"] == s) for s in np.unique(data["site"])]
    draws = []
    for _ in range(600):
        rows = np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
        resample = {k: v[rows] for k, v in data.items() if k != "site"}
        draws.append(_hajek_of_mass(resample, np.ones(len(rows))))
    reference = float(np.std(draws, ddof=1))
    # 200 product and 600 reference replicates estimate an SD to about 6 percent together.
    assert result.estimate.se_bootstrap == pytest.approx(reference, rel=0.25)


def test_a_causal_forest_with_clusters_subsamples_and_splits_whole_clusters():
    data = _survey(400)
    data["site"] = np.array([f"s{i // 10}" for i in range(400)])
    result = _run(data, CausalForest(n_trees=12, min_leaf=5, cluster="site"), bootstrap=0)
    # The forest's marginal ATE is the mean of the whole-cluster cross-fitted AIPW contrast
    # scores and its SE their cluster sandwich: both recomputed here from the retained scores.
    mean, se = _score_table_sandwich(result.estimate, data["site"])
    assert result.estimate.ate == pytest.approx(mean, rel=1e-10)
    assert result.estimate.se_analytic == pytest.approx(se, rel=1e-10)
    with pytest.raises(CausalUnsupportedError, match="too_few_clusters"):
        data["site"] = np.array([f"s{i // 100}" for i in range(400)])
        _run(data, CausalForest(n_trees=12, min_leaf=5, cluster="site"), bootstrap=0)


# --- randomized trial, panel difference-in-differences, temporal window --------------------


def _trial(n: int = 200, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    assigned = rng.uniform(size=n) < 0.5
    y = x + np.where(assigned, 1.0 + 0.5 * x, 0.0) + 0.7 * rng.normal(size=n)
    w = 0.5 + 3.0 * rng.uniform(size=n)
    return {"y": y, "w": w, "assigned": assigned.astype(float)}


def _trial_hajek(data) -> tuple[float, float]:
    assigned = data["assigned"] > 0.5
    keep = data["w"] > 0
    effects, variances = [], []
    for arm in (True, False):
        rows = keep & (assigned == arm)
        a = data["w"][rows] / 0.5
        mean = (a * data["y"][rows]).sum() / a.sum()
        n = rows.sum()
        spread = ((a * (data["y"][rows] - mean)) ** 2).sum() * n / (n - 1) / a.sum() ** 2
        effects.append(mean)
        variances.append(spread)
    return effects[0] - effects[1], float(np.sqrt(sum(variances)))


def _trial_query(data, **extra):
    n = len(data["y"])
    design = ant.experiment.ExperimentDesign(
        ant.interference.BernoulliAssignment(0.5),
        [bool(a) for a in data["assigned"]],
        [f"account-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    return ant.experiment.RandomizedEffect("y", design, **extra)


def test_a_weighted_randomized_trial_is_the_hajek_contrast_and_reports_its_record():
    data = _trial()
    effect, se = _trial_hajek(data)
    result = require_success(
        ant.analyze(
            {"y": data["y"], "w": data["w"]},
            query=_trial_query(data, weights="w"),
            refute="none",
        )
    )
    fit = result.randomized_effect
    assert fit.effect == pytest.approx(effect, rel=1e-9)
    assert fit.standard_error is None and fit.interval_95 is None
    assert np.sqrt(fit.variance_upper_bound) == pytest.approx(se, rel=1e-9)
    assert fit.uncertainty == "sampling_weighted_hajek_linearized_variance_no_interval"
    record = result.estimate.sampling_weights
    assert record.rows_used == 200 and record.zero_weight_rows == 0
    scaled = require_success(
        ant.analyze(
            {"y": data["y"], "w": data["w"] * 1e4},
            query=_trial_query(data, weights="w"),
            refute="none",
        )
    )
    assert scaled.randomized_effect.effect == pytest.approx(fit.effect, rel=1e-9)


def test_a_randomized_trial_refuses_weights_it_cannot_read():
    data = _trial()
    design = ant.experiment.ExperimentDesign(
        ant.interference.ClusterRandomization([i // 20 for i in range(200)], 5),
        [bool((i // 20) % 2 == 0) for i in range(200)],
        [f"cluster-{i // 20}" for i in range(200)],
        [f"row-{i}" for i in range(200)],
    )
    with pytest.raises(CausalUnsupportedError, match="unadjusted Bernoulli and complete"):
        ant.experiment.RandomizedEffect("y", design, weights="w")
    with pytest.raises(CausalValueError, match="source row 3"):
        weights = data["w"].copy()
        weights[3] = -1.0
        require_success(
            ant.analyze(
                {"y": data["y"], "w": weights}, query=_trial_query(data, weights="w"), refute="none"
            )
        )


def _panel(clusters_per_arm: int = 20, seed: int = 5):
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"y": [], "id": [], "group": [], "after": [], "site": [], "w": []}
    subject = 0
    for cluster in range(2 * clusters_per_arm):
        active = cluster >= clusters_per_arm
        shock = 0.6 * rng.normal()
        for _ in range(4):
            weight = 0.5 + 3.0 * rng.uniform()
            baseline = rng.normal() + 0.5 * shock
            change = 0.3 + (1.0 if active else 0.0) + shock + 0.5 * rng.normal()
            for after, value in ((0, baseline), (1, baseline + change)):
                rows["y"].append(value)
                rows["id"].append(f"s{subject}")
                rows["group"].append(int(active))
                rows["after"].append(after)
                rows["site"].append(f"c{cluster}")
                rows["w"].append(weight)
            subject += 1
    return {key: np.asarray(value) for key, value in rows.items()}


def _did_oracle(data) -> tuple[float, float]:
    n = len(data["y"]) // 2
    change = data["y"][1::2] - data["y"][0::2]
    weight = data["w"][0::2]
    weight = weight * n / weight.sum()
    treated = data["group"][0::2] == 1
    clusters = data["site"][0::2]
    means, scores = [], {}
    for arm, sign in ((True, 1.0), (False, -1.0)):
        rows = treated == arm
        mass = weight[rows].sum()
        mean = (weight[rows] * change[rows]).sum() / mass
        means.append(mean)
        for site, w, c in zip(clusters[rows], weight[rows], change[rows], strict=True):
            scores[site] = scores.get(site, 0.0) + sign * w * (c - mean) / mass
    g = len(scores)
    return means[0] - means[1], float(np.sqrt(g / (g - 1) * sum(v * v for v in scores.values())))


def test_a_weighted_panel_did_is_the_weighted_change_contrast_with_a_cluster_sandwich():
    data = _panel()
    effect, se = _did_oracle(data)
    result = require_success(
        ant.analyze(
            data,
            query=ant.quasi.PanelDifferenceInDifferences(
                "y", "id", "group", "after", cluster="site", weights="w"
            ),
            refute="none",
        )
    )
    fit = result.panel_did
    assert fit.estimate == pytest.approx(effect, rel=1e-9)
    assert fit.standard_error == pytest.approx(se, rel=1e-9)
    assert fit.interval_95 is None
    assert result.estimate.sampling_weights.rows_used == len(data["y"]) // 2
    with pytest.raises(CausalUnsupportedError, match="repeated cross-section"):
        ant.quasi.PanelDifferenceInDifferences(
            "y", "id", "group", "after", sampling="repeated_cross_section", weights="w"
        )


def test_a_weighted_temporal_pulse_reads_one_weight_per_time_step():
    rng = np.random.default_rng(9)
    n = 240
    t = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 3.0 * (t[i - 2] if i >= 2 else 0.0)
        y[i] += 0.8 * rng.normal()
    w = 0.3 + 3.0 * rng.uniform(size=n)
    graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
    kwargs = {
        "graph": graph,
        "refute": False,
        "inference": ant.Frequentist(bootstrap=0),
        "execution": ant.Execution(seed=1),
    }
    weighted = require_success(
        ant.analyze(
            {"t": t, "y": y, "w": w}, query=ant.PulseEffect("t", "y", weights="w"), **kwargs
        )
    )
    # The pulse at lag 1 adjusts for no other lag (the lag-2 treatment is independent of it),
    # so the windows are y_t on [1, t_{t-1}] for t = 1..n-1, each weighted by w_t.
    rows = np.arange(1, n)
    x = np.column_stack([np.ones(len(rows)), t[rows - 1]])
    sqrt_w = np.sqrt(w[rows])[:, None]
    beta = np.linalg.lstsq(sqrt_w * x, np.sqrt(w[rows]) * y[rows], rcond=None)[0]
    assert weighted.estimate.ate == pytest.approx(beta[1], rel=1e-8)
    assert weighted.estimate.sampling_weights.rows_used == n - 1
    with pytest.raises(CausalUnsupportedError, match="sampling_weights.route_unsupported|refuters"):
        require_success(
            ant.analyze(
                {"t": t, "y": y, "w": w},
                query=ant.PulseEffect("t", "y", weights="w"),
                graph=graph,
                refute="cheap",
                inference=ant.Frequentist(bootstrap=0),
            )
        )


@pytest.mark.parametrize(
    "estimator",
    [
        DistanceMatching(weights="w", bootstrap=0),
        GlmAdjustment(weights="w", bootstrap=0, family="gaussian_identity"),
        DML(weights="w"),
        DML(weights="w", score="partially_linear"),
        DRLearner(weights="w"),
        Aipw(weights="w", bootstrap=0, overlap=Overlap(trim=0.05)),
    ],
)
def test_the_newly_weighted_estimators_move_with_the_weights_and_report_the_record(estimator):
    data = _survey(240)
    weighted = _run(data, estimator)
    plain = _run(
        {k: v for k, v in data.items() if k != "w"}, dataclasses.replace(estimator, weights=None)
    )
    assert abs(weighted.estimate.ate - plain.estimate.ate) > 1e-6
    record = weighted.estimate.sampling_weights
    assert record.rows_used == 240
    assert record.effective_sample_size < 240


def test_the_classical_se_is_refused_with_glm_weights():
    with pytest.raises(CausalValueError, match="precision"):
        GlmAdjustment(weights="w", se="homoskedastic")


def test_cluster_dml_with_weights_reads_the_weighted_cluster_sandwich():
    # Integer weights are row multiplicities. The weighted cluster-DML contrast and its cluster
    # sandwich therefore equal those of the unweighted fit on the replicated rows (same cluster
    # labels, so the same whole-cluster folds); the unweighted fit's SE is checked against the
    # sandwich of its own retained scores, computed here from the definition.
    data = _survey(400)
    data["w"] = np.array([1.0 + (i % 3) for i in range(400)])
    data["site"] = np.array([f"s{i // 10}" for i in range(400)])
    repeats = data["w"].astype(int)
    replicated = {k: np.repeat(v, repeats) for k, v in data.items() if k != "w"}
    # The default outcome learner is a ridge with a fixed penalty, whose fit is not invariant to
    # duplicating rows, so the outcome nuisance is the unpenalised linear fit here.
    for estimator in (DML, DRLearner):
        unit = ClusterDml(cluster="site")
        weighted = _run(data, estimator(outcome="linear", weights="w", cluster_dml=unit))
        plain = _run(replicated, estimator(outcome="linear", cluster_dml=unit))
        assert weighted.estimate.sampling_weights.rows_used == 400
        assert weighted.estimate.ate == pytest.approx(plain.estimate.ate, rel=1e-6)
        assert weighted.estimate.se_analytic == pytest.approx(plain.estimate.se_analytic, rel=1e-6)
        mean, se = _score_table_sandwich(plain.estimate, replicated["site"])
        assert plain.estimate.ate == pytest.approx(mean, rel=1e-10)
        assert plain.estimate.se_analytic == pytest.approx(se, rel=1e-10)
        flat = _run(data, estimator(outcome="linear", weights="w"))
        assert weighted.estimate.se_analytic != pytest.approx(flat.estimate.se_analytic, rel=1e-3)


def test_a_penalized_propensity_reads_weights_through_a_weighted_penalty_selection():
    data = _survey(300)
    penalty = ant.estimators.PropensityPenalty()
    weighted = _run(data, Aipw(weights="w", bootstrap=0, propensity_penalty=penalty))
    plain = _run(
        {k: v for k, v in data.items() if k != "w"},
        Aipw(bootstrap=0, propensity_penalty=penalty),
    )
    assert abs(weighted.estimate.ate - plain.estimate.ate) > 1e-6
    assert weighted.estimate.sampling_weights.rows_used == 300
