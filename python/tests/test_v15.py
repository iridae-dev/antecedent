"""1.5 Python surface: functionals, retarget, tiered background, EconML extras."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")

import antecedent
from antecedent.graph import TieredBackground, WithinTier
from antecedent.handoff import econml
from antecedent.query import Exceedance, ExceedanceGrid, Mean, Quantile, coerce_outcome_functional


def _binary_confounded(n: int = 400, seed: int = 15):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _codetermined_siblings(n: int = 800, seed: int = 214):
    """CoDetermined `{z, u} | {t} | {y}` — same-tier z↔u in the closure ADMG."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.3 + 0.9 * z + 0.5 * u)))).astype(float)
    y = (1.0 + z) * t + 0.4 * z + 0.3 * u + 0.35 * rng.normal(size=n)
    weights = np.exp(-0.5 * ((z - 0.6) / 0.7) ** 2)
    data = {"t": t, "y": y, "z": z, "u": u}
    background = TieredBackground(
        tiers=[["z", "u"], ["t"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    return data, background, weights


def _assert_same_effect(batch, solo, *, abs_=1e-12):
    a, b = batch.estimate.ate, solo.estimate.ate
    if np.isnan(a) and np.isnan(b):
        assert batch.estimate.exceedance_cdf is not None
        assert solo.estimate.exceedance_cdf is not None
        np.testing.assert_allclose(
            batch.estimate.exceedance_cdf, solo.estimate.exceedance_cdf, atol=abs_, rtol=0.0
        )
        return
    assert a == pytest.approx(b, abs=abs_)
    if batch.estimate.exceedance_cdf is not None or solo.estimate.exceedance_cdf is not None:
        np.testing.assert_allclose(
            batch.estimate.exceedance_cdf, solo.estimate.exceedance_cdf, atol=abs_, rtol=0.0
        )


def test_outcome_functional_wire():
    assert coerce_outcome_functional(None) is None
    assert coerce_outcome_functional(Mean()) is None
    assert coerce_outcome_functional(Exceedance(1.2)) == {"kind": "exceedance", "threshold": 1.2}
    assert coerce_outcome_functional(ExceedanceGrid([0.0, 0.5])) == {
        "kind": "exceedance_grid",
        "thresholds": [0.0, 0.5],
    }
    assert coerce_outcome_functional(Quantile(0.5)) == {"kind": "quantile", "tau": 0.5}


def test_analyze_exceedance_and_mean():
    data = _binary_confounded()
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    mean = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    exc = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y", outcome_functional=Exceedance(1.0)),
        refute="none",
        bootstrap=0,
        estimator="aipw",
    )
    assert np.isfinite(mean.estimate.ate)
    assert np.isfinite(exc.estimate.ate)
    qte = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y", outcome_functional=Quantile(0.5)),
        refute="none",
        bootstrap=0,
        estimator="aipw",
    )
    assert np.isfinite(qte.estimate.ate)


def test_prepared_retarget():
    data = _binary_confounded(800)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        query=antecedent.AverageEffect("t", "y"),
        graph=graph,
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    weights = np.exp(-0.5 * ((data["z"] - 0.5) / 0.8) ** 2)
    out = plan.retarget(weights, ["z"])
    assert np.isfinite(out.estimate.ate)
    table = out.estimate.score_table
    assert table is not None
    assert table.n_rows == len(data["t"])
    assert len(table.observed_outcome) == table.n_rows
    assert table.treatment == 0
    assert table.intervened == []
    assert table.adjustment_set == [2]
    assert len(table.observed_arm) == table.n_rows
    assert len(table.propensities) == table.n_rows * len(table.columns)


def test_tiered_background():
    rng = np.random.default_rng(22)
    n = 500
    era = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-era))).astype(float)
    y = 1.5 * t + 0.4 * era + 0.3 * rng.normal(size=n)
    result = antecedent.analyze(
        {"era": era, "t": t, "y": y},
        graph=TieredBackground(tiers=[["era"], ["t"], ["y"]], within_tier=WithinTier.CODETERMINED),
        query=antecedent.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    assert np.isfinite(result.estimate.ate)


def test_econml_modifier_set_and_exceedance():
    data = _binary_confounded()
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    spec = econml(
        result,
        modifiers=["z"],
        target_weights=np.ones(len(data["t"])),
        outcome_functional=Exceedance(0.5),
    )
    assert spec.modifiers == ("z",)
    assert spec.outcome_functional is not None
    cols = spec.columns(data)
    assert "W" in cols


def test_exceedance_grid_on_fresh_estimate():
    data = _binary_confounded(800)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    query = antecedent.AverageEffect("t", "y", outcome_functional=ExceedanceGrid([0.0, 0.5, 1.0]))
    fresh = antecedent.analyze(
        data, graph=graph, query=query, estimator="aipw", refute="none", bootstrap=0
    )
    cdf = fresh.estimate.exceedance_cdf
    assert cdf is not None
    assert len(cdf) == 6
    assert all(np.isfinite(cdf))
    assert np.isnan(fresh.estimate.ate)
    assert fresh.estimate.monotone_rearranged
    assert fresh.estimate.simultaneous_interval is None
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=query, graph=graph, estimator="aipw", refute="none", bootstrap=0
    )
    click = plan.estimate(data)
    assert click.estimate.exceedance_cdf is not None
    assert len(click.estimate.exceedance_cdf) == 6


def test_class_aware_prepare_conditional_exceedance():
    data = _binary_confounded(600)
    graph = antecedent.Cpdag.from_directed_undirected(
        ["t", "y", "z"],
        [("z", "t"), ("z", "y"), ("t", "y")],
        [],
    )
    mean = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        query=antecedent.ConditionalEffect("t", "y", "z"),
        graph=graph,
        refute="none",
        bootstrap=0,
    ).estimate(data)
    query = antecedent.ConditionalEffect("t", "y", "z", outcome_functional=Exceedance(0.5))
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=query, graph=graph, refute="none", bootstrap=0
    )
    out = plan.estimate(data)
    assert np.isfinite(out.estimate.ate)
    assert out.estimate.ate != pytest.approx(mean.estimate.ate, abs=1e-6)
    grid = antecedent.ConditionalEffect(
        "t", "y", "z", outcome_functional=ExceedanceGrid([0.5, 1.0, 1.5])
    )
    out_grid = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=grid, graph=graph, refute="none", bootstrap=0
    ).estimate(data)
    cdf = out_grid.estimate.exceedance_cdf
    assert cdf is not None
    assert len(cdf) == 6
    assert all(0.0 <= value <= 1.0 for value in cdf)
    assert np.all(np.diff(np.array(cdf).reshape(-1, 2), axis=0) >= 0.0)
    assert np.isnan(out_grid.estimate.ate)
    inf = out_grid.estimate.score_inference
    assert inf is not None
    assert len(inf.raw_means) == 6
    assert len(inf.lower) == 6
    assert len(inf.threshold_supported) == 6
    assert any(inf.threshold_supported)
    assert all(
        np.isfinite(lo) for lo, ok in zip(inf.lower, inf.threshold_supported, strict=True) if ok
    )
    extreme = antecedent.ConditionalEffect(
        "t", "y", "z", outcome_functional=ExceedanceGrid([0.0, 1.0e6])
    )
    tail = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=extreme, graph=graph, refute="none", bootstrap=0
    ).estimate(data)
    tail_inf = tail.estimate.score_inference
    assert tail_inf is not None
    assert tail_inf.threshold_supported[-2:] == [False, False]
    assert all(not np.isfinite(v) for v in tail_inf.lower[-2:])


def test_analyze_does_not_always_return_scores():
    data = _binary_confounded()
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    linear = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        refute="none",
        bootstrap=0,
    )
    assert linear.estimate.score_table is None
    aipw = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    assert aipw.estimate.score_table is not None
    ones = np.ones(aipw.estimate.score_table.n_rows)
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        query=antecedent.AverageEffect("t", "y"),
        graph=graph,
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    retargeted = plan.retarget(ones, [])
    assert aipw.estimate.ate == pytest.approx(retargeted.estimate.ate, abs=1e-12)


def test_prepared_batch_and_candidate_selection():
    data = _binary_confounded(500)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    q1 = antecedent.AverageEffect("t", "y")
    q2 = antecedent.AverageEffect("t", "y", control_level=0.0, active_level=1.0)
    n = len(data["t"])
    screen = antecedent.estimation.CandidateScreen(
        screen_id="py.screen",
        procedure="bh",
        screen_rows=list(range(n // 2)),
        estimate_rows=list(range(n // 2, n)),
    )
    batch = antecedent.estimation.PreparedBatch.prepare(
        data,
        graph=graph,
        queries=[q1, q2],
        estimator="aipw",
        refute="none",
        bootstrap=0,
        candidate_screen=screen,
    )
    results = batch.estimate(data)
    assert len(results) == 2
    sel = results[0].estimate.candidate_selection
    assert sel is not None
    assert sel.screen_id == "py.screen"
    assert sel.procedure == "bh"
    assert sel.disjoint
    assert len(sel.screen_rows) == n // 2
    assert len(sel.estimate_rows) == n - n // 2
    assert results[0].estimate.adjusted_p_values is not None
    design = batch.shared_design
    assert design is not None
    assert design.n_folds == 5
    assert design.shares_covariates
    assert not design.shares_propensity
    assert not design.shares_outcome_residualization
    assert design.fold_ids


def test_validator_computation_failure_is_typed_and_does_not_abort_claim():
    data = _binary_confounded()
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])

    def broken_validator(**kwargs):
        raise RuntimeError("deliberate computation failure")

    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        refute="none",
        validators=[broken_validator],
        bootstrap=0,
    )
    assert np.isfinite(result.estimate.ate)
    assert result.validation.ran
    assert not result.validation.passed
    assert result.validation.count == 1
    assert not result.validation.reports
    (failure,) = result.validation.computation_failures
    assert "deliberate computation failure" in failure.reason
    assert failure.validator


def test_dag_intervention_response_cheap_runs():
    data = _binary_confounded(600)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    query = antecedent.InterventionResponse("y", intervention=antecedent.intervention.Set("t", 1.0))
    one_shot = antecedent.analyze(data, graph=graph, query=query, refute="cheap", bootstrap=0)
    assert np.isfinite(one_shot.estimate)
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=query, graph=graph, refute="none", bootstrap=0
    )
    click = plan.estimate(data)
    assert np.isfinite(click.estimate)
    validated = plan.refute(data, suite="cheap")
    assert np.isfinite(validated.estimate.ate)
    assert validated.validation.ran


def test_analyze_many_keeps_exceedance_functional():
    rng = np.random.default_rng(218)
    n = 1200
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = rng.normal(size=n) * (1.0 + 0.4 * t)
    data = {"t": t, "y": y}
    graph = antecedent.Dag.from_edges(["t", "y"], [("t", "y")])
    q90 = 1.2815515655446004
    mean_q = antecedent.AverageEffect("t", "y")
    exc_q = antecedent.AverageEffect("t", "y", outcome_functional=Exceedance(q90))
    mean, exc = antecedent.estimation.analyze_many(
        data,
        graph=graph,
        queries=[mean_q, exc_q],
        estimator="aipw",
        refute=False,
        bootstrap=0,
    )
    assert mean.estimate.exceedance_cdf is None
    assert exc.estimate.ate != pytest.approx(mean.estimate.ate, abs=1e-6)
    solo_mean = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=mean_q, graph=graph, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    solo_exc = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=exc_q, graph=graph, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    _assert_same_effect(mean, solo_mean)
    _assert_same_effect(exc, solo_exc)


def test_analyze_many_tiered_codetermined_aipw():
    data, background, _ = _codetermined_siblings(800, 219)
    mean_q = antecedent.AverageEffect("t", "y")
    grid_q = antecedent.AverageEffect("t", "y", outcome_functional=ExceedanceGrid([0.0, 0.5]))
    mean, grid = antecedent.estimation.analyze_many(
        data,
        graph=background,
        queries=[mean_q, grid_q],
        estimator="aipw",
        refute=False,
        bootstrap=0,
    )
    assert np.isfinite(mean.estimate.ate)
    assert mean.estimate.exceedance_cdf is None
    assert np.isnan(grid.estimate.ate)
    assert grid.estimate.exceedance_cdf is not None
    solo_mean = antecedent.analyze(
        data, graph=background, query=mean_q, estimator="aipw", refute="none", bootstrap=0
    )
    solo_grid = antecedent.analyze(
        data, graph=background, query=grid_q, estimator="aipw", refute="none", bootstrap=0
    )
    _assert_same_effect(mean, solo_mean)
    _assert_same_effect(grid, solo_grid)


def test_prepared_batch_keeps_exceedance_functional():
    rng = np.random.default_rng(216)
    n = 1200
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = rng.normal(size=n) * (1.0 + 0.4 * t)
    data = {"t": t, "y": y}
    graph = antecedent.Dag.from_edges(["t", "y"], [("t", "y")])
    q90 = 1.2815515655446004
    mean_q = antecedent.AverageEffect("t", "y")
    exc_q = antecedent.AverageEffect("t", "y", outcome_functional=Exceedance(q90))
    batch = antecedent.estimation.PreparedBatch.prepare(
        data,
        graph=graph,
        queries=[mean_q, exc_q],
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    mean, exc = batch.estimate(data)
    assert mean.estimate.exceedance_cdf is None
    assert exc.estimate.ate != pytest.approx(mean.estimate.ate, abs=1e-6)
    solo_mean = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=mean_q, graph=graph, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    solo_exc = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=exc_q, graph=graph, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    _assert_same_effect(mean, solo_mean)
    _assert_same_effect(exc, solo_exc)


def test_prepared_batch_tiered_codetermined_aipw():
    data, background, _ = _codetermined_siblings(800, 217)
    mean_q = antecedent.AverageEffect("t", "y")
    grid_q = antecedent.AverageEffect("t", "y", outcome_functional=ExceedanceGrid([0.0, 0.5]))
    batch = antecedent.estimation.PreparedBatch.prepare(
        data,
        graph=background,
        queries=[mean_q, grid_q],
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    mean, grid = batch.estimate(data)
    assert np.isfinite(mean.estimate.ate)
    assert mean.estimate.exceedance_cdf is None
    assert np.isnan(grid.estimate.ate)
    assert grid.estimate.exceedance_cdf is not None
    solo_mean = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=mean_q, graph=background, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    solo_grid = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=grid_q, graph=background, estimator="aipw", refute="none", bootstrap=0
    ).estimate(data)
    _assert_same_effect(mean, solo_mean)
    _assert_same_effect(grid, solo_grid)


def test_prepared_retarget_codetermined_exceedance_grid():
    data, background, weights = _codetermined_siblings(800, 214)
    query = antecedent.AverageEffect("t", "y", outcome_functional=ExceedanceGrid([0.0, 0.5, 1.0]))
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data, query=query, graph=background, estimator="aipw", refute="none", bootstrap=0
    )
    out = plan.retarget(weights, ["z"])
    cdf = out.estimate.exceedance_cdf
    assert cdf is not None
    assert len(cdf) == 6
    assert np.isnan(out.estimate.ate)
    # Walking ↔ as descendants from t would refuse a treatment-tier peer.
    rng = np.random.default_rng(214)
    n = 800
    z = rng.normal(size=n)
    latent = rng.normal(size=n)
    u = latent + 0.35 * rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * z + 0.45 * latent)))).astype(
        float
    )
    y = (1.0 + z) * t + 0.4 * z + 0.3 * u + 0.35 * rng.normal(size=n)
    w_u = np.exp(-0.5 * ((u - 0.3) / 0.8) ** 2)
    peer = antecedent.estimation.PreparedAnalysis.prepare(
        {"t": t, "y": y, "z": z, "u": u},
        query=query,
        graph=TieredBackground(
            tiers=[["z"], ["t", "u"], ["y"]], within_tier=WithinTier.CODETERMINED
        ),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    ).retarget(w_u, ["u"])
    assert peer.estimate.exceedance_cdf is not None
    assert len(peer.estimate.exceedance_cdf) == 6


def test_unrecorded_batch_has_no_invented_winner():
    data = _binary_confounded(300)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    batch = antecedent.estimation.PreparedBatch.prepare(
        data,
        graph=graph,
        queries=[antecedent.AverageEffect("t", "y")],
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    (result,) = batch.estimate(data)
    assert result.estimate.candidate_selection is not None
    assert result.estimate.candidate_selection.winner_index is None
