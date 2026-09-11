"""1.5 Python surface: functionals, retarget, tiered background, EconML extras."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")

import antecedent
from antecedent.errors import CausalIdentifyError, CausalUnsupportedError
from antecedent.graph import TieredBackground, WithinTier
from antecedent.handoff import econml
from antecedent.intervention import Set
from antecedent.query import (
    Exceedance,
    ExceedanceGrid,
    InterventionResponse,
    Mean,
    Quantile,
    coerce_outcome_functional,
)


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
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y", outcome_functional=Quantile(0.5)),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    click = plan.estimate(data)
    ones = plan.retarget(np.ones(len(data["t"])), [])
    assert click.estimate.ate == pytest.approx(qte.estimate.ate, abs=1e-12)
    assert ones.estimate.ate == pytest.approx(qte.estimate.ate, abs=1e-12)


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
    assert isinstance(fresh.estimate.monotone_rearranged, bool)
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


def test_codetermined_joint_cells_are_cell_aipw():
    rng = np.random.default_rng(17)
    n = 2000
    z = rng.normal(size=n)
    latent = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.9 * z + 0.7 * latent)))).astype(
        float
    )
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.8 * z + 0.65 * latent)))).astype(
        float
    )
    y = 1.2 * t1 + 0.8 * t2 + 1.5 * t1 * t2 + 0.55 * z + 0.3 * rng.normal(size=n)
    data = {"z": z, "t1": t1, "t2": t2, "y": y}
    background = TieredBackground(
        tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    query = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
    result = antecedent.analyze(
        data,
        graph=background,
        query=query,
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    assert np.isfinite(result.estimate)
    assert result.uncertainty.standard_error is not None
    assert np.isfinite(result.uncertainty.standard_error)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        query=query,
        graph=background,
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    click = prepared.estimate(data)
    assert np.isfinite(click.estimate)
    unknown = TieredBackground(tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.UNKNOWN)
    with pytest.raises((CausalUnsupportedError, CausalIdentifyError), match="no single ADMG"):
        antecedent.analyze(
            data,
            graph=unknown,
            query=query,
            estimator="cell.aipw",
            refute="none",
            bootstrap=0,
        )


def _codetermined_pair_family(n: int = 2000, seed: int = 17):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    latent = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.9 * z + 0.7 * latent)))).astype(
        float
    )
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.8 * z + 0.65 * latent)))).astype(
        float
    )
    y = 1.2 * t1 + 0.8 * t2 + 1.5 * t1 * t2 + 0.55 * z + 0.3 * rng.normal(size=n)
    data = {"z": z, "t1": t1, "t2": t2, "y": y}
    background = TieredBackground(
        tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    q11 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
    q10 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 0.0)])
    return data, background, q11, q10


def test_prepared_batch_prepare_cells_codetermined_pair_family():
    data, background, q11, q10 = _codetermined_pair_family()
    batch = antecedent.estimation.PreparedBatch.prepare_cells(
        data,
        graph=background,
        queries=[q11, q10],
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    design = batch.shared_design
    assert design is not None
    assert design.n_folds == 5
    assert design.shares_covariates
    assert not design.shares_propensity
    first, second = batch.estimate(data)
    assert first.estimate.joint_covariance is not None
    assert len(first.estimate.joint_covariance) == 2
    assert second.estimate.joint_covariance is not None
    assert len(second.estimate.joint_covariance) == 2
    assert first.estimate.adjusted_p_values is not None
    assert second.estimate.adjusted_p_values is not None
    assert first.estimate.simultaneous_interval is not None
    assert any(d.startswith("batch.joint_if:") for d in first.diagnostics)
    solo = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        query=q11,
        graph=background,
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    ).estimate(data)
    fresh = antecedent.analyze(
        data,
        graph=background,
        query=q11,
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    assert first.estimate.ate == pytest.approx(solo.estimate, abs=1e-12)
    assert first.estimate.ate == pytest.approx(fresh.estimate, abs=1e-12)
    unknown = TieredBackground(tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.UNKNOWN)
    with pytest.raises((CausalUnsupportedError, CausalIdentifyError), match="no single ADMG"):
        antecedent.estimation.PreparedBatch.prepare_cells(
            data,
            graph=unknown,
            queries=[q11],
            estimator="cell.aipw",
            refute="none",
            bootstrap=0,
        )


def _zero_effect_pair_family(n: int = 2400, seed: int = 221):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.7 * z)))).astype(float)
    y = 2.0 + 0.4 * z + 0.3 * rng.normal(size=n)
    data = {"z": z, "t1": t1, "t2": t2, "y": y}
    graph = antecedent.Dag.from_edges(
        ["z", "t1", "t2", "y"],
        [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
    )
    q11 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
    q10 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 0.0)])
    return data, graph, q11, q10


def test_prepared_batch_zero_effect_pair_family_is_not_significant():
    data, graph, q11, q10 = _zero_effect_pair_family()
    batch = antecedent.estimation.PreparedBatch.prepare_cells(
        data,
        graph=graph,
        queries=[q11, q10],
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    first, second = batch.estimate(data)
    solo = antecedent.analyze(
        data, graph=graph, query=q11, estimator="cell.aipw", refute="none", bootstrap=0
    )
    assert first.estimate.ate == pytest.approx(solo.estimate, abs=1e-12)
    for i, result in enumerate((first, second)):
        assert result.estimate.ate > 1.0
        assert abs(result.estimate.ate) / result.estimate.se_analytic > 8.0
        bh, by_q = result.estimate.adjusted_p_values
        assert bh > 0.05 and by_q > 0.05
        value, se = result.estimate.family_contrast
        assert abs(value) < 0.25
        _clo, chi, clevel = result.estimate.family_contrast_interval
        assert clevel == pytest.approx(0.95)
        contrast_crit = (chi - value) / se
        _llo, lhi, _ = result.estimate.simultaneous_interval
        level_se = result.estimate.joint_covariance[i][i] ** 0.5
        level_crit = (lhi - result.estimate.ate) / level_se
        if abs(contrast_crit - level_crit) > 1e-4:
            assert abs(value + level_crit * se - chi) > 1e-6
    omitted = antecedent.estimation.PreparedBatch.prepare_cells(
        data,
        graph=graph,
        queries=[q11, q10],
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
        family_contrast=None,
    ).estimate(data)
    assert all(r.estimate.adjusted_p_values is None for r in omitted)
    assert all(r.estimate.family_contrast is None for r in omitted)
    assert all(r.estimate.family_contrast_interval is None for r in omitted)
    assert all(r.estimate.simultaneous_interval is not None for r in omitted)


def test_prepared_batch_codetermined_distinct_pairs_share_folds_not_z():
    rng = np.random.default_rng(17)
    n = 2400
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.7 * z)))).astype(float)
    t3 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.15 + 0.75 * z)))).astype(float)
    y = 2.0 + 0.4 * z + 0.3 * rng.normal(size=n)
    data = {"z": z, "t1": t1, "t2": t2, "t3": t3, "y": y}
    background = TieredBackground(
        tiers=[["z"], ["t1", "t2", "t3"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    q12 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
    q13 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t3", 1.0)])
    q23 = InterventionResponse("y", intervention=[Set("t2", 1.0), Set("t3", 1.0)])
    batch = antecedent.estimation.PreparedBatch.prepare_cells(
        data,
        graph=background,
        queries=[q12, q13, q23],
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    design = batch.shared_design
    assert design is not None
    assert design.n_folds == 5
    assert not design.shares_covariates
    results = batch.estimate(data)
    assert len(results) == 3
    assert len(results[0].estimate.joint_covariance) == 3
    solo = antecedent.analyze(
        data, graph=background, query=q12, estimator="cell.aipw", refute="none", bootstrap=0
    )
    assert results[0].estimate.ate == pytest.approx(solo.estimate, abs=1e-12)
    for result in results:
        assert result.estimate.ate > 1.0
        bh, by_q = result.estimate.adjusted_p_values
        assert bh > 0.05 and by_q > 0.05
        value, _se = result.estimate.family_contrast
        assert abs(value) < 0.25


def test_prepared_batch_prepare_cells_dag_multi_pair():
    rng = np.random.default_rng(618)
    n = 3200
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.9 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.8 * z)))).astype(float)
    y = 1.2 * t1 + 0.8 * t2 + 1.5 * t1 * t2 + 0.55 * z + 0.8 * np.exp(rng.normal(size=n)) - 0.8
    data = {"z": z, "t1": t1, "t2": t2, "y": y}
    graph = antecedent.Dag.from_edges(
        ["z", "t1", "t2", "y"],
        [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
    )
    q11 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
    q10 = InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 0.0)])
    batch = antecedent.estimation.PreparedBatch.prepare_cells(
        data,
        graph=graph,
        queries=[q11, q10],
        estimator="cell.aipw",
        refute="none",
        bootstrap=0,
    )
    first, second = batch.estimate(data)
    assert len(first.estimate.joint_covariance) == 2
    assert first.estimate.adjusted_p_values is not None
    solo = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=graph, query=q11, estimator="cell.aipw", refute="none", bootstrap=0
    ).estimate(data)
    assert first.estimate.ate == pytest.approx(solo.estimate, abs=1e-12)
    assert np.isfinite(second.estimate.ate)


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


def test_conditional_grid_outcome_translation():
    data = _binary_confounded(700)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    cdfs = []
    for offset in [0.0, 10.0]:
        shifted = {**data, "y": data["y"] + offset}
        result = antecedent.analyze(
            shifted,
            graph=graph,
            query=antecedent.ConditionalEffect(
                "t",
                "y",
                "z",
                outcome_functional=ExceedanceGrid([offset, offset + 0.5, offset + 1.0]),
            ),
            refute="none",
            bootstrap=0,
        )
        cdfs.append(result.estimate.exceedance_cdf)
    np.testing.assert_allclose(cdfs[0], cdfs[1], atol=1e-12)


def test_conditional_quantile_fresh_prepared_and_class():
    data = _binary_confounded(1600, seed=617)
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    dag = antecedent.Dag.from_edges(["t", "y", "z"], edges)
    cpdag = antecedent.Cpdag.from_directed_undirected(["t", "y", "z"], edges, [])
    query = antecedent.ConditionalEffect("t", "y", "z", outcome_functional=Quantile(0.5))
    results = []
    for graph in (dag, cpdag):
        fresh = antecedent.analyze(data, graph=graph, query=query, refute="none", bootstrap=0)
        plan = antecedent.estimation.PreparedAnalysis.prepare(
            data, graph=graph, query=query, refute="none", bootstrap=0
        )
        click = plan.estimate(data)
        assert fresh.estimate.ate == pytest.approx(click.estimate.ate, abs=1e-10)
        assert fresh.estimate.se_analytic == pytest.approx(click.estimate.se_analytic, abs=1e-10)
        assert fresh.estimate.ate == pytest.approx(2.0, abs=0.2)
        assert any(
            diagnostic.startswith("estimate.functional.quantile_grid:")
            and "control then active" in diagnostic
            for diagnostic in fresh.diagnostics
        )
        results.append(fresh)
    assert results[0].estimate.ate == pytest.approx(results[1].estimate.ate, abs=1e-10)


@pytest.mark.parametrize("tiered", [False, True])
@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("first_level", [0.0, 1.0])
def test_dag_cell_aipw_public_route_and_quantile(quantile, first_level, tiered):
    rng = np.random.default_rng(618)
    n = 3200
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.9 * z)))).astype(float)
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.1 + 0.8 * z)))).astype(float)
    # Skewed noise distinguishes the median from the mean. z confounds T and Y.
    y = 1.2 * t1 + 0.8 * t2 + 1.5 * t1 * t2 + 0.55 * z + 0.8 * np.exp(rng.normal(size=n)) - 0.8
    data = {"z": z, "t1": t1, "t2": t2, "y": y}
    graph = antecedent.Dag.from_edges(
        ["z", "t1", "t2", "y"],
        [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
    )
    if tiered:
        graph = TieredBackground(
            tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.CODETERMINED
        )
    query = InterventionResponse(
        "y",
        intervention=[Set("t1", first_level), Set("t2", 1.0)],
        outcome_functional=Quantile(0.5) if quantile else Mean(),
    )
    fresh = antecedent.analyze(
        data, graph=graph, query=query, estimator="cell.aipw", refute="none", bootstrap=0
    )
    plan = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=graph, query=query, estimator="cell.aipw", refute="none", bootstrap=0
    )
    click = plan.estimate(data)
    retargeted = plan.retarget(np.ones(n), depends_on=[])
    assert fresh.estimate == pytest.approx(click.estimate, abs=1e-10)
    assert retargeted.estimate.ate == pytest.approx(fresh.estimate, abs=1e-10)
    assert fresh.uncertainty.standard_error > 0.0
    truth = 0.8 + 2.7 * first_level
    if not quantile:
        truth += 0.8 * (np.exp(0.5) - 1.0)
    assert fresh.estimate == pytest.approx(truth, abs=0.2)

    if quantile:
        weights = np.exp(-0.1 * z**2)
        weighted = plan.retarget(weights, depends_on=["z"])
        scaled = plan.retarget(7.0 * weights, depends_on=["z"])
        assert weighted.estimate.ate == pytest.approx(scaled.estimate.ate, abs=1e-10)
        assert weighted.estimate.se_analytic == pytest.approx(
            scaled.estimate.se_analytic, abs=1e-10
        )
        shifted = {**data, "y": y + 10.0}
        moved = plan.estimate(shifted)
        assert moved.estimate == pytest.approx(fresh.estimate + 10.0, abs=1e-9)
        plan.refresh(shifted)
        retained = plan.retarget(np.ones(n), depends_on=[])
        assert retained.estimate.ate == pytest.approx(moved.estimate, abs=1e-9)


@pytest.mark.parametrize("refute", ["cheap", "full"])
def test_conditional_quantile_refuses_mean_effect_refuters(refute):
    data = _binary_confounded(400)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    with pytest.raises(CausalUnsupportedError, match="mean-effect refuters"):
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.ConditionalEffect("t", "y", "z", outcome_functional=Quantile(0.5)),
            refute=refute,
            bootstrap=0,
        )


@pytest.mark.parametrize("prepared", [False, True])
def test_response_quantile_refuses_mean_only_estimator(prepared):
    data = _binary_confounded(400)
    graph = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    query = InterventionResponse("y", intervention=Set("t", 1.0), outcome_functional=Quantile(0.5))
    run = antecedent.estimation.PreparedAnalysis.prepare if prepared else antecedent.analyze
    with pytest.raises(CausalUnsupportedError, match="quantiles require"):
        run(data, graph=graph, query=query, refute="none", bootstrap=0)
