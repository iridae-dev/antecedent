"""1.5 Python surface: functionals, retarget, tiered background, EconML extras."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")

import antecedent
from antecedent.graph import TieredBackground, WithinTier
from antecedent.handoff import econml
from antecedent.query import Exceedance, ExceedanceGrid, Mean, coerce_outcome_functional


def _binary_confounded(n: int = 400, seed: int = 15):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def test_outcome_functional_wire():
    assert coerce_outcome_functional(None) is None
    assert coerce_outcome_functional(Mean()) is None
    assert coerce_outcome_functional(Exceedance(1.2)) == {"kind": "exceedance", "threshold": 1.2}
    assert coerce_outcome_functional(ExceedanceGrid([0.0, 0.5])) == {
        "kind": "exceedance_grid",
        "thresholds": [0.0, 0.5],
    }


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
    assert all(np.isfinite(cdf))
    assert np.isnan(out_grid.estimate.ate)


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
    query = antecedent.InterventionResponse(
        "y", intervention=antecedent.intervention.Set("t", 1.0)
    )
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
