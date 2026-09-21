"""Prepared analysis re-estimate (Python dual of Rust backlog B)."""

from __future__ import annotations

import math
import random

import antecedent
import numpy as np
import pytest


def _confounded_scm(n: int = 500, seed: int = 19):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_prepared_reestimate_matches_fresh_analyze():
    data, edges = _confounded_scm()
    fresh = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    first = prepared.estimate(data, seed=1)
    second = prepared.refresh(data, seed=1)
    assert math.isfinite(first.ate)
    assert abs(first.ate - 2.0) < 0.5
    assert abs(first.ate - fresh.ate) < 1e-12
    assert abs(second.ate - fresh.ate) < 1e-12
    assert first.identification.status == fresh.identification.status
    assert first.identification.adjustment_set == fresh.identification.adjustment_set
    assert first.performance.plan_id == fresh.performance.plan_id
    # Result.refresh uses the retained prepared handle.
    via_result = first.refresh(data, seed=1)
    assert abs(via_result.ate - fresh.ate) < 1e-12


def test_oneshot_analyze_result_retains_study():
    data, edges = _confounded_scm(n=200, seed=5)
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    assert isinstance(result.study, antecedent.estimation.PreparedAnalysis)
    assert result.refresh(data).effect == pytest.approx(result.effect)


def test_prepared_prior_transfer_validates_artifact_at_estimation():
    data, edges = _confounded_scm(n=80, seed=7)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(prior_from=b"not-used"),
        refute=False,
    )
    with pytest.raises(antecedent.errors.CausalSerializationError, match="bad artifact magic"):
        prepared.estimate(data)


def test_prepared_second_shot_reuses_identification():
    data, edges = _confounded_scm(n=800, seed=31)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    first = prepared.estimate(data, seed=1)
    second = prepared.estimate(data, seed=1)
    assert any(d.startswith("exec.identify.cached") for d in second.diagnostics)
    assert first.effect == second.effect


def test_prepared_response_curve_matches_analyze():
    rng = np.random.default_rng(19)
    z = rng.normal(size=400)
    t = 0.7 * z + rng.normal(size=400)
    y = 2.0 * t + z + rng.normal(scale=0.2, size=400)
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    query = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])
    fresh = antecedent.analyze(data, graph=edges, query=query, refute=False, bootstrap=0, seed=1)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=edges, query=query, latency="interactive", seed=1
    )
    click = prepared.estimate(data, seed=1)
    assert click.response is not None
    assert fresh.response is not None
    assert click.response.values == fresh.response.values
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="not_applicable:"):
        prepared.refute(data, suite="cheap")


def test_prepared_response_curve_cheap_refute_at_prepare_is_not_applicable():
    """Staged ResponseCurve must agree with ``analyze()``: cheap/full denote
    the ATE-shaped scalar refuter suite, which a function-valued estimand has
    no state for, so this is a typed impossibility (``not_applicable:``), not
    a bespoke ``CausalTypeError``."""
    rng = np.random.default_rng(31)
    z = rng.normal(size=400)
    t = 0.7 * z + rng.normal(size=400)
    y = 2.0 * t + z + rng.normal(scale=0.2, size=400)
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    query = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="not_applicable:"):
        antecedent.estimation.PreparedAnalysis.prepare(
            data, graph=edges, query=query, refute="cheap", seed=1
        )


def test_prepared_temporal_mediation_cheap_refute_runs():
    """The staged path executes a native mediator/RCC suite."""
    n = 80
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.1 * math.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * math.cos(i)
        y[i] = 0.5 * m[i] + 0.02 * math.sin(i)
    data = {"t": t, "m": m, "y": y}
    edges = [("t", 1, "m", 0), ("m", 0, "y", 0)]
    query = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated")
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data, graph=edges, query=query, refute="cheap", seed=1, latency=None
    )
    result = prepared.estimate(data)
    assert len(result.validation.reports) == 2


def test_prepared_temporal_mediation_bootstrap_follows_latency_tier():
    """An omitted bootstrap follows the latency tier like Pulse / Sustained."""
    n = 160
    t = np.asarray([math.sin(0.071 * i) + 0.35 * math.cos(0.137 * i) for i in range(n)])
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        m[i] = 0.8 * t[i - 1] + 0.12 * math.sin(0.43 * i)
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * math.cos(0.29 * i)
    data = {"t": t, "m": m, "y": y}
    edges = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
    query = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated")

    def click(**kwargs):
        prepared = antecedent.estimation.PreparedAnalysis.prepare(
            data, graph=edges, query=query, refute=False, seed=3, **kwargs
        )
        return prepared.estimate(data, seed=3)

    interactive = click(latency="interactive")
    assert interactive.estimate.se_bootstrap is None
    # An omitted tier keeps the omitted replicate budget, as everywhere else.
    assert click().estimate.se_bootstrap is not None
    for latency in ("standard", "report"):
        tiered = click(latency=latency)
        assert tiered.ate == pytest.approx(interactive.ate, abs=1e-12)
        assert tiered.estimate.se_bootstrap is not None
        assert math.isfinite(tiered.estimate.se_bootstrap) and tiered.estimate.se_bootstrap > 0
    # An explicit bootstrap still wins over the tier.
    assert click(latency="standard", bootstrap=0).estimate.se_bootstrap is None


def test_prepared_multi_horizon_mediation_retains_every_slice_and_artifact_axis():
    n = 96
    t = np.asarray([(-1.0, 0.0, 1.0)[i % 3] for i in range(n)])
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(2, n):
        m[i] = 0.8 * t[i - 1] + 0.05 * math.cos(i)
        y[i] = 0.2 * t[i - 1] + 0.5 * m[i] + 0.02 * math.sin(i)
    data = {"t": t, "m": m, "y": y}
    query = antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated", horizons=[1, 2])
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=[("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)],
        query=query,
        refute=False,
        seed=11,
    )
    result = prepared.estimate(data, seed=11)
    assert result.mediation is None
    assert result.estimate.ate is None
    assert result.ate is None
    assert result.mediation_grid is not None
    assert [slice_.horizon for slice_ in result.mediation_grid] == [1, 2]
    assert result.mediation_grid.joint_posterior is False

    artifact = antecedent.artifacts.loads(prepared.export_artifact())
    assert artifact.payload_kind == "analysis_result"
    assert [slice_["horizon"] for slice_ in artifact.payload["mediation_grid"]["slices"]] == [1, 2]
