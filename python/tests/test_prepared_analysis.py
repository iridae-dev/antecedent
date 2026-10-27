"""Prepared analysis re-estimate (Python dual of Rust backlog B)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _scm import confounded_scm


def test_prepared_reestimate_matches_fresh_analyze():
    data, edges = confounded_scm(n=500, seed=19)
    fresh = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(profile="interactive"),
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    first_outcome = antecedent.estimate(prepared)
    first = require_success(first_outcome)
    second = require_success(antecedent.estimate(antecedent.rebind(prepared, data=data)))
    assert math.isfinite(first.ate)
    assert abs(first.ate - 2.0) < 0.5
    assert abs(first.ate - fresh.ate) < 1e-12
    assert abs(second.ate - fresh.ate) < 1e-12
    assert first.identification.status == fresh.identification.status
    assert first.identification.adjustment_set == fresh.identification.adjustment_set
    assert first.performance.plan_id == fresh.performance.plan_id
    # The outcome retains its completed study, which rebinds without refitting a new preparation.
    via_result = require_success(
        antecedent.estimate(antecedent.rebind(first_outcome.study, data=data))
    )
    assert abs(via_result.ate - fresh.ate) < 1e-12


def test_oneshot_analyze_result_retains_study():
    data, edges = confounded_scm(n=200, seed=5)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(profile="interactive"),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert isinstance(result.study, antecedent.PreparedAnalysis)
    rebound = antecedent.estimate(antecedent.rebind(result.study, data=data))
    assert require_success(rebound).effect == pytest.approx(result.effect)


def test_prepared_prior_transfer_validates_artifact_at_estimation():
    data, edges = confounded_scm(n=80, seed=7)
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(prior_from=b"not-used"),
        refute=False,
    )
    with pytest.raises(antecedent.errors.CausalSerializationError, match="bad artifact magic"):
        require_success(antecedent.estimate(prepared))


def test_prepared_second_shot_reuses_identification():
    data, edges = confounded_scm(n=800, seed=31)
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    first = require_success(antecedent.estimate(prepared))
    second = require_success(antecedent.estimate(prepared))
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
    fresh = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=query,
        inference=antecedent.Frequentist(profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    clicked = antecedent.estimate(prepared)
    click = require_success(clicked)
    assert click.response is not None
    assert fresh.response is not None
    assert click.response.values == fresh.response.values
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="not_applicable:"):
        require_success(clicked.study.refute(suite="cheap"))


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
        require_success(
            antecedent.estimate(
                antecedent.prepare(
                    data,
                    graph=edges,
                    query=query,
                    refute="cheap",
                    execution=antecedent.Execution(seed=1),
                )
            )
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
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=query,
        refute="cheap",
        inference=antecedent.Frequentist(profile=None),
        execution=antecedent.Execution(seed=1),
    )
    result = require_success(antecedent.estimate(prepared))
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

    def click(*, latency=None, bootstrap=None):
        prepared = antecedent.prepare(
            data,
            graph=edges,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=bootstrap, profile=latency),
            execution=antecedent.Execution(seed=3),
        )
        return require_success(antecedent.estimate(prepared))

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
    prepared = antecedent.prepare(
        data,
        graph=[("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)],
        query=query,
        refute=False,
        execution=antecedent.Execution(seed=11),
    )
    outcome = antecedent.estimate(prepared)
    result = require_success(outcome)
    assert result.mediation is None
    assert result.estimate.ate is None
    assert result.ate is None
    assert result.mediation_grid is not None
    assert [slice_.horizon for slice_ in result.mediation_grid] == [1, 2]
    assert result.mediation_grid.joint_posterior is False

    artifact = antecedent.artifacts.loads(bytes(outcome.export(quantity="record")))
    assert artifact.payload_kind == "analysis_result"
    assert [
        slice_["horizon"] for slice_ in artifact.to_dict()["payload"]["mediation_grid"]["slices"]
    ] == [1, 2]
