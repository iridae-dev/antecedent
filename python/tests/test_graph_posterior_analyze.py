"""P1-D: graph-posterior × Bayesian effect mixture via analyze."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def test_exact_dag_posterior_bayesian_ate_mixture():
    n = 200
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z

    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=80, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    assert np.isfinite(result.posterior.effect_mean)
    assert result.ate is None or np.isfinite(result.ate)
    assert result.evidence_status == "licensed"
    if mass > 0.0:
        assert result.posterior.envelope is not None
        assert result.posterior.envelope.unidentified_mass == mass


def test_exact_dag_posterior_frequentist_ate_mixture():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        discovery=antecedent.discovery.ExactDagPosterior(),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.posterior is None
    assert result.ate is None or np.isfinite(result.ate)
    assert result.evidence_status == "licensed"
    assert any("unidentified_mass=" in diagnostic for diagnostic in result.diagnostics)


def test_dbn_posterior_bayesian_pulse_mixture():
    n = 400
    rng = np.random.default_rng(42)
    # White-noise treatment keeps BIC mass on the lag edge (AR loops often
    # fail temporal backdoor history caps under the DBN mixture).
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = antecedent.analyze(
        {"pressure": pressure, "defect": defect},
        discovery=antecedent.discovery.DbnPosterior(max_lag=1),
        query=antecedent.PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    assert np.isfinite(result.posterior.effect_mean)
    assert abs(result.posterior.effect_mean - 0.9) < 0.35
    assert result.evidence_status == "licensed"


def test_dbn_posterior_bayesian_sustained_mixture():
    n = 400
    rng = np.random.default_rng(42)
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = antecedent.analyze(
        {"pressure": pressure, "defect": defect},
        discovery=antecedent.discovery.DbnPosterior(max_lag=1),
        query=antecedent.SustainedEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    assert np.isfinite(result.posterior.effect_mean)
    assert abs(result.posterior.effect_mean - 0.9) < 0.35
    assert result.evidence_status == "licensed"


def test_cpdag_graph_posterior_ate_from_graphs():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")], []
    )
    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        discovery=antecedent.discovery.GraphPosterior.from_graphs(["t", "y", "z"], [1.0], [cpdag]),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.evidence_status == "licensed"
    assert result.ate is None or np.isfinite(result.ate)
    assert result.study is not None


def test_admg_graph_posterior_ate_and_intervention_response():
    n = 80
    t = (np.arange(n) % 2).astype(np.float64)
    y = 1.0 + 2.0 * t
    admg = antecedent.Admg.from_edges(["t", "y"], directed=[("t", "y")], bidirected=[])
    discovery = antecedent.discovery.GraphPosterior.from_graphs(["t", "y"], [1.0], [admg])
    ate = antecedent.analyze(
        {"t": t, "y": y},
        discovery=discovery,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert ate.evidence_status == "licensed"
    response = antecedent.analyze(
        {"t": t, "y": y},
        discovery=discovery,
        query=antecedent.InterventionResponse(
            "y", intervention=antecedent.intervention.Set("t", 1.0)
        ),
        inference=antecedent.Frequentist(),
        refute="cheap",
        bootstrap=0,
        seed=1,
    )
    assert response.evidence_status == "licensed"
    assert response.response is not None


def test_dbn_posterior_temporal_response_curve_and_intervention():
    n = 400
    rng = np.random.default_rng(42)
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]
    data = {"pressure": pressure, "defect": defect}
    curve = antecedent.analyze(
        data,
        discovery=antecedent.discovery.DbnPosterior(max_lag=1),
        query=antecedent.ResponseCurve("pressure", "defect", grid=[0.0, 1.0], horizons=[1]),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert curve.evidence_status == "licensed"
    assert curve.response is not None
    assert curve.study is not None
    ir = antecedent.analyze(
        data,
        discovery=antecedent.discovery.DbnPosterior(max_lag=1),
        query=antecedent.InterventionResponse(
            "defect",
            intervention=antecedent.intervention.Set("pressure", 1.0),
            horizons=[1],
        ),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert ir.evidence_status == "licensed"
    assert ir.response is not None


def test_temporal_cpdag_graph_posterior_pulse():
    n = 200
    rng = np.random.default_rng(7)
    treatment = rng.normal(size=n)
    outcome = np.zeros(n)
    for i in range(1, n):
        outcome[i] = 0.8 * treatment[i - 1]
    lag_mask = 1 << (0 * 2 + 1)
    discovery = antecedent.discovery.GraphPosterior.from_atoms(
        ["t", "y"],
        [1.0],
        [0],
        atom_kind="Cpdag",
        lag_masks=[lag_mask],
        max_lag=1,
        lagged_edge_marginals=[0.0, 1.0, 0.0, 0.0],
    )
    result = antecedent.analyze(
        {"t": treatment, "y": outcome},
        discovery=discovery,
        query=antecedent.PulseEffect(
            treatment="t",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=3,
    )
    assert result.evidence_status == "licensed"
