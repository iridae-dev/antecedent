"""P1-D: graph-posterior × Bayesian effect mixture via analyze."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def test_exact_dag_posterior_bayesian_ate_mixture():
    n = 200
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z

    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.ExactDagPosterior().run(
                {"t": t, "y": y, "z": z}, execution=antecedent.Execution(seed=7)
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=80, prior_scale=100.0, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=7),
        )
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 < mass < 1.0  # this posterior splits between identified and unidentified graphs
    assert result.ate is None  # a mixture with unidentified mass publishes no single point
    assert result.evidence_status == "licensed"
    assert result.posterior.envelope is not None
    assert result.posterior.envelope.unidentified_mass == mass
    # The identified-graph mixture is a convex combination of the per-graph effects. Here the
    # per-graph total effects are the adjusted 2 (z a common cause: y = 1 + 2 t + 3 z) and the
    # unadjusted 3.5 = 2 + 3 (E[z | t=1] - E[z | t=0]) (z downstream of or independent of t), so
    # the mixture mean lies in [2, 3.5] up to the conjugate prior's shrinkage.
    assert 2.0 - 0.05 <= result.posterior.effect_mean <= 3.5 + 0.05, result.posterior.effect_mean


def test_exact_dag_posterior_frequentist_ate_mixture():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.ExactDagPosterior().run(
                {"t": t, "y": y, "z": z}, execution=antecedent.Execution(seed=1)
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.posterior is None
    assert result.ate is None  # unidentified mass: no single point is published
    assert result.evidence_status == "licensed"
    masses = [d for d in result.diagnostics if "unidentified_mass=" in d]
    assert masses, result.diagnostics
    # The structural aggregation states a partition of the posterior into identified and
    # unidentified mass; the two sum to one.
    line = next(d for d in masses if "identified_mass=" in d and "structural_aggregation" in d)
    identified = float(line.split("; identified_mass=")[1].split(";")[0])
    unidentified = float(line.split("unidentified_mass=")[1].split(";")[0])
    assert 0.0 < unidentified < 1.0
    assert identified + unidentified == pytest.approx(1.0, abs=1e-9), line


def test_dbn_posterior_bayesian_pulse_mixture():
    n = 400
    rng = np.random.default_rng(42)
    # White-noise treatment keeps BIC mass on the lag edge (AR loops often
    # fail temporal backdoor history caps under the DBN mixture).
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = require_success(
        antecedent.analyze(
            {"pressure": pressure, "defect": defect},
            discovery=antecedent.discovery.DbnPosterior(max_lag=1).run(
                {"pressure": pressure, "defect": defect}, execution=antecedent.Execution(seed=11)
            ),
            query=antecedent.PulseEffect(
                treatment="pressure",
                outcome="defect",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    # y[t] = 0.9 pressure[t-1] exactly: a unit pulse at lag 1 moves the outcome by 0.9.
    assert result.posterior.effect_mean == pytest.approx(0.9, abs=1e-3)
    assert result.evidence_status == "licensed"


def test_dbn_posterior_bayesian_sustained_mixture():
    n = 400
    rng = np.random.default_rng(42)
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]

    result = require_success(
        antecedent.analyze(
            {"pressure": pressure, "defect": defect},
            discovery=antecedent.discovery.DbnPosterior(max_lag=1).run(
                {"pressure": pressure, "defect": defect}, execution=antecedent.Execution(seed=11)
            ),
            query=antecedent.SustainedEffect(
                treatment="pressure",
                outcome="defect",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Bayesian(n_draws=64, prior_scale=100.0, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    assert result.posterior is not None
    mass = result.posterior.unidentified_mass
    assert mass is not None
    assert 0.0 <= mass <= 1.0
    # y[t] = 0.9 pressure[t-1] exactly: a unit pulse at lag 1 moves the outcome by 0.9.
    assert result.posterior.effect_mean == pytest.approx(0.9, abs=1e-3)
    assert result.evidence_status == "licensed"


def test_cpdag_graph_posterior_ate_from_graphs():
    n = 80
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")], []
    )
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.discovery.GraphPosterior.from_graphs(
                ["t", "y", "z"], [1.0], [cpdag]
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.evidence_status == "licensed"
    # z -> t, z -> y, t -> y with y = 1 + 2 t + 3 z: the adjusted total effect of t is exactly 2
    # (the unadjusted contrast is 3.5).
    assert result.ate == pytest.approx(2.0, abs=1e-9)
    assert result.study is not None


def test_admg_graph_posterior_ate_and_intervention_response():
    n = 80
    t = (np.arange(n) % 2).astype(np.float64)
    y = 1.0 + 2.0 * t
    admg = antecedent.Admg.from_edges(["t", "y"], directed=[("t", "y")], bidirected=[])
    discovery = antecedent.discovery.GraphPosterior.from_graphs(["t", "y"], [1.0], [admg])
    ate = require_success(
        antecedent.analyze(
            {"t": t, "y": y},
            discovery=discovery,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert ate.evidence_status == "licensed"
    assert ate.ate == pytest.approx(2.0, abs=1e-9)  # y = 1 + 2 t exactly
    response = require_success(
        antecedent.analyze(
            {"t": t, "y": y},
            discovery=discovery,
            query=antecedent.InterventionResponse(
                "y", intervention=antecedent.intervention.Set("t", 1.0)
            ),
            inference=antecedent.Frequentist(bootstrap=0),
            refute="cheap",
            execution=antecedent.Execution(seed=1),
        )
    )
    assert response.evidence_status == "licensed"
    assert response.response is not None
    assert response.response.values[0][0] == pytest.approx(3.0, abs=1e-9)  # E[y | do(t=1)] = 1 + 2


def test_dbn_posterior_temporal_response_curve_and_intervention():
    n = 400
    rng = np.random.default_rng(42)
    pressure = rng.normal(size=n).astype(np.float64)
    defect = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1]
    data = {"pressure": pressure, "defect": defect}
    curve = require_success(
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.DbnPosterior(max_lag=1).run(
                data, execution=antecedent.Execution(seed=11)
            ),
            query=antecedent.ResponseCurve("pressure", "defect", grid=[0.0, 1.0], horizons=[1]),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    assert curve.evidence_status == "licensed"
    assert curve.response is not None
    # defect[t] = 0.9 pressure[t-1]: the curve at horizon 1 is 0 at pressure 0 and 0.9 at pressure 1.
    assert curve.response.points == ((0.0, 1.0), (1.0, 1.0))
    assert [v[0] for v in curve.response.values] == pytest.approx([0.0, 0.9], abs=1e-9)
    assert curve.study is not None
    ir = require_success(
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.DbnPosterior(max_lag=1).run(
                data, execution=antecedent.Execution(seed=11)
            ),
            query=antecedent.InterventionResponse(
                "defect",
                intervention=antecedent.intervention.Set("pressure", 1.0),
                horizons=[1],
            ),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=11),
        )
    )
    assert ir.evidence_status == "licensed"
    assert ir.response is not None
    assert ir.response.values[0][0] == pytest.approx(0.9, abs=1e-9)  # E[defect | do(pressure=1)]


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
    result = require_success(
        antecedent.analyze(
            {"t": treatment, "y": outcome},
            discovery=discovery,
            query=antecedent.PulseEffect(
                treatment="t",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=3),
        )
    )
    assert result.evidence_status == "licensed"
    assert result.ate == pytest.approx(0.8, abs=1e-9)  # outcome[i] = 0.8 treatment[i-1] exactly
