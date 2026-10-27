"""First enforced-refusal wave: closed cells raise; licensed and default-refused still run."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _two_node_table(n: int = 48, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.2, 1.8, n)
    y = 2.0 * t + rng.normal(scale=0.1, size=n)
    return {"t": t, "y": y}


_DATA = _two_node_table()
_EDGES = [("t", "y")]
_DAG = antecedent.Dag.from_edges(["t", "y"], _EDGES)
_PAG_DATA = {**_DATA, "r": np.arange(len(_DATA["t"]), dtype=float) % 2}
_PAG = antecedent.Pag.from_marked_edges(
    ["t", "y", "r"], [("r", "t", "tail", "arrow"), ("t", "y", "tail", "arrow")]
)
_ACCEPTED = antecedent.AcceptedGraph.from_graph(_DAG, algorithm_id="hand")
_CURVE = antecedent.ResponseCurve("t", "y", grid=[0.5, 1.0, 1.5])


def test_path_specific_graph_posterior_runs_and_is_never_licensed():
    # A path-restricted effect reads a finite-discrete table, so the posterior runs on binary
    # columns; every atom is answered by the executor its own structure has.
    rng = np.random.default_rng(1)
    treatment = (rng.random(400) < 0.5).astype(float)
    outcome = (rng.random(400) < 0.3 + 0.4 * treatment).astype(float)
    result = require_success(
        antecedent.analyze(
            {"t": treatment, "y": outcome},
            query=antecedent.PathSpecificEffect("t", "y"),
            discovery=antecedent.discovery.ExactDagPosterior().run(
                {"t": treatment, "y": outcome}, seed=1
            ),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert np.isfinite(result.estimate.ate)
    assert result.evidence_status != "licensed"


def test_licensed_pag_response_curve_runs():
    result = require_success(
        antecedent.analyze(
            _PAG_DATA,
            graph=_PAG,
            query=_CURVE,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.response is not None
    assert np.isfinite(result.response.values).all()
    assert result.evidence_status == "licensed"


@pytest.mark.parametrize(
    "graph",
    [_EDGES, _DAG],
    ids=["edges", "dag"],
)
def test_licensed_response_curve_still_runs(graph):
    result = require_success(
        antecedent.analyze(
            _DATA,
            graph=graph,
            query=_CURVE,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.response is not None
    assert np.isfinite(result.response.values).all()


def test_licensed_average_effect_on_dag_frequentist():
    result = require_success(
        antecedent.analyze(
            _DATA,
            graph=_DAG,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


def test_licensed_pag_ate_runs():
    result = require_success(
        antecedent.analyze(
            _PAG_DATA,
            graph=_PAG,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


def test_licensed_admg_ate_runs():
    n = 300
    u = np.array([1.0 if (i % 5) < 2 else 0.0 for i in range(n)])
    t = np.array([1.0 if (i % 3) == 0 else 0.0 for i in range(n)])
    m = np.array([float(int(ti + ui) % 2) for ti, ui in zip(t, u, strict=True)])
    y = np.array([float(int(mi + ui) % 2) for mi, ui in zip(m, u, strict=True)])
    data = {"t": t, "m": m, "y": y}
    admg = antecedent.Admg.from_edges(
        ["t", "m", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
    )
    result = require_success(
        antecedent.analyze(
            data,
            graph=admg,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


# Allowlist is empty: graph-posterior Bayesian ATE and Pulse DBN posterior none are licensed.


def test_licensed_pulse_effect_temporal_dag_runs():
    """PulseEffect x TemporalDag x explicit is licensed (the allowlist is
    empty): the temporal backdoor path is fully wired for both inference
    modes and every validation suite, pinned on a known-truth fixture the
    same way the static AverageEffect family is."""
    n = 200
    t = np.array([float(i % 2) for i in range(n)])
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * t[i - 1] + 0.1 * y[i - 1] + 0.01 * (i % 5)
    data = {"t": t, "y": y}
    graph = [("t", 1, "y", 0)]
    result = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.PulseEffect(
                treatment="t", outcome="y", treatment_lag=1, horizon_steps=1
            ),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


def test_admg_bayesian_average_effect_uses_functional_estimator():
    """AverageEffect × Admg × Bayesian is licensed via functional.effect."""
    n = 300
    u = np.array([1.0 if (i % 5) < 2 else 0.0 for i in range(n)])
    t = np.array([1.0 if (i % 3) == 0 else 0.0 for i in range(n)])
    m = np.array([float(int(ti + ui) % 2) for ti, ui in zip(t, u, strict=True)])
    y = np.array([float(int(mi + ui) % 2) for mi, ui in zip(m, u, strict=True)])
    data = {"t": t, "m": m, "y": y}
    admg = antecedent.Admg.from_edges(
        ["t", "m", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
    )
    result = require_success(
        antecedent.analyze(
            data,
            graph=admg,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=64),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.plan.estimator == "functional.effect"
    assert result.posterior is not None


def test_path_specific_cheap_refute_runs_native_suite():
    data = {
        "t": np.array([0.0] * 100 + [1.0] * 100),
        "m": np.array([0.0] * 50 + [1.0] * 50 + [0.0] * 20 + [1.0] * 80),
        "y": np.array(
            [0.0] * 40
            + [1.0] * 10
            + [0.0] * 10
            + [1.0] * 40
            + [0.0] * 10
            + [1.0] * 10
            + [0.0] * 10
            + [1.0] * 70
        ),
    }
    dag = antecedent.Dag.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")])
    result = require_success(
        antecedent.analyze(
            data,
            graph=dag,
            query=antecedent.PathSpecificEffect("t", "y", path_nodes=["m"]),
            refute="cheap",
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )

    assert result.ate == pytest.approx(0.3)
    assert len(result.validation.reports) == 1


def test_distribution_full_refute_runs_native_suite():
    t = np.array([0.0, 1.0] * 40)
    y = t.copy()
    z = np.zeros(80)
    data = {"t": t, "y": y, "z": z}
    dag = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    result = require_success(
        antecedent.analyze(
            data,
            graph=dag,
            query=antecedent.InterventionalDistribution("y", interventions={"t": 1.0}),
            refute="full",
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )

    assert result.ate == pytest.approx(1.0)
    assert len(result.validation.reports) == 3
