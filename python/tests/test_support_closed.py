"""First enforced-refusal wave: closed cells raise; licensed and default-refused still run."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError

_REASON_PATH_DIST = "refused: Graph-posterior path and distribution mixtures are not staged."
_REASON_ADMG_RESPONSE = (
    "refused: Admg response has no functional plug-in; licensed general-ID ATE does "
    "not estimate a curve."
)


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
_ADMG = antecedent.Admg.from_edges(["t", "y"], _EDGES)


_REFUSED = [
    (
        "path_specific_exact_dag_posterior",
        antecedent.PathSpecificEffect("t", "y"),
        {"discovery": antecedent.discovery.ExactDagPosterior()},
        _REASON_PATH_DIST,
    ),
    # InterventionResponse × Admg is the Python routing-gate pin for the closed
    # Admg-response rule. Cpdag/Pag response is now licensed (see
    # `test_licensed_pag_response_curve_runs`). ConditionalEffect × Cpdag/Pag
    # is licensed; ConditionalEffect × Admg stays closed.
    # PathSpecificEffect/InterventionalDistribution × Admg/Pag, and
    # TemporalMediationEffect × TemporalCpdag/TemporalPag remain closed on the Rust
    # matrix. AverageEffect × graph_posterior × Frequentist is licensed; other
    # Frequentist graph-posterior queries stay closed.
    (
        "intervention_response_admg",
        antecedent.InterventionResponse("y", intervention=antecedent.intervention.Set("t", 1.0)),
        {"graph": _ADMG},
        _REASON_ADMG_RESPONSE,
    ),
    (
        "conditional_effect_admg",
        antecedent.ConditionalEffect("t", "y", "t"),
        {"graph": _ADMG},
        "refused: ConditionalEffect on Admg has no compile arm",
    ),
]


@pytest.mark.parametrize(
    "query, kwargs, prefix",
    [row[1:] for row in _REFUSED],
    ids=[row[0] for row in _REFUSED],
)
def test_closed_cells_raise_refused(query, kwargs, prefix):
    with pytest.raises(CausalUnsupportedError) as ei:
        antecedent.analyze(
            _DATA,
            query=query,
            refute=False,
            bootstrap=0,
            seed=1,
            **kwargs,
        )
    msg = str(ei.value)
    assert msg.startswith("refused:"), msg
    assert msg.startswith(prefix), msg


def test_licensed_pag_response_curve_runs():
    result = antecedent.analyze(
        _PAG_DATA,
        graph=_PAG,
        query=_CURVE,
        refute=False,
        bootstrap=0,
        seed=1,
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
    result = antecedent.analyze(
        _DATA,
        graph=graph,
        query=_CURVE,
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert result.response is not None
    assert np.isfinite(result.response.values).all()


def test_licensed_average_effect_on_dag_frequentist():
    result = antecedent.analyze(
        _DATA,
        graph=_DAG,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


def test_licensed_pag_ate_runs():
    result = antecedent.analyze(
        _PAG_DATA,
        graph=_PAG,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
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
    result = antecedent.analyze(
        data,
        graph=admg,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
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
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.PulseEffect(treatment="t", outcome="y", treatment_lag=1, horizon_steps=1),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)
    assert result.evidence_status == "licensed"


def test_newly_enforced_admg_bayesian_average_effect_raises_refused():
    """AverageEffect x Admg(bidirected) x Bayesian is a newly-enforced closure
    (parity/support_closed.toml, 2026-08-19 addition): general ID is the only
    identifier compile.rs wires for a bidirected ADMG, and it is not compatible
    with the bayesian.gcomp estimator that inference=Bayesian selects. Reachable
    from Python because AverageEffect x Admg passes a bare Admg graph straight
    through to native support-matrix consultation. Licensed ConditionalEffect /
    TemporalMediationEffect Bayesian cells now take the staged prepare path."""
    n = 300
    u = np.array([1.0 if (i % 5) < 2 else 0.0 for i in range(n)])
    t = np.array([1.0 if (i % 3) == 0 else 0.0 for i in range(n)])
    m = np.array([float(int(ti + ui) % 2) for ti, ui in zip(t, u, strict=True)])
    y = np.array([float(int(mi + ui) % 2) for mi, ui in zip(m, u, strict=True)])
    data = {"t": t, "m": m, "y": y}
    admg = antecedent.Admg.from_edges(
        ["t", "m", "y"], [("t", "m"), ("m", "y")], bidirected=[("t", "y")]
    )
    with pytest.raises(CausalUnsupportedError) as ei:
        antecedent.analyze(
            data,
            graph=admg,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(),
            refute=False,
            bootstrap=0,
            seed=1,
        )
    msg = str(ei.value)
    assert msg.startswith("refused: General ID"), msg


def test_path_specific_cheap_refute_runs_native_suite():
    t = np.array([0.0, 1.0] * 40)
    m = t.copy()
    y = t.copy()
    data = {"t": t, "m": m, "y": y}
    dag = antecedent.Dag.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")])
    result = antecedent.analyze(
        data,
        graph=dag,
        query=antecedent.PathSpecificEffect("t", "y", path_nodes=["m"]),
        refute="cheap",
        bootstrap=0,
        seed=1,
    )

    assert result.ate == pytest.approx(1.0)
    assert len(result.validation.reports) == 1


def test_distribution_full_refute_runs_native_suite():
    t = np.array([0.0, 1.0] * 40)
    y = t.copy()
    z = np.zeros(80)
    data = {"t": t, "y": y, "z": z}
    dag = antecedent.Dag.from_edges(["t", "y", "z"], [("z", "t"), ("z", "y"), ("t", "y")])
    result = antecedent.analyze(
        data,
        graph=dag,
        query=antecedent.InterventionalDistribution("y", interventions={"t": 1.0}),
        refute="full",
        bootstrap=0,
        seed=1,
    )

    assert result.ate == pytest.approx(1.0)
    assert len(result.validation.reports) == 3
