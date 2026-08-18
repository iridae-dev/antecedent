"""First enforced-refusal wave: closed cells raise; licensed and default-refused still run."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError

_REASON_DERIVATIVE = (
    "refused: Derivative cells are not licensed; only ResponseCurve on a Dag is staged."
)
_REASON_RESPONSE_PAG = "refused: ResponseCurve is licensed only on a Dag."
_REASON_PATH_DIST = (
    "refused: Path and distribution queries are licensed only as explicit "
    "Dag cells; accepted and graph-posterior structures are not staged."
)
_REASON_COUNTERFACTUAL = "refused: Counterfactual is not on the staged handle."
_REASON_MEDIATION = "refused: MediationEffect is not on the staged handle."
_REASON_SUSTAINED = "refused: SustainedEffect is not on the staged handle."
_REASON_INTERVENTION_RESPONSE_OFF_DAG = (
    "refused: InterventionResponse executes only on a supplied static Dag, the same "
    "requirement ResponseCurve is closed on above; Cpdag/Admg/Pag have no Response "
    "compile arm."
)


def _two_node_table(n: int = 48, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.2, 1.8, n)
    y = 2.0 * t + rng.normal(scale=0.1, size=n)
    return {"t": t, "y": y}


_DATA = _two_node_table()
_EDGES = [("t", "y")]
_DAG = antecedent.Dag.from_edges(["t", "y"], _EDGES)
_PAG = antecedent.Pag.from_marked_edges(["t", "y"], [("t", "y", "circle", "arrow")])
_ACCEPTED = antecedent.AcceptedGraph.from_graph(_DAG, algorithm_id="hand")
_CURVE = antecedent.ResponseCurve("t", "y", grid=[0.5, 1.0, 1.5])
_ADMG = antecedent.Admg.from_edges(["t", "y"], _EDGES)


_REFUSED = [
    (
        "elasticity",
        antecedent.Elasticity("t", "y", at=1.0),
        {"graph": _EDGES},
        _REASON_DERIVATIVE,
    ),
    (
        "response_curve_pag",
        _CURVE,
        {"graph": _PAG},
        _REASON_RESPONSE_PAG,
    ),
    (
        "path_specific_exact_dag_posterior",
        antecedent.PathSpecificEffect("t", "y"),
        {"discovery": antecedent.discovery.ExactDagPosterior()},
        _REASON_PATH_DIST,
    ),
    (
        "interventional_distribution_accepted",
        antecedent.InterventionalDistribution("y", interventions={"t": 1.0}),
        {"graph": _ACCEPTED},
        _REASON_PATH_DIST,
    ),
    (
        "counterfactual",
        antecedent.Counterfactual("t", "y"),
        {"graph": _DAG},
        _REASON_COUNTERFACTUAL,
    ),
    (
        "mediation",
        antecedent.MediationEffect("t", "y", mediators=["m"]),
        {"graph": [("t", "m"), ("m", "y")]},
        _REASON_MEDIATION,
    ),
    (
        "sustained",
        antecedent.SustainedEffect("t", "y", treatment_lag=1, horizon_steps=1),
        {"graph": [("t", 1, "y", 0)]},
        _REASON_SUSTAINED,
    ),
    # Second enforced-refusal wave (parity/support_closed.toml): ConditionalEffect /
    # InterventionResponse / TemporalMediationEffect off a supplied Dag, and any
    # graph-posterior structure under Frequentist inference.
    #
    # Only InterventionResponse × Admg/Pag has a representative row here.
    # ConditionalEffect × Cpdag/Admg/Pag, PathSpecificEffect/InterventionalDistribution ×
    # Admg/Pag (explicit), and TemporalMediationEffect × TemporalCpdag/TemporalPag are all
    # closed on the Rust support matrix (see crates/antecedent/src/support.rs's
    # `closed_conditional_effect_off_dag_is_enforced` /
    # `closed_path_and_distribution_on_explicit_admg_pag_is_enforced` /
    # `closed_temporal_mediation_off_temporal_dag_is_enforced` tests), but none of them are
    # reachable through `antecedent.analyze()`: `_static_edges`/`_lagged_edges`
    # (python/antecedent/estimation.py) only coerce `Dag`/`Cpdag`/`TemporalDag`/edge-list
    # inputs, so a bare `Admg`/`Pag`/`TemporalCpdag`/`TemporalPag` fails the
    # `for a, b in graph` unpack *before* the call ever reaches the native support-matrix
    # consultation (a `ValueError`, not `CausalUnsupportedError`); a `Cpdag` is converted to
    # a plain oriented edge list client-side, so it is indistinguishable from an explicit Dag
    # by the time it reaches native code and never carries a `Cpdag` class tag to classify.
    # Likewise graph_posterior × Frequentist is pre-empted by `_analyze.py`'s own
    # `discovery=`/`inference=` check (`handle_static_ate_discover`, line ~817), which raises
    # `TypeError("graph-posterior discovery requires inference=Bayesian(...) for effect
    # mixture")` before `Study::build` ever runs. The Rust closures still apply to the Rust
    # surface (`crates/antecedent/src/support.rs`); they are simply unreachable from this
    # Python entry point today.
    (
        "intervention_response_admg",
        antecedent.InterventionResponse("y", intervention={"t": 1.0}),
        {"graph": _ADMG},
        _REASON_INTERVENTION_RESPONSE_OFF_DAG,
    ),
    (
        "intervention_response_pag",
        antecedent.InterventionResponse("y", intervention={"t": 1.0}),
        {"graph": _PAG},
        _REASON_INTERVENTION_RESPONSE_OFF_DAG,
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


def test_default_refused_pag_ate_still_runs():
    result = antecedent.analyze(
        _DATA,
        graph=_PAG,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)


# -- 2026-08-19: allowlist introduction (parity/support_allowlist.toml). PAG ATE
# above is one allowlisted family; the temporal PulseEffect family is another.


def test_allowlisted_pulse_effect_temporal_dag_still_runs():
    """PulseEffect x TemporalDag x explicit is on the allowlist (running,
    unlicensed, not closed): the temporal backdoor path is fully wired for both
    inference modes and every validation suite, but has no cell-shaped
    known-truth fixture the way the static AverageEffect family does."""
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
        query=antecedent.PulseEffect(
            treatment="t", outcome="y", treatment_lag=1, horizon_steps=1
        ),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert np.isfinite(result.ate)


def test_newly_enforced_admg_bayesian_average_effect_raises_refused():
    """AverageEffect x Admg(bidirected) x Bayesian is a newly-enforced closure
    (parity/support_closed.toml, 2026-08-19 addition): general ID is the only
    identifier compile.rs wires for a bidirected ADMG, and it is not compatible
    with the bayesian.gcomp estimator that inference=Bayesian selects. Reachable
    from Python (unlike ConditionalEffect/TemporalMediationEffect x Bayesian,
    which _analyze.py itself pre-empts with a TypeError before reaching native
    code) because AverageEffect x Admg passes a bare Admg graph straight through
    to native support-matrix consultation."""
    n = 300
    u = np.array([1.0 if (i % 5) < 2 else 0.0 for i in range(n)])
    t = np.array([1.0 if (i % 3) == 0 else 0.0 for i in range(n)])
    m = np.array([float(int(ti + ui) % 2) for ti, ui in zip(t, u)])
    y = np.array([float(int(mi + ui) % 2) for mi, ui in zip(m, u)])
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
