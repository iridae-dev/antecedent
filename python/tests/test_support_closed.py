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
    "Dag cells, and those cells are not staged yet."
)
_REASON_COUNTERFACTUAL = "refused: Counterfactual is not on the staged handle."
_REASON_MEDIATION = "refused: MediationEffect is not on the staged handle."
_REASON_SUSTAINED = "refused: SustainedEffect is not on the staged handle."


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
