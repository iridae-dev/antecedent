"""Supplied-posterior binding: schema order, atom validity, provenance, refusals."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError

from known_truth import (
    BAYES,
    STATIC,
    TEMPORAL,
    set_edge,
    static_data,
    static_posterior,
    temporal_posterior,
    white_noise_pulse_series,
)

_ATE = antecedent.AverageEffect(treatment="t", outcome="y")
_PULSE = antecedent.PulseEffect(
    treatment="pressure",
    outcome="defect",
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
)


def _reordered(data: dict[str, np.ndarray], order: list[str]) -> dict[str, np.ndarray]:
    return {name: data[name] for name in order}


def test_supplied_static_posterior_refuses_column_order_mismatch():
    data = _reordered(static_data(int(STATIC["n"])), ["z", "t", "y"])
    with pytest.raises(ValueError, match="names must match data column names and order"):
        antecedent.analyze(
            data, discovery=static_posterior(), query=_ATE, inference=BAYES, refute=False
        )
    with pytest.raises(ValueError, match="names must match data column names and order"):
        # lgtm[py/call/wrong-named-argument]
        antecedent.estimation.PreparedAnalysis.prepare(
            data, discovery=static_posterior(), query=_ATE, inference=BAYES, refute=False
        )


def test_supplied_dbn_posterior_refuses_column_order_mismatch():
    series = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    data = _reordered(series, ["defect", "pressure"])
    with pytest.raises(ValueError, match="names must match data column names and order"):
        antecedent.analyze(
            data, discovery=temporal_posterior(), query=_PULSE, inference=BAYES, refute=False
        )
    with pytest.raises(ValueError, match="names must match data column names and order"):
        # lgtm[py/call/wrong-named-argument]
        antecedent.estimation.PreparedAnalysis.prepare(
            data, discovery=temporal_posterior(), query=_PULSE, inference=BAYES, refute=False
        )


def test_supplied_posterior_refuses_variable_count_mismatch():
    data = static_data(int(STATIC["n"]))
    two_variable = antecedent.discovery.GraphPosterior.from_atoms(
        ["t", "y"], [1.0], [set_edge(0, 2, 0, 1)]
    )
    with pytest.raises(ValueError, match="names must match data column names and order"):
        antecedent.analyze(data, discovery=two_variable, query=_ATE, inference=BAYES, refute=False)


def test_from_atoms_validates_atoms_and_weights():
    from_atoms = antecedent.discovery.GraphPosterior.from_atoms
    names = ["t", "y", "z"]
    cyclic = set_edge(set_edge(0, 3, 0, 1), 3, 1, 0)
    with pytest.raises(ValueError, match="not a DAG"):
        from_atoms(names, [1.0], [cyclic])
    with pytest.raises(ValueError, match="beyond 3 variables"):
        from_atoms(names, [1.0], [1 << 6])
    with pytest.raises(ValueError, match="sum to 1"):
        from_atoms(names, [0.5, 0.3], [set_edge(0, 3, 0, 1), 0])
    with pytest.raises(ValueError, match="at most 8 variables"):
        from_atoms([f"v{i}" for i in range(9)], [1.0], [0])
    with pytest.raises(ValueError, match="lag_masks require max_lag"):
        from_atoms(["pressure", "defect"], [1.0], [0], lag_masks=[2])
    with pytest.raises(ValueError, match="beyond max_lag=1"):
        from_atoms(
            ["pressure", "defect"],
            [1.0],
            [0],
            lagged_edge_marginals=[0.0, 1.0, 0.0, 0.0],
            lag_masks=[1 << 4],
            max_lag=1,
        )
    built = from_atoms(names, [0.7, 0.3], [set_edge(0, 3, 0, 1), 0])
    assert built.algorithm == "from_atoms"
    assert built.converged is True
    assert built.n_graphs == 2


def test_supplied_dbn_posterior_refuses_a_static_query():
    series = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    static_query = antecedent.AverageEffect(treatment="pressure", outcome="defect")
    with pytest.raises(CausalUnsupportedError, match="DBN lag structure"):
        antecedent.analyze(
            series,
            discovery=temporal_posterior(),
            query=static_query,
            inference=BAYES,
            refute=False,
        )
    with pytest.raises(CausalUnsupportedError, match="DBN lag structure"):
        # lgtm[py/call/wrong-named-argument]
        antecedent.estimation.PreparedAnalysis.prepare(
            series,
            discovery=temporal_posterior(),
            query=static_query,
            inference=BAYES,
            refute=False,
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("identifier", "backdoor.adjustment"),
        ("estimator", "linear.adjustment.ate"),
        ("validators", []),
        ("return_posterior_artifact", True),
    ],
)
def test_supplied_posterior_refuses_options_it_cannot_honour(option, value):
    data = static_data(int(STATIC["n"]))
    with pytest.raises(CausalUnsupportedError, match=option):
        antecedent.analyze(
            data,
            discovery=static_posterior(),
            query=_ATE,
            inference=BAYES,
            refute=False,
            **{option: value},
        )


def test_supplied_posterior_refuses_prior_transfer():
    data = static_data(int(STATIC["n"]))
    transfer = antecedent.Bayesian(backend="conjugate", n_draws=32, prior_from=b"\x00")
    with pytest.raises(CausalUnsupportedError, match="prior_artifact"):
        antecedent.analyze(
            data, discovery=static_posterior(), query=_ATE, inference=transfer, refute=False
        )


def test_supplied_posterior_keeps_discovery_provenance():
    n = 200
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    data = {"t": t, "y": y, "z": z}
    posterior = antecedent.discovery.ExactDagPosterior().run(data, seed=7)
    assert posterior.algorithm == "exact_dag_posterior"
    result = antecedent.analyze(
        data,
        discovery=posterior,
        query=_ATE,
        inference=antecedent.Bayesian(n_draws=48, prior_scale=100.0, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert result.plan.discovery_algorithm == "exact_dag_posterior"
    assert result.evidence_status == "licensed"
