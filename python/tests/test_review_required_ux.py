"""Structured CausalReviewError attrs and TemporalPag completion."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _lag1_series(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.9 * x[t - 1] + 0.05 * rng.normal()
    return {"x": x, "y": y}


def test_fci_review_required_attrs():
    n = 80
    rng = np.random.default_rng(11)
    z = rng.normal(size=n)
    t = z + rng.normal(size=n) * 0.3
    y = 1.2 * t + z + rng.normal(size=n) * 0.3
    with pytest.raises(antecedent.CausalReviewError) as ei:
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            discovery=antecedent.FCI(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            accept_discovered=False,
            refute=False,
            bootstrap=0,
            seed=1,
        )
    err = ei.value
    assert getattr(err, "kind", None) == "static_pag"
    assert getattr(err, "algorithm", None) == "fci"
    assert isinstance(getattr(err, "pending_edge_count", None), int)
    assert getattr(err, "hint", None)


def test_complete_temporal_pag_estimates():
    data = _lag1_series()
    pag = antecedent.TemporalPag.from_marked_lagged_edges(
        ["x", "y"],
        [("x", 1, "y", 0, "tail", "arrow")],
    )
    result = antecedent.analyze(
        data,
        graph=pag,
        query=antecedent.PulseEffect(
            treatment="x",
            outcome="y",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        bootstrap=0,
        seed=1,
        refute=False,
    )
    assert isinstance(result.ate, float)
    assert abs(result.ate - 0.9) < 0.15
    assert any("temporal.pag.completed_to_dag" in str(d) for d in result.diagnostics)


def test_incomplete_temporal_pag_review_attrs():
    data = _lag1_series(n=60, seed=9)
    pag = antecedent.TemporalPag.from_marked_lagged_edges(
        ["x", "y"],
        [("x", 1, "y", 0, "circle", "arrow")],
    )
    with pytest.raises(antecedent.CausalReviewError) as ei:
        antecedent.analyze(
            data,
            graph=pag,
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
    err = ei.value
    assert getattr(err, "kind", None) == "temporal_pag"
    assert getattr(err, "pending_edge_count", 0) >= 1
    assert getattr(err, "hint", None)
