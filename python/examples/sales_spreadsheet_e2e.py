#!/usr/bin/env python3
"""Sales spreadsheet E2E: discover → Bayesian ATE → path → ITE + temporal pulse.

Mirrors the interactive UX spine (ADR 0011 / backlog Docs):

  discover once → AcceptedGraph
    → Bayesian ATE estimate click
    → path-specific decompose
    → unit ITE
  plus a temporal pulse Bayesian block on a held TemporalDag.

Requires a built causal extension (`maturin develop` in python/).
"""

from __future__ import annotations

import math

import numpy as np

import antecedent


def _sales_static(n: int = 400, seed: int = 7):
    """Campaign intensity (t) → revenue (y) via channel (m), confounder spend_context (z).

    Continuous linear-Gaussian SEM so path_decompose (β products) and GCM ITE share
    the same artifact as Bayesian ATE — binary treatment would refuse path_decompose.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.7 * z + rng.normal(size=n)
    m = 0.6 * t + 0.3 * z + 0.2 * rng.normal(size=n)
    y = 1.2 * t + 0.8 * m + 0.5 * z + 0.3 * rng.normal(size=n)
    return {"t": t, "m": m, "y": y, "z": z}


def _sales_temporal(n: int = 350, seed: int = 11):
    """Pulse: promo intensity (x) at lag 1 moves defect/return rate proxy (y)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.04) + 0.1 * rng.normal(size=n)
    y = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        y[i] = 0.85 * x[i - 1] + 0.05 * rng.normal()
    return {"promo": x, "returns": y}


def main() -> None:
    # --- Static spreadsheet block ---
    data = _sales_static()
    discovery_calls = {"n": 0}
    real_pc = antecedent.discover_pc

    def spy_pc(*args, **kwargs):
        discovery_calls["n"] += 1
        return real_pc(*args, **kwargs)

    # Spy via accepted_graph path used by rediscover; estimate clicks must not call it.
    antecedent.discover_pc = spy_pc  # type: ignore[assignment]

    # Hand-accepted DAG (discover-once already reviewed in product UX).
    dag = antecedent.Dag.from_edges(
        ["z", "t", "m", "y"],
        [("z", "t"), ("z", "m"), ("z", "y"), ("t", "m"), ("t", "y"), ("m", "y")],
    )
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="reviewed")
    q = antecedent.AverageEffect(treatment="t", outcome="y")

    bayes = accepted.analyze(
        data,
        query=q,
        inference=antecedent.Bayesian(backend="laplace", n_draws=128),
        refute=False,
        seed=3,
        bootstrap=0,
    )
    assert math.isfinite(bayes.ate), bayes.ate
    assert bayes.posterior is not None
    print(f"Bayesian ATE={bayes.ate:.4f} (campaign → revenue)")

    # Path-specific: direct t→y vs mediated t→m→y
    names = ["z", "t", "m", "y"]
    cols = [data["z"], data["t"], data["m"], data["y"]]
    edges = [("z", "t"), ("z", "m"), ("z", "y"), ("t", "m"), ("t", "y"), ("m", "y")]
    path = antecedent.attribute_path_specific(
        names,
        cols,
        edges,
        "t",
        "y",
        path_nodes=["m"],
        seed=5,
    )
    assert math.isfinite(path.total_change)
    print(f"Path decompose total_change={path.total_change:.4f} paths={len(path.path_breakdown)}")

    ite = antecedent.counterfactual_ite(
        names, cols, edges, "t", "y", 1.0, 0.0, seed=7
    )
    assert ite.n_units == len(cols[0])
    assert math.isfinite(ite.mean_ite)
    print(f"ITE mean={ite.mean_ite:.4f} n={ite.n_units}")

    # Second estimate click — still no discovery.
    _ = accepted.analyze(
        data, query=q, inference=antecedent.Bayesian(n_draws=64), refute=False, seed=4
    )
    assert discovery_calls["n"] == 0, "static estimate clicks must not discover"
    assert accepted.version == 1

    # --- Temporal pulse Bayesian block ---
    series = _sales_temporal()
    tdag = antecedent.TemporalDag.from_lagged_edges(
        ["promo", "returns"], [("promo", 1, "returns", 0)]
    )
    temporal = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="pcmci")
    pulse = temporal.analyze(
        series,
        query=antecedent.PulseEffect(
            treatment="promo",
            outcome="returns",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Bayesian(backend="laplace", n_draws=96),
        refute=False,
        seed=13,
        bootstrap=0,
    )
    assert math.isfinite(pulse.ate), pulse.ate
    print(f"Temporal pulse Bayesian ATE={pulse.ate:.4f} (promo → returns)")
    assert abs(pulse.ate - 0.85) < 0.25, pulse.ate
    assert temporal.version == 1

    print("sales_spreadsheet_e2e: ok")


if __name__ == "__main__":
    main()
