#!/usr/bin/env python3
"""Explore several causal questions in a simulated sales analysis.

Discover and accept a graph, estimate an average effect with Bayesian
inference, and examine effects along particular paths and for individual
rows. A separate temporal example estimates the effect of a brief intervention.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import math

import antecedent
import numpy as np


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
    real_run = antecedent.discovery.PC.run

    def spy_run(self, *args, **kwargs):
        discovery_calls["n"] += 1
        return real_run(self, *args, **kwargs)

    # Spy on the config's own `run` — every path into PC discovery goes through it,
    # so the "estimate clicks must not discover" assertion below is not vacuous.
    antecedent.discovery.PC.run = spy_run  # type: ignore[method-assign]

    # Hand-accepted DAG (discover-once already reviewed in product UX).
    dag = antecedent.Dag.from_edges(
        ["z", "t", "m", "y"],
        [("z", "t"), ("z", "m"), ("z", "y"), ("t", "m"), ("t", "y"), ("m", "y")],
    )
    accepted = antecedent.AcceptedGraph.from_graph(dag, algorithm_id="reviewed")
    q = antecedent.AverageEffect(treatment="t", outcome="y")

    bayes = antecedent.analyze(
        data,
        graph=accepted,
        query=q,
        inference=antecedent.Bayesian(backend="laplace", n_draws=128),
        refute="none",
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
    path = antecedent.attribution.attribute_path_specific(
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

    ite = antecedent.analyze(
        data, graph=accepted, query=antecedent.Counterfactual("t", "y"), seed=7
    )
    assert len(ite.unit_effects) == len(cols[0])
    assert math.isfinite(ite.mean_ite)
    print(f"ITE mean={ite.mean_ite:.4f} n={len(ite.unit_effects)}")

    # Second estimate click — still no discovery.
    study = bayes.study
    second = study.estimate(seed=4)
    assert second.data_snapshot_id == bayes.data_snapshot_id
    acceptance = antecedent.load(bayes.export()).acceptance
    assert acceptance.verified or acceptance.sealed
    print("Calibration:", bayes.calibration.status)
    assert discovery_calls["n"] == 0, "static estimate clicks must not discover"
    assert accepted.version == 1

    # --- Temporal pulse Bayesian block ---
    series = _sales_temporal()
    tdag = antecedent.TemporalDag.from_lagged_edges(
        ["promo", "returns"], [("promo", 1, "returns", 0)]
    )
    temporal = antecedent.AcceptedGraph.from_graph(tdag, algorithm_id="pcmci")
    pulse = antecedent.analyze(
        series,
        graph=temporal,
        query=antecedent.PulseEffect(
            treatment="promo",
            outcome="returns",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        inference=antecedent.Bayesian(backend="laplace", n_draws=96),
        refute="none",
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
