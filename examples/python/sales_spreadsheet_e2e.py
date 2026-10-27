#!/usr/bin/env python3
"""Explore several causal questions in a simulated sales analysis.

Discover and accept a graph, estimate an average effect with Bayesian
inference, and examine effects along particular paths and for individual
rows. A separate temporal block estimates the effect of a brief intervention.

The data are simulated from declared linear mechanisms, so the known targets are
fixture truths of that model: campaign intensity reaches revenue directly with
coefficient 1.2 and through the channel with 0.6 x 0.8 = 0.48, a total of 1.68,
and the lagged promotion pulse has coefficient 0.85. The checks compare each
answer with those values at tolerances sized for a few hundred rows.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import math

import antecedent
import numpy as np
from antecedent.outcomes import require_success


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

    bayes = require_success(
        antecedent.analyze(
            data,
            graph=accepted,
            query=q,
            inference=antecedent.Bayesian(backend="laplace", n_draws=128),
            refute="none",
            execution=antecedent.Execution(seed=3),
        )
    )
    assert math.isfinite(bayes.ate), bayes.ate
    assert bayes.posterior is not None
    # Declared total effect: 1.2 direct + 0.6 * 0.8 through the channel.
    assert abs(bayes.ate - 1.68) < 0.25, bayes.ate
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
        execution=antecedent.Execution(seed=5),
    )
    assert math.isfinite(path.total_change)
    mediated = next(
        (c for p, c in path.path_breakdown if len(p) == 3 and p[1] == "m"), None
    )
    assert mediated is not None, path.path_breakdown
    # Declared channel path: 0.6 * 0.8 = 0.48 (Monte Carlo tolerance).
    assert abs(mediated - 0.48) < 0.25, mediated
    print(
        f"Path decompose total_change={path.total_change:.4f} paths={len(path.path_breakdown)}"
    )

    ite = require_success(
        antecedent.analyze(
            data,
            graph=accepted,
            query=antecedent.Counterfactual("t", "y"),
            execution=antecedent.Execution(seed=7),
        )
    )
    assert len(ite.unit_effects) == len(cols[0])
    assert math.isfinite(ite.mean_ite)
    assert all(math.isfinite(float(u)) for u in ite.unit_effects)
    print(f"ITE mean={ite.mean_ite:.4f} n={len(ite.unit_effects)}")

    # Second estimate click — still no discovery.
    study = bayes.study
    second = require_success(
        antecedent.estimate(
            antecedent.reconfigure(study, execution=antecedent.Execution(seed=4))
        )
    )
    assert second.data_snapshot_id == bayes.data_snapshot_id
    artifact = bayes.export()
    acceptance = antecedent.load(artifact, expected=artifact.expectation).acceptance
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
    pulse = require_success(
        antecedent.analyze(
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
            execution=antecedent.Execution(seed=13),
        )
    )
    assert math.isfinite(pulse.ate), pulse.ate
    print(f"Temporal pulse Bayesian ATE={pulse.ate:.4f} (promo → returns)")
    assert abs(pulse.ate - 0.85) < 0.25, pulse.ate
    assert temporal.version == 1

    print("sales_spreadsheet_e2e: ok")


if __name__ == "__main__":
    main()
