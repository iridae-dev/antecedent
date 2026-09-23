#!/usr/bin/env python3
"""DoWhy ↔ Antecedent handoff on a small backdoor ATE SCM.

Shared toy: confounder z, treatment t, outcome y, structural ATE = +2.
Antecedent identifies via backdoor adjustment on {z} and estimates with
propensity weighting. When ``dowhy`` is installed, the same graph DOT is
handed to DoWhy (and a DoWhy DOT is imported back into Antecedent).

``dowhy`` is optional. Without it, the Antecedent half still runs and asserts
the adjustment set and effect sign. See docs/interop_dowhy.md.
Install with `python -m pip install antecedent`; optionally `pip install dowhy`.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from antecedent import AcceptedGraph, AverageEffect, Dag, analyze, identify

EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
NAMES = ["z", "t", "y"]
TREATMENT = "t"
OUTCOME = "y"
EXPECTED_ADJUSTMENT = frozenset({"z"})
TRUE_ATE = 2.0


def confounded_scm(n: int = 800, seed: int = 7) -> dict[str, np.ndarray]:
    """Linear-Gaussian outcome with propensity selection on z (ATE = 2)."""
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = TRUE_ATE * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}


def antecedent_side(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Identify and estimate under Antecedent on a reviewed DAG."""
    dag = Dag.from_edges(NAMES, EDGES)
    query = AverageEffect(treatment=TREATMENT, outcome=OUTCOME)
    identified = identify(graph=dag, query=query)
    adjustment = frozenset(identified.adjustment_set)
    accepted = AcceptedGraph.from_graph(dag, algorithm_id="hand")
    result = analyze(
        data,
        graph=accepted,
        query=query,
        estimator="propensity.weighting",
        bootstrap=0,
        refute=False,
        seed=11,
    )
    return {
        "dot": dag.to_dot(),
        "adjustment_set": adjustment,
        "ate": float(result.ate),
        "status": identified.status,
        "method": identified.method,
    }


def dowhy_side(data: dict[str, np.ndarray], antecedent_dot: str) -> dict[str, Any] | None:
    """Optional DoWhy half: Antecedent DOT → DoWhy, and DoWhy DOT → Antecedent."""
    try:
        import pandas as pd
        from dowhy import CausalModel
    except ImportError:
        return None

    frame = pd.DataFrame(data)
    model = CausalModel(
        data=frame,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=antecedent_dot,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    dowhy_adjustment = frozenset(estimand.get_backdoor_variables())
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    dowhy_ate = float(estimate.value)

    # DoWhy → Antecedent: import a DoWhy-style graph string after review.
    dowhy_dot = "digraph { z -> t; z -> y; t -> y; }"
    imported = Dag.from_dot(dowhy_dot)
    accepted = AcceptedGraph.from_graph(imported, algorithm_id="dowhy-reviewed")
    query = AverageEffect(treatment=TREATMENT, outcome=OUTCOME)
    round_trip = identify(graph=accepted.graph, query=query)
    round_trip_adj = frozenset(round_trip.adjustment_set)

    return {
        "adjustment_set": dowhy_adjustment,
        "ate": dowhy_ate,
        "round_trip_adjustment_set": round_trip_adj,
    }


def main() -> None:
    data = confounded_scm()
    ant = antecedent_side(data)
    assert ant["adjustment_set"] == EXPECTED_ADJUSTMENT, ant["adjustment_set"]
    assert ant["ate"] > 0.0 and abs(ant["ate"] - TRUE_ATE) < 0.5, ant["ate"]
    print(
        f"Antecedent: adjustment={sorted(ant['adjustment_set'])} "
        f"method={ant['method']} ATE={ant['ate']:.4f}"
    )

    dowhy = dowhy_side(data, ant["dot"])
    if dowhy is None:
        print("DoWhy not installed; skipped DoWhy half (soft dependency).")
        return

    assert dowhy["adjustment_set"] == EXPECTED_ADJUSTMENT, dowhy["adjustment_set"]
    assert dowhy["round_trip_adjustment_set"] == EXPECTED_ADJUSTMENT
    assert dowhy["ate"] > 0.0 and abs(dowhy["ate"] - TRUE_ATE) < 0.5, dowhy["ate"]
    print(
        f"DoWhy: adjustment={sorted(dowhy['adjustment_set'])} "
        f"ATE={dowhy['ate']:.4f} (same sign / set as Antecedent)"
    )


if __name__ == "__main__":
    main()
