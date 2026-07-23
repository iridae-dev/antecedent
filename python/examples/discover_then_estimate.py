#!/usr/bin/env python3
"""Spreadsheet-style discover-once → many interactive estimates (backlog D).

Contrast with one-shot ``analyze(..., discovery=...)`` (script path). Interactive
products should: discover → accept into ``AcceptedGraph`` → estimate clicks with
``graph=`` / ``AcceptedGraph.analyze``; rediscover only on explicit refresh.

Requires a built causal extension (`maturin develop` in python/).
"""

from __future__ import annotations

import math
import random

import numpy as np

import causal


def _confounded_scm(n: int = 500, seed: int = 7):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}


def main() -> None:
    data = _confounded_scm()
    discovery_calls = {"n": 0}
    real_discover = causal.discover_pc

    def counted_discover(*args, **kwargs):
        discovery_calls["n"] += 1
        return real_discover(*args, **kwargs)

    # Structure-ready click (once).
    causal.discover_pc = counted_discover  # type: ignore[method-assign]
    result = causal.discover_pc(
        data, alpha=0.5, fdr=False, max_cond_size=0, seed=1
    )
    evidence = causal.AcceptedGraph.from_discovery(result, algorithm_id="pc")
    assert discovery_calls["n"] == 1
    assert isinstance(evidence.graph, (causal.Dag, causal.Cpdag))

    # Spreadsheet review: accept a fully oriented DAG for estimate clicks.
    # (Incomplete CPDAG marks stay on the evidence handle until explicit rediscover.)
    accepted = causal.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")],
        algorithm_id=evidence.algorithm_id,
        version=evidence.version,
    )
    structure_version = accepted.version

    query = causal.AverageEffect(treatment="t", outcome="y")

    # Effect-ready clicks (many) — must not re-enter discovery.
    first = accepted.analyze(data, query=query, seed=1)
    second = accepted.analyze(data, query=query, seed=1, bootstrap=0)
    prepared = accepted.prepare(data, query=query, seed=1)
    third = prepared.estimate(data, seed=1)

    assert discovery_calls["n"] == 1, (
        f"second estimate re-ran discovery (calls={discovery_calls['n']})"
    )
    assert accepted.version == structure_version
    assert math.isfinite(first.ate) and abs(first.ate - 2.0) < 0.75
    assert math.isfinite(second.ate) and math.isfinite(third.ate)

    # Durable hold for the next session.
    restored = causal.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == accepted.version

    print(
        f"ATE={first.ate:.4f} version={accepted.version} "
        f"discovery_calls={discovery_calls['n']} "
        f"latency={first.performance.latency_mode}"
    )


if __name__ == "__main__":
    main()
