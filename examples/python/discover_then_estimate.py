#!/usr/bin/env python3
"""Discover a graph once and reuse a reviewed graph for later estimates.

We first run PC discovery. For this simulation, we then supply the known
causal directions as an AcceptedGraph. In a real analysis, those directions
would need justification from subject-matter knowledge or other evidence.

Keeping result.study lets us estimate again or refresh the data without
repeating discovery. The counter below checks that discovery runs only once.
Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import math
import random

import antecedent
import numpy as np


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
    real_run = antecedent.discovery.PC.run

    def counted_run(self, *args, **kwargs):
        discovery_calls["n"] += 1
        return real_run(self, *args, **kwargs)

    # Spy on the config's own `run` — every path into PC discovery goes through
    # it, so "estimate clicks never rediscover" is a claim this can actually check.
    antecedent.discovery.PC.run = counted_run  # type: ignore[method-assign]

    # Run discovery once.
    evidence = antecedent.discovery.PC(alpha=0.5, fdr=False, max_cond_size=0).accept(data, seed=1)
    assert discovery_calls["n"] == 1
    assert isinstance(evidence.graph, (antecedent.Dag, antecedent.Cpdag))

    # Spreadsheet review: accept a fully oriented DAG for estimate clicks.
    # (Incomplete CPDAG marks stay on the evidence handle until explicit rediscover.)
    accepted = antecedent.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")],
        algorithm_id=evidence.algorithm_id,
        version=evidence.version,
    )
    structure_version = accepted.version

    query = antecedent.AverageEffect(treatment="t", outcome="y")

    # Reuse the reviewed graph for repeated estimates.
    first = antecedent.analyze(data, graph=accepted, query=query, seed=1, latency="interactive")
    study = first.study
    second = study.estimate()
    third = study.refresh(_confounded_scm(seed=8))
    assert first.data_snapshot_id != third.data_snapshot_id
    print("Calibration:", first.calibration.status)

    assert discovery_calls["n"] == 1, (
        f"second estimate re-ran discovery (calls={discovery_calls['n']})"
    )
    assert accepted.version == structure_version
    assert math.isfinite(first.ate) and abs(first.ate - 2.0) < 0.75
    assert math.isfinite(second.ate) and math.isfinite(third.ate)

    # Durable hold for the next session.
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == accepted.version

    print(
        f"ATE={first.ate:.4f} version={accepted.version} "
        f"discovery_calls={discovery_calls['n']} "
        f"latency={first.performance.latency_mode}"
    )


if __name__ == "__main__":
    main()
