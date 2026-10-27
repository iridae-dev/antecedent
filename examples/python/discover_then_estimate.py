#!/usr/bin/env python3
"""Discover a graph once and reuse a reviewed graph for later estimates.

We first run PC discovery. For this simulation, we then supply the known
causal directions as an AcceptedGraph. In a real analysis, those directions
would need justification from subject-matter knowledge or other evidence.

Keeping result.study lets us estimate again or rebind new data without
repeating discovery. The counter below checks that discovery is not re-run.
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
    real_discover = antecedent.discovery.discover

    def counted_run(self, *args, **kwargs):
        discovery_calls["n"] += 1
        return real_run(self, *args, **kwargs)

    def counted_discover(*args, **kwargs):
        discovery_calls["n"] += 1
        return real_discover(*args, **kwargs)

    # Spy on both entry points into PC discovery, so "estimate clicks never
    # rediscover" is a claim this can actually check.
    antecedent.discovery.PC.run = counted_run  # type: ignore[method-assign]
    antecedent.discovery.discover = counted_discover  # type: ignore[assignment]

    # Run discovery once, explicitly, and accept its structure for review.
    evidence = antecedent.discovery.discover(
        data,
        method=antecedent.discovery.PC(alpha=0.5, fdr=False, max_cond_size=0),
        accept_discovered=True,
        execution=antecedent.Execution(seed=1),
    )
    discovered = discovery_calls["n"]
    assert discovered >= 1
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
    first = antecedent.analyze(
        data,
        graph=accepted,
        query=query,
        inference=antecedent.Frequentist(profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    study = first.study
    second = antecedent.estimate(study)
    third = antecedent.estimate(antecedent.rebind(study, data=_confounded_scm(seed=8)))
    assert first.value.data_snapshot_id != third.value.data_snapshot_id
    print("Calibration:", first.value.calibration.status)

    assert discovery_calls["n"] == discovered, (
        f"a later estimate re-ran discovery (calls={discovery_calls['n']})"
    )
    assert accepted.version == structure_version
    first_ate, second_ate, third_ate = (o.answer.value for o in (first, second, third))
    assert math.isfinite(first_ate) and abs(first_ate - 2.0) < 0.75
    assert math.isfinite(second_ate) and math.isfinite(third_ate)

    # Durable hold for the next session.
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == accepted.version

    print(
        f"ATE={first_ate:.4f} version={accepted.version} "
        f"discovery_calls={discovery_calls['n']} "
        f"latency={first.value.performance.latency_mode}"
    )


if __name__ == "__main__":
    main()
