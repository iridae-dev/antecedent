#!/usr/bin/env python3
"""Discover a graph once, review it, and reuse the accepted graph for later estimates.

The simulation has a confounder z of a binary treatment t whose effect on y is 2. PC discovery
runs once; the script then supplies the known directions as an AcceptedGraph. In a real
analysis those directions need justification from subject-matter knowledge or other evidence.

Keeping result.study lets later estimates and a rebind onto new data skip discovery. The
script counts discovery calls to assert that none is repeated, and asserts the first estimate
within 0.75 of 2. Accepting a graph does not make discovery certain.

Run: ``python examples/python/discover_then_estimate.py``."""

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

    # Count calls through both entry points into PC discovery, so the claim that later
    # estimates never rediscover is checked and not assumed.
    antecedent.discovery.PC.run = counted_run  # type: ignore[method-assign]
    antecedent.discovery.discover = counted_discover  # type: ignore[assignment]

    # Run discovery once, explicitly; its result is the evidence a reviewer reads.
    evidence = antecedent.discovery.discover(
        data,
        method=antecedent.discovery.PC(alpha=0.5, fdr=False, max_cond_size=0),
        accept_discovered=True,
        execution=antecedent.Execution(seed=1),
    )
    discovered = discovery_calls["n"]
    assert discovered >= 1
    assert isinstance(evidence.graph, (antecedent.Dag, antecedent.Cpdag))

    # Review: accept a fully oriented DAG for the later estimates.
    # (Edges discovery left unoriented stay on the evidence handle; the directions here are
    # the reviewer's.)
    accepted = antecedent.AcceptedGraph.from_graph(
        [("z", "t"), ("z", "y"), ("t", "y")],
        algorithm_id=evidence.algorithm_id,
        version=evidence.version,
    )
    structure_version = accepted.version

    query = antecedent.AverageEffect(treatment="t", outcome="y")

    # Every estimate below rests on the accepted graph: a first analysis, a repeat on the
    # retained study, and a rebind onto a fresh sample.
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

    # The accepted graph round-trips through JSON with its version, for a later session.
    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())
    assert restored.version == accepted.version

    print(
        f"ATE={first_ate:.4f} version={accepted.version} "
        f"discovery_calls={discovery_calls['n']} "
        f"latency={first.value.performance.latency_mode}"
    )


if __name__ == "__main__":
    main()
