#!/usr/bin/env python3
"""Class-preserving CPDAG ATE: the graph stays a Cpdag.

A partial CPDAG estimates a MEC envelope. A fully-oriented CPDAG stays a
Cpdag and can hand off its adjustment set. Completing the graph yourself
is still the Dag cell.

Requires a built antecedent extension (`maturin develop` in python/).
"""

from __future__ import annotations

import numpy as np
from antecedent import AverageEffect, Cpdag, analyze, identify
from antecedent.handoff import econml


def confounded(n: int = 200) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    return {"t": t, "y": y, "z": z}


def main() -> None:
    data = confounded()
    query = AverageEffect(treatment="t", outcome="y")
    names = ["t", "y", "z"]

    partial = Cpdag.from_directed_undirected(
        names,
        [("z", "y"), ("t", "y")],
        [("z", "t")],
    )
    identified = identify(graph=partial, query=query)
    print(f"partial status={identified.status} method={identified.method}")
    assert identified.status == "PartiallyIdentified"

    envelope = analyze(data, graph=partial, query=query, refute="none", bootstrap=0)
    print(f"partial ate={envelope.ate:.4f} class=Cpdag")
    assert np.isfinite(envelope.ate)

    oriented = Cpdag.from_directed_undirected(
        names,
        [("z", "t"), ("z", "y"), ("t", "y")],
    )
    point = identify(graph=oriented, query=query)
    spec = econml(point)
    print(f"oriented status={point.status} W={spec.confounders}")
    assert "Identified" in point.status
    assert spec.confounders == ("z",)


if __name__ == "__main__":
    main()
