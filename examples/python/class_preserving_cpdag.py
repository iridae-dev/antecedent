#!/usr/bin/env python3
"""Estimate an effect when some causal directions are unknown.

A CPDAG represents several possible causal graphs. First, we leave one edge
undirected and get an answer that preserves that uncertainty. Then we supply
its direction and pass the resulting adjustment set to an EconML handoff.
The graph object remains a Cpdag in both cases.

Install with `python -m pip install antecedent`; see examples/README.md."""

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
    print("Partial answer:", envelope.answer)
    print("Identification:", envelope.inspect().identification)
    print("Calibration:", envelope.calibration.status)
    assert envelope.answer.kind in {"bounds", "partial"}
    assert envelope.answer.value is None
    assert envelope.study.estimate().answer == envelope.answer

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
