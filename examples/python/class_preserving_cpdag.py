#!/usr/bin/env python3
"""What is identified when one edge direction is unknown?

A CPDAG stands for every DAG in one equivalence class. With the z - t edge left undirected,
z is a confounder in one completion and a mediator in the other, so the effect of t on y is
partially identified and the answer carries no point. Supplying the direction z -> t identifies the
effect with adjustment set {z}, which is passed to an EconML handoff specification. The graph
object remains a Cpdag in both cases.

The fixture is noise-free with structural effect 2. The script asserts the partial status,
the absent point and the adjustment set; it does not assert the estimate. The orientation is
a reviewer's declaration, not learned from the data.

Run: ``python examples/python/class_preserving_cpdag.py``."""

from __future__ import annotations

import antecedent as _ant_api
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

    envelope = analyze(
        data,
        graph=partial,
        query=query,
        refute="none",
        inference=_ant_api.Frequentist(bootstrap=0),
    )
    print("Partial answer:", envelope.answer)
    print("Identification:", envelope.inspect().identification)
    print("Calibration:", envelope.value.calibration.status)
    assert envelope.answer.kind in {"bounds", "partial"}
    assert envelope.answer.value is None
    assert _ant_api.estimate(envelope.study).answer == envelope.answer

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
