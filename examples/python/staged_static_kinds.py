#!/usr/bin/env python3
"""Staged MediationEffect and Counterfactual on a confounded linear SCM.

Requires a built antecedent extension (`maturin develop` in python/).

Z confounds A, M, and Y. For control 0.2 vs active 0.8 the structural
contrasts are NDE=1.8, NIE=4.8, total/mean ITE=6.6. Unadjusted parent
regressions that omit Z are not those numbers.
"""

from __future__ import annotations

import numpy as np
from antecedent import Counterfactual, MediationEffect, analyze


def confounded_scm(n: int = 500) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    i = np.arange(n, dtype=float)
    z = np.cos(i * 0.41)
    a = np.sin(i * 0.71) + 0.4 * z
    m = 2 * a + 0.5 * z + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 5 * z + 0.1 * np.sin(i * 0.31)
    graph = [
        ("a", "m"),
        ("a", "y"),
        ("m", "y"),
        ("z", "a"),
        ("z", "m"),
        ("z", "y"),
    ]
    return {"a": a, "m": m, "y": y, "z": z}, graph


def main() -> None:
    data, graph = confounded_scm()
    control, active = 0.2, 0.8
    for contrast, expected in (
        ("natural_direct", 1.8),
        ("natural_indirect", 4.8),
        ("total", 6.6),
    ):
        result = analyze(
            data,
            graph=graph,
            query=MediationEffect(
                "a",
                "y",
                mediators=["m"],
                contrast=contrast,
                control_level=control,
                active_level=active,
            ),
            refute="none",
            bootstrap=0,
        )
        print(f"{contrast}={result.effect:.4f} estimator={result.estimate.estimator_id}")
        assert abs(result.effect - expected) < 0.03, result.effect
        assert result.estimate.estimator_id == "mediation.linear"

    ite = analyze(
        data,
        graph=graph,
        query=Counterfactual("a", "y", control_level=control, active_level=active),
        refute="none",
    )
    print(
        f"mean_ite={ite.mean_ite:.4f} n={len(ite.unit_effects)} "
        f"estimator={ite.estimate.estimator_id}"
    )
    assert abs(ite.mean_ite - 6.6) < 0.03, ite.mean_ite
    assert ite.estimate.estimator_id == "gcm.fit"
    assert len(ite.unit_effects) == len(data["a"])


if __name__ == "__main__":
    main()
