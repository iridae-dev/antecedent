#!/usr/bin/env python3
"""Hand off Antecedent's adjustment set to EconML for CATE.

Antecedent identifies a backdoor adjustment set on a small confounded SCM and
estimates the ATE. ``antecedent.handoff.econml`` exports that set; callers fit
EconML themselves. Soft-depends on ``econml``: without it, the Antecedent half
still runs and asserts. Install Antecedent with ``python -m pip install
antecedent``; optionally ``python -m pip install econml`` for the CATE half.
"""

from __future__ import annotations

import math
import random

import numpy as np
from antecedent import AverageEffect, Dag, analyze, identify
from antecedent.handoff import econml

TRUE_ATE = 2.0

# Populated by ``main`` for earning tests that run this file via runpy.
LAST_HANDOFF: dict[str, object] = {}


def _confounded_scm(n: int = 800, seed: int = 11) -> dict[str, np.ndarray]:
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


def main() -> None:
    data = _confounded_scm()
    graph = Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    query = AverageEffect(treatment="t", outcome="y")

    identified = identify(graph=graph, query=query)
    print(
        f"Identify: status={identified.status} "
        f"method={identified.method} adjustment={list(identified.adjustment_set)}"
    )

    result = analyze(
        data,
        graph=graph,
        query=query,
        refute=False,
        bootstrap=0,
        seed=11,
    )
    assert result.answer.kind == "point"
    ate = float(result.answer.value)
    print(f"Antecedent ATE={ate:.4f} (truth={TRUE_ATE})")

    spec = econml(result)
    print(
        f"Handoff: treatment={spec.treatment!r} outcome={spec.outcome!r} "
        f"confounders={spec.confounders} identifier={spec.identifier} "
        f"status={spec.status}"
    )
    assert spec.confounders == ("z",), spec.confounders
    assert abs(ate - TRUE_ATE) < 0.35, ate
    cols = spec.columns(data)
    assert cols["W"] is not None and cols["W"].shape[1] == 1

    LAST_HANDOFF.clear()
    LAST_HANDOFF.update(
        confounders=spec.confounders,
        ate=ate,
        identifier=spec.identifier,
    )

    try:
        from econml.dml import LinearDML
    except ImportError:
        print("econml not installed; skipping LinearDML CATE half")
        return

    # Caller-owned fit: Antecedent only supplied the adjustment columns.
    learner = LinearDML(random_state=11)
    learner.fit(Y=cols["Y"], T=cols["T"], X=None, W=cols["W"])
    cate_mean = float(np.asarray(learner.const_marginal_ate()).reshape(-1)[0])
    print(f"EconML LinearDML CATE (const marginal ATE)={cate_mean:.4f}")
    print(
        f"Summary: Antecedent ATE={ate:.4f} vs EconML CATE={cate_mean:.4f} "
        f"on W={list(spec.confounders)}"
    )
    LAST_HANDOFF["econml_cate"] = cate_mean


if __name__ == "__main__":
    main()
