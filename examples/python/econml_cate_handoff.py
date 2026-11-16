#!/usr/bin/env python3
"""Hand an identified adjustment set to EconML for conditional effects.

The simulation has a confounder z of a binary treatment t whose effect on y is 2. Antecedent
identifies the backdoor adjustment set and estimates the average effect;
``antecedent.handoff.econml`` exports the set as the Y, T and W columns a caller passes to an
EconML learner. The script asserts the set {z} and an effect within 0.35 of 2. The EconML fit
is the caller's: its output is printed, not asserted, and Antecedent makes no claim about it.
``econml`` is optional: without it only the Antecedent half runs.

Run: ``python examples/python/econml_cate_handoff.py`` (optionally after ``pip install econml``).
"""

from __future__ import annotations

import math
import random

import antecedent as _ant_api
import numpy as np
from antecedent import AverageEffect, Dag, analyze, identify
from antecedent.handoff import econml

TRUE_ATE = 2.0

# Populated by ``main`` for tests that run this file via runpy.
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
        inference=_ant_api.Frequentist(bootstrap=0),
        execution=_ant_api.Execution(seed=11),
    )
    assert result.answer.kind == "point"
    ate = float(result.answer.value)
    print(f"Antecedent ATE={ate:.4f} (truth={TRUE_ATE})")

    spec = econml(result.value)
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

    # The fit below belongs to the caller and to EconML; Antecedent supplied only the
    # adjustment columns.
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
