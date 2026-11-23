"""Second-click refute via PreparedAnalysis (BACKLOG E)."""

from __future__ import annotations

import math
import random

import antecedent
import numpy as np
from antecedent.outcomes import require_success


def _confounded(n: int = 400, seed: int = 11):
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
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_prepared_refute_second_click():
    data, edges = _confounded()
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0, profile="interactive"),
        execution=antecedent.Execution(seed=5),
    )
    first_outcome = antecedent.estimate(prepared)
    first = require_success(first_outcome)
    assert not first.validation.ran
    ate = first.ate

    second_outcome = first_outcome.study.refute(suite="placebo")
    second = require_success(second_outcome)
    assert abs(second.ate - ate) < 1e-12
    assert second.validation.ran
    assert second.validation.count >= 1
    # The completed first estimate is untouched by the deferred suite.
    assert not first.validation.ran
    assert second_outcome.study is not first_outcome.study
