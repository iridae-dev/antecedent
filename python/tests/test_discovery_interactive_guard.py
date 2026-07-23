"""discovery= + latency=interactive must fail closed (backlog D)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 400, seed: int = 11):
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


def test_discovery_plus_interactive_raises_unsupported():
    data, _edges = _confounded_scm()
    with pytest.raises(antecedent.CausalUnsupportedError, match="interactive estimate path"):
        antecedent.analyze(
            data,
            discovery=antecedent.PC(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            latency="interactive",
            seed=1,
        )


def test_discovery_plus_standard_still_allowed():
    data, _edges = _confounded_scm(n=500, seed=13)
    # May Ready-estimate or fail closed on incomplete CPDAG; both prove the path
    # is not blocked by the Interactive guard.
    try:
        result = antecedent.analyze(
            data,
            discovery=antecedent.PC(alpha=0.5, fdr=False, max_cond_size=0),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            latency="standard",
            seed=1,
            refute=False,
            accept_discovered=True,
        )
        assert math.isfinite(result.ate)
    except (antecedent.CausalReviewError, ValueError, antecedent.CausalUnsupportedError) as exc:
        # Incomplete auto-accept / review is fine; Interactive-style refuse is not.
        assert "interactive estimate path" not in str(exc).lower()


def test_interactive_with_supplied_graph_ok():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        seed=1,
    )
    assert math.isfinite(result.ate)
    assert abs(result.ate - 2.0) < 0.5
    assert result.performance.latency_mode == "interactive"
