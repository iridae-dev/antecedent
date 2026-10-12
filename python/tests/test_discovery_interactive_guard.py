"""discovery= + latency=interactive must fail closed (backlog D)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest

from _scm import confounded_scm


def test_discovery_plus_interactive_raises_unsupported():
    data, _edges = confounded_scm(n=400, seed=11)
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="interactive estimate path"):
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.PC(alpha=0.2, fdr=False, max_cond_size=2),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            latency="interactive",
            seed=1,
        )


def test_discovery_plus_standard_still_allowed():
    # A collider design (t -> y <- w, independent causes) that PC orients fully, so the
    # standard path must estimate: the Interactive guard does not block it, and the
    # estimate is the structural coefficient, not merely a finite number.
    rng = np.random.default_rng(13)
    n = 1000
    t = rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.5 * t + w + rng.normal(size=n) * 0.3
    result = antecedent.analyze(
        {"t": t, "y": y, "w": w},
        discovery=antecedent.discovery.PC(alpha=0.001, fdr=False, max_cond_size=2),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="standard",
        seed=1,
        refute=False,
        accept_discovered=True,
    )
    assert abs(result.ate - 1.5) < 4.0 * result.estimate.se_analytic


def test_interactive_with_supplied_graph_ok():
    data, edges = confounded_scm(n=400, seed=11)
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
