"""A live discovery configuration never enters preparation; executed discovery feeds any profile."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _scm import confounded_scm


@pytest.mark.parametrize("profile", ["interactive", "standard"])
def test_discovery_plus_interactive_raises_unsupported(profile):
    data, _edges = confounded_scm(n=400, seed=11)
    with pytest.raises(antecedent.errors.CausalTypeError, match="executed discovery result"):
        require_success(
            antecedent.analyze(
                data,
                discovery=antecedent.discovery.PC(alpha=0.2, fdr=False, max_cond_size=2),
                query=antecedent.AverageEffect(treatment="t", outcome="y"),
                inference=antecedent.Frequentist(profile=profile),
                execution=antecedent.Execution(seed=1),
            )
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
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "w": w},
            graph=antecedent.discovery.discover(
                {"t": t, "y": y, "w": w},
                method=antecedent.discovery.PC(alpha=0.001, fdr=False, max_cond_size=2),
                execution=antecedent.Execution(seed=1),
                accept_discovered=True,
            ),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(profile="standard"),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )
    assert abs(result.ate - 1.5) < 4.0 * result.estimate.se_analytic


def test_interactive_with_supplied_graph_ok():
    data, edges = confounded_scm(n=400, seed=11)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(profile="interactive"),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert math.isfinite(result.ate)
    assert abs(result.ate - 2.0) < 0.5
    assert result.performance.latency_mode == "interactive"
