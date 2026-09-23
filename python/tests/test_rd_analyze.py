"""Regression discontinuity via analyze(..., estimator='rd.sharp')."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest


def test_rd_sharp_via_analyze():
    rng = np.random.default_rng(25)
    n = 3000
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    data = {"t": t, "y": y, "r": r}
    # The design as a graph: the running variable is the treatment's only cause.
    result = antecedent.analyze(
        data,
        graph=[("r", "t"), ("t", "y"), ("r", "y")],
        query=antecedent.AverageEffect("t", "y"),
        estimator="rd.sharp",
        identifier="rd.sharp",
        running_variable="r",
        cutoff=0.0,
        bandwidth=1.5,
        refute=False,
        bootstrap=0,
        seed=26,
    )
    assert abs(result.ate - 2.0) < 0.35
    assert result.estimate.estimator_id in ("rd.sharp", "rd.sharp.local_linear", "")
    # The jump is the effect for units at the cutoff; the result says so rather than
    # presenting it as a population average effect.
    target = result.inspect().to_dict()["target"]["query"]
    assert "local_at_cutoff" in str(target["target_population"])


def test_rd_sharp_refuses_a_treatment_column_that_breaks_the_rule():
    rng = np.random.default_rng(27)
    n = 3000
    r = rng.uniform(-2.0, 2.0, size=n)
    # Imperfect compliance: the outcome jump is an intent-to-treat contrast, not the
    # effect of `t`.
    t = (rng.uniform(size=n) < np.where(r >= 0.0, 0.75, 0.25)).astype(np.float64)
    y = 1.0 + 3.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    with pytest.raises(Exception, match="not the threshold rule"):
        antecedent.analyze(
            {"t": t, "y": y, "r": r},
            graph=[("r", "t"), ("t", "y"), ("r", "y")],
            query=antecedent.AverageEffect("t", "y"),
            estimator="rd.sharp",
            identifier="rd.sharp",
            running_variable="r",
            cutoff=0.0,
            bandwidth=1.5,
            refute=False,
            bootstrap=0,
            seed=28,
        )
