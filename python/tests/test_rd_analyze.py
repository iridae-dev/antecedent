"""Sharp regression discontinuity via the quasi.SharpRegressionDiscontinuity query."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.quasi import SharpRegressionDiscontinuity


def test_rd_sharp_via_query():
    rng = np.random.default_rng(25)
    n = 3000
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    data = {"t": t, "y": y, "r": r}
    # No graph, no estimator, no kwargs: the fixed sharp-RD design is synthesized.
    result = antecedent.analyze(
        data,
        query=SharpRegressionDiscontinuity("y", "t", "r", 0.0, 1.5),
        seed=26,
    )
    assert abs(result.answer.value - 2.0) < 0.35
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
    # effect of `t`. Sharp RD refuses it (use FuzzyRegressionDiscontinuity instead).
    t = (rng.uniform(size=n) < np.where(r >= 0.0, 0.75, 0.25)).astype(np.float64)
    y = 1.0 + 3.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    with pytest.raises(Exception, match="not the threshold rule"):
        antecedent.analyze(
            {"t": t, "y": y, "r": r},
            query=SharpRegressionDiscontinuity("y", "t", "r", 0.0, 1.5),
            seed=28,
        )


def test_sharp_rd_se_kind_reaches_the_estimator():
    rng = np.random.default_rng(25)
    n = 1500
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    data = {"t": t, "y": y, "r": r}

    def fit(se):
        return antecedent.analyze(
            data, query=SharpRegressionDiscontinuity("y", "t", "r", 0.0, 1.5, se=se), seed=26
        ).estimate

    default, hc1, homoskedastic = fit(None), fit("hc1"), fit("homoskedastic")
    assert default.se_analytic == hc1.se_analytic
    assert homoskedastic.ate == hc1.ate
    assert homoskedastic.se_analytic != hc1.se_analytic
    with pytest.raises(Exception, match="se must be one of"):
        SharpRegressionDiscontinuity("y", "t", "r", 0.0, 1.5, se="cluster")


def test_sharp_rd_estimator_spelling_points_to_the_query():
    data = {"t": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "r": np.array([-1.0, 1.0])}
    with pytest.raises(CausalUnsupportedError, match="SharpRegressionDiscontinuity"):
        antecedent.analyze(
            data,
            graph=[("r", "t"), ("t", "y"), ("r", "y")],
            query=antecedent.AverageEffect("t", "y"),
            estimator="rd.sharp",
        )
