"""Native CausalForest analyze path."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.estimators import CausalForest
from antecedent.outcomes import require_success


def test_causal_forest_returns_cate():
    rng = np.random.default_rng(5)
    n = 280
    z = rng.normal(size=n)
    t = (0.4 * z + rng.normal(size=n) * 0.8 > 0).astype(float)
    y = (1.0 + z) * t + 0.3 * z + rng.normal(size=n) * 0.35
    data = {"t": t, "y": y, "z": z}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    result = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            estimator=CausalForest(n_trees=80, min_leaf=8, max_depth=4),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=3),
        )
    )
    assert result.estimate.estimator_id == "causal.forest"
    assert result.estimate.cate is not None
    assert len(result.estimate.cate) == n
    # A forest publishes a leaf-dispersion diagnostic, never a pointwise SE.
    assert result.estimate.cate_se is None
    assert result.estimate.cate_leaf_dispersion is not None
    assert len(result.estimate.cate_leaf_dispersion) == n
    assert all(d >= 0.0 for d in result.estimate.cate_leaf_dispersion)
    # The structural CATE is 1 + z (population ATE 1; the sample average of 1 + z is 0.974).
    # The forest's ATE sits within 0.11 of that (observed 0.871); a constant or unadjusted
    # answer does not recover the heterogeneity checked next.
    assert abs(result.ate - float(np.mean(1.0 + z))) < 0.2
    cate = np.asarray(result.estimate.cate)
    assert np.corrcoef(cate, 1.0 + z)[0, 1] > 0.9  # observed 0.976
    assert float(np.sqrt(np.mean((cate - (1.0 + z)) ** 2))) < 0.4  # observed 0.243
