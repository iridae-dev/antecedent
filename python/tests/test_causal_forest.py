"""Native CausalForest analyze path."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.estimators import CausalForest


def test_causal_forest_returns_cate():
    rng = np.random.default_rng(5)
    n = 280
    z = rng.normal(size=n)
    t = (0.4 * z + rng.normal(size=n) * 0.8 > 0).astype(float)
    y = (1.0 + z) * t + 0.3 * z + rng.normal(size=n) * 0.35
    data = {"t": t, "y": y, "z": z}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    result = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=CausalForest(n_trees=80, min_leaf=8, max_depth=4),
        bootstrap=0,
        refute=False,
        seed=3,
    )
    assert result.estimate.estimator_id == "causal.forest"
    assert result.estimate.cate is not None
    assert len(result.estimate.cate) == n
    # A forest publishes a leaf-dispersion diagnostic, never a pointwise SE.
    assert result.estimate.cate_se is None
    assert result.estimate.cate_leaf_dispersion is not None
    assert len(result.estimate.cate_leaf_dispersion) == n
    assert all(d >= 0.0 for d in result.estimate.cate_leaf_dispersion)
    assert abs(result.ate - 1.0) < 0.4
