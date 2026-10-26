"""Population registry + custom-distribution IPW via analyze()."""

from __future__ import annotations

import antecedent
from antecedent.outcomes import require_success

from _scm import confounded_scm


def test_custom_distribution_ipw_via_analyze():
    data, _edges = confounded_scm(n=1200, seed=21)
    n = len(data["t"])
    weights = [1.0] * n
    for i in range(n // 2):
        weights[i] = 0.5
    reg = antecedent.population.PopulationRegistry()
    reg.insert_distribution(7, weights)
    result = require_success(
        antecedent.analyze(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=antecedent.AverageEffect(
                treatment="t",
                outcome="y",
                target_population=antecedent.population.CustomDistribution(7),
            ),
            population_registry=reg,
            estimator="propensity.weighting",
            refute=False,
            bootstrap=0,
            seed=11,
        )
    )
    assert abs(result.ate - 2.0) < 0.35
