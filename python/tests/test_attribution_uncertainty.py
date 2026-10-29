"""Posterior over the mechanisms and refit bootstrap for the fitted-model attributions.

Two periods of ``x -> y`` with ``y = intercept + 2x + noise`` and the intercept shifting by 5
in the second period, so the change of ``y`` between the periods is 5 up to sampling noise.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.attribution import (
    AttributionWithUncertainty,
    anomaly_attribution,
    attribute_distribution_change,
    attribute_distribution_change_robust,
    attribute_feature_relevance,
    attribute_path_specific,
    attribute_root_causes,
    attribute_structure_change,
    attribute_unit_change,
)

EDGES = [("x", "y")]


def _data(n: int = 200, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    shift = np.where(np.arange(n) < n // 2, 1.0, 6.0)
    y = shift + 2.0 * x + 0.5 * rng.normal(size=n)
    return ["x", "y"], [x, y]


def _change_args():
    names, cols = _data()
    return names, cols, EDGES, "y", 0, 100, 100, 200


def _period_oracle():
    """Difference of the period means of y and the standard error of that difference."""
    _, (_, y) = _data()
    first, second = y[:100], y[100:]
    return (
        float(second.mean() - first.mean()),
        float(np.sqrt(first.var(ddof=1) / 100 + second.var(ddof=1) / 100)),
    )


def test_point_results_are_unchanged_without_an_uncertainty_option():
    result = attribute_distribution_change(
        *_change_args(), n_samples=300, execution=antecedent.Execution(seed=2)
    )
    assert not isinstance(result, AttributionWithUncertainty)
    change, _ = _period_oracle()
    # The change of y between the periods is the difference of the period means (about 5.6
    # here, 5 up to sampling noise); the sampled attribution is within Monte Carlo error.
    assert result.total_change == pytest.approx(change, abs=0.05)


def test_change_attribution_posterior_and_bootstrap_agree_on_the_total():
    bayes = attribute_distribution_change(
        *_change_args(), n_samples=300, execution=antecedent.Execution(seed=2), posterior_draws=24
    )
    assert isinstance(bayes, AttributionWithUncertainty)
    posterior = bayes.uncertainty.posterior
    assert posterior is not None and posterior.draws == 24
    mean, sd, low, high = posterior.quantity("total_change")
    change, standard_error = _period_oracle()
    assert mean == pytest.approx(change, abs=0.15) and low < mean < high
    # The spread of the total is the sampling error of a difference of two period means.
    assert 0.5 * standard_error < sd < 1.5 * standard_error
    assert low < change < high
    assert bayes.uncertainty.bootstrap is None
    assert bayes.uncertainty.point["total_change"] == pytest.approx(bayes.result.total_change)

    freq = attribute_distribution_change(
        *_change_args(),
        n_samples=300,
        execution=antecedent.Execution(seed=2),
        bootstrap_replicates=30,
    )
    bootstrap = freq.uncertainty.bootstrap
    assert bootstrap is not None and bootstrap.replicates_ok >= 15
    se = bootstrap.se["total_change"]
    assert se is not None and 0.5 * standard_error < se < 1.5 * standard_error
    assert freq.uncertainty.posterior is None
    assert 0.3 < sd / se < 3.0


@pytest.mark.parametrize("draws", [0, 12])
def test_every_other_attribution_carries_the_requested_uncertainty(draws):
    names, cols = _data()
    options = {"posterior_draws": draws} if draws else {"bootstrap_replicates": 12}
    results = {
        "robust": attribute_distribution_change_robust(
            names, cols, EDGES, "y", 0, 100, 100, 200, n_samples=200, **options
        ),
        "structure": attribute_structure_change(
            names, cols, EDGES, [], "y", 0, 100, 100, 200, n_samples=200, **options
        ),
        "paths": attribute_path_specific(names, cols, EDGES, "x", "y", **options),
        "features": attribute_feature_relevance(names, cols, EDGES, "y", n_samples=100, **options),
        "anomaly": anomaly_attribution(names, cols, EDGES, ["y"], **options),
        "unit": attribute_unit_change(names, cols, EDGES, "y", **options),
        "root": attribute_root_causes(
            names, cols, EDGES, "y", 0, 100, 100, 200, n_samples=200, **options
        ),
    }
    for name, result in results.items():
        assert isinstance(result, AttributionWithUncertainty), name
        uncertainty = result.uncertainty
        if draws:
            assert uncertainty.posterior is not None and uncertainty.posterior.draws == 12, name
            assert all(np.isfinite(uncertainty.posterior.mean)), name
        else:
            assert uncertainty.bootstrap is not None, name
            assert uncertainty.bootstrap.replicates_ok >= 6, name
        assert uncertainty.point, name


def test_the_options_exclude_each_other_and_mechanism_change_has_no_posterior():
    with pytest.raises(
        ValueError, match="ask for posterior_draws or bootstrap_replicates, not both"
    ):
        attribute_distribution_change(*_change_args(), posterior_draws=8, bootstrap_replicates=8)
    names, cols = _data()
    with pytest.raises(ValueError, match="mechanism-change"):
        antecedent.attribution.attribution_uncertainty(
            "mechanism_change", names, cols, EDGES, {}, posterior_draws=8
        )
