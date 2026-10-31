"""Unit-level change attribution reports a signed total and validates sample counts."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest


def _opposing_effects(n: int = 300, seed: int = 4):
    """Two parents of `y` whose per-unit contributions have opposite signs on average."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=3.0, size=n)
    b = rng.normal(loc=3.0, size=n)
    y = 2.0 * a - 1.5 * b + rng.normal(scale=0.05, size=n)
    return ["a", "b", "y"], [a, b, y], [("a", "y"), ("b", "y")]


def test_total_change_equals_signed_contribution_sum():
    names, cols, edges = _opposing_effects()
    result = antecedent.attribution.attribute_unit_change(
        names, cols, edges, "y", execution=antecedent.Execution(seed=1)
    )
    contributions = [value for _, value in result.contributions]
    assert abs(result.total_change - sum(contributions)) < 1e-12


@pytest.mark.parametrize(
    "call",
    [
        antecedent.attribution.attribute_distribution_change,
        antecedent.attribution.attribute_distribution_change_robust,
    ],
)
def test_zero_samples_refused(call):
    names, cols, edges = _opposing_effects()
    with pytest.raises(ValueError, match="n_samples"):
        call(names, cols, edges, "y", 0, 150, 150, 300, n_samples=0)
