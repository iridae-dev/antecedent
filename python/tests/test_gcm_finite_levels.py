"""Non-finite intervention levels and contrast widths are refused at the binding."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest


def _model(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    names, cols, edges = ["x", "y"], [x, y], [("x", "y")]
    return names, cols, edges, antecedent.counterfactual.fit_gcm(names, cols, edges)


@pytest.mark.parametrize("level", [float("nan"), float("inf")])
def test_sample_do_refuses_a_non_finite_level(level):
    *_, fitted = _model()
    with pytest.raises(ValueError, match="finite"):
        fitted.sample_do({"x": level}, 10)


@pytest.mark.parametrize("level", [float("nan"), float("-inf")])
def test_counterfactual_ite_refuses_a_non_finite_arm(level):
    names, cols, edges, fitted = _model()
    with pytest.raises(ValueError, match="finite"):
        fitted.counterfactual_ite("x", "y", level, 0.0)
    with pytest.raises(ValueError, match="finite"):
        antecedent.counterfactual.counterfactual_ite(names, cols, edges, "x", "y", 1.0, level)


@pytest.mark.parametrize("delta", [float("nan"), 0.0])
def test_population_do_contrast_refuses_an_undefined_width(delta):
    *_, fitted = _model()
    with pytest.raises(ValueError, match="delta"):
        fitted.population_do_contrast("x", "y", delta)
