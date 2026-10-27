"""Conditional / context effect OO dual of conformance/context/conditional_effect."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success


def test_conditional_effect_recovers_interaction():
    n = 200
    t = np.asarray([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    w = np.asarray([(i % 5) for i in range(n)], dtype=np.float64)
    y = 1.0 + 2.0 * t + 0.5 * t * w
    data = {"t": t, "y": y, "w": w}
    edges = [("t", "y"), ("w", "y")]
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.ConditionalEffect("t", "y", "w"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert abs(result.ate - 3.0) < 0.3


def test_conditional_bayesian_consults_prior():
    rng = np.random.default_rng(12)
    t, w = rng.normal(size=(2, 100))
    data = {"t": t, "w": w, "y": 2 * t + 0.5 * t * w + rng.normal(scale=0.2, size=100)}
    result = require_success(
        antecedent.analyze(
            data,
            graph=[("t", "y"), ("w", "y")],
            query=antecedent.ConditionalEffect("t", "y", "w"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=512),
            refute=False,
        )
    )
    assert result.posterior is not None
    assert require_success(antecedent.load(result.export(quantity="posterior"))).n_draws == 512
    assert result.ate == pytest.approx(2 + 0.5 * w.mean(), abs=0.1)
    assert result.estimate.estimator_id == "conditional.bayesian"


def test_handle_conditional_bayesian_uses_staged_estimator():
    """analyze is prepare().estimate(); Bayesian ConditionalEffect stays licensed."""
    rng = np.random.default_rng(12)
    t, w = rng.normal(size=(2, 100))
    data = {"t": t, "w": w, "y": 2 * t + 0.5 * t * w + rng.normal(scale=0.2, size=100)}
    result = require_success(
        antecedent.analyze(
            data,
            query=antecedent.ConditionalEffect("t", "y", "w"),
            graph=antecedent.Dag.from_edges(["t", "y", "w"], [("t", "y"), ("w", "y")]),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=512),
            refute=False,
            execution=antecedent.Execution(seed=1, threads=1),
        )
    )
    assert result.estimate.estimator_id == "conditional.bayesian"
    assert result.posterior is not None
    assert np.isfinite(result.ate)


def test_the_frequentist_conditional_effect_is_the_linear_adjustment_with_one_modifier():
    n = 200
    t = np.asarray([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    w = np.asarray([(i % 5) for i in range(n)], dtype=np.float64)
    y = 1.0 + 2.0 * t + 0.5 * t * w
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "w": w},
            graph=[("t", "y"), ("w", "y")],
            query=antecedent.ConditionalEffect("t", "y", "w"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.estimate.estimator_id == "conditional.linear.adjustment"
    # The effect at the mean modifier: 2 + 0.5 * mean(w) = 3 for this noise-free design.
    assert result.ate == pytest.approx(2.0 + 0.5 * float(w.mean()), abs=0.1)
    # The regression is fitted for one modifier: a modifier set is refused with a remedy.
    with pytest.raises(CausalUnsupportedError) as refusal:
        antecedent.ConditionalEffect("t", "y", ["w", "z"])
    assert refusal.value.remedy
