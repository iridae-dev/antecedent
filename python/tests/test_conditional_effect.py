"""Conditional / context effect OO dual of conformance/context/conditional_effect."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalValueError
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
    # One modifier carries no modifier-set interval assumption.
    assert not any(
        "conditional.joint_modifier_interval_uncalibrated" in str(a)
        for a in result.assumptions or []
    )


_JOINT_ASSUMPTION = "conditional.joint_modifier_interval_uncalibrated"


def _joint_modifier_table(n: int = 4000, seed: int = 7):
    # y = 1 + t * (1 + 0.5 w1 - 0.3 w2) + noise, with t randomized.
    rng = np.random.default_rng(seed)
    w1 = rng.normal(1.0, 1.0, n)
    w2 = rng.normal(-0.5, 1.0, n)
    t = rng.integers(0, 2, n).astype(np.float64)
    y = 1.0 + t * (1.0 + 0.5 * w1 - 0.3 * w2) + rng.normal(scale=0.3, size=n)
    return {"t": t, "y": y, "w1": w1, "w2": w2}


def _analyze_conditional(data, modifier):
    return require_success(
        antecedent.analyze(
            data,
            graph=[("t", "y"), ("w1", "y"), ("w2", "y")],
            query=antecedent.ConditionalEffect("t", "y", modifier),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )


def test_conditional_effect_over_a_modifier_set_recovers_the_joint_interaction():
    data = _joint_modifier_table()
    result = _analyze_conditional(data, ["w1", "w2"])
    assert result.estimate.estimator_id == "conditional.linear.adjustment"
    # The published point is the effect at the modifier means:
    # 1 + 0.5 * mean(w1) - 0.3 * mean(w2).
    truth = 1.0 + 0.5 * float(data["w1"].mean()) - 0.3 * float(data["w2"].mean())
    assert result.ate == pytest.approx(truth, abs=0.05)
    # The interval's coverage is measured for one modifier only: the set says so.
    assert any(_JOINT_ASSUMPTION in str(a) for a in result.assumptions or [])


def test_conditional_effect_modifier_order_does_not_change_the_effect():
    data = _joint_modifier_table(n=2000, seed=3)
    forward = _analyze_conditional(data, ["w1", "w2"])
    backward = _analyze_conditional(data, ("w2", "w1"))
    assert backward.ate == pytest.approx(forward.ate, abs=1e-9)


def test_conditional_effect_over_one_modifier_is_unchanged_by_a_singleton_sequence():
    data = _joint_modifier_table(n=2000, seed=5)
    bare = _analyze_conditional(data, "w1")
    wrapped = _analyze_conditional(data, ["w1"])
    assert wrapped.ate == pytest.approx(bare.ate, abs=1e-12)
    assert not any(_JOINT_ASSUMPTION in str(a) for a in bare.assumptions or [])
    assert not any(_JOINT_ASSUMPTION in str(a) for a in wrapped.assumptions or [])


def test_conditional_effect_refuses_empty_duplicate_and_non_string_modifiers():
    for bad in ([], ["w1", "w1"], ["w1", 2], [""], 7):
        with pytest.raises(CausalValueError):
            antecedent.ConditionalEffect("t", "y", bad)
    query = antecedent.ConditionalEffect("t", "y", ["w1", "w2"])
    assert query.modifiers == ("w1", "w2")


def test_conditional_effect_over_a_modifier_set_identifies_through_the_structure_route():
    graph = antecedent.Dag.from_edges(
        ["t", "y", "w1", "w2"], [("t", "y"), ("w1", "y"), ("w2", "y")]
    )
    identified = antecedent.identify(
        graph=graph, query=antecedent.ConditionalEffect("t", "y", ["w1", "w2"])
    )
    baseline = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
    assert identified.status == baseline.status
    # The statement names the whole modifier set.
    assert "w1" in identified.statement and "w2" in identified.statement
