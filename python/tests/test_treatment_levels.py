"""Declared treatment levels of the propensity estimators.

The query's ``active_level`` and ``control_level`` pick the two arms of a propensity contrast
whatever the treatment column's own coding: ``1/2``, reversed and arbitrary declared values
estimate the same contrast as ``0/1`` (negated when active and control swap). A row at neither
declared level is refused with ``treatment_level_undeclared``; nothing is dropped silently.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError
from antecedent.estimators import Aipw, PropensityMatching, PropensityWeighting

from _refusal import assert_registered_refusal

STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y")]


def _binary(n: int = 1200, seed: int = 4) -> dict[str, np.ndarray]:
    """``P(T=1|z) = expit(-0.5 + z)`` and ``Y = 2 T + z + noise``: the effect is 2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.5 + z)))).astype(float)
    y = 2.0 * t + z + 0.5 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _estimate(data: dict[str, np.ndarray], estimator: Any, *, control: float, active: float) -> Any:
    return ant.analyze(
        data,
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y", control_level=control, active_level=active),
        estimator=estimator,
        refute="none",
        bootstrap=0,
        seed=3,
    )


@pytest.mark.parametrize("estimator", [Aipw, PropensityWeighting, PropensityMatching])
def test_a_one_two_coding_gives_the_zero_one_answer(estimator: Any) -> None:
    data = _binary()
    shifted = {**data, "t": data["t"] + 1.0}
    base = _estimate(data, estimator(bootstrap=0), control=0.0, active=1.0)
    recoded = _estimate(shifted, estimator(bootstrap=0), control=1.0, active=2.0)
    assert recoded.answer.value == base.answer.value
    assert abs(base.answer.value - 2.0) < 0.4


def test_swapping_active_and_control_flips_the_sign() -> None:
    data = _binary()
    forward = _estimate(data, PropensityWeighting(bootstrap=0), control=0.0, active=1.0)
    reversed_ = _estimate(data, PropensityWeighting(bootstrap=0), control=1.0, active=0.0)
    assert abs(forward.answer.value + reversed_.answer.value) < 1e-9


def test_rows_at_an_undeclared_level_are_refused_not_dropped() -> None:
    data = _binary()
    data["t"] = data["t"].copy()
    data["t"][:5] = 2.0
    with pytest.raises(CausalError) as raised:
        _estimate(data, Aipw(bootstrap=0), control=0.0, active=1.0)
    error = raised.value
    assert error.reason_code == "treatment_level_undeclared"
    assert_registered_refusal(error)
    assert "5 complete-case rows" in str(error)


def _three_level(n: int = 1500, seed: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    level = rng.integers(0, 3, size=n).astype(float)
    level[z > 0.5] = np.where(rng.uniform(size=int((z > 0.5).sum())) < 0.6, 2.0, level[z > 0.5])
    y = 2.0 * (level == 2.0) + (level == 1.0) + z + 0.5 * rng.normal(size=n)
    return {"t": level, "y": y, "z": z}


def test_a_declared_exclusion_equals_dropping_the_rows_by_hand() -> None:
    data = _three_level()
    declared = _estimate(
        data,
        PropensityWeighting(bootstrap=0, other_levels="exclude_undeclared"),
        control=0.0,
        active=2.0,
    )
    keep = data["t"] != 1.0
    by_hand = _estimate(
        {name: values[keep] for name, values in data.items()},
        PropensityWeighting(bootstrap=0),
        control=0.0,
        active=2.0,
    )
    assert declared.answer.value == by_hand.answer.value


def test_the_default_refuses_and_a_bad_policy_is_rejected() -> None:
    data = _three_level()
    with pytest.raises(CausalError) as raised:
        _estimate(data, PropensityWeighting(bootstrap=0), control=0.0, active=2.0)
    assert raised.value.reason_code == "treatment_level_undeclared"
    with pytest.raises(ValueError, match="other_levels"):
        PropensityWeighting(other_levels="drop")._wire()  # type: ignore[arg-type]
