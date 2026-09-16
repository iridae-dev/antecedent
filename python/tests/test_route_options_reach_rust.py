"""Options the Rust study honours reach it on every route that licenses them.

A query field or ``estimator_config`` key that one prepare route forwards and
another drops (or refuses although Rust would honour it) is a capability the
Python API does not reach. Each test changes one option and checks that the
executed study changed accordingly.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import DbnPosterior


def _series(n: int = 600, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.9 * x[i - 1] + 0.3 * rng.normal()
    return {"x": x, "y": y}


@pytest.mark.parametrize(
    ("query_type", "options"),
    [
        (ant.PulseEffect, {"graph": [("x", 1, "y", 0)]}),
        (ant.SustainedEffect, {"graph": [("x", 1, "y", 0)]}),
        (ant.PulseEffect, {"graph": [("x", 1, "y", 0)], "inference": ant.Bayesian(n_draws=200)}),
        (
            ant.PulseEffect,
            {"discovery": DbnPosterior(max_lag=1), "inference": ant.Bayesian(n_draws=200)},
        ),
    ],
    ids=["pulse", "sustained", "pulse-bayesian", "pulse-dbn-posterior"],
)
def test_temporal_control_level_sets_the_contrast(query_type: Any, options: dict) -> None:
    """A linear lagged effect scales with ``active_level - control_level``."""
    data = _series()

    def effect(control: float) -> float:
        query = query_type("x", "y", treatment_lag=1, control_level=control, active_level=1.0)
        return float(ant.analyze(data, query=query, refute=False, seed=3, **options).ate)

    full, half = effect(0.0), effect(0.5)
    assert full == pytest.approx(0.9, abs=0.1)
    assert half == pytest.approx(full / 2, rel=0.05)
