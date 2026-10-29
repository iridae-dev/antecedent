"""A pulse far outside the observed treatment range is flagged, then refused past a bound."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _series(n: int = 160) -> dict[str, np.ndarray]:
    t = np.sin(0.7 * np.arange(n))
    y = np.zeros(n)
    y[1:] = 2.0 * t[:-1] + 0.1 * np.sin(0.13 * np.arange(1, n))
    return {"t": t, "y": y}


def _pulse(level: float, execution: ant.Execution | None = None):
    return ant.analyze(
        _series(),
        graph=[("t", 1, "y", 0)],
        query=ant.PulseEffect("t", "y", treatment_lag=1, active_level=level),
        inference=ant.Frequentist(bootstrap=0),
        execution=execution or ant.Execution(seed=5),
    )


def test_a_level_inside_the_range_has_no_extrapolation_flag() -> None:
    assert require_success(_pulse(0.5)) is not None


def test_a_level_beyond_the_default_bound_is_refused_with_a_next_attempt() -> None:
    with pytest.raises(ant.errors.CausalError) as caught:
        _pulse(1000.0)
    assert "reason=positivity_violated" in str(caught.value)
    assert "Next attempt" in str(caught.value)


def test_the_bound_is_a_declared_execution_policy() -> None:
    wide = ant.Execution(seed=5, pulse_extrapolation_ranges=1000.0)
    assert require_success(_pulse(1000.0, wide)) is not None
    tight = ant.Execution(seed=5, pulse_extrapolation_ranges=1.0)
    with pytest.raises(ant.errors.CausalError):
        _pulse(7.0, tight)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_an_invalid_bound_is_refused(bad: float) -> None:
    with pytest.raises(ant.errors.CausalValueError):
        ant.Execution(pulse_extrapolation_ranges=bad)
    with pytest.raises(ant.errors.CausalTypeError):
        ant.Execution(pulse_extrapolation_ranges="10")  # type: ignore[arg-type]
