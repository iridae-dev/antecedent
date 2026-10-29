"""Pulse, sustained and conditional contrasts on a ``TemporalAdmg`` over one series.

The compiler unrolls the template over the periods the policy needs, identifies the contrast on
that window (the complete ID algorithm, IDC for a conditional contrast) and evaluates the
identified functional on the lag-aligned window table. The cell executes unlicensed: the support
slot reports ``refused``, the result discloses the window, and the expected values below are
counted from the series, not taken from the engine.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

DISCLOSURE = "estimate.temporal_admg.window"
N = 300


def _series(n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 2, n).astype(float)
    w = rng.integers(0, 2, n).astype(float)
    y = np.zeros(n)
    y[1:] = np.where(w[:-1] == 1.0, x[:-1], 0.0)
    for t in (9, 33, 71, 120):
        y[t] = 1.0 - y[t]
    return {"x": x, "y": y, "w": w}


def _graph(confounded: bool = False) -> antecedent.graph.TemporalAdmg:
    return antecedent.graph.TemporalAdmg.from_lagged_edges(
        ["x", "y", "w"],
        [("x", 1, "y", 0), ("w", 1, "y", 0)],
        [("x", 1, "y", 0)] if confounded else None,
    )


def _counted_unadjusted(series: dict[str, np.ndarray]) -> float:
    x, y = series["x"], series["y"]
    positions = np.arange(1, len(x) - 1)
    treated = positions[x[positions] == 1.0]
    control = positions[x[positions] == 0.0]
    return float(y[treated + 1].mean() - y[control + 1].mean())


def _counted_conditional(series: dict[str, np.ndarray]) -> float:
    """The contrast within each level of ``w`` averaged over the law of ``w`` on the window.

    ``w`` at the treated time is a parent of the outcome with no parent in the window, so
    this is also the identified unconditional pulse (the standardized contrast); the
    unadjusted difference of means is a different finite-sample quantity.
    """
    x, y, w = series["x"], series["y"], series["w"]
    positions = np.arange(1, len(x) - 1)
    total = 0.0
    for stratum in (0.0, 1.0):
        in_stratum = positions[w[positions] == stratum]
        weight = len(in_stratum) / len(positions)
        treated = in_stratum[x[in_stratum] == 1.0]
        control = in_stratum[x[in_stratum] == 0.0]
        total += weight * float(y[treated + 1].mean() - y[control + 1].mean())
    return total


def _run(query, graph=None, inference=None, **kwargs):
    return require_success(
        antecedent.analyze(
            _series(),
            graph=graph if graph is not None else _graph(),
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0) if inference is None else inference,
            execution=antecedent.Execution(seed=3),
            **kwargs,
        )
    )


def _discloses(result) -> bool:
    return any(DISCLOSURE in str(diagnostic) for diagnostic in result.diagnostics)


def test_a_pulse_is_the_counted_contrast_and_stays_unlicensed() -> None:
    series = _series()
    result = _run(antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1))
    assert abs(result.ate - _counted_conditional(series)) < 1e-12
    assert abs(result.ate - _counted_unadjusted(series)) > 1e-3
    support = result.reasoning.support.payload
    assert support["matrix_status"] == "refused"
    assert str(support["matrix_coordinate"]).startswith("PulseEffect:TemporalAdmg")
    assert result.evidence_status is None
    assert _discloses(result)


def test_overlapping_windows_are_resampled_in_blocks() -> None:
    result = _run(
        antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1),
        inference=antecedent.Frequentist(bootstrap=40),
    )
    se = result.estimate.se_bootstrap
    assert se is not None and se > 0.0
    # Overlapping windows are resampled as circular blocks, not as independent rows.
    window = next(str(d) for d in result.diagnostics if DISCLOSURE in str(d))
    assert "resamples circular blocks of" in window
    block = int(window.split("circular blocks of ")[1].split()[0])
    assert block >= 3  # never shorter than the three-period window


def test_a_conditional_pulse_is_idc_on_the_same_window() -> None:
    series = _series()
    conditional = antecedent.TemporalConditionalPulse(
        "x", "y", "w", treatment_lag=1, horizon_steps=1
    )
    result = _run(conditional)
    assert abs(result.ate - _counted_conditional(series)) < 1e-12
    assert abs(result.ate - _counted_unadjusted(series)) > 1e-3
    assert _discloses(result)


def test_the_bayesian_form_shares_one_weight_per_block() -> None:
    series = _series()
    result = _run(
        antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1),
        inference=antecedent.Bayesian(n_draws=300, backend="conjugate"),
    )
    # Observed deviation from the counted contrast is 1.4e-3 (conjugate posterior mean).
    assert abs(result.ate - _counted_conditional(series)) < 0.01
    assert result.posterior is not None
    assert _discloses(result)


def test_a_window_that_holds_a_hedge_is_refused_with_the_next_attempts() -> None:
    with pytest.raises(CausalUnsupportedError, match="hedge") as refusal:
        _run(
            antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1),
            graph=_graph(confounded=True),
        )
    assert "next attempts" in str(refusal.value)


def test_a_graph_in_another_variable_order_is_refused() -> None:
    reordered = antecedent.graph.TemporalAdmg.from_lagged_edges(["y", "x", "w"], [("x", 1, "y", 0)])
    with pytest.raises(ValueError, match="variable names must match"):
        _run(antecedent.PulseEffect("x", "y", treatment_lag=1, horizon_steps=1), graph=reordered)
