"""Joint draws across the horizons of a real temporal response, read back from the result."""

from __future__ import annotations

from typing import Any

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.estimation import PreparedAnalysis
from antecedent.external import ExternalRefusal
from antecedent.horizon_composition import JointHorizonDraws

_EDGES = [("x", 1, "y", 0), ("x", 2, "y", 0)]
_HORIZONS = [1, 2]
_GRID = [0.0, 1.0]


def _series(n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.3 + 0.2 * x[t - 1] + 0.3 * rng.standard_normal()
        y[t] = 1.0 + 2.0 * x[t - 1] + 3.0 * x[t - 2] + 0.5 * rng.standard_normal()
    return {"x": x, "y": y}


def _query() -> antecedent.ResponseCurve:
    return antecedent.ResponseCurve(
        "x", "y", grid=_GRID, horizons=_HORIZONS, policy="pulse", treatment_lag=1
    )


def _prepare(data: dict[str, np.ndarray], **kwargs: Any) -> Any:
    return PreparedAnalysis.prepare(
        data, graph=_EDGES, query=_query(), refute=False, seed=21, **kwargs
    )


def _cells(result: Any) -> np.ndarray:
    assert result.response is not None
    return np.asarray([row[0] for row in result.response.values], dtype=float)


def _column(rows: Any) -> np.ndarray:
    return np.asarray([row[0] for row in rows], dtype=float)


def test_frequentist_horizon_draws_are_the_joint_bootstrap_replicates_of_the_surface():
    data = _series()
    result = _prepare(data, bootstrap=60, retain_joint_horizons=True).estimate(data, seed=21)
    cells = _cells(result)
    assert cells.shape == (len(_GRID) * len(_HORIZONS),)
    lower = _column(result.uncertainty.lower)
    upper = _column(result.uncertainty.upper)

    first = result.horizon_draws(0)
    second = result.horizon_draws(1)
    assert isinstance(second, JointHorizonDraws)
    assert second.horizons == tuple(_HORIZONS)
    # Shape: one column per horizon, one row per retained replicate, all paired by index.
    assert len(second.columns) == len(_HORIZONS)
    replicates = len(second.columns[0])
    assert 30 <= replicates <= 60
    assert all(len(column) == replicates for column in second.columns)
    assert np.isfinite(np.asarray(second.columns)).all()
    # Alignment: both doses read one joint law, so the identity is shared and the rows pair.
    assert first.alignment_id == second.alignment_id
    assert second.alignment_id.startswith(second.evidence_id + "#joint:")
    assert second.alignment_id.split("#joint:")[1].startswith(f"{replicates}:")
    assert not np.allclose(first.columns[0], second.columns[0])
    # The columns are the surface cells: dose-major, horizon-minor.
    for dose, draws in enumerate((first, second)):
        for k, horizon in enumerate(draws.horizons):
            cell = dose * len(_HORIZONS) + k
            assert draws.center[k] == pytest.approx(cells[cell])
            column = np.asarray(draws.columns[k])
            spread = column.std(ddof=1)
            assert spread > 0.0
            assert abs(column.mean() - cells[cell]) < 4.0 * spread / np.sqrt(replicates), horizon
            # The reported pointwise band is the spread of these replicates.
            half = 0.5 * (upper[cell] - lower[cell])
            assert 0.6 < half / (1.959964 * spread) < 1.6, (dose, horizon)
    # Pairing by index is what makes a cumulative effect honest: the composed draws are the row
    # sums, and a contrast between the horizons has the spread of the replicates' difference.
    cumulative = second.compose(operation="sum", coefficients=[1.0, 1.0])
    np.testing.assert_allclose(
        cumulative.draws, np.asarray(second.columns[0]) + np.asarray(second.columns[1])
    )
    contrast = second.compose(operation="contrast", coefficients=[-1.0, 1.0])
    difference = np.asarray(second.columns[1]) - np.asarray(second.columns[0])
    assert contrast.standard_deviation == pytest.approx(difference.std(ddof=1), rel=1e-6)
    assert contrast.alignment_id == second.alignment_id


def test_horizon_draws_are_off_by_default_and_refuse_a_dose_outside_the_grid():
    data = _series()
    plain = _prepare(data, bootstrap=60).estimate(data, seed=21)
    with pytest.raises(ExternalRefusal) as plain_refusal:
        plain.horizon_draws(0)
    assert plain_refusal.value.reason_code == "joint_law_required"
    assert "horizon_draws_not_joint" in str(plain_refusal.value)

    kept = _prepare(data, bootstrap=60, retain_joint_horizons=True).estimate(data, seed=21)
    with pytest.raises(ExternalRefusal) as outside:
        kept.horizon_draws(len(_GRID))
    assert "horizon_draws_dose" in str(outside.value)
    with pytest.raises(CausalValueError):
        kept.horizon_draws(-1)


def test_retaining_joint_horizons_needs_a_temporal_response_and_a_bootstrap():
    data = _series()
    static = antecedent.ResponseCurve("x", "y", grid=_GRID)
    with pytest.raises(CausalUnsupportedError):
        PreparedAnalysis.prepare(
            data,
            graph=[("x", "y")],
            query=static,
            refute=False,
            retain_joint_horizons=True,
        )
    # A Frequentist response retains replicates only when a bootstrap produces them.
    with pytest.raises((CausalUnsupportedError, CausalValueError)):
        _prepare(data, bootstrap=0, retain_joint_horizons=True)


def test_bayesian_horizon_draws_are_the_block_weight_posterior_draws():
    data = _series()
    result = _prepare(
        data,
        inference=antecedent.Bayesian(n_draws=60, backend="conjugate"),
        retain_joint_horizons=True,
    ).estimate(data, seed=21)
    cells = _cells(result)
    draws = result.horizon_draws(1)
    assert draws.horizons == tuple(_HORIZONS)
    assert all(len(column) == 60 for column in draws.columns)
    for k in range(len(_HORIZONS)):
        cell = len(_HORIZONS) + k
        column = np.asarray(draws.columns[k])
        spread = column.std(ddof=1)
        assert spread > 0.0
        assert draws.center[k] == pytest.approx(cells[cell])
        assert abs(column.mean() - cells[cell]) < 0.5 * spread
    # The per-horizon conjugate fits are independent marginals and keep no joint draws.
    marginal = _prepare(
        data, inference=antecedent.Bayesian(n_draws=60, backend="conjugate")
    ).estimate(data, seed=21)
    with pytest.raises(ExternalRefusal):
        marginal.horizon_draws(1)
