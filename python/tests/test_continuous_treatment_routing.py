"""AverageEffect routing: binary 0/1 vs any other complete-case treatment encoding."""

from __future__ import annotations

import antecedent as ac
import numpy as np
import pytest
from antecedent.estimators import PropensityWeighting


def _frame(t: np.ndarray, y_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = t.shape[0]
    y = 1.0 + 2.0 * t + 0.01 * np.arange(n, dtype=np.float64)
    if y_mask is not None:
        y = y.copy()
        y[~y_mask] = np.nan
    z = np.arange(n, dtype=np.float64) % 5
    return {"t": t.astype(np.float64), "y": y, "z": z}


GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ac.AverageEffect("t", "y")


def _cheap(data: dict[str, np.ndarray], *, estimator=None):
    return ac.analyze(
        data,
        graph=GRAPH,
        query=QUERY,
        estimator=estimator,
        refute="cheap",
        bootstrap=0,
        seed=9,
    )


def _names(result) -> list[str]:
    return [report.refuter for report in result.validation.reports]


@pytest.mark.parametrize(
    "treatment, binary",
    [
        (np.tile([0.0, 1.0], 40), True),
        (np.linspace(0.0, 1.0, 80, endpoint=False), False),
        (np.arange(80) % 4, False),
        (np.arange(80) % 3, False),
        (np.tile([2.0, 5.0], 40), False),
    ],
)
def test_python_facade_classifies_treatment_encoding(treatment, binary):
    names = _names(_cheap(_frame(np.asarray(treatment, dtype=np.float64))))
    if binary:
        assert "overlap.assessment" in names
        assert "overlap.continuous_support" not in names
    else:
        assert "overlap.continuous_support" in names
        assert "overlap.assessment" not in names


def test_degenerate_one_level_stays_off_continuous_overlap():
    try:
        names = _names(_cheap(_frame(np.ones(80))))
    except ac.errors.CausalError as err:
        assert "continuous_support" not in str(err)
        return
    assert "overlap.continuous_support" not in names


def test_missingness_after_complete_case_uses_remaining_sample():
    t = np.tile([0.0, 1.0], 40)
    t[5] = 2.7
    mask = np.ones(80, dtype=bool)
    mask[5] = False
    try:
        names = _names(_cheap(_frame(t, mask)))
    except ac.errors.CausalError as err:
        assert "continuous_support" not in str(err)
        return
    assert "overlap.assessment" in names


def test_propensity_weighting_refuses_continuous_treatment():
    with pytest.raises(ac.errors.CausalEstimateError, match="binary treatment"):
        ac.analyze(
            _frame(np.linspace(0.0, 1.0, 80, endpoint=False)),
            graph=GRAPH,
            query=QUERY,
            estimator=PropensityWeighting(),
            refute=False,
            bootstrap=0,
            seed=9,
        )


def test_propensity_weighting_accepts_two_valued_float():
    result = ac.analyze(
        _frame(np.tile([0.0, 1.0], 40)),
        graph=GRAPH,
        query=QUERY,
        estimator=PropensityWeighting(),
        refute=False,
        bootstrap=0,
        seed=9,
    )
    assert np.isfinite(result.ate)
