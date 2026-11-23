"""AverageEffect routing: binary 0/1 vs any other complete-case treatment encoding."""

from __future__ import annotations

import antecedent as ac
import numpy as np
import pytest
from antecedent.estimators import PropensityWeighting
from antecedent.outcomes import require_success


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


def _cheap(data: dict[str, np.ndarray], *, estimator=None, query=QUERY):
    return require_success(
        ac.analyze(
            data,
            graph=GRAPH,
            query=query,
            estimator=estimator,
            refute="cheap",
            inference=ac.Frequentist(bootstrap=0),
            execution=ac.Execution(seed=9),
        )
    )


def _names(result) -> list[str]:
    return [report.refuter for report in result.validation.reports]


@pytest.mark.parametrize(
    "treatment, binary",
    [
        (np.tile([0.0, 1.0], 40), True),
        (np.linspace(0.0, 1.0, 80, endpoint=False), False),
        (np.tile([2.0, 5.0], 40), False),
    ],
)
def test_python_facade_classifies_treatment_encoding(treatment, binary):
    values = np.asarray(treatment, dtype=np.float64)
    levels = sorted(set(values.tolist()))
    # A two-valued column that does not carry 0 and 1 names its own arms; the contrast
    # levels must be rows the column actually holds.
    query = (
        ac.AverageEffect("t", "y", control_level=levels[0], active_level=levels[1])
        if len(levels) == 2 and levels != [0.0, 1.0]
        else QUERY
    )
    names = _names(_cheap(_frame(values), query=query))
    if binary:
        assert "overlap.assessment" in names
        assert "overlap.continuous_support" not in names
    else:
        assert "overlap.continuous_support" in names
        assert "overlap.assessment" not in names


def test_a_contrast_level_no_row_carries_is_refused():
    with pytest.raises(ac.errors.CausalUnsupportedError, match="arm_not_populated"):
        _cheap(_frame(np.tile([2.0, 5.0], 40)))


@pytest.mark.parametrize("levels", [4, 3])
def test_extra_discrete_levels_are_refused_or_restricted_to_the_binary_contrast(levels):
    # A discrete treatment with levels beyond the contrasted 0 and 1 is not read as a dose:
    # it is refused, or restricted by declaration to the two contrasted levels, whose
    # overlap is then the binary one.
    data = _frame((np.arange(80) % levels).astype(np.float64))
    with pytest.raises(ac.errors.CausalError, match="treatment_level_undeclared"):
        _cheap(data)
    restricted = PropensityWeighting(other_levels="exclude_undeclared")
    names = _names(_cheap(data, estimator=restricted))
    assert "overlap.assessment" in names
    assert "overlap.continuous_support" not in names


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
    with pytest.raises(ac.errors.CausalError, match="treatment_level_undeclared"):
        require_success(
            ac.analyze(
                _frame(np.linspace(0.0, 1.0, 80, endpoint=False)),
                graph=GRAPH,
                query=QUERY,
                estimator=PropensityWeighting(),
                refute=False,
                inference=ac.Frequentist(bootstrap=0),
                execution=ac.Execution(seed=9),
            )
        )


def test_propensity_weighting_accepts_two_valued_float():
    result = require_success(
        ac.analyze(
            _frame(np.tile([0.0, 1.0], 40)),
            graph=GRAPH,
            query=QUERY,
            estimator=PropensityWeighting(),
            refute=False,
            inference=ac.Frequentist(bootstrap=0),
            execution=ac.Execution(seed=9),
        )
    )
    # y = 1 + 2 t + 0.01 i: the treated rows are the odd indices, 0.01 later on average.
    assert result.ate == pytest.approx(2.01, abs=1e-9)
