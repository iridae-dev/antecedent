"""Sufficient-statistics slots keep their dimension and refuse an unbounded one."""

import antecedent
import pytest
from antecedent.execution import CachePolicy, Execution


def _state():
    return antecedent.state.CausalState(execution=Execution(cache=CachePolicy.bounded(1 << 20)))


def test_ols_ensure_different_dimension_refused():
    state = _state()
    state.ols_ensure("k", 2)
    with pytest.raises(ValueError, match="already holds"):
        state.ols_ensure("k", 3)
    state.ols_ensure("k", 2)
    assert state.ols_get("k")["ncols"] == 2


def test_ols_ensure_huge_dimension_refused():
    state = _state()
    with pytest.raises(ValueError, match="exceeds the bound"):
        state.ols_ensure("k", 10**9)


def test_cov_ensure_different_dimension_refused():
    state = _state()
    state.cov_ensure("c", 2)
    with pytest.raises(ValueError, match="already holds"):
        state.cov_ensure("c", 4)
    with pytest.raises(ValueError, match="exceeds the bound"):
        state.cov_ensure("big", 10**9)
