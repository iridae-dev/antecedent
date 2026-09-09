"""Intervention constructor contracts (no native estimation)."""

from __future__ import annotations

import pytest
from antecedent.errors import CausalValueError
from antecedent.intervention import Bernoulli, Categorical, Gaussian, Sequence, Set, Shift, Soft


def test_set_shift_and_stochastic_guards():
    Set("a", 1.0)
    Shift("a", -0.5)
    Bernoulli("a", 0.0)
    Bernoulli("a", 1.0)
    Gaussian("a", 0.0, 0.25)
    Categorical("a", [0.25, 0.75])

    with pytest.raises(CausalValueError, match="non-empty string"):
        Set("", 1.0)
    with pytest.raises(CausalValueError, match="finite"):
        Set("a", float("inf"))
    with pytest.raises(CausalValueError, match="finite"):
        Shift("a", float("nan"))
    with pytest.raises(CausalValueError, match=r"\[0, 1\]"):
        Bernoulli("a", 1.5)
    with pytest.raises(CausalValueError, match="positive"):
        Gaussian("a", 0.0, 0.0)
    with pytest.raises(CausalValueError, match="non-empty"):
        Categorical("a", [])
    with pytest.raises(CausalValueError, match="non-negative"):
        Categorical("a", [-0.1, 1.1])
    with pytest.raises(CausalValueError, match="positive total mass"):
        Categorical("a", [0.0, 0.0])


def test_soft_and_sequence_guards():
    Soft("a", "constant", parameters=[1.0])
    Soft("a", "additive_shift", parameters=[-0.25])
    Soft("a", "replacement")
    Sequence([Set("a", 1.0)])

    with pytest.raises(CausalValueError, match="mechanism"):
        Soft("a", "")
    with pytest.raises(CausalValueError, match="finite"):
        Soft("a", "replacement", parameters=[float("nan")])
    with pytest.raises(CausalValueError, match="exactly one"):
        Soft("a", "constant", parameters=[1.0, 2.0])
    with pytest.raises(CausalValueError, match="must not be empty"):
        Sequence([])
