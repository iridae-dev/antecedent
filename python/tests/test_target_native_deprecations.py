"""The direct target-native checks warn and name the compiled refuter, and still return the same
answers on known cases: the derivative of ``sin`` is ``cos``, a line has no curvature, and an
observed target in the middle of symmetric replicates sits at the stated tail fraction.

The compiled replacement (``PreparedAnalysis.refute_target``) runs the same kernels on the study's
own fitted response and draws; its twin equality with the kernels is asserted in Rust
(``crates/antecedent/src/analysis/target_validation/native_twin_tests.rs``)."""

from __future__ import annotations

import math

import pytest
from antecedent import validation as v
from antecedent._deprecation import AntecedentDeprecationWarning


def test_derivative_consistency_warns_and_keeps_its_answer():
    points = [0.1, 0.7, 1.3]
    with pytest.warns(AntecedentDeprecationWarning, match="derivative_consistency"):
        good = v.derivative_consistency(math.sin, points, [math.cos(x) for x in points])
    assert good.passed
    with pytest.warns(AntecedentDeprecationWarning, match="refute_target"):
        bad = v.derivative_consistency(
            math.sin, points, [math.cos(x) + 0.1 for x in points], tolerance=1e-3
        )
    assert not bad.passed


def test_local_linearity_warns_and_keeps_its_answer():
    with pytest.warns(AntecedentDeprecationWarning, match="local_linearity"):
        assert v.local_linearity(lambda x: 2 * x + 1, [0.0, 0.5], step=0.5).passed
    with pytest.warns(AntecedentDeprecationWarning, match="local_linearity"):
        assert not v.local_linearity(lambda x: math.exp(5 * x), [0.0, 0.5], step=0.5).passed


def test_posterior_predictive_target_warns_and_keeps_its_answer():
    draws = [[0.0]] * 4
    replicated = [[-1.0], [-0.5], [0.5], [1.0]]
    with pytest.warns(AntecedentDeprecationWarning, match="posterior_predictive_target"):
        check = v.posterior_predictive_target(["a"], [1.0], draws, replicated, alpha=0.5)
    assert check.p_value == 0.5
    assert check.passed
