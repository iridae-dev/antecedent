"""Observation mechanism constructors and stage helpers (no native calls)."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.observation import (
    AdjustedOutcome,
    Complete,
    IndependentGiven,
    IntervalCensored,
    LeftCensored,
    OutcomeIndependentGiven,
    RightCensored,
    Selected,
    Structural,
    Truncated,
    _assumption_kwargs,
    _ensure_latent_schema_column,
    _mechanism_kwargs,
    _stage_inputs,
    gaussian_log_likelihood,
)
from antecedent.query import ResponseCurve


def test_mechanism_and_assumption_constructors():
    RightCensored("y", "obs", "c", "e")
    LeftCensored("y", "obs", "c", "e")
    IntervalCensored("y", "lo", "hi")
    Selected("y", "obs", "r")
    Truncated("y", "obs", lower="lo")
    Truncated("y", "obs", upper="hi")
    IndependentGiven(["x"])
    OutcomeIndependentGiven(["x"])
    Structural("gaussian_observation_likelihood")
    AdjustedOutcome([1.0], [0.5], "ipw")

    with pytest.raises(CausalValueError, match="non-empty variable"):
        RightCensored(" ", "obs", "c", "e")
    with pytest.raises(CausalValueError, match="at least one"):
        Truncated("y", "obs")
    with pytest.raises(CausalValueError, match="model_id"):
        Structural(" ")
    with pytest.raises(CausalValueError, match="same length"):
        AdjustedOutcome([1.0], [0.5, 0.5], "ipw")
    with pytest.raises(CausalValueError, match="method"):
        AdjustedOutcome([1.0], [0.5], " ")
    with pytest.raises(CausalValueError, match="non-empty variable"):
        IndependentGiven(["x", ""])


def test_stage_helpers_fail_closed_before_native():
    with pytest.raises(CausalValueError, match="scalar response query"):
        _stage_inputs(object())
    query = ResponseCurve("a", "y", grid=[0.0, 1.0])
    with pytest.raises(CausalValueError, match="non-complete"):
        _stage_inputs(query)
    with pytest.raises(CausalValueError, match="exactly one"):
        _stage_inputs(
            ResponseCurve(
                "a",
                "y",
                grid=[0.0, 1.0],
                observation=Selected("y", "obs", "r"),
                observation_assumptions=[],
            )
        )
    with pytest.raises(CausalValueError, match="latent outcome"):
        _stage_inputs(
            ResponseCurve(
                "a",
                "y",
                grid=[0.0, 1.0],
                observation=Selected("latent", "obs", "r"),
                observation_assumptions=[OutcomeIndependentGiven(["x"])],
            )
        )

    right = RightCensored("y", "obs", "c", "e")
    left = LeftCensored("y", "obs", "c", "e")
    interval = IntervalCensored("y", "lo", "hi")
    truncated = Truncated("y", "obs", lower="lo", upper="hi")
    selected = Selected("y", "obs", "r")
    assert _mechanism_kwargs(right)["observation_kind"] == "right_censored"
    assert _mechanism_kwargs(left)["observation_kind"] == "left_censored"
    assert _mechanism_kwargs(interval)["observation_kind"] == "interval_censored"
    assert _mechanism_kwargs(truncated)["observation_kind"] == "truncated"
    assert _mechanism_kwargs(selected)["observation_kind"] == "selected"
    with pytest.raises(CausalValueError, match="unsupported observation mechanism"):
        _mechanism_kwargs(Complete())

    assert _assumption_kwargs(IndependentGiven(["x"]))["assumption_kind"] == "independent_given"
    assert (
        _assumption_kwargs(OutcomeIndependentGiven(["x"]))["assumption_kind"]
        == "outcome_independent_given"
    )
    assert _assumption_kwargs(Structural("m"))["assumption_kind"] == "structural"
    with pytest.raises(CausalValueError, match="unsupported observation assumption"):
        _assumption_kwargs(object())

    names, cols = _ensure_latent_schema_column(
        ["a", "y"],
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        selected,
    )
    assert names == ["a", "y"]
    names, cols = _ensure_latent_schema_column(["a"], [np.array([1.0, 2.0])], selected)
    assert names[-1] == "y"
    assert np.isnan(cols[-1]).all()
    with pytest.raises(CausalValueError, match="at least one recorded"):
        _ensure_latent_schema_column([], [], selected)

    with pytest.raises(CausalValueError, match="Structural"):
        gaussian_log_likelihood(
            {"a": [0.0], "y": [0.0], "lo": [-1.0], "hi": [1.0]},
            ResponseCurve(
                "a",
                "y",
                grid=[0.0, 1.0],
                observation=interval,
                observation_assumptions=[IndependentGiven(())],
            ),
            means=[0.0],
            sigma=1.0,
        )
