from __future__ import annotations

from typing import Any, cast

import antecedent
import numpy as np
import pytest
from antecedent import observation
from antecedent._native import analyze_observation_response
from antecedent.errors import CausalEstimateError, CausalValueError
from antecedent.query import DirectionalDerivative, ResponseCurve


def test_selected_outcome_stage_returns_pseudo_values_and_diagnostic_weights() -> None:
    treatment = np.linspace(0.0, 1.0, 80)
    selected = np.asarray([float(i % 3 != 0) for i in range(80)])
    outcome = np.where(selected == 1.0, 2.0 + 3.0 * treatment, np.nan)
    data = {"a": treatment, "y_observed": outcome, "r": selected, "x": treatment.copy()}
    query = ResponseCurve(
        "a",
        "y_latent",
        grid=[0.0, 1.0],
        observation=observation.Selected("y_latent", "y_observed", "r"),
        observation_assumptions=[observation.OutcomeIndependentGiven(["x"])],
    )

    adjusted = observation.adjusted_outcome(data, query)

    assert len(adjusted.values) == 80
    assert len(adjusted.weights) == 80
    assert all(np.isfinite(adjusted.values))
    assert adjusted.method == "observation.selected.crossfit_logistic_aipw.v1"
    assert all(
        weight == 0.0
        for weight, r in zip(adjusted.weights, selected, strict=True)
        if r == 0.0
    )


def test_gaussian_interval_likelihood_requires_and_honours_structural_opt_in() -> None:
    data = {
        "a": np.asarray([0.0, 1.0, 0.5]),
        "y": np.asarray([0.0, 0.0, 0.0]),
        "lower": np.asarray([-0.2, 0.8, 0.3]),
        "upper": np.asarray([0.2, 1.2, 0.7]),
    }
    mechanism = observation.IntervalCensored("y", "lower", "upper")
    opted_in = ResponseCurve(
        "a",
        "y",
        grid=[0.0, 1.0],
        observation=mechanism,
        observation_assumptions=[
            observation.Structural("gaussian_observation_likelihood")
        ],
    )

    value = observation.gaussian_log_likelihood(
        data, opted_in, means=[0.0, 1.0, 0.5], sigma=0.5
    )
    assert np.isfinite(value)

    not_opted_in = ResponseCurve(
        "a",
        "y",
        grid=[0.0, 1.0],
        observation=mechanism,
        observation_assumptions=[observation.IndependentGiven(())],
    )
    with pytest.raises(CausalValueError, match="Structural"):
        observation.gaussian_log_likelihood(
            data, not_opted_in, means=[0.0, 1.0, 0.5], sigma=0.5
        )


def test_observation_stage_rejects_vector_response_query() -> None:
    query = DirectionalDerivative(
        ["a"],
        ["y"],
        at=[0.0],
        direction=[1.0],
        observation=observation.Selected("y", "y", "r"),
        observation_assumptions=[observation.OutcomeIndependentGiven(["x"])],
    )
    with pytest.raises(CausalValueError, match="scalar response query"):
        observation.adjusted_outcome({}, query)


def test_observation_stage_does_not_accept_complete_observation() -> None:
    query = ResponseCurve("a", "y", grid=[0.0, 1.0])
    with pytest.raises(CausalValueError, match="non-complete"):
        observation.adjusted_outcome({}, query)


def test_native_selected_outcome_response_is_point_only_with_explicit_warning() -> None:
    treatment = np.linspace(0.0, 1.0, 80)
    selected = np.asarray([float(i % 3 != 0) for i in range(80)])
    outcome = np.where(selected == 1.0, 2.0 + 3.0 * treatment, np.nan)

    result = analyze_observation_response(
        ["a", "y", "r"],
        [treatment, outcome, selected],
        [("a", "y")],
        "a",
        "y",
        [0.2, 0.8],
        "selected",
        "y",
        observed="y",
        indicator="r",
        assumption_kind="outcome_independent_given",
        assumption_variables=["a"],
    )

    assert result.uncertainty_kind == "none"
    assert result.lower is None
    assert result.upper is None
    assert result.provenance_id == "estimate.response.observation_adjusted"
    assert any("joint_uncertainty_unavailable" in warning for warning in result.warnings)


def test_public_analyze_composes_selected_outcome_into_point_only_curve() -> None:
    treatment = np.linspace(0.0, 1.0, 100)
    selected = np.asarray([float(i % 3 != 0) for i in range(100)])
    confounder = np.sin(np.linspace(0.0, 3.0, 100))
    observed = np.where(
        selected == 1.0, 1.0 + 2.0 * treatment + 0.5 * confounder, np.nan
    )
    query = antecedent.ResponseCurve(
        "a",
        "latent_y",
        grid=[0.2, 0.8],
        observation=observation.Selected("latent_y", "observed_y", "r"),
        observation_assumptions=[observation.OutcomeIndependentGiven(["a", "z"])],
    )

    result = cast(
        Any,
        antecedent.analyze(
            {"a": treatment, "z": confounder, "observed_y": observed, "r": selected},
            query=query,
            graph=[("z", "a"), ("z", "latent_y"), ("a", "latent_y")],
        ),
    )

    assert result.response is not None
    assert result.uncertainty.kind == "none"
    assert result.identification.status == "NonparametricallyIdentified"
    assert result.identification.method == "response.backdoor"
    assert result.identification.adjustment_set == ["z"]
    assert result.provenance["operation_id"] == "estimate.response.observation_adjusted"
    assert any(
        "joint_uncertainty_unavailable" in warning
        for warning in result.support.warnings
    )


@pytest.mark.parametrize("side", ["right", "left"])
def test_public_analyze_composes_marginal_censoring_into_point_only_curve(
    side: str,
) -> None:
    treatment = np.linspace(0.0, 1.0, 100)
    observed = 1.0 + 2.0 * treatment
    event = np.ones(100)
    censoring = np.full(100, 10.0 if side == "right" else -10.0)
    mechanism = (
        observation.RightCensored("latent_y", "observed_y", "c", "event")
        if side == "right"
        else observation.LeftCensored("latent_y", "observed_y", "c", "event")
    )
    query = antecedent.ResponseCurve(
        "a",
        "latent_y",
        grid=[0.2, 0.8],
        observation=mechanism,
        observation_assumptions=[observation.IndependentGiven(())],
    )

    result = cast(
        Any,
        antecedent.analyze(
            {"a": treatment, "observed_y": observed, "c": censoring, "event": event},
            query=query,
            graph=[("a", "latent_y")],
        ),
    )

    assert result.response is not None
    assert result.uncertainty.kind == "none"
    assert any(
        f"observation.{side}_censored" in warning for warning in result.support.warnings
    )


def test_public_analyze_keeps_interval_censoring_on_likelihood_only_path() -> None:
    treatment = np.linspace(0.0, 1.0, 40)
    query = antecedent.ResponseCurve(
        "a",
        "latent_y",
        grid=[0.2, 0.8],
        observation=observation.IntervalCensored("latent_y", "lower", "upper"),
        observation_assumptions=[
            observation.Structural("gaussian_observation_likelihood")
        ],
    )
    with pytest.raises(CausalEstimateError, match="opt-in Gaussian likelihood"):
        antecedent.analyze(
            {
                "a": treatment,
                "lower": treatment,
                "upper": treatment + 0.2,
            },
            query=query,
            graph=[("a", "latent_y")],
        )


def test_public_analyze_keeps_truncation_on_likelihood_only_path() -> None:
    treatment = np.linspace(0.0, 1.0, 40)
    query = antecedent.ResponseCurve(
        "a",
        "latent_y",
        grid=[0.2, 0.8],
        observation=observation.Truncated(
            "latent_y", "observed_y", lower="lower", upper="upper"
        ),
        observation_assumptions=[
            observation.Structural("gaussian_observation_likelihood")
        ],
    )
    with pytest.raises(CausalEstimateError, match="opt-in Gaussian likelihood"):
        antecedent.analyze(
            {
                "a": treatment,
                "observed_y": treatment + 0.1,
                "lower": np.full(40, -1.0),
                "upper": np.full(40, 2.0),
            },
            query=query,
            graph=[("a", "latent_y")],
        )
