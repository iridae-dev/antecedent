"""Python-only semantic surface for the 0.5 causal-response primitives."""

from __future__ import annotations

import dataclasses

import antecedent
import numpy as np
import pytest
from antecedent import interference, intervention, observation, transport
from antecedent._native import analyze_response_pag
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.results import (
    CausalResponseView,
    ResponseUncertainty,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
)


def test_only_queries_are_reexported_at_root():
    for name in (
        "ResponseCurve",
        "AverageDerivative",
        "PointDerivative",
        "Elasticity",
        "SemiElasticity",
        "DirectionalDerivative",
        "ResponseJacobian",
        "InterventionResponse",
    ):
        assert name in antecedent.__all__
    for name in ("SelectionDiagram", "RightCensored", "NeighborFraction"):
        assert name not in antecedent.__all__
        assert not hasattr(antecedent, name)


def test_response_query_validation_and_repr():
    curve = antecedent.ResponseCurve("a", "y", grid=[0.0, 0.5, 1.0])
    assert curve.treatment == "a"
    assert curve.outcome == "y"
    with pytest.raises(CausalValueError, match="strictly increasing"):
        antecedent.ResponseCurve("a", "y", grid=[0.0, 0.0])
    with pytest.raises(CausalValueError, match="positive"):
        antecedent.Elasticity("a", "y", at=0.0)
    with pytest.raises(CausalValueError, match="one value per treatment"):
        antecedent.ResponseJacobian(["a", "b"], ["y"], at=[0.0])
    with pytest.raises(CausalValueError, match="non-zero"):
        antecedent.DirectionalDerivative(
            ["a", "b"], ["y"], at=[0.0, 0.0], direction=[0.0, 0.0]
        )


def test_stage_specs_are_frozen_and_keep_assumptions_separate():
    intervention_spec = intervention.Set("dose", 1.0)
    assert intervention_spec.variable == "dose"
    with pytest.raises(dataclasses.FrozenInstanceError):
        intervention_spec.value = 2.0  # type: ignore[misc]

    mechanism = observation.RightCensored("event_time", "time", "censor_time", "event")
    assumption = observation.IndependentGiven(["age", "group"])
    assert mechanism.censoring == "censor_time"
    assert assumption.variables == ["age", "group"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mechanism.latent = "changed"  # type: ignore[misc]

    diagram = transport.SelectionDiagram("trial", "target", ["age"])
    assert transport.TransportQuery(antecedent.AverageEffect("a", "y"), diagram).diagram is diagram

    exposure = interference.NeighborFraction()
    query = interference.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        exposure,
        interference.ExposureContrast(
            "y",
            interference.ExposureLevel(0.0, 0.0),
            interference.ExposureLevel(0.0, 1.0),
        ),
    )
    assert query.exposure is exposure


def test_response_result_views_validate_shape_and_report_orthogonal_axes():
    response = ResponseView(
        treatments=["a"],
        outcomes=["y"],
        points=[[0.0], [1.0]],
        values=[[2.0], [3.0]],
    )
    support = SupportReport(
        "extrapolative",
        {"a": (0.0, 1.0)},
        diagnostics=[SupportDiagnostic("local_ess", [18.0], "below the configured threshold")],
        warnings=["upper grid point depends on extrapolation"],
    )
    uncertainty = ResponseUncertainty(
        "pointwise", lower=[[1.5], [2.4]], upper=[[2.5], [3.6]], level=0.95
    )
    result = CausalResponseView(
        estimand=antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0]),
        response=response,
        estimate=None,
        uncertainty=uncertainty,
        support=support,
        identification="identified",
    )
    assert len(response) == 2
    assert not support
    assert "extrapolative" in repr(result)
    assert "95.0%" in repr(uncertainty)

    with pytest.raises(CausalValueError, match="same number of rows"):
        ResponseView(["a"], ["y"], [[0.0]], [])
    with pytest.raises(CausalValueError, match="both be provided"):
        ResponseUncertainty("pointwise", lower=[[0.0]], upper=None)


def test_response_curve_runs_through_public_analyze_api():
    rng = np.random.default_rng(17)
    confounder = rng.normal(size=400)
    treatment = 0.7 * confounder + rng.normal(size=400)
    outcome = 2.0 * treatment + confounder + rng.normal(scale=0.2, size=400)
    query = antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5])

    result = antecedent.analyze(
        {"x": confounder, "a": treatment, "y": outcome},
        query=query,
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
    )

    assert result.estimand is query
    assert result.response is not None
    assert list(result.response.points) == [[-0.5], [0.0], [0.5]]
    assert result.provenance["operation_id"] == "estimate.response.kennedy_dr"
    assert result.identification.status == "NonparametricallyIdentified"


@pytest.mark.parametrize(
    "spec",
    [
        intervention.Set("a", 0.25),
        intervention.Shift("a", 0.25),
        intervention.Bernoulli("a", 0.4),
        intervention.Gaussian("a", 0.25, 0.01),
        intervention.Categorical("a", [0.2, 0.8]),
    ],
)
def test_intervention_response_runs_through_public_analyze_api(spec):
    rng = np.random.default_rng(1701)
    x = rng.normal(size=400)
    a = 0.7 * x + rng.normal(size=400)
    y = 1.0 + 2.0 * a + 0.8 * x + rng.normal(scale=0.1, size=400)
    query = antecedent.InterventionResponse("y", intervention=spec)
    result = antecedent.analyze(
        {"x": x, "a": a, "y": y},
        query=query,
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
    )
    assert np.isfinite(result.estimate)
    assert result.provenance["operation_id"] == "estimate.response.intervention_gcomp"


def test_intervention_response_checks_strategy_and_fails_closed():
    data = {"a": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)}
    query = antecedent.InterventionResponse("y", intervention=intervention.Set("a", 1.0))
    with pytest.raises(ValueError, match="requires estimator"):
        antecedent.analyze(
            data,
            query=query,
            graph=[("a", "y")],
            estimator="response.kennedy_dr",
        )
    with pytest.raises(CausalUnsupportedError, match="structural/temporal"):
        antecedent.analyze(
            data,
            query=antecedent.InterventionResponse(
                "y", intervention=intervention.Soft("a", "replacement")
            ),
            graph=[("a", "y")],
        )


def test_response_strategy_options_are_checked_not_silently_ignored():
    rng = np.random.default_rng(18)
    treatment = rng.normal(size=240)
    outcome = 2.0 * treatment + rng.normal(scale=0.2, size=240)
    query = antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5])
    data = {"a": treatment, "y": outcome}

    accepted = antecedent.analyze(
        data,
        query=query,
        graph=[("a", "y")],
        identifier=antecedent.Identifier.RESPONSE_BACKDOOR,
        estimator=antecedent.Estimator.RESPONSE_KENNEDY_DR,
    )
    assert accepted.response is not None
    with pytest.raises(ValueError, match="requires identifier"):
        antecedent.analyze(
            data,
            query=query,
            graph=[("a", "y")],
            identifier=antecedent.Identifier.FRONTDOOR,
        )
    with pytest.raises(ValueError, match="requires estimator"):
        antecedent.analyze(
            data,
            query=query,
            graph=[("a", "y")],
            estimator=antecedent.Estimator.AIPW,
        )


def test_accepted_discovery_dag_is_an_artifact_first_response_input():
    rng = np.random.default_rng(170)
    a = rng.normal(size=320)
    y = 1.8 * a + rng.normal(scale=0.25, size=320)
    accepted = antecedent.AcceptedGraph.from_graph(
        antecedent.Dag.from_edges(["a", "y"], [("a", "y")]),
        algorithm_id="ges",
    )
    query = antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5])

    result = antecedent.analyze({"a": a, "y": y}, query=query, graph=accepted)

    assert result.response is not None
    assert result.identification.status == "NonparametricallyIdentified"
    assert accepted.version == 1
    with pytest.raises(CausalUnsupportedError, match="already accepted"):
        antecedent.analyze(
            {"a": a, "y": y},
            query=query,
            graph=accepted,
            discovery=antecedent.discovery.PC(),
        )


def test_pag_mean_curve_preserves_unidentified_completion_mass():
    rng = np.random.default_rng(171)
    a = rng.normal(size=360)
    y = 2.0 * a + rng.normal(scale=0.3, size=360)
    pag = antecedent.Pag.from_marked_edges(
        ["a", "y"], [("a", "y", "circle", "arrow")]
    )

    result = antecedent.analyze(
        {"a": a, "y": y},
        query=antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]),
        graph=pag,
    )

    assert result.response is None
    assert result.envelope is not None
    assert result.envelope.identified_mass == pytest.approx(0.5)
    assert result.envelope.unidentified_mass == pytest.approx(0.5)
    assert result.envelope.completion_count == 2
    assert not result.envelope.enumeration_capped
    assert result.envelope.mass_scope == "full_class"
    assert len(result.envelope) == 3
    assert result.uncertainty.kind == "identified_set"
    assert result.identification.status == "GraphDependent"


def test_pag_curve_labels_capped_mass_as_examined_not_full_class():
    rng = np.random.default_rng(173)
    a = rng.normal(size=240)
    y = 1.2 * a + rng.normal(scale=0.2, size=240)
    pag = antecedent.Pag.from_marked_edges(
        ["a", "y"], [("a", "y", "circle", "arrow")]
    )

    raw = analyze_response_pag(
        ["a", "y"], [a, y], pag, "a", "y", [-0.25, 0.25], max_completions=1
    )

    assert raw.enumeration_capped is True
    assert raw.mass_scope == "examined_completions"
    assert raw.identified_mass == pytest.approx(1.0)
    assert raw.unidentified_mass == pytest.approx(0.0)


def test_curve_validation_runs_support_and_subset_but_skips_scalar_refuters():
    rng = np.random.default_rng(172)
    a = rng.normal(size=300)
    y = 1.4 * a + rng.normal(scale=0.2, size=300)
    result = antecedent.analyze(
        {"a": a, "y": y},
        query=antecedent.ResponseCurve("a", "y", grid=[-0.4, 0.0, 0.4]),
        graph=[("a", "y")],
        refute="cheap",
        seed=9,
    )

    assert result.validation is not None
    assert [check.id for check in result.validation.checks] == [
        "overlap.support",
        "data.subset",
        "scalar_ate_refuters",
    ]
    assert result.validation.checks[1].status == "informative"
    assert result.validation.checks[1].replicates == 10
    assert result.validation.skipped[0].id == "scalar_ate_refuters"
    assert result.provenance["validation_operation_ids"] == [
        "validate.overlap",
        "validate.response_data_subset",
    ]


def test_response_analyze_derivative_and_jacobian_shapes():
    rng = np.random.default_rng(29)
    x = rng.normal(size=500)
    a = 0.4 * x + rng.normal(size=500)
    b = -0.2 * x + rng.normal(size=500)
    y = 8.0 + 1.5 * a - 0.75 * b + x + rng.normal(scale=0.15, size=500)
    data = {"x": x, "a": a, "b": b, "y": y}
    graph = [("x", "a"), ("x", "b"), ("x", "y"), ("a", "y"), ("b", "y")]

    derivative = antecedent.analyze(
        data,
        query=antecedent.PointDerivative("a", "y", at=0.5),
        graph=graph,
        estimator_config={"bandwidth": 0.4},
    )
    average = antecedent.analyze(
        data,
        query=antecedent.AverageDerivative("a", "y"),
        graph=graph,
    )
    jacobian = antecedent.analyze(
        data,
        query=antecedent.ResponseJacobian(["a", "b"], ["y"], at=[0.0, 0.0]),
        graph=graph,
    )

    assert isinstance(derivative.estimate, float)
    assert isinstance(average.estimate, float)
    assert jacobian.estimate is not None
    assert len(jacobian.estimate) == 1
    assert len(jacobian.estimate[0]) == 2


def test_response_analyze_refuses_unwired_semantic_options():
    data = {"a": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)}
    graph = [("a", "y")]
    with pytest.raises(ValueError, match="target_population"):
        antecedent.analyze(
            data,
            query=antecedent.ResponseCurve(
                "a", "y", grid=[1.0, 2.0], target_population="target"
            ),
            graph=graph,
        )


def test_response_returns_standard_identification_view_and_adjustment_set():
    rng = np.random.default_rng(61)
    z = rng.normal(size=320)
    a = 0.5 * z + rng.normal(size=320)
    y = 2.0 * a + z + rng.normal(scale=0.1, size=320)
    result = antecedent.analyze(
        {"z": z, "a": a, "y": y},
        query=antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]),
        graph=[("z", "a"), ("z", "y"), ("a", "y")],
    )
    assert isinstance(result.identification, antecedent.results.IdentificationView)
    assert result.identification.method == "response.backdoor"
    assert result.identification.adjustment_set == ["z"]
    assert result.identification


def test_response_simultaneous_band_is_public_and_requires_explicit_bandwidth():
    rng = np.random.default_rng(67)
    a = rng.normal(size=360)
    y = 1.0 + 1.5 * a + rng.normal(scale=0.2, size=360)
    query = antecedent.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5])
    with pytest.raises(ValueError, match="explicit.*bandwidth"):
        antecedent.analyze(
            {"a": a, "y": y},
            query=query,
            graph=[("a", "y")],
            estimator_config={"simultaneous_replicates": 100},
        )
    result = antecedent.analyze(
        {"a": a, "y": y},
        query=query,
        graph=[("a", "y")],
        estimator_config={
            "bandwidth": 0.35,
            "simultaneous_replicates": 100,
            "multiplier_seed": 19,
        },
    )
    assert result.uncertainty.kind == "simultaneous"
    assert result.uncertainty.replicates == 100
    assert result.provenance["operation_id"] == "estimate.response.kennedy_dr_simultaneous"


def test_response_refuses_ignored_threads():
    with pytest.raises(ValueError, match="threads=1"):
        antecedent.analyze(
            {"a": np.arange(30.0), "y": np.arange(30.0)},
            query=antecedent.ResponseCurve("a", "y", grid=[1.0, 2.0]),
            graph=[("a", "y")],
            threads=2,
        )


def test_response_refuses_admg_with_explicit_error():
    admg = antecedent.Admg.from_edges(["a", "y"], directed=[("a", "y")], bidirected=[])
    with pytest.raises(TypeError, match="Dag or Pag"):
        antecedent.analyze(
            {"a": np.arange(30.0), "y": np.arange(30.0)},
            query=antecedent.ResponseCurve("a", "y", grid=[1.0, 2.0]),
            graph=admg,
        )


def test_response_refuses_cpdag_with_explicit_error():
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["a", "y"], directed=[], undirected=[("a", "y")]
    )
    with pytest.raises(TypeError, match="Dag or Pag"):
        antecedent.analyze(
            {"a": np.arange(30.0), "y": np.arange(30.0)},
            query=antecedent.ResponseCurve("a", "y", grid=[1.0, 2.0]),
            graph=cpdag,
        )


def test_elasticity_and_semi_elasticity_analyze_execute():
    rng = np.random.default_rng(91)
    a = np.exp(rng.normal(size=400) * 0.25)
    y = np.exp(0.5 * np.log(a) + rng.normal(scale=0.15, size=400))
    data = {"a": a, "y": y}
    graph = [("a", "y")]
    elasticity = antecedent.analyze(
        data,
        query=antecedent.Elasticity("a", "y", at=float(np.median(a))),
        graph=graph,
        estimator_config={"bandwidth": 0.25},
    )
    assert isinstance(elasticity.estimate, float)
    assert np.isfinite(elasticity.estimate)
    semi = antecedent.analyze(
        data,
        query=antecedent.SemiElasticity(
            "a", "y", at=float(np.median(a)), log_scale="treatment"
        ),
        graph=graph,
        estimator_config={"bandwidth": 0.25},
    )
    assert isinstance(semi.estimate, float)
    assert np.isfinite(semi.estimate)


def test_response_refuses_discovery_and_bayesian():
    data = {"a": np.arange(40.0), "y": np.arange(40.0)}
    query = antecedent.ResponseCurve("a", "y", grid=[1.0, 2.0])
    with pytest.raises(ValueError, match="discovery="):
        antecedent.analyze(
            data,
            query=query,
            graph=[("a", "y")],
            discovery=antecedent.discovery.PC(alpha=0.05),
        )
    with pytest.raises(TypeError, match="Bayesian"):
        antecedent.analyze(
            data,
            query=query,
            graph=[("a", "y")],
            inference=antecedent.Bayesian(n_draws=8),
        )
