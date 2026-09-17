"""Live executions render their identification state honestly.

One verdict renderer (banner, reprs, statement), nested views that never show a lone
point and interval for a set-identified claim, one ``answer`` vocabulary across live
and loaded results, readable report targets, and a calibration row with no ``None``.
"""

from __future__ import annotations

import math
import re
import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent._verdict import describe_status
from antecedent.results import Answer, CalibrationInfo
from antecedent.results._execution import ANSWER_KINDS, CLAIM_KIND_ANSWERS

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
# z - t undirected: both completions identify the effect with different adjustment sets.
PARTIAL = ant.Cpdag.from_directed_undirected(
    ["t", "y", "z"], [("t", "y"), ("z", "y")], [("z", "t")]
)
# Fully undirected triangle: one completion leaves the effect unidentified.
GRAPH_DEPENDENT = ant.Cpdag.from_directed_undirected(
    ["t", "y", "z"], [], [("z", "t"), ("t", "y"), ("z", "y")]
)


def _continuous(n: int = 400, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.8 * z + rng.normal(size=n)
    y = 1.5 * t + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _run(graph, *, bayesian: bool):
    inference = ant.Bayesian(n_draws=64, backend="conjugate") if bayesian else ant.Frequentist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ant.analyze(
            _continuous(),
            graph=graph,
            query=ant.AverageEffect("t", "y"),
            inference=inference,
            refute="none",
            bootstrap=0 if bayesian else 20,
        )


def _banner(html: str) -> tuple[str, str]:
    match = re.search(r'antecedent-ar-banner (ar-[a-z]+)"><span>([^<]*)</span>', html)
    assert match is not None, html[:400]
    return match.group(1), match.group(2)


@pytest.fixture(scope="module")
def partial_frequentist():
    return _run(PARTIAL, bayesian=False)


@pytest.fixture(scope="module")
def partial_bayesian():
    return _run(PARTIAL, bayesian=True)


@pytest.fixture(scope="module")
def graph_dependent():
    return _run(GRAPH_DEPENDENT, bayesian=False)


@pytest.fixture(scope="module")
def identified():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ant.analyze(
            _continuous(),
            graph=GRAPH,
            query=ant.AverageEffect("t", "y"),
            refute="none",
            bootstrap=0,
        )


# --- one verdict renderer ------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["partial_frequentist", "partial_bayesian"])
def test_partially_identified_banner_and_repr_say_partially_identified(fixture, request):
    result = request.getfixturevalue(fixture)
    assert result.identification.status == "PartiallyIdentified"
    tone, text = _banner(result._repr_html_())
    assert (tone, text) == ("ar-caution", "Partially identified")
    assert "Not identified" not in result._repr_html_()
    assert repr(result).startswith("<AnalysisResult partially identified ")


def test_graph_dependent_banner_and_repr_say_graph_dependent(graph_dependent):
    assert graph_dependent.identification.status == "GraphDependent"
    assert _banner(graph_dependent._repr_html_()) == ("ar-caution", "Graph-dependent")
    assert repr(graph_dependent).startswith("<AnalysisResult graph-dependent ")


def test_identified_banner_and_repr(identified):
    assert _banner(identified._repr_html_()) == ("ar-ok", "Identified")
    assert repr(identified).startswith("<AnalysisResult identified effect=")


def test_parametric_identification_keeps_its_qualifier():
    rng = np.random.default_rng(1)
    z = rng.normal(size=300)
    t = (rng.uniform(size=300) < 1 / (1 + np.exp(-z))).astype(float)
    y = t + z + rng.normal(scale=0.2, size=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ant.analyze(
            {"t": t, "y": y, "z": z},
            graph=GRAPH,
            query=ant.Counterfactual("t", "y"),
            refute="none",
            bootstrap=0,
        )
    assert result.identification.status == "IdentifiedUnderParametricRestrictions"
    assert repr(result).startswith("<AnalysisResult identified under parametric restrictions ")
    assert "identified under parametric restrictions" in repr(result.identification)
    assert _banner(result._repr_html_()) == ("ar-ok", "Identified under parametric restrictions")
    staged = ant.Identification.from_view(result.identification, graph=GRAPH, query=result.query)
    assert "is identified under parametric restrictions by gcm.parametric" in staged.statement
    assert staged.to_dict()["qualified_verdict"] == "identified under parametric restrictions"
    assert staged.verdict == "identified"


def test_licensed_design_routes_are_nonparametric():
    from test_transport_interference_lifecycle import (
        _interference_design,
        _interference_query,
        _transport_data,
        _transport_graph,
        _transport_query,
    )

    transport_result = ant.analyze(
        _transport_data(80, 1), graph=_transport_graph(), query=_transport_query()
    )
    assignment, data = _interference_design()
    interference_result = ant.analyze(data, graph=[], query=_interference_query(assignment))
    for result, method_prefix in (
        (transport_result, "transport.sid"),
        (interference_result, "interference.design"),
    ):
        assert result.identification.status == "NonparametricallyIdentified"
        assert result.identification.method.startswith(method_prefix)
        assert _banner(result._repr_html_()) == ("ar-ok", "Identified")
        assert repr(result).startswith("<AnalysisResult identified ")
        assert "parametric restrictions" not in repr(result)
        assert "parametric restrictions" not in repr(result.identification)


# --- nested views of a set-identified result -----------------------------------------------


@pytest.mark.parametrize("fixture", ["partial_frequentist", "partial_bayesian"])
def test_nested_estimate_view_withholds_the_point(fixture, request):
    result = request.getfixturevalue(fixture)
    assert result.estimate.limitation == "identified_set"
    text = repr(result.estimate)
    assert "(identified_set)" in text
    assert "ate=" not in text and "se=" not in text


def test_nested_posterior_view_withholds_mean_and_interval(partial_bayesian):
    posterior = partial_bayesian.posterior
    assert posterior is not None and posterior.limitation == "identified_set"
    text = repr(posterior)
    assert "(identified_set)" in text
    assert "mean=" not in text and "ci95" not in text
    html = posterior._repr_html_()
    assert "identified_set" in html
    assert "95% CI" not in html and "±" not in html


def test_posterior_unidentified_mass_is_never_negative_zero(partial_bayesian):
    mass = partial_bayesian.posterior.unidentified_mass
    assert mass == 0.0
    assert math.copysign(1.0, mass) == 1.0


def test_identified_nested_views_still_show_the_point(identified):
    assert identified.estimate.limitation is None
    assert "ate=" in repr(identified.estimate)


# --- one answer vocabulary -------------------------------------------------------------------


def test_answer_kind_is_closed():
    assert set(CLAIM_KIND_ANSWERS.values()) <= set(ANSWER_KINDS)
    with pytest.raises(ValueError, match="Answer.kind"):
        Answer("mixture")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fixture", "kind", "detail"),
    [
        ("identified", "point", None),
        # A partially identified static class answer carries the identified set
        # over its completions, so it is bounds, not an unquantified partial.
        ("partial_frequentist", "bounds", "identified_set"),
        ("partial_bayesian", "bounds", "identified_set"),
        ("graph_dependent", "bounds", "unidentified_mass"),
    ],
)
def test_live_and_loaded_answers_agree(fixture, kind, detail, request):
    result = request.getfixturevalue(fixture)
    loaded = ant.load(result.export())
    assert loaded.acceptance.verified
    assert result.answer.kind == kind
    assert loaded.answer.kind == kind
    assert loaded.answer.detail == result.answer.detail == detail
    assert loaded.answer.bounds == result.answer.bounds
    if kind == "point":
        assert loaded.answer.value == result.answer.value


def test_bounds_answer_carries_the_identified_set_live_and_loaded():
    from test_temporal_class_bayesian_envelope import _PIN, _cpdag, _pulse, _series

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ant.analyze(
            _series(_PIN),
            graph=_cpdag(),
            query=_pulse(_PIN),
            inference=ant.Bayesian(n_draws=64, backend="conjugate"),
            refute=False,
            bootstrap=0,
            seed=7,
        )
    loaded = ant.load(result.export())
    assert result.answer.kind == loaded.answer.kind == "bounds"
    assert result.answer.bounds is not None
    assert loaded.answer.bounds == result.answer.bounds
    assert result.answer.bounds == result.structural_identified_set
    assert "bounds=[" in repr(result)


def test_function_valued_answers_agree_live_and_loaded():
    rng = np.random.default_rng(3)
    z = rng.normal(size=200)
    t = z + rng.normal(scale=0.35, size=200)
    y = t + z + rng.normal(scale=0.2, size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ant.analyze(
            {"t": t, "y": y, "z": z},
            graph=GRAPH,
            query=ant.ResponseJacobian(["t"], ["y"], at={"t": 0.5}),
            estimator_config={"bandwidth": 0.45},
            refute="none",
            bootstrap=0,
        )
    assert result.answer.kind == "response"
    assert ant.load(result.export()).answer.kind == "response"


def test_limited_function_valued_answer_is_partial_live_and_loaded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ant.analyze(
            _continuous(n=300),
            graph=PARTIAL,
            query=ant.ResponseCurve("t", "y", grid=[0.0, 0.5, 1.0]),
            bootstrap=0,
            refute="none",
        )
    loaded = ant.load(result.export())
    assert result.answer.kind == loaded.answer.kind == "partial"
    assert result.answer.detail == loaded.answer.detail is not None
    verdict = describe_status(result.identification.status)
    assert verdict in {"partially identified", "graph-dependent"}
    assert repr(result).startswith(f"<CausalResponseView {verdict} answer=partial limitation=")


# --- readable report target ------------------------------------------------------------------


def test_report_target_query_uses_variable_names_live_and_loaded(identified):
    for report in (
        identified.inspect().to_dict(),
        ant.load(identified.export()).inspect().to_dict(),
    ):
        query = report["target"]["query"]
        assert query["kind"] == "average_effect"
        assert (query["treatment"], query["outcome"]) == ("t", "y")
        assert (query["treatment_id"], query["outcome_id"]) == (0, 1)
        assert query["target_population"] == "all_observed"


# --- calibration row ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        (
            {"status": "calibrated", "record_id": "cov.a"},
            "calibrated · record cov.a",
        ),
        (
            {"status": "scope_not_assessed", "record_id": "cov.b"},
            "scope_not_assessed · record cov.b",
        ),
        (
            {"status": "unavailable", "reason": "estimator_grid_not_measured"},
            "unavailable · reason estimator_grid_not_measured",
        ),
        (
            {
                "status": "boundary",
                "record_id": "cov.c",
                "reason": "boundary_record",
                "level": 0.95,
                "observed_coverage": 0.912,
                "replicates": 400,
            },
            "boundary · reason boundary_record · record cov.c · level 0.95"
            " · observed coverage 0.912 · 400 replicates",
        ),
        (
            {"status": "some_future_status", "record_id": "cov.d", "new_field": "x"},
            "some_future_status · record cov.d · new_field x",
        ),
    ],
)
def test_calibration_describe_renders_every_status(slot, expected):
    info = CalibrationInfo.from_contract({"claim": {"calibration": slot}})
    assert info.status == slot["status"]
    assert info.describe() == expected
    assert "None" not in info.describe()


def test_calibration_row_in_notebook_card_has_no_none(partial_bayesian, identified):
    for result in (partial_bayesian, identified):
        html = result._repr_html_()
        row = re.search(r"Calibration</span><span>([^<]*)</span>", html)
        assert row is not None
        assert "None" not in row.group(1)
        assert row.group(1) == result.calibration.describe()
        assert row.group(1).startswith(result.calibration.status)
