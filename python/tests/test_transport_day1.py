"""Day-1 transport verbs: one query, analyze/identify/estimate, one result shape."""

import pytest
from antecedent import Admg, AverageEffect, ResponseCurve, analyze, identify, transport
from antecedent.results import AnalysisResult, CausalResponseView


def _graph():
    return Admg.from_edges(["x", "y"], [("x", "y")])


def _single_source():
    return transport.Evidence(
        source=transport.Source(
            "source", kind="experimental", interventions=["x"], sampling="independent"
        ),
        target_sampling="representative_sample",
    )


def _curve():
    return transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=_single_source(),
    )


def _ate():
    return transport.Transport(
        AverageEffect("x", "y"),
        target="target",
        evidence=_single_source(),
    )


def _statistical():
    return transport.StatisticalTransportData(
        samples=(
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 50 + [1.0] * 50},
                interventions=(("x", 0.0),),
            ),
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 20 + [1.0] * 80},
                interventions=(("x", 1.0),),
            ),
        )
    )


def _exact():
    return transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.5, 0.5),
                "v1",
                interventions=(("x", 0.0),),
            ),
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.2, 0.8),
                "v1",
                interventions=(("x", 1.0),),
            ),
        )
    )


def _meta():
    graph = Admg.from_edges(
        ["x", "z", "y"], [("x", "z"), ("z", "y")], bidirected=[("x", "z"), ("x", "y")]
    )
    evidence = transport.Evidence(
        source=[
            transport.Source(
                "a",
                kind="experimental",
                interventions=["x"],
                sampling="independent",
                selections=["y"],
            ),
            transport.Source(
                "b",
                kind="experimental",
                interventions=["z"],
                sampling="independent",
                selections=["z"],
            ),
        ],
        target_sampling="representative_sample",
    )
    query = transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=evidence,
    )
    laws = []
    for population, regime, treatment, outcome, probabilities in [
        ("a", "x_trial", "x", "z", (0.2, 0.8)),
        ("b", "z_trial", "z", "y", (0.1, 0.9)),
    ]:
        for value, probability in enumerate(probabilities):
            laws.append(
                transport.ExactDiscreteLaw(
                    population,
                    regime,
                    ((outcome, (0.0, 1.0)),),
                    (1 - probability, probability),
                    "supplied-v1",
                    interventions=((treatment, float(value)),),
                )
            )
    return graph, query, transport.ExactTransportData(tuple(laws))


def test_identify_estimate_analyze_loop():
    graph = _graph()
    query = _curve()
    ident = identify(graph=graph, query=query)
    assert ident.status == "NonparametricallyIdentified"
    report = ident.inspect()
    assert report.identification.available
    result = ident.estimate(_statistical())
    assert isinstance(result, CausalResponseView)
    assert result.answer.kind == "response"
    assert list(result.response.values) == [[0.5], [0.8]]
    again = analyze(_statistical(), graph=graph, query=query)
    assert again.answer.kind == result.answer.kind
    assert list(again.response.values) == list(result.response.values)


def test_clerical_inference_from_single_table():
    data = {
        "x": [0.0] * 50 + [1.0] * 100,
        "y": [0.0] * 25 + [1.0] * 25 + [0.0] * 20 + [1.0] * 80,
    }
    result = analyze(data, graph=_graph(), query=_curve())
    assert result.answer.kind == "response"
    assert [row[0] for row in result.response.values] == [0.5, 0.8]
    assert result.transport.provider == "empirical_table"
    assert result.transport.bindings


def test_provider_opt_in_is_explicit():
    result = analyze(_statistical(), graph=_graph(), query=_curve())
    assert result.transport.provider == "empirical_table"
    learned = analyze(
        _statistical(),
        graph=_graph(),
        query=_curve(),
        provider=transport.LearnedCategorical(),
    )
    assert learned.transport.provider == "learned_categorical"
    assert list(learned.response.values) == list(result.response.values)


def test_missing_evidence_inspect_and_unavailable_answer():
    query = _ate()
    ident = identify(graph=_graph(), query=query)
    report = ident.inspect()
    assert "AverageEffect of y from x" in report.identification.summary
    assert "engine" in report.identification.payload
    assert "formula" not in report.identification.payload
    support = report.support
    assert "AverageEffect of y from x is identified for target" in support.payload["detail"]
    assert "unbound" in support.payload["detail"]
    partial = transport.ExactTransportData((_exact().laws[1],))
    result = analyze(partial, graph=_graph(), query=query)
    assert result.answer.kind == "unavailable"
    assert "unbound" in result.answer.detail


def test_result_shape_matches_across_regimes():
    curve = _curve()
    statistical = analyze(_statistical(), graph=_graph(), query=curve)
    exact = analyze(_exact(), graph=_graph(), query=curve)
    graph, meta_query, meta_data = _meta()
    complementary = analyze(meta_data, graph=graph, query=meta_query)
    for result, expected in (
        (statistical, [[0.5], [0.8]]),
        (exact, [[0.5], [0.8]]),
        (complementary, [[0.26], [0.74]]),
    ):
        assert isinstance(result, CausalResponseView)
        assert result.answer.kind == "response"
        assert hasattr(result, "inspect")
        assert result.transport.formula
        assert [round(row[0], 2) for row in result.response.values] == [
            round(value[0], 2) for value in expected
        ]
    contrast = analyze(_exact(), graph=_graph(), query=_ate())
    assert isinstance(contrast, AnalysisResult)
    assert contrast.answer.kind == "point"
    assert contrast.answer.value == pytest.approx(0.3)


def test_day1_surface_excludes_stage_types():
    assert "Transport" in transport.__all__
    assert "ExactTransportData" in transport.__all__
    assert "StatisticalTransportQuery" not in transport.__all__
    assert not hasattr(transport, "StatisticalTransportQuery")
    assert not hasattr(transport, "TransportQuery")
    assert transport.advanced.StatisticalTransportQuery is not None


def test_day1_export_load_rehydrates_view():
    from antecedent import load

    statistical = analyze(_statistical(), graph=_graph(), query=_ate())
    loaded = load(statistical.export())
    assert isinstance(loaded, AnalysisResult)
    assert loaded.answer.kind == statistical.answer.kind
    assert loaded.answer.value == pytest.approx(statistical.answer.value)
    assert loaded.inspect().support.summary
    assert not loaded.export().startswith(b"ANTECEDENT-STATISTICAL-TRANSPORT")
    exact = analyze(_exact(), graph=_graph(), query=_curve())
    restored = load(exact.export())
    assert isinstance(restored, CausalResponseView)
    assert restored.answer.kind == "response"
    assert list(restored.response.values) == list(exact.response.values)
    assert restored.support.status == exact.support.status
    assert restored.inspect().support.summary == exact.inspect().support.summary


def test_trial_aipw_on_analyze():
    from antecedent.errors import CausalUnsupportedError
    from antecedent.learners import Linear

    data = transport.TrialAipwData(
        {},
        [1.0, 3.0] * 60 + [0.0] * 80,
        [False, True] * 60 + [False] * 80,
        [True] * 120 + [False] * 80,
        [0.5] * 200,
        "independent_samples",
    )
    query = transport.Transport(
        AverageEffect("a", "y"),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "trial", kind="experimental", interventions=["a"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )
    try:
        analyze(data, graph=Admg.from_edges(["a", "y"], [("a", "y")]), query=query)
    except CausalUnsupportedError as error:
        assert error.reason_code == "option_not_applicable"
    else:
        raise AssertionError("TrialAipwData must not auto-promote")
    result = analyze(
        data,
        graph=Admg.from_edges(["a", "y"], [("a", "y")]),
        query=query,
        provider=transport.TrialAipw(outcome=Linear(), folds=3),
        inference=transport.TransportInference(bootstrap=0, seed=8),
    )
    assert isinstance(result, AnalysisResult)
    assert result.answer.kind == "point"
    assert result.answer.value == pytest.approx(2.0)
    assert result.transport.provider == "trial_aipw"
    curve = transport.Transport(
        ResponseCurve("a", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=query.evidence,
    )
    try:
        analyze(
            data,
            graph=Admg.from_edges(["a", "y"], [("a", "y")]),
            query=curve,
            provider=transport.TrialAipw(outcome=Linear(), folds=3),
        )
    except CausalUnsupportedError as error:
        assert error.reason_code == "option_not_applicable"
    else:
        raise AssertionError("Trial AIPW must refuse a response curve")


def test_provider_refused_on_ordinary_query():
    from antecedent.errors import CausalUnsupportedError

    try:
        analyze(
            {"x": [0.0, 1.0], "y": [1.0, 2.0]},
            graph=[("x", "y")],
            query=AverageEffect("x", "y"),
            provider=transport.EmpiricalTable(),
        )
    except CausalUnsupportedError as error:
        assert error.reason_code == "option_not_applicable"
    else:
        raise AssertionError("provider= must be refused on non-transport queries")
