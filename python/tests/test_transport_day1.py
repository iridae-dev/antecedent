"""Day-1 transport verbs: one query, analyze/identify/estimate, one result shape."""

import json

import pytest
from antecedent import Admg, AverageEffect, ResponseCurve, analyze, identify, transport
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
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


def test_inspect_does_not_require_data_or_fit(monkeypatch):
    from antecedent import prepare

    ident = identify(graph=_graph(), query=_curve())
    prepared = prepare(_statistical(), graph=_graph(), query=_curve())
    calls: list[str] = []

    def boom(name):
        def _boom(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"inspect must not {name}")

        return _boom

    # inspect_catalog does bounded catalog search over regime metadata, not sample tables.
    monkeypatch.setattr(transport.EmpiricalTable, "_wire", boom("fit"))
    monkeypatch.setattr(transport.advanced, "evaluate_exact", boom("evaluate"))
    monkeypatch.setattr(transport.advanced, "inspect_catalog", boom("catalog_search"))
    report = ident.inspect()
    assert report.identification.available
    assert report.support.summary
    assert ident.estimate is not None
    prepared_report = prepared.inspect()
    assert prepared_report.identification.available
    assert calls == []


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
    # Two providers compute the same probabilities by different arithmetic, so they
    # agree to rounding (one ulp differs across platforms), not bit for bit.
    learned_values = [value for row in learned.response.values for value in row]
    empirical_values = [value for row in result.response.values for value in row]
    assert learned_values == pytest.approx(empirical_values, rel=1e-12)


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
    assert hasattr(transport.advanced, "StatisticalTransportQuery")


def test_route_transport_tabular_explicit_day1():
    from antecedent import load

    result = analyze(_statistical(), graph=_graph(), query=_curve())
    assert result.study is not None
    refreshed = result.refresh(_statistical())
    assert list(refreshed.response.values) == list(result.response.values)
    loaded = load(result.export())
    assert list(loaded.response.values) == list(result.response.values)
    again = result.study.estimate()
    assert list(again.response.values) == list(result.response.values)


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


@pytest.mark.parametrize(
    "name",
    [
        "DirectFormula",
        "NonTransportableCertificate",
        "PopulationFactor",
        "RecursiveFactorizationFormula",
        "SelectionDiagram",
        "StandardizationFormula",
        "TransportCertificate",
        "TransportIdentification",
        "TransportQuery",
        "TrialTransportEstimate",
        "estimate_trial_effect",
        "identify",
    ],
)
def test_names_moved_in_2_0_point_at_advanced(name):
    # The 1.11 spelling is gone, the 2.0 spelling exists, and the error says where.
    with pytest.raises(AttributeError, match=f"transport.advanced.{name}"):
        getattr(transport, name)
    assert hasattr(transport.advanced, name)
    assert name not in transport.__all__


def _surrogate_graph():
    return Admg.from_edges(
        ["w", "z", "x", "y"],
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )


def _surrogate_law():
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                probabilities.append(
                    (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                )
    return transport.ExactDiscreteLaw(
        "source",
        "do_z",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
    )


def _restricted_query():
    return transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source", kind="experimental", interventions=["z"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )


def test_restricted_experiment_uses_z_transport_not_classical_sid():
    graph = _surrogate_graph()
    query = _restricted_query()
    result = analyze(transport.ExactTransportData((_surrogate_law(),)), graph=graph, query=query)
    assert result.answer.kind == "response"
    assert [row[0] for row in result.response.values] == pytest.approx([0.2, 0.8])
    assert result.transport.formula == "single_source_z_transport_cited_joints_sound_incomplete"
    assert result.transport.provider == "exact_law"
    # A grid execution carries one independently consumable artifact per point.
    artifacts = result.transport.distribution.artifacts
    assert len(artifacts) == 2
    consumed = json.loads(transport.advanced.consume_z_transport_artifact(artifacts[0]))
    assert consumed["probabilities"]
    assert consumed["outcomes"] == ["y"]
    with pytest.raises(CausalUnsupportedError, match="one artifact per target assignment"):
        result.transport.distribution.export()
    contrast = analyze(
        transport.ExactTransportData((_surrogate_law(),)),
        graph=graph,
        query=transport.Transport(
            AverageEffect("x", "y"),
            target="target",
            evidence=query.evidence,
        ),
    )
    assert contrast.answer.value == pytest.approx(0.6)


def test_restricted_column_table_publishes_the_nominal_interval():
    probabilities = []
    rows = {"w": [], "z": [], "x": [], "y": []}
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                probability = (
                    (0.25 if w else 0.75) * (0.35 if x else 0.65) * (0.8 if y == x else 0.2)
                )
                count = round(probability * 10_000)
                probabilities.append(probability)
                rows["w"].extend([float(w)] * count)
                rows["z"].extend([0.0] * count)
                rows["x"].extend([float(x)] * count)
                rows["y"].extend([float(y)] * count)
    result = analyze(rows, graph=_surrogate_graph(), query=_restricted_query())
    assert [row[0] for row in result.response.values] == pytest.approx([0.2, 0.8], abs=0.002)
    assert result.transport.formula == "single_source_z_transport_cited_joints_sound_incomplete"


def test_two_restricted_sources_are_not_combined_by_meta_transport():
    graph = _surrogate_graph()
    query = transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=[
                transport.Source(
                    "alpha", kind="experimental", interventions=["z"], sampling="independent"
                ),
                transport.Source(
                    "beta", kind="experimental", interventions=["z"], sampling="independent"
                ),
            ],
            target_sampling="representative_sample",
        ),
    )
    ident = identify(graph=graph, query=query)
    assert ident.status == "NotIdentified"
    assert (
        ident.certificate["engine"]["reason"] == "z_transport.multi_source_combination_not_searched"
    )


def test_one_restricted_source_identifies_without_the_other():
    graph = _surrogate_graph()
    query = transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=[
                transport.Source(
                    "alpha", kind="experimental", interventions=["z"], sampling="independent"
                ),
                transport.Source(
                    "beta", kind="experimental", interventions=["z"], sampling="independent"
                ),
            ],
            target_sampling="representative_sample",
        ),
    )
    law = _surrogate_law()
    law = transport.ExactDiscreteLaw(
        "alpha",
        law.regime,
        law.axes,
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
    )
    result = analyze(transport.ExactTransportData((law,)), graph=graph, query=query)
    assert [row[0] for row in result.response.values] == pytest.approx([0.2, 0.8])
    assert result.transport.formula == "single_source_z_transport_cited_joints_sound_incomplete"


def test_restricted_experiment_keeps_the_z_transport_bounds():
    names = [f"v{index}" for index in range(13)]
    graph = Admg.from_edges(names, [(names[index], names[index + 1]) for index in range(12)])
    query = transport.Transport(
        ResponseCurve("v12", "v11", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source", kind="experimental", interventions=["v0"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="z_transport.unsupported_observed_count"):
        identify(graph=graph, query=query)
    controllable = ["c0", "c1", "c2", "c3", "c4"]
    small = Admg.from_edges(
        ["x", "y", *controllable],
        [("x", "y"), *[(name, "x") for name in controllable]],
    )
    too_many = transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source",
                kind="experimental",
                interventions=controllable,
                sampling="independent",
            ),
            target_sampling="representative_sample",
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="z_transport.unsupported_controllable_count"):
        identify(graph=small, query=too_many)


def test_root_transport_query_removal_names_its_replacement():
    import antecedent

    with pytest.raises(AttributeError, match="antecedent.transport.advanced.TransportQuery"):
        antecedent.TransportQuery  # noqa: B018
    assert "TransportQuery" not in antecedent.__all__


def _three_source_query():
    return transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=[
                transport.Source(
                    name, kind="experimental", interventions=["z"], sampling="independent"
                )
                for name in ("alpha", "beta", "gamma")
            ],
            target_sampling="representative_sample",
        ),
    )


def test_three_restricted_sources_are_refused_with_a_registered_code():
    with pytest.raises(
        CausalUnsupportedError, match="multi_source_combination_not_searched"
    ) as refused:
        identify(graph=_surrogate_graph(), query=_three_source_query())
    assert refused.value.reason_code == "transport_not_certified"
    with pytest.raises(CausalUnsupportedError, match="multi_source_combination_not_searched"):
        analyze(
            transport.ExactTransportData((_surrogate_law(),)),
            graph=_surrogate_graph(),
            query=_three_source_query(),
        )


def _snapshot_law(snapshot):
    law = _surrogate_law()
    return transport.ExactDiscreteLaw(
        law.population,
        law.regime,
        law.axes,
        law.probabilities,
        snapshot,
        interventions=law.interventions,
    )


def test_restricted_prepared_study_is_a_first_class_handle():
    from antecedent import load, prepare
    from antecedent.results import PhysicalPlanView

    graph = _surrogate_graph()
    study = prepare(
        transport.ExactTransportData((_surrogate_law(),)), graph=graph, query=_restricted_query()
    )
    with pytest.raises(CausalUnsupportedError, match="estimate before exporting") as refused:
        study.export()
    assert refused.value.reason_code == "not_executed"
    result = study.estimate()
    assert [row[0] for row in result.response.values] == pytest.approx([0.2, 0.8])
    assert isinstance(study.plan, PhysicalPlanView)
    assert study.plan.kernels == "z_transport_exact_point"
    assert study.structure_source == "explicit"
    preview = study.preview_transform("display_precision")
    assert preview["refused"] == "false"
    refused_preview = study.preview_transform("change_graph")
    assert refused_preview["refused"] == "true"
    assert refused_preview["refusal_code"] == "option_not_applicable"
    study_view = load(study.export())
    assert [row[0] for row in study_view.response.values] == pytest.approx([0.2, 0.8])
    assert study.inspect().identification.available

    # The result view carries one z-transport artifact per grid point, and
    # every reloaded number is recomputed by the native consumer.
    blob = result.export()
    payload = json.loads(blob[len(b"ANTECEDENT-TRANSPORT-VIEW\x01") :])
    assert len(payload["specialist_artifacts"]) == 2
    loaded = load(blob)
    assert isinstance(loaded, CausalResponseView)
    assert [row[0] for row in loaded.response.values] == pytest.approx([0.2, 0.8])
    assert loaded.identification.status == "NonparametricallyIdentified"
    assert loaded.transport.formula == result.transport.formula
    assert loaded.transport.bindings == result.transport.bindings
    # Editing a point artifact inside the envelope is refused on load.
    import base64
    import struct

    first = base64.b64decode(payload["specialist_artifacts"][0])
    probabilities = json.loads(transport.advanced.consume_z_transport_artifact(first))[
        "probabilities"
    ]

    def cbor_bytes(value):
        # The point artifact travels as a CBOR byte vector (an array of small ints).
        raw = b"\xfb" + struct.pack(">d", value)
        return b"".join(bytes([b]) if b < 24 else b"\x18" + bytes([b]) for b in raw)

    pattern = cbor_bytes(probabilities[0])
    assert pattern in first
    edited = first.replace(pattern, cbor_bytes(probabilities[0] + 0.05), 1)
    payload["specialist_artifacts"][0] = base64.b64encode(edited).decode("ascii")
    with pytest.raises(CausalSerializationError, match="does not replay"):
        load(b"ANTECEDENT-TRANSPORT-VIEW\x01" + json.dumps(payload).encode())

    contrast = analyze(
        transport.ExactTransportData((_surrogate_law(),)),
        graph=graph,
        query=transport.Transport(
            AverageEffect("x", "y"), target="target", evidence=_restricted_query().evidence
        ),
    )
    reloaded = load(contrast.export())
    assert isinstance(reloaded, AnalysisResult)
    assert reloaded.answer.value == pytest.approx(0.6)


def test_restricted_refresh_with_a_new_snapshot_identity_rebinds_laws_and_catalog():
    from antecedent import prepare

    study = prepare(
        transport.ExactTransportData((_snapshot_law("snap_one"),)),
        graph=_surrogate_graph(),
        query=_restricted_query(),
    )
    first = study.estimate()
    assert first.transport.bindings == ("source:z=0.0:snap_one",)
    second = study.refresh(transport.ExactTransportData((_snapshot_law("snap_two"),)))
    assert second.transport.bindings == ("source:z=0.0:snap_two",)
    assert [row[0] for row in second.response.values] == pytest.approx([0.2, 0.8])
    # Same identity, native rebind: still the same claim.
    third = study.refresh(transport.ExactTransportData((_snapshot_law("snap_two"),)))
    assert third.transport.bindings == ("source:z=0.0:snap_two",)
    study.replace_snapshot(transport.ExactTransportData((_snapshot_law("snap_three"),)))
    with pytest.raises(CausalUnsupportedError, match="estimate before exporting"):
        study.export()
    assert study.estimate().transport.bindings == ("source:z=0.0:snap_three",)


def test_restricted_missing_evidence_is_named_in_variables_and_survives_export():
    from antecedent import load

    query = transport.Transport(
        AverageEffect("x", "y"), target="target", evidence=_restricted_query().evidence
    )
    ident = identify(graph=_surrogate_graph(), query=query)
    assert ident.status == "NotIdentified"
    assert ident.certificate["outcome"] == "missing_evidence"
    assert "VariableId" not in json.dumps(ident.certificate)
    assert ident.certificate["engine"]["missing"]["kind"] in {
        "unassigned_controllable",
        "cited_factor",
    }
    assert (
        "names no level" in ident.certificate["missing_detail"]
        or "not supplied" in ident.certificate["missing_detail"]
    )
    result = analyze(None, graph=_surrogate_graph(), query=query)
    assert result.answer.kind == "unavailable"
    blob = result.export()
    payload = json.loads(blob[len(b"ANTECEDENT-TRANSPORT-VIEW\x01") :])
    assert "identification_snapshot" in payload
    loaded = load(blob)
    assert loaded.answer.kind == "unavailable"
    assert loaded.identification.status == "NotIdentified"


def test_statistical_payload_shape_and_support_budget_are_enforced():
    from antecedent.errors import CausalResourceError, CausalValueError

    from test_transport_statistical import fixture

    identified, catalog, data = fixture()
    with pytest.raises(CausalResourceError, match="max_support_rows"):
        transport.advanced.prepare_statistical(
            identified, catalog, data, at={"x": 1.0}, max_support_rows=1
        )
    study = transport.advanced.prepare_statistical(
        identified, catalog, data, at={"x": 1.0}, bootstrap=0
    )
    result = study.estimate()
    with pytest.raises(CausalValueError, match="samples field"):
        study._native.refresh(tuple(data.laws))
    with pytest.raises(CausalValueError, match="RegimeSample"):
        study._native.refresh(transport.StatisticalTransportData(samples=(_surrogate_law(),)))
    from antecedent.state import CancellationToken

    token = CancellationToken()
    token.cancel()
    before = study.export()
    from antecedent.errors import CausalCancelledError

    with pytest.raises(CausalCancelledError):
        study.estimate(cancel=token)
    assert study.export() == before
    assert study.estimate().probabilities == pytest.approx(result.probabilities)
