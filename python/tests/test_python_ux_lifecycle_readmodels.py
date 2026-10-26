"""Scientific read models and frozen designs keep the notebook/system boundary."""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.outcomes import require_success
from antecedent.results._report import InspectionReport, ResultModel, SlotModel

from test_column_name_inputs import _bernoulli, _randomized_named_query
from test_composition_bundle import _consume, _joint_bundle
from test_longitudinal_retained import fixture


@pytest.mark.parametrize("operation", ["estimate", "refresh", "replace_snapshot", "refute"])
@pytest.mark.parametrize("change", ["assignment", "ids", "order"])
def test_every_design_rebinding_path_refuses_and_retains_result(operation, change):
    assignment, outcome, ids = _bernoulli(n=64)
    data = {"y": outcome, "arm": assignment, "id": ids}
    study = ant.prepare(data, query=_randomized_named_query())
    result = require_success(study.estimate())
    replacement = {name: values.copy() for name, values in data.items()}
    if change == "assignment":
        replacement["arm"] = ~assignment
        column = "arm"
    elif change == "ids":
        replacement["id"][0] = "other"
        column = "id"
    else:
        replacement = {name: values[::-1] for name, values in data.items()}
        column = "(?:arm|id)"
    with pytest.raises(ant.errors.CausalValueError, match=f"{column}.*reprepare") as refusal:
        getattr(study, operation)(replacement)
    assert refusal.value.reason_code == "invalid_argument"
    assert study.last_result is result
    assert study.inspect().answer == result.answer
    assert require_success(study.estimate()).answer == result.answer


def test_design_normalization_and_omission_preserve_frozen_design():
    assignment, outcome, ids = _bernoulli(n=64)
    study = ant.prepare(
        {"y": outcome, "arm": assignment, "id": ids}, query=_randomized_named_query()
    )
    original = require_success(study.estimate())
    normalized = require_success(
        study.estimate({"y": outcome, "arm": assignment.astype(float), "id": ids.tolist()})
    )
    assert normalized.answer == original.answer
    assert require_success(
        study.refresh({"y": outcome + assignment})
    ).answer.value == pytest.approx(original.answer.value + 2 * assignment.mean())
    assert require_success(study.estimate({"y": outcome})).answer == original.answer


def test_exact_snapshot_replacement_invalidates_execution_view_without_running():
    from dataclasses import replace

    from antecedent.transport import advanced as transport

    from test_transport_exact import fixture as exact_fixture

    identification, catalog, data = exact_fixture()
    study = ant.prepare(
        data, query=transport.ExactTransportQuery(identification, catalog, {"x": 1.0})
    )
    original = require_success(study.estimate())
    assert study.last_result is original
    assert study.inspect().stage == "executed"
    updated = transport.ExactTransportData(
        (replace(data.laws[0], probabilities=(0.1, 0.4, 0.4, 0.1), snapshot_identity="new"),)
    )
    study.replace_snapshot(updated)
    assert study.last_result is None
    assert study.inspect().stage == "prepared"
    assert require_success(study.estimate()).mean("y") != original.mean("y")


def test_graphless_design_description_retains_native_standing_without_graph_certificate():
    data, query = fixture()
    study = ant.prepare(data, query=query)
    identification = study.identification
    assert identification.status == "DesignBased"
    assert identification.verdict == "design based"
    assert identification.certificate is None
    assert identification.design["basis"] == "declared_design"
    assert identification.inspect().identification.summary == "design based"
    assert study.inspect_plan().identification.summary == "design based"
    assert "declared design" in identification.statement
    assert "design based" in identification._repr_html_()
    with pytest.raises(TypeError, match="does not support item assignment"):
        identification.design["basis"] = "native_graph"
    export = identification.to_dict()
    export["design"]["basis"] = "changed"
    assert identification.design["basis"] == "declared_design"
    result = require_success(study.estimate())
    assert result.longitudinal_regime.support_status == "unlicensed_point_utility"
    assert result.longitudinal_regime.uncertainty == "point_only_no_interval"


class _TypedReadModel(ResultModel):
    values: np.ndarray
    metadata: dict[str, Any]


def test_typed_result_models_freeze_arrays_nested_fields_and_extra_records():
    values = np.array([1.0, 2.0])
    metadata = {"rows": [{"label": "accepted"}]}
    extras = {"nested": [{"identity": "retained"}]}
    handle = object()
    model = _TypedReadModel(values=values, metadata=metadata, receipt=extras, _raw=handle)
    values[0] = 99
    metadata["rows"][0]["label"] = "changed"
    extras["nested"][0]["identity"] = "changed"
    assert model._raw is handle
    assert model.values[0] == 1
    assert model.metadata["rows"][0]["label"] == "accepted"
    assert model.receipt["nested"][0]["identity"] == "retained"
    with pytest.raises(ValueError, match="WRITEABLE"):
        model.values.setflags(write=True)
    with pytest.raises(TypeError, match="does not support item assignment"):
        model.metadata["rows"][0]["label"] = "changed"
    with pytest.raises(TypeError, match="does not support item assignment"):
        model.model_extra["receipt"] = {}
    editable = model.to_dict()
    assert editable["values"] == [1.0, 2.0]
    assert model.model_dump(mode="json")["values"] == [1.0, 2.0]
    editable["metadata"]["rows"][0]["label"] = "changed"
    assert model.metadata["rows"][0]["label"] == "accepted"


def test_inspection_answers_are_detached_and_render_values_and_operations():
    value = {"estimate": 1.25, "scope": {"units": "mmHg"}}
    slot = SlotModel(available=False, summary="point only", reason="not_calibrated")
    report = InspectionReport(
        identification=slot,
        support=slot,
        uncertainty=slot,
        assumptions=slot,
        answer=value,
        stage="executed",
        operations=("inspect", "export"),
    )
    value["scope"]["units"] = "changed"
    assert report.answer["scope"]["units"] == "mmHg"
    with pytest.raises(TypeError, match="does not support item assignment"):
        report.answer["scope"]["units"] = "changed"
    html = report._repr_html_()
    assert "1.25" in html and "point only" in html and "inspect, export" in html


def test_consumed_bundle_node_facts_and_editable_exports_are_independent():
    consumed = _consume(_joint_bundle())
    node = consumed.nodes[0]
    before = node.to_dict()
    with pytest.raises(TypeError, match="does not support item assignment"):
        node.facts["standing"] = "upgraded"
    with pytest.raises(TypeError, match="does not support item assignment"):
        node.values["injected"] = 123
    editable = consumed.to_dict()
    editable["nodes"][0]["facts"]["standing"] = "changed"
    assert node.to_dict() == before
    assert consumed.inspect().stage == "consumed"


def test_declared_law_inspection_has_no_execution_stage_or_export_operation():
    from antecedent.mechanism_composition import JointGaussianLaw

    law = JointGaussianLaw(("x", "y"), [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
    report = law.inspect()
    assert report.stage == "declared"
    assert "export" not in report.operations
    assert not report.assumptions.available
    assert report.assumptions.reason == "not_recorded"
    assert "1.0" in report._repr_html_()
    assert "1.0" in repr(law)
    assert len(repr(law)) < 400


@pytest.mark.parametrize(
    ("status", "available", "limitation", "uncertainty"),
    [
        ("nonparametrically_identified", True, None, "point only"),
        ("partially_identified", True, "identified_set", "identified set"),
        ("unavailable", False, "identification_unavailable", "unavailable"),
        ("nonparametrically_identified", True, None, "uncalibrated"),
    ],
)
def test_inspection_rendering_preserves_scientific_standing(
    status, available, limitation, uncertainty
):
    identification = SlotModel(available=available, summary=status)
    slot = SlotModel(available=False, summary=uncertainty, reason="not_calibrated")
    report = InspectionReport(
        identification=identification,
        support=slot,
        uncertainty=slot,
        assumptions=slot,
        stage="executed",
        family="ExampleScientificResult",
        details={"quantity": "mean blood pressure", "value": 12.5},
        operations=("inspect",),
    )
    assert report.rendering_limitation() == limitation
    html = report._repr_html_()
    assert status in html and uncertainty in html and "12.5" in html
    assert "mean blood pressure" in html
    assert "executed" in repr(report)
    assert report.to_dict()["uncertainty"]["reason"] == "not_calibrated"


def test_interference_design_has_no_fabricated_empty_dag_identification():
    from antecedent import interference as inf

    from test_interference_pointwise_inference import _network

    clusters, edges = _network(8)
    assigned = [cluster >= 4 for cluster in clusters]
    query = inf.InterferenceQuery(
        inf.ClusterRandomization(clusters, treated_clusters=4),
        inf.NeighborFraction(),
        inf.ExposureContrast("y", inf.ExposureLevel(0.0, 0.0), inf.ExposureLevel(1.0, 1.0)),
        network=edges,
        realized_assignment=assigned,
        partial_interference=inf.PartialInterference(clusters),
    )
    study = ant.prepare(
        {"y": [2.0 * float(a) + i % 3 for i, a in enumerate(assigned)]}, graph=[], query=query
    )
    identification = study.identification
    assert identification.certificate is None
    assert identification.design["query_family"] == "InterferenceQuery"
    assert identification.verdict == "design based"
    assert study.inspect().identification.summary == "design based"
    assert require_success(
        study.estimate()
    ).interference.contrast.horvitz_thompson == pytest.approx(2.0)


def test_copied_inspection_and_result_read_models_freeze_replacement_payloads():
    from test_notebook_to_system import _study

    _, study = _study()
    inspection = study.inspect_plan()
    with pytest.raises(TypeError, match="does not support item assignment"):
        inspection.details["status"] = "changed"
    replacement = {"nested": [{"label": "retained"}]}
    copied = inspection.model_copy(update={"details": replacement})
    replacement["nested"][0]["label"] = "changed"
    assert copied.details["nested"][0]["label"] == "retained"
    model = _TypedReadModel(values=np.array([1.0]), metadata={})
    replacement_array = np.array([2.0])
    replacement_metadata = {"rows": [{"label": "accepted"}]}
    revised = model.model_copy(
        update={"values": replacement_array, "metadata": replacement_metadata}
    )
    replacement_array[0] = 99
    replacement_metadata["rows"][0]["label"] = "changed"
    assert revised.values[0] == 2
    assert revised.metadata["rows"][0]["label"] == "accepted"
    assert model.values[0] == 1


def test_prepared_question_and_repreparation_detach_mutable_caller_declarations():
    rng = np.random.default_rng(91)
    treatment = rng.normal(size=80)
    data = {"t": treatment, "y": 2.0 * treatment + rng.normal(scale=0.1, size=80)}
    grid = [-0.5, 0.0, 0.5]
    query = ant.ResponseCurve("t", "y", grid=grid)
    edges = [("t", "y")]
    study = ant.prepare(
        data, graph=edges, query=query, refute=False, inference=ant.Frequentist(bootstrap=0)
    )
    grid[0] = -99.0
    edges.clear()
    identification = study.identification
    assert tuple(identification.query.grid) == (-0.5, 0.0, 0.5)
    result = require_success(study.estimate())
    assert tuple(result.estimand.grid) == (-0.5, 0.0, 0.5)
    repeated = require_success(study.reprepare(data).estimate())
    assert repeated.response.values == result.response.values
    assert repeated.identification == result.identification


def test_repreparation_keeps_graph_and_configuration_despite_caller_edits():
    rng = np.random.default_rng(217)
    z = rng.normal(size=96)
    t = 0.4 * z + rng.normal(size=96)
    data = {"z": z, "t": t, "y": 2.0 * t + z + rng.normal(size=96)}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    configuration = {"se_kind": "hc1"}
    study = ant.prepare(
        data,
        graph=edges,
        query=ant.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        estimator_config=configuration,
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    original = require_success(study.estimate())
    edges.clear()
    configuration["se_kind"] = "invalid"
    repeated = require_success(study.reprepare(data).estimate())
    assert repeated.answer == original.answer
    assert repeated.estimate.se_analytic == original.estimate.se_analytic
    assert study.identification.graph == (("z", "t"), ("z", "y"), ("t", "y"))
    assert study.identification.adjustment_set == ("z",)


@pytest.mark.parametrize("refusal", ["invalid_law", "cancelled"])
def test_restricted_transport_refresh_refusal_preserves_prior_execution_and_export(refusal):
    from dataclasses import replace

    from antecedent.state import CancellationToken
    from antecedent.transport import advanced as transport

    from test_transport_day1 import _restricted_query, _snapshot_law, _surrogate_graph

    study = ant.prepare(
        transport.ExactTransportData((_snapshot_law("original"),)),
        graph=_surrogate_graph(),
        query=_restricted_query(),
    )
    original = require_success(study.estimate())
    exported = bytes(study.export())
    kwargs = {}
    if refusal == "invalid_law":
        law = replace(_snapshot_law("refused"), probabilities=(1.0,))
        expected = ant.CausalError
    else:
        law = _snapshot_law("refused")
        token = CancellationToken()
        token.cancel()
        kwargs["cancel"] = token
        expected = ant.errors.CausalCancelledError
    with pytest.raises(expected):
        require_success(study.refresh(transport.ExactTransportData((law,)), **kwargs))
    assert study.last_result is original
    assert bytes(study.export()) == exported
    assert require_success(study.estimate()).transport.bindings == ("source:z=0.0:original",)
    assert require_success(ant.load(exported)).transport.bindings == original.transport.bindings


def test_restricted_refresh_reuses_identification_and_preserves_original_result(monkeypatch):
    from antecedent.transport import _restricted
    from antecedent.transport import advanced as transport

    from test_transport_day1 import _restricted_query, _snapshot_law, _surrogate_graph

    study = ant.prepare(
        transport.ExactTransportData((_snapshot_law("original"),)),
        graph=_surrogate_graph(),
        query=_restricted_query(),
    )
    original = require_success(study.estimate())
    original_bytes = bytes(original.export())
    stage = study._native.stage

    def no_identification(*args, **kwargs):
        raise AssertionError("refresh must reuse the retained Rust identification stage")

    monkeypatch.setattr(_restricted, "identify_z_transport", no_identification)
    updated = require_success(
        study.refresh(transport.ExactTransportData((_snapshot_law("updated"),)))
    )
    assert study._native.stage is stage
    assert updated.transport.bindings == ("source:z=0.0:updated",)
    assert original.transport.bindings == ("source:z=0.0:original",)
    assert bytes(original.export()) == original_bytes
    assert (
        require_success(ant.load(original_bytes)).transport.bindings == original.transport.bindings
    )
