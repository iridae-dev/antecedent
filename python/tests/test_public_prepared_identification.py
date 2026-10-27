"""Prepared identification navigates retained proofs without projecting or executing."""

from __future__ import annotations

import gc
import json
from dataclasses import FrozenInstanceError

import antecedent as ant
import pytest
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.execution import Execution
from antecedent.execution_attempt import observe_native_attempts
from antecedent.identify import Identification, PreparedIdentification
from antecedent.outcomes import Refusal


def _study():
    return ant.prepare(
        {
            "z": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "t": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 2.0, 1.0, 3.0, 0.1, 1.9, 1.1, 2.9],
        },
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=Execution(seed=7, threads=1),
        refute=False,
    )


def test_public_prepared_identification_is_bounded_and_never_uses_legacy_conversion(monkeypatch):
    study = _study()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bounded reading must not materialize a proof")

    monkeypatch.setattr(Identification, "from_prepared", forbidden)
    monkeypatch.setattr(PreparedIdentification, "to_dict", forbidden)

    def read():
        view = study.identification
        assert isinstance(view, PreparedIdentification)
        assert not isinstance(view, Identification)
        summary = view.inspect()
        assert summary.kind == "identification"
        assert summary.child_count == view.child_count
        assert summary.metadata["graph_class"] == "Dag"
        assert view.children(limit=1)[0].kind == "layer"
        assert len(repr(view)) < 400
        assert len(view._repr_html_()) < 600
        assert study.inspect().stage == "prepared"
        with pytest.raises(FrozenInstanceError):
            view.execution = Execution()
        return summary

    observed = observe_native_attempts(read)
    observed.unwrap()
    for operation in ("glm_fit", "posterior_fit", "posterior_draw", "score_construction"):
        assert observed.report.operations[operation].attempted == 0


def test_public_prepared_identification_child_retains_owner_and_explicit_full_proof():
    study = _study()
    root = study.identification
    children = root.children()
    original = [child.summary.to_dict() for child in children]
    del root, study
    gc.collect()
    assert [child.summary.to_dict() for child in children] == original
    materialized = [child.to_dict() for child in children]

    def proofs(node):
        if "identification" in node:
            yield node["identification"]
        for child in node.get("children", ()):
            yield from proofs(child)

    retained = [proof for tree in materialized for proof in proofs(tree)]
    assert retained
    for tree in materialized:
        assert tree["names"] == ["z", "t", "y"]
    assert any(
        estimand["adjustment_set"] == [0] for proof in retained for estimand in proof["estimands"]
    )


class _NativeNode:
    """Read-model boundary fixture; scientific correctness uses real native cases above."""

    def __init__(self, summary, children=()):
        self.summary = summary
        self.nodes = children
        self.materializations = 0

    def summary_json(self):
        return json.dumps(self.summary)

    def child(self, index):
        return self.nodes[index]

    def children(self, *, offset, limit):
        return self.nodes[offset : offset + limit]

    def materialize(self, *, seed, threads):
        self.materializations += 1
        summary = self.summary

        class Payload:
            def to_dict(self):
                return {"summary": summary, "children": [], "seed": seed, "names": ["z", "t", "y"]}

        return Payload()

    def schema_names(self):
        return ["z", "t", "y"]


def test_composite_summary_preserves_native_mass_and_horizon_without_single_verdict():
    first = _NativeNode({"kind": "horizon", "horizon": 1, "status": "first-status", "children": 0})
    second = _NativeNode(
        {"kind": "horizon", "horizon": 3, "status": "second-status", "children": 0}
    )
    native = _NativeNode(
        {
            "kind": "temporal_class",
            "status": None,
            "status_unavailable": "composite_identification",
            "children": 2,
            "identified_weight": 0.7,
            "unidentified_weight": 0.3,
            "truncated_completions": 4,
        },
        (first, second),
    )
    view = PreparedIdentification.from_native(native, Execution(seed=7))
    assert view.status is None
    assert view.status_unavailable == "composite_identification"
    assert view.summary.metadata["identified_weight"] == 0.7
    assert view.summary.metadata["unidentified_weight"] == 0.3
    assert view.summary.metadata["truncated_completions"] == 4
    page = view.children(offset=1, limit=1)
    assert len(page) == 1 and page[0].status == "second-status"
    assert page[0].summary.metadata["horizon"] == 3
    assert view.child(0).summary.metadata["horizon"] == 1
    assert native.materializations == first.materializations == second.materializations == 0
    assert view.children(offset=2) == ()
    with pytest.raises(CausalTypeError):
        bool(view)
    with pytest.raises(TypeError, match="does not support item assignment"):
        view.summary.metadata["identified_weight"] = 1.0


def test_explicit_materialization_uses_execution_without_changing_retained_policy():
    native = _NativeNode({"kind": "proof", "status": "native-status", "children": 0})
    policy = Execution(seed=7, threads=1)
    view = PreparedIdentification.from_native(native, policy)
    converted = view.to_dict(execution=Execution(seed=29, threads=1))
    assert converted["seed"] == 29
    assert converted["names"] == ["z", "t", "y"]
    assert native.materializations == 1
    assert view.execution is policy
    with pytest.raises(CausalTypeError):
        view.to_dict(execution={})


def test_explicit_unavailable_identification_preserves_original_refusal():
    refusal = Refusal(code="effect_not_identified", message="native scientific refusal")
    details = {"provider_requirements": ["declared-only"]}
    view = PreparedIdentification.unavailable(
        "provider_identification_not_independently_verified",
        kind="provider",
        refusal=refusal,
        details=details,
    )
    details["provider_requirements"].append("mutated")
    assert view.refusal is refusal
    assert view.status is None
    assert view.status_unavailable == "provider_identification_not_independently_verified"
    assert view.summary.metadata["detail_count"] == 1
    assert "details" not in view.summary.metadata
    assert view.to_dict()["details"]["provider_requirements"] == ["declared-only"]
    assert view.children() == ()
    assert view.to_dict()["children"] == []
    with pytest.raises(CausalValueError):
        view.child(0)


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"limit": 257}, {"limit": True}])
def test_identification_pages_reject_invalid_controls(kwargs):
    view = PreparedIdentification.unavailable("not_executed")
    with pytest.raises((CausalTypeError, CausalValueError)):
        view.children(**kwargs)
