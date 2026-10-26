# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cyclic presentation retains native meanings without hidden execution."""

from dataclasses import FrozenInstanceError

import pytest
from antecedent.cyclic import (
    CyclicResult,
    Equation,
    EquilibriumModel,
    EvidenceSchema,
    EvidenceVariable,
    LinearCyclicModel,
    PreparedCyclicFamily,
    PreparedCyclicSystem,
    World,
    WorldSelector,
    consume_artifact,
    consume_family_artifact,
)
from antecedent.outcomes import Outcome, require_success


def _worlds():
    return LinearCyclicModel(("A", "B"), (0.0, 0.4, 0.5, 0.0), (1.0, 0.0, 0.0, 1.0)).worlds(
        [World(7)], component_id="loop", source_id="specified", certifying_evidence=["bench"]
    )


def test_large_payload_repr_and_inspect_only_read_retained_metadata(monkeypatch):
    payload = {
        "identity": "a" * 64,
        "worlds": [[{"values": [3.0] * 100_000}]],
        "claim": "specified_or_fitted_model_conditional",
        "calibration": "unmeasured",
    }
    value = CyclicResult(payload, b"large artifact" * 100_000)
    monkeypatch.setattr(CyclicResult, "to_dict", lambda self: pytest.fail("materialized report"))
    monkeypatch.setattr(CyclicResult, "export", lambda self: pytest.fail("exported artifact"))
    for _ in range(2):
        assert len(repr(value)) < 300
        assert len(value.inspect()._repr_html_()) < 3000
        assert value.inspect() is value.inspect()
    with pytest.raises(TypeError, match="does not support item assignment"):
        value.report["worlds"][0][0]["values"][0] = 4.0
    payload["worlds"][0][0]["values"][0] = 5.0
    assert value.report["worlds"][0][0]["values"][0] == 3.0


def test_prepared_repr_does_not_read_request_or_run_solver(monkeypatch):
    class Unrenderable:
        def __repr__(self):
            pytest.fail("rendered request")

    for cls in (PreparedCyclicSystem, PreparedCyclicFamily):
        prepared = object.__new__(cls)
        object.__setattr__(prepared, "request_json", Unrenderable())
        assert len(repr(prepared)) < 50
    prepared = _worlds()
    monkeypatch.setattr(PreparedCyclicFamily, "execute", lambda *a, **k: pytest.fail("solved"))
    # Inspection must not decode the full request again, even for large draws/data.
    object.__setattr__(prepared, "request_json", "not JSON")
    assert prepared.inspect()["worlds"] == [7]
    assert prepared.inspect()["preparation_evaluations"] == 0


def test_live_common_inspection_retains_actual_contract_and_loaded_has_no_study():
    prepared = _worlds()
    expected = prepared.contract()
    outcome = prepared.execute()
    assert outcome.study is prepared
    report = outcome.inspect()
    assert report.identification.summary == expected["identification_kind"]
    assert report.contract["identities"] == expected["identities"]
    assert report.to_dict()["target"] == expected["layers"]["target"]
    assert report.uncertainty.payload["method"] == "none"
    assert not report.uncertainty.available
    assert outcome.answer.kind == "structured"
    assert outcome.distribution is None
    assert outcome.identities["target_id"] == expected["identities"]["target"]
    assert outcome.provenance["native_contract"]["request_identity"] == outcome.value.identity
    value = require_success(outcome)
    loaded = consume_family_artifact(bytes(value.export()), expected_identity=value.identity)
    assert loaded.study is None
    assert loaded.value.to_dict() == value.to_dict()
    assert bytes(loaded.export()) == bytes(value.export())
    assert loaded.inspect().stage == "consumed"
    assert not loaded.inspect().identification.available


def test_system_live_study_and_native_scientific_metadata():
    prepared = EquilibriumModel((Equation("A", 0.0, (0.0,), "identity"),), (1.0,)).prepare(
        [[]], [[0.0]]
    )
    outcome = prepared.execute()
    assert outcome.study is prepared
    assert outcome.inspect().calibration == "unmeasured"
    value = require_success(outcome)
    loaded = consume_artifact(bytes(value.export()), expected_identity=value.identity)
    assert loaded.study is None
    assert loaded.value.to_dict() == value.to_dict()
    native = CyclicResult(
        {
            "identity": "b" * 64,
            "answer": {
                "family": "graph",
                "result": {"assumptions": ["declared ioSCM"], "diagnostics": ["native warning"]},
            },
        },
        b"",
    )
    inspection = Outcome(value=native).inspect()
    assert inspection.assumptions.payload["assumptions"] == ("declared ioSCM",)
    assert inspection.diagnostics == ("native warning",)


def test_typed_world_read_models_preserve_law_and_declared_evidence_coordinates():
    prepared = _worlds()
    worlds = prepared.interventions()
    assert worlds.study is prepared
    world = worlds.value[7]
    assert len(repr(world)) < 300
    with pytest.raises(TypeError, match="does not support item assignment"):
        world.outcome["mean"][0] = 9.0
    with pytest.raises(FrozenInstanceError):
        world.world = 9
    inspection = world.inspect()
    assert inspection.support.payload["certificate"] == world.certificate
    assert inspection.details["physical_stability"] == "unclaimed"
    # A world mean is a scalar reader; the common world answer stays structured.
    assert isinstance(world.mean("B"), float)
    assert Outcome(value=world).answer.kind == "structured"
    assert Outcome(value=world).distribution is None
    evidence_outcome = prepared.evidence(
        EvidenceSchema(
            "bench",
            "bench population",
            (EvidenceVariable("A", "meter"), EvidenceVariable("B", "meter")),
        ),
        [WorldSelector(7, "B")],
    )
    assert evidence_outcome.study is prepared
    evidence = evidence_outcome.value
    assert evidence.quantities[0].units == "meter"
    assert evidence.quantities[0].population_id == "bench population"
    assert evidence.report["columns"][0]["samples"] == pytest.approx([world.mean("B")])
