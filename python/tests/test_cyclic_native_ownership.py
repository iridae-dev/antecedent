"""Actual native cyclic owners separate execution, inspection and materialization."""

import gc

import numpy as np
import pytest
from antecedent.cyclic import (
    AsymptoticNormal,
    CyclicResult,
    Equation,
    EquilibriumModel,
    EvidenceSchema,
    EvidenceVariable,
    LinearCyclicModel,
    PreparedCyclicFamily,
    PreparedCyclicSystem,
    World,
    WorldAnswer,
)
from antecedent.intervention import Set
from antecedent.outcomes import require_success

from test_cyclic import _recovery


def _system():
    return EquilibriumModel(
        (Equation("A", 0.0, (0.0, 0.4), "identity"), Equation("B", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    ).prepare([[]], [[1.0, 0.0]])


def test_execution_and_inspection_do_not_materialize_report_export_or_full_contract(monkeypatch):
    system = _system()

    def forbidden(*args, **kwargs):
        pytest.fail("execution/inspection materialized a report, artifact, or full contract")

    with monkeypatch.context() as bounded:
        bounded.setattr(CyclicResult, "to_dict", forbidden)
        bounded.setattr(CyclicResult, "export", forbidden)
        bounded.setattr(PreparedCyclicSystem, "contract", forbidden)
        bounded.setattr(PreparedCyclicFamily, "contract", forbidden)
        value = require_success(system.execute())
        assert len(value.identity) == 64
        assert len(repr(value)) < 500
        assert value.inspect() is value.inspect()
        assert value.inspect().contract is None
        assert value.inspect().details["contract_is_summary"]
    np.testing.assert_allclose(value.to_dict()["worlds"][0][0]["values"], [1.25, 0.625], atol=1e-10)


def test_native_numeric_buffer_is_readonly_and_survives_python_result_and_study():
    prepared = _system()
    result = require_success(prepared.execute())
    selected = result.numeric(0)
    assert selected.readonly
    np.testing.assert_allclose(selected, [1.25, 0.625], atol=1e-10)
    with pytest.raises(TypeError, match="read.only"):
        selected[0] = 0.0
    array = np.asarray(selected)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE|writeable"):
        array.setflags(write=True)
    del prepared, result, selected
    gc.collect()
    np.testing.assert_allclose(array, [1.25, 0.625], atol=1e-10)


def test_selected_world_composition_uses_retained_native_projection(monkeypatch):
    family = LinearCyclicModel(("A", "B"), (0.0, 0.4, 0.5, 0.0), (1.0, 0.0, 0.0, 1.0)).worlds(
        [World(7, Set("A", 2.0))],
        component_id="feedback",
        source_id="specified",
        certifying_evidence=["analytic"],
    )
    worlds = require_success(family.interventions())
    world = worlds[7]
    assert world.mean("B") == pytest.approx(1.0, abs=1e-10)
    selected = world.numeric(field="mean")
    schema = EvidenceSchema(
        "bench",
        "declared population",
        (EvidenceVariable("A", "meter"), EvidenceVariable("B", "meter")),
    )
    with monkeypatch.context() as bounded:
        bounded.setattr(WorldAnswer, "to_dict", lambda *args, **kwargs: pytest.fail("world DTO"))
        bounded.setattr(CyclicResult, "to_dict", lambda *args, **kwargs: pytest.fail("result DTO"))
        evidence = require_success(world.evidence(schema, "B"))
        assert evidence.quantities[0].units == "meter"
        assert evidence.quantities[0].population_id == "declared population"
        assert evidence.columns(limit=1).items[0]["world"] == 7
    assert evidence.to_dict()["columns"][0]["samples"] == pytest.approx([1.0])
    del world, worlds, family
    gc.collect()
    assert selected.readonly
    assert selected[1] == pytest.approx(1.0, abs=1e-10)


def test_recovered_law_stays_on_result_and_leaves_original_scientific_record_unchanged():
    prepared = _recovery(AsymptoticNormal(4), designs=(None, "A"))
    before = prepared.inspect()
    result = require_success(prepared.execute())
    for key in (
        "draw_origin",
        "law_meaning",
        "design_unresolved",
        "unresolved_directions",
        "fit_dependencies",
        "worlds",
    ):
        assert prepared.inspect()[key] == before[key]
    assert "draw_status" not in prepared.inspect()
    law = result.law
    assert law.inspect().details["draw_count"] == 4
    assert law.inspect().details["draw_status_counts"]["unresolved"] == 4
    statuses = law.draw_status(limit=2)
    assert statuses.count == 4
    assert len(statuses.items) == len(statuses.next().items) == 2
    assert all(item["status"] == "unresolved" for item in statuses.items)
    assert law.unresolved(0).items
    del result, prepared
    gc.collect()
    assert law.inspect().details["draw_count"] == 4
    assert law.to_dict()["origin"]["origin"] == "asymptotic_normal"
