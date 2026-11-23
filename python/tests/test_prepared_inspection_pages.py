"""Canonical prepared inspection borrows native metadata and explicitly pages records."""

from __future__ import annotations

import gc
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalStateError, CausalTypeError, CausalValueError
from antecedent.prepared import InspectionPage, PreparedAnalysis


def _prepare():
    z = np.tile([0.0, 0.0, 1.0, 1.0], 20)
    t = np.tile([0.0, 1.0, 0.0, 1.0], 20)
    return ant.prepare(
        ant.data.table({"z": z, "t": t, "y": 1 + 2 * t + z}),
        graph=ant.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]),
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=19, threads=1),
        refute="none",
    )


def test_prepared_identity_domains_and_all_four_slots_match_native_header():
    study = _prepare()
    native = study._engine._native
    raw = json.loads(native.prepared_inspection_json())
    report = study.inspect()
    assert report.stage == "prepared"
    assert report.contract is None
    for identity in (
        "target_id",
        "identification_id",
        "identification_product_id",
        "program_id",
        "inference_binding_id",
        "observation_id",
        "data_snapshot_id",
    ):
        assert getattr(report, identity) == raw["identities"][identity]
    assert report.details["snapshot_unavailable"] == raw["identities"]["snapshot_unavailable"]
    for name in ("identification", "support", "uncertainty", "assumptions"):
        slot = getattr(report, name)
        assert slot.available is raw[name]["available"]
        assert slot.reason == raw[name].get("reason")
    assert report.identification.payload["retained"] == study.identification.inspect().metadata
    # This explicit legacy contract call belongs only to the identity comparison,
    # never to canonical inspection. Its genuine snapshot does not rewrite the
    # earlier sealed prepared header or get substituted from another identity.
    contract = native.contract()
    for field, coordinate in (
        ("target_id", "target"),
        ("identification_id", "identification"),
        ("identification_product_id", "identification_product"),
        ("program_id", "program"),
        ("inference_binding_id", "inference_binding"),
        ("observation_id", "observation"),
    ):
        assert getattr(report, field) == contract.get(coordinate)
    assert study.inspect().data_snapshot_id == raw["identities"]["data_snapshot_id"]
    assert json.loads(native.prepared_inspection_json()) == raw


def test_bounded_inspection_does_not_enter_legacy_contract_or_explicit_pages(monkeypatch):
    from antecedent._prepared_identification import PreparedIdentification
    from antecedent.estimation import _PreparedEngine as Engine
    from antecedent.identify import Identification

    study = _prepare()

    def forbidden(*args, **kwargs):
        pytest.fail("bounded prepared inspection entered work or materialization")

    monkeypatch.setattr(ant.Execution, "_call", forbidden)
    monkeypatch.setattr(Engine, "inspect_plan", forbidden)
    monkeypatch.setattr(Engine, "identification", property(forbidden))
    monkeypatch.setattr(Identification, "from_prepared", forbidden)
    monkeypatch.setattr(PreparedIdentification, "to_dict", forbidden)
    for name in (
        "assumption_obligations",
        "uncertainty_components",
        "evidence_obligations",
        "evidence_actions",
    ):
        monkeypatch.setattr(PreparedAnalysis, name, forbidden)
    for _ in range(2):
        report = study.inspect()
        assert report.stage == "prepared" and report.contract is None
        assert len(repr(report)) < 500
        assert len(report._repr_html_()) < 4000
    assert study._engine.last_result is None


@pytest.mark.parametrize(
    ("public", "native"),
    [
        ("assumption_obligations", "prepared_obligations_json"),
        ("uncertainty_components", "prepared_uncertainty_json"),
        ("evidence_obligations", "prepared_evidence_obligations_json"),
        ("evidence_actions", "prepared_evidence_actions_json"),
    ],
)
def test_pages_preserve_native_records_and_retained_owner_after_study_drop(public, native):
    study = _prepare()
    source = getattr(study._engine._native, native)
    expected = json.loads(source(offset=0, limit=256))
    assert expected["total"] <= 256
    page = getattr(study, public)(limit=1)
    assert isinstance(page, InspectionPage)
    assert page.total == expected["total"]
    assert page.to_dict()["items"] == expected["items"][:1]
    with pytest.raises(FrozenInstanceError):
        page.total = 0
    if page.items:
        with pytest.raises(TypeError, match="does not support item assignment"):
            page.items[0]["description"] = "changed"
    assert len(repr(page)) < 300
    del study, source
    gc.collect()
    records = []
    while page is not None:
        records.extend(page.to_dict()["items"])
        page = page.next()
    assert records == expected["items"]


def test_page_bounds_end_of_range_and_explicit_zero_length_navigation():
    study = _prepare()
    page = study.assumption_obligations(limit=1)
    assert page.total > 0
    assert study.assumption_obligations(offset=page.total + 1).items == ()
    assert study.assumption_obligations(offset=page.total + 1).next() is None
    empty = study.assumption_obligations(limit=0)
    assert empty.items == ()
    with pytest.raises(CausalValueError, match="positive limit"):
        empty.next()
    assert empty.next(limit=1).items == page.items
    with pytest.raises(CausalValueError, match="at most 256"):
        study.assumption_obligations(limit=257)
    with pytest.raises(CausalTypeError, match="integer"):
        study.assumption_obligations(offset=True)
    with pytest.raises(CausalValueError, match="nonnegative"):
        study.assumption_obligations(offset=-1)


def test_missing_specialized_native_adapter_is_an_implementation_error():
    study = _prepare()
    missing = replace(study, _engine=SimpleNamespace(_native=object(), _last_result=None))
    with pytest.raises(CausalStateError, match="native prepared-inspection adapter"):
        missing.inspect()
    with pytest.raises(CausalStateError, match="native adapter for inspection pages"):
        missing.assumption_obligations()
