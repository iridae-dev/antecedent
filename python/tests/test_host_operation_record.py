"""The host record of an operation report, shaped as the schema's ``operation_report``."""

from __future__ import annotations

import pytest
from antecedent import host
from antecedent.errors import CausalTypeError, CausalValueError


def _report(**overrides):
    base = {
        "operation": "estimate",
        "applicability": "licensed",
        "readiness": "executable",
        "coordinate": "static/binary/ate",
        "blockers": [],
        "next_actions": [],
        "neighbors": [],
        "unmet_obligations": [],
    }
    return {**base, **overrides}


def test_a_licensed_executable_operation_is_supported_and_carries_every_field():
    record = host.operation_record(_report())
    assert record == {
        "operation": "estimate",
        "applicability": "licensed",
        "readiness": "executable",
        "coordinate": "static/binary/ate",
        "blockers": [],
        "next_actions": [],
        "neighbors": [],
        "unmet_obligations": [],
        "status": host.operation_status("licensed", "executable"),
    }
    assert record["status"] == "supported"


def test_blockers_actions_neighbors_and_obligations_are_carried_with_the_status():
    record = host.operation_record(
        _report(
            applicability="licensed",
            readiness="binding_missing",
            blockers=[
                {"id": "treatment_unbound", "reason": "no treatment column", "scientific": 0}
            ],
            next_actions=[{"id": "measure_variable:z", "description": "measure z"}],
            neighbors=[
                {
                    "coordinate": "static/binary/att",
                    "changed": [{"axis": "estimand", "from": "ate", "to": "att"}],
                    "required_action": "declare the att",
                }
            ],
            unmet_obligations=[
                {
                    "id": "ignorability",
                    "scope": "assumption",
                    "kind": "assumption",
                    "status": "uncheckable",
                    "required_check": None,
                    "description": "no unmeasured confounding",
                }
            ],
        )
    )
    assert record["status"] == host.operation_status("licensed", "binding_missing")
    assert record["blockers"] == [
        {"id": "treatment_unbound", "reason": "no treatment column", "scientific": False}
    ]
    assert record["next_actions"] == [{"id": "measure_variable:z", "description": "measure z"}]
    assert record["neighbors"][0]["changed"][0]["axis"] == "estimand"
    assert record["unmet_obligations"][0]["status"] == "uncheckable"


def test_an_unlicensed_operation_is_never_supported_and_a_missing_readiness_is_none():
    record = host.operation_record(
        _report(applicability="impossible", readiness=None, coordinate=None)
    )
    assert record["readiness"] is None
    assert record["coordinate"] is None
    assert record["status"] == host.operation_status("impossible", None)
    assert record["status"] != "supported"


def test_a_report_that_is_not_a_mapping_of_the_capability_fields_is_refused():
    with pytest.raises(CausalTypeError):
        host.operation_record(["estimate"])  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        host.operation_record({"operation": "estimate"})
