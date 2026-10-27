"""The host record of an operation report, shaped as the schema's ``operation_report``."""

from __future__ import annotations

import antecedent as ant
import numpy as np
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


_AXES = ("query", "graph_class", "structure", "inference", "validation")


def _unlicensed_single_axis_cells(cells):
    """Coordinates one axis value away from a licensed cell that are not licensed, by status."""
    licensed = set(cells)
    values = [sorted({c.split(":")[i] for c in cells}) for i in range(len(_AXES))]
    found: dict[tuple[str, bool], tuple[str, str, int, dict]] = {}
    for cell in cells:
        parts = cell.split(":")
        for axis, options in enumerate(values):
            for option in options:
                if option == parts[axis]:
                    continue
                candidate = ":".join([*parts[:axis], option, *parts[axis + 1 :]])
                if candidate in licensed:
                    continue
                report = host.support_cell_report(candidate)
                found.setdefault((report["status"], axis == 1), (candidate, cell, axis, report))
        if ("refused", False) in found and ("not_applicable", False) in found:
            break
    return found


def test_every_listed_support_cell_reports_licensed_with_no_neighbors():
    cells = host.support_cells()
    assert cells and len(set(cells)) == len(cells)
    for coordinate in cells:
        assert len(coordinate.split(":")) == len(_AXES)
        report = host.support_cell_report(coordinate)
        assert report == {
            "coordinate": coordinate,
            "status": "licensed",
            "message": None,
            "neighbors": [],
        }


def test_a_prepared_study_is_licensed_exactly_when_its_cell_is_listed():
    n = 120
    z = np.sin(np.arange(n) / 7.0)
    t = (z + np.cos(np.arange(n) / 5.0) > 0).astype(float)
    prepared = ant.prepare(
        {"t": t, "y": 1.0 + 2.0 * t + 0.5 * z, "z": z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=3),
    )
    payload = prepared.inspect().support.payload
    assert payload["matrix_status"] == "licensed"
    assert payload["matrix_coordinate"] in host.support_cells()
    assert host.support_cell_report(payload["matrix_coordinate"])["neighbors"] == []


def test_an_unlicensed_cell_reports_its_blocker_and_same_class_changed_premise_neighbors():
    cells = host.support_cells()
    licensed = set(cells)
    found = _unlicensed_single_axis_cells(cells)
    candidate, origin, axis, report = found[("refused", False)]
    assert report["coordinate"] == candidate
    assert report["status"] == "refused"
    assert isinstance(report["message"], str) and report["message"]
    assert host.support_cell_report(candidate) == report, "the report is a pure read"

    neighbors = report["neighbors"]
    assert neighbors, "a refused cell of a licensed graph class names licensed alternatives"
    sizes = [len(n["changed"]) for n in neighbors]
    assert sizes == sorted(sizes) and min(sizes) >= 1, "nearest first, each changes a premise"
    for neighbor in neighbors:
        assert neighbor["coordinate"] in licensed
        assert neighbor["coordinate"].split(":")[1] == candidate.split(":")[1], "no class relabel"
        assert isinstance(neighbor["required_action"], str) and neighbor["required_action"]
        for change in neighbor["changed"]:
            position = _AXES.index(change["axis"])
            assert change["from"] == candidate.split(":")[position]
            assert change["to"] == neighbor["coordinate"].split(":")[position]
    # Undoing the one axis that was moved is exactly one changed premise.
    undo = next(n for n in neighbors if n["coordinate"] == origin)
    assert [(c["axis"], c["from"], c["to"]) for c in undo["changed"]] == [
        (_AXES[axis], candidate.split(":")[axis], origin.split(":")[axis])
    ]

    if ("not_applicable", False) in found:
        impossible = found[("not_applicable", False)][3]
        assert isinstance(impossible["message"], str) and impossible["message"]
        assert impossible["message"] != report["message"]


def test_a_malformed_or_off_matrix_coordinate_is_refused():
    with pytest.raises(CausalValueError):
        host.support_cell_report("static:dag")
    with pytest.raises(CausalValueError):
        host.support_cell_report("no:such:axis:value:here")
    with pytest.raises(CausalTypeError):
        host.support_cell_report(None)  # type: ignore[arg-type]
