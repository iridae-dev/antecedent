"""Scenario matrix: the pending_measurement state and its negative controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import scenario_matrix as matrix  # noqa: E402

ROW = "FAM.static.frequentist"
SURFACE = "core crates/x/\nsuite.closing_calibration crates/y/\n"


def counterpart_row(identifier: str, closure: str, gap: str, execution: str) -> str:
    return f"""[[row]]
id = "{identifier}"
kind = "method"
method = "FAM"
design = "static"
inference = "frequentist"
route = "r"
status = "executing_unlicensed"
uncertainty = "design_aware_resampling"
uncertainty_basis = "b"
target = "t"
sampling_design = "independent rows"
execution = "{execution}"
reuse_key = "k"
decision_port = "unlisted"
artifact = "a"
closure = "{closure}"
scope_status = "mandatory"
gap = "{gap}"
reason = "calibration of the interval is unmeasured"
next_attempt = "measure it"
"""


def build_tree(root: Path, pending: str = "", curated: str = "", execution: str = "none.rs::t") -> dict:
    (root / "parity").mkdir()
    (root / "scripts").mkdir()
    (root / "parity/counterparts.toml").write_text(
        counterpart_row(ROW, "narrowed_promise", "validation", execution), encoding="utf-8"
    )
    (root / "parity/missing_policy_support.toml").write_text("", encoding="utf-8")
    (root / "parity/scenario_matrix.toml").write_text("schema = 1\n" + curated, encoding="utf-8")
    (root / "scripts/calibration_surface.list").write_text(SURFACE, encoding="utf-8")
    if pending:
        (root / "parity/pending_measurement.toml").write_text(pending, encoding="utf-8")
    return matrix.build(root)


def entry(facet: str = "suite.closing_calibration", owed: str = "interval_coverage", axes: str = "") -> str:
    facet_line = f'facet = "{facet}"\n' if facet is not None else ""
    return f'[[pending]]\nrow = "{ROW}"\n{facet_line}owed = "{owed}"\n{axes}\n'


def cells_of(result: dict, axis: str = "design_inference") -> list[dict]:
    return [c for c in result["cells"] if c["axis"] == axis and c.get("row") == ROW]


def run_case(**kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return build_tree(Path(tmp), **kwargs)


def test_a_registered_calibration_owed_cell_is_pending_and_names_its_closing_facet():
    result = run_case(pending=entry())
    assert result["findings"] == [], result["findings"]
    for axis in ("design_inference", "uncertainty"):
        (cell,) = [c for c in cells_of(result, axis) if c["value"] == "static.frequentist"]
        assert cell["status"] == matrix.PENDING
        assert cell["facet"] == "suite.closing_calibration"
        assert cell["owed"] == "interval_coverage"
    assert result["counts"].get("pending_measurement") == 2


def test_the_same_cell_without_an_entry_is_a_gap_not_pending():
    result = run_case()
    cell = [c for c in cells_of(result) if c["value"] == "static.frequentist"][0]
    assert cell["status"] == matrix.GAP and cell["gap_kind"] == "narrowed_promise"


def test_a_pending_entry_with_no_facet_is_rejected_and_the_cell_stays_a_gap():
    result = run_case(pending=entry(facet=None))
    assert any("no calibration facet" in f for f in result["findings"]), result["findings"]
    cell = [c for c in cells_of(result) if c["value"] == "static.frequentist"][0]
    assert cell["status"] == matrix.GAP


def test_a_pending_entry_whose_facet_the_surface_does_not_define_is_rejected():
    result = run_case(pending=entry(facet="suite.no_such_facet"))
    assert any("is not defined in scripts/calibration_surface.list" in f for f in result["findings"])
    cell = [c for c in cells_of(result) if c["value"] == "static.frequentist"][0]
    assert cell["status"] == matrix.GAP


def test_a_pending_entry_with_an_unknown_owed_kind_or_axis_is_rejected():
    kinds = run_case(pending=entry(owed="vibes"))
    assert any("owed 'vibes'" in f for f in kinds["findings"]), kinds["findings"]
    axes = run_case(pending=entry(axes='axes = ["graph_class"]'))
    assert any("are not among" in f for f in axes["findings"]), axes["findings"]


def test_a_pending_entry_that_names_no_open_cell_is_stale():
    stale = entry(axes='axes = ["uncertainty"]').replace(ROW, "NOPE.static.frequentist")
    result = run_case(pending=stale)
    assert any("names no open cell" in f for f in result["findings"]), result["findings"]


def test_a_delivered_curated_cell_without_an_exact_test_is_rejected_and_stays_a_gap():
    curated = """
[[axis]]
id = "weights_missingness"
title = "w"
values = ["sampling_weights"]

[[cell]]
family = "FAM"
axis = "weights_missingness"
value = "sampling_weights"
status = "delivered"
test = "crates/x/tests/none.rs"
what = "w"
"""
    result = run_case(curated=curated)
    assert any("delivered but" in f for f in result["findings"]), result["findings"]
    (cell,) = [c for c in result["cells"] if c["axis"] == "weights_missingness"]
    assert cell["status"] == matrix.GAP


def test_every_pending_cell_of_the_live_matrix_names_a_defined_facet_and_nothing_is_delivered_unproven():
    built = matrix.build(REPO_ROOT)
    assert built["findings"] == [], built["findings"]
    facets = matrix.surface_facets(REPO_ROOT)
    pending = matrix.pending(built)
    assert pending, "the live matrix owes calibration measurements"
    for cell in pending:
        assert cell["facet"] in facets, cell
        assert cell["owed"] in matrix.OWED_KINDS, cell
        assert cell["evidence"] == "", cell
    assert all(c["evidence"] for c in built["cells"] if c["status"] == matrix.DELIVERED)


_CASES = [
    test_a_registered_calibration_owed_cell_is_pending_and_names_its_closing_facet,
    test_the_same_cell_without_an_entry_is_a_gap_not_pending,
    test_a_pending_entry_with_no_facet_is_rejected_and_the_cell_stays_a_gap,
    test_a_pending_entry_whose_facet_the_surface_does_not_define_is_rejected,
    test_a_pending_entry_with_an_unknown_owed_kind_or_axis_is_rejected,
    test_a_pending_entry_that_names_no_open_cell_is_stale,
    test_a_delivered_curated_cell_without_an_exact_test_is_rejected_and_stays_a_gap,
    test_every_pending_cell_of_the_live_matrix_names_a_defined_facet_and_nothing_is_delivered_unproven,
]


def load_tests(loader, tests, pattern):  # noqa: ARG001
    suite = unittest.TestSuite()
    suite.addTests(unittest.FunctionTestCase(case) for case in _CASES)
    return suite


if __name__ == "__main__":
    unittest.main()
