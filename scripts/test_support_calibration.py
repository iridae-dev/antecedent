"""Generated supported cells bind no record list; an unmeasured cell states the reason."""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import gate_support  # noqa: E402
import support_calibration  # noqa: E402

CELL = {
    "query": "InterventionalDistribution",
    "graph_class": "Pag",
    "structure": "accepted",
    "inference": "Bayesian",
    "validation": "none",
    "estimators": ["functional.distribution"],
}


class CalibrationLinesTest(unittest.TestCase):
    def test_a_cell_no_record_measures_states_the_reason(self) -> None:
        self.assertEqual(
            support_calibration.calibration_lines(CELL, {}),
            'calibration_reason = "estimator_grid_not_measured"\n',
        )

    def test_a_cell_the_index_links_records_to_states_nothing(self) -> None:
        linked = {gate_support.cell_ident(CELL): ["cov.a"]}
        self.assertEqual(support_calibration.calibration_lines(CELL, linked), "")

    def test_another_cells_records_do_not_count(self) -> None:
        linked = {gate_support.cell_ident({**CELL, "graph_class": "Cpdag"}): ["cov.a"]}
        self.assertEqual(
            support_calibration.calibration_lines(CELL, linked),
            'calibration_reason = "estimator_grid_not_measured"\n',
        )


if __name__ == "__main__":
    unittest.main()
