"""The ready-coordinate calibration receipt: measured, with every failure resolved."""
from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import generate_calibration_backlog as backlog  # noqa: E402

READY = (("AverageEffect", "Pag", "graph_posterior", "Bayesian"),)


def cell(**extra: object) -> dict:
    base = {"query": "AverageEffect", "graph_class": "Pag", "structure": "graph_posterior",
            "inference": "Bayesian", "validation": "none", "calibration": ["rec"]}
    return {**base, **extra}


class ReadyCoordinateTest(unittest.TestCase):
    def test_ready_coordinates_must_be_measured_with_failures_resolved(self) -> None:
        passing = {"id": "rec", "boundary": False, "grid": [{"boundary": False}]}
        self.assertEqual(backlog.ready_coordinate_problems([cell()], [passing], READY), [])

        unmeasured = cell(calibration_reason="estimator_grid_not_measured", calibration=[])
        found = backlog.ready_coordinate_problems([unmeasured], [passing], READY)
        self.assertTrue(any("estimator_grid_not_measured" in p for p in found))
        self.assertTrue(any("cites no coverage record" in p for p in found))

        failed = {"id": "rec", "boundary": True, "grid": [{"boundary": True}]}
        self.assertTrue(any("unresolved failed grid point" in p
                            for p in backlog.ready_coordinate_problems([cell()], [failed], READY)))
        gridded = {"id": "rec", "boundary": False, "grid": [{"boundary": False}, {"boundary": True}]}
        self.assertTrue(backlog.ready_coordinate_problems([cell()], [gridded], READY))
        self.assertTrue(any("does not exist" in p for p in backlog.ready_coordinate_problems([cell()], [], READY)))
        self.assertTrue(any("boundary_record" in p for p in backlog.ready_coordinate_problems(
            [cell(calibration_reason="boundary_record")], [passing], READY)))
        self.assertTrue(any("no licensed cell" in p for p in backlog.ready_coordinate_problems([], [], READY)))

    def test_the_frozen_coordinates_are_the_seventeen_the_case_names(self) -> None:
        self.assertEqual(len(set(backlog.READY_COORDINATES)), 17)


if __name__ == "__main__":
    unittest.main()
