"""Generated supported cells state the calibration the collector bound onto them."""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import support_calibration  # noqa: E402

CELL = {
    "query": "InterventionalDistribution",
    "graph_class": "Pag",
    "structure": "accepted",
    "inference": "Bayesian",
    "validation": "none",
    "estimators": ["functional.distribution"],
}

SUFFIX = ".v2_validation_none.v2_structure_accepted"
ID_A, ID_B = f"cov.a{SUFFIX}", f"cov.b{SUFFIX}"


def record(rid: str, estimator: str = "functional.distribution", **extra: object) -> dict:
    return {
        "id": f"{rid}.v2_validation_none.v2_structure_accepted",
        "validation_schema": 2,
        "query": "InterventionalDistribution",
        "graph_class": "Pag",
        "inference": "Bayesian",
        "structure": "accepted",
        "validation": "none",
        "estimator": estimator,
        "nominal": 0.95,
        "boundary": False,
        **extra,
    }


class CalibrationLinesTest(unittest.TestCase):
    def test_a_cell_without_records_states_the_reason(self) -> None:
        self.assertEqual(
            support_calibration.calibration_lines(CELL, {}),
            'calibration_reason = "estimator_grid_not_measured"\n',
        )

    def test_owned_records_are_bound_sorted_and_carry_no_reason(self) -> None:
        records = {r["id"]: r for r in (record("cov.b"), record("cov.a"))}
        self.assertEqual(
            support_calibration.calibration_lines(CELL, records),
            f'calibration = ["{ID_A}", "{ID_B}"]\n',
        )

    def test_a_foreign_estimator_or_coordinate_is_not_owned(self) -> None:
        records = {
            "f": record("cov.foreign", estimator="other.estimator"),
            "c": {**record("cov.other_class"), "graph_class": "Cpdag"},
        }
        self.assertEqual(
            support_calibration.calibration_lines(CELL, records),
            'calibration_reason = "estimator_grid_not_measured"\n',
        )

    def test_a_boundary_only_cell_states_the_boundary_reason(self) -> None:
        boundary = record("cov.a", boundary=True)
        self.assertEqual(
            support_calibration.calibration_lines(CELL, {boundary["id"]: boundary}),
            f'calibration = ["{ID_A}"]\ncalibration_reason = "boundary_record"\n',
        )


if __name__ == "__main__":
    unittest.main()
