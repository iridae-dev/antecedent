#!/usr/bin/env python3
"""A type-compatible recipe does not open a route, and a license matches the matrix."""

from __future__ import annotations

import unittest
from pathlib import Path

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import gate_recipe_obligations as gate  # noqa: E402

CLOSED = [
    {
        "queries": ["Counterfactual"],
        "graph_classes": ["Admg"],
        "reason": "Counterfactual requires a supplied static Dag.",
    }
]
NOT_APPLICABLE = [
    {
        "queries": ["Counterfactual"],
        "graph_classes": ["Admg"],
        "reason": "typed impossibility that must not replace the closed reason",
    }
]
AXES = {
    "query": ["AverageEffect", "Counterfactual", "PulseEffect"],
    "graph_class": ["Dag", "Admg"],
    "structure": ["explicit"],
    "inference": ["Frequentist"],
    "validation": ["none"],
}


def route(**extra) -> dict:
    cell = {
        "query": "AverageEffect",
        "graph_class": "Dag",
        "structure": "explicit",
        "inference": "Frequentist",
        "validation": "none",
    }
    cell.update(extra)
    return cell


def licensed_row(**extra) -> dict:
    row = route(
        estimators=["linear.adjustment.ate"],
        checked_execution=[
            {
                "estimator": "linear.adjustment.ate",
                "test": "crates/antecedent/tests/example.rs",
                "assertion": "retains_the_plan",
            }
        ],
        evidence_kind="internal_known_truth",
        evidence_test="crates/antecedent/tests/example.rs",
        evidence_assertion="known_truth",
        known_truth_fixture="conformance/estimate/example",
        calibration_reason="no_interval_reported",
    )
    row.update(extra)
    return row


def matrix_cell(**extra) -> dict:
    cell = route(
        route_estimator="linear.adjustment.ate",
        estimators=["linear.adjustment.ate"],
    )
    cell.update(extra)
    return cell


class RecipeObligationTest(unittest.TestCase):
    def test_a_type_compatible_recipe_does_not_open_a_route(self) -> None:
        compatible = route()
        self.assertEqual(gate.standing(compatible, [], [], set()), "unlicensed")
        self.assertNotEqual(gate.standing(compatible, [], [], set()), "not_applicable")
        obligations = gate.emit_obligations(compatible)
        self.assertEqual(obligations["routes"], [])
        self.assertEqual(obligations["tests"], [])
        self.assertFalse(set(obligations) & set(gate.IDENTIFICATION))
        self.assertFalse(set(obligations) & set(gate.CALIBRATION))
        self.assertEqual(gate.cross_check([], [], [], [], axes=AXES), [])

    def test_a_closed_cell_stays_closed_when_a_recipe_could_run(self) -> None:
        cell = route(query="Counterfactual", graph_class="Admg")
        self.assertEqual(gate.standing(cell, CLOSED, NOT_APPLICABLE, set()), "closed")
        self.assertNotEqual(gate.standing(cell, CLOSED, NOT_APPLICABLE, set()), "not_applicable")
        self.assertEqual(gate.cross_check([], [], CLOSED, NOT_APPLICABLE, axes=AXES), [])
        licensed = licensed_row(query="Counterfactual", graph_class="Admg")
        matrix = matrix_cell(query="Counterfactual", graph_class="Admg")
        problems = gate.cross_check([licensed], [matrix], CLOSED, NOT_APPLICABLE)
        self.assertTrue(
            any("Counterfactual" in problem and "stays closed" in problem for problem in problems)
        )
        self.assertTrue(all("not_applicable" not in problem for problem in problems))

    def test_licensed_obligations_match_the_matrix_and_leave_calibration_on_the_license(
        self,
    ) -> None:
        licensed = licensed_row()
        matrix = matrix_cell(estimators=["linear.adjustment.ate", "cell.aipw"])
        self.assertEqual(gate.cross_check([licensed], [matrix], [], []), [])
        obligations = gate.emit_obligations(licensed)
        self.assertEqual(
            [item["estimator"] for item in obligations["routes"]],
            ["linear.adjustment.ate"],
        )
        self.assertNotIn("cell.aipw", [item["estimator"] for item in obligations["routes"]])
        self.assertNotIn("evidence_kind", obligations)
        self.assertNotIn("calibration_reason", obligations)
        self.assertEqual(licensed["evidence_kind"], "internal_known_truth")
        self.assertEqual(licensed["calibration_reason"], "no_interval_reported")

        invented = licensed_row(estimators=["other.estimator"])
        invented["checked_execution"][0]["estimator"] = "other.estimator"
        problems = gate.cross_check([invented], [matrix_cell()], [], [])
        self.assertTrue(any("AverageEffect" in problem for problem in problems))
        self.assertTrue(all("not_applicable" not in problem for problem in problems))

    def test_an_unreachable_citation_fails_the_cross_check(self) -> None:
        def cite(test: str, assertion: str) -> list[str]:
            return [f"evidence assertion {assertion!r} is not reachable"]

        problems = gate.cross_check([licensed_row()], [matrix_cell()], [], [], cite=cite)
        self.assertTrue(
            any("AverageEffect" in problem and "not reachable" in problem for problem in problems)
        )

    def test_recorded_licenses_match_the_support_matrix(self) -> None:
        problems = gate.matrix_alignment_problems(Path(REPO_ROOT), cite=_path_exists)
        self.assertEqual(problems, [])


def _path_exists(test: str, assertion: str) -> list[str]:
    path = Path(REPO_ROOT) / test
    if not path.is_file():
        return [f"test {test!r} does not exist"]
    if not assertion.strip():
        return ["assertion is blank"]
    return []


if __name__ == "__main__":
    unittest.main()
