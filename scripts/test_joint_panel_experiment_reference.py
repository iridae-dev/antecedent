"""Independent checks of the experiment reference, separate from production Rust.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "conformance/response/joint_panel"
SPEC = importlib.util.spec_from_file_location(
    "experiment_reference", HERE / "experiment_reference.py"
)
assert SPEC and SPEC.loader
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class ExperimentReference(unittest.TestCase):
    def test_retained_reference_reproduces(self):
        self.assertEqual(
            reference.reference_artifact(),
            json.loads((HERE / "experiment_reference.json").read_text(encoding="utf-8")),
        )

    def test_two_state_analytic_update_and_constrained_optimum(self):
        weights = reference.posterior([0.5, 0.5], [-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], 0.5, 1.0, 1)
        self.assertAlmostEqual(weights[1], math.e / (1 + math.e), places=14)
        result = reference.decision(weights, [[0, 0], [-1, 1], [100, 100]], [True, True, False])
        self.assertEqual(result["chosen"], [1])
        self.assertAlmostEqual(result["expected_utility"], 2 * weights[1] - 1, places=14)
        self.assertAlmostEqual(result["evpi"], weights[0], places=14)

    def test_quadrature_error_contains_analytic_two_state_evsi(self):
        # Symmetric +/-1 states, Gaussian noise variance1 and one observation.
        # Optimal action is sign(Y), hence E[max_a E[a theta|Y]] = 2 Phi(1)-1.
        utilities = [[-1, 1], [1, -1]]
        exact = math.erf(1 / math.sqrt(2))
        for cells in (128, 1024, 4096):
            result = reference.future_value(
                [0.5, 0.5], [-1, 1], utilities, [True, True], 1, 1, cells
            )
            self.assertLessEqual(abs(result["gross_evsi"] - exact), result["absolute_error_bound"])

    def test_uninformative_study_has_zero_value(self):
        result = reference.future_value([0.3, 0.7], [0, 0], [[0, 0], [-1, 1]], [True, True], 1, 2)
        self.assertLessEqual(abs(result["gross_evsi"]), result["absolute_error_bound"])

    def test_population_dose_duration_translations_change_actual_estimands(self):
        artifact = reference.reference_artifact()
        rows = artifact["population_time_dose_estimands"]
        self.assertEqual(len(rows), 12)
        for duration in (2, 12, 24):
            subset = [r for r in rows if r["duration"] == duration]
            self.assertEqual(len({r["mean"] for r in subset}), 4)
            for mix in ([1.0] * 4, [0.1, 0.2, 0.3, 0.4]):
                dose1, dose4 = [r["mean"] for r in subset if r["population_weights"] == mix]
                self.assertGreater(dose4, dose1)

    def test_invalid_translation_and_no_feasible_action_refuse(self):
        with self.assertRaises(ValueError):
            reference.posterior([1], [1], [0], [0], 1, 1, 1)
        with self.assertRaises(ValueError):
            reference.decision([1], [[1]], [False])


if __name__ == "__main__":
    unittest.main()
