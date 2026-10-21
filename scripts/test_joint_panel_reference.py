"""Independent X06.1 fixture checks, with closed forms separate from the generator.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "conformance/response/joint_panel"
SPEC = importlib.util.spec_from_file_location("joint_panel_reference", HERE / "reference.py")
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class JointPanelReference(unittest.TestCase):
    def setUp(self):
        self.truth = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    def test_seed_and_all_retained_artifacts_reproduce(self):
        for name, generated in reference.artifacts().items():
            self.assertTrue(
                reference.matches_artifact(
                    name, (HERE / name).read_text(encoding="utf-8"), generated
                ),
                name,
            )
        self.assertEqual(self.truth["seed"], 20261021)

    def test_truth_comparison_preserves_metadata_shapes_and_rejects_numeric_drift(self):
        self.assertTrue(
            reference.same_truth({"seed": 42, "truth": [0.5]}, {"seed": 42, "truth": [0.5 + 1e-15]})
        )
        self.assertFalse(reference.same_truth({"seed": 42}, {"seed": 43}))
        self.assertFalse(reference.same_truth([0.5], [0.5 + 1e-8]))
        self.assertFalse(reference.same_truth([0.5], [0.5, 0.5]))
        self.assertFalse(reference.same_truth(0.5, "0.5"))
        self.assertFalse(reference.same_truth(float("nan"), float("nan")))
        self.assertFalse(reference.matches_artifact("data.csv", "0.5\n", "0.500000000000001\n"))

    def test_steady_state_interaction_has_an_independent_closed_form(self):
        # After L=2, exposure under sustained dose4 is4 regardless of history.
        # The additive terms cancel, leaving beta_AB*(4/(4+2))^2 = 2/3.
        delta = self.truth["variants"]["interaction"]["delta_ab_by_time"]
        for value in delta[2:]:
            self.assertAlmostEqual(value, 2.0 / 3.0, places=13)
        for value in self.truth["variants"]["additive"]["delta_ab_by_time"]:
            self.assertAlmostEqual(value, 0.0, places=13)

    def test_transient_contrast_respects_shared_initial_history(self):
        groups = self.truth["variants"]["interaction"]["latent_truth"]
        for t in (0, 1):
            expected = []
            for group in groups:
                # Calculate finite-lag totals directly, without the generator's
                # carryover/saturation/response functions.
                totals = []
                for key, weights in (
                    ("initial_a", (0.6, 0.3, 0.1)),
                    ("initial_b", (0.5, 0.35, 0.15)),
                ):
                    history = group[key]
                    inherited = sum(weights[k] * history[2 + t - k] for k in range(t + 1, 3))
                    zero, four = inherited, inherited + 4.0 * sum(weights[: t + 1])
                    totals.append(four / (four + 2.0) - zero / (zero + 2.0))
                expected.append(1.5 * math.prod(totals))
            actual = self.truth["variants"]["interaction"]["delta_ab_by_time"][t]
            self.assertAlmostEqual(actual, sum(expected) / 4.0, places=13)
        self.assertLess(self.truth["variants"]["interaction"]["delta_ab_by_time"][0], 2.0 / 3.0)

    def test_observed_outcomes_keep_declared_ar_errors_and_finite_lags(self):
        for design in ("randomized", "confounded"):
            source = (
                "confounded_interaction.csv"
                if design == "confounded"
                else "randomized_interaction.csv"
            )
            with (HERE / source).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            groups = (
                self.truth["designs"][design]["latent_truth"]
                if design == "confounded"
                else self.truth["variants"]["interaction"]["latent_truth"]
            )
            for group in groups:
                series = [row for row in rows if int(row["group"]) == group["group"]]
                for t, period in enumerate(group["periods"]):
                    a = sum(
                        weight * float(series[t + 2 - lag]["channel_a"])
                        for lag, weight in enumerate((0.6, 0.3, 0.1))
                    )
                    b = sum(
                        weight * float(series[t + 2 - lag]["channel_b"])
                        for lag, weight in enumerate((0.5, 0.35, 0.15))
                    )
                    sa, sb = a / (a + 2.0), b / (b + 2.0)
                    mean = (
                        2.0
                        + group["intercept"]
                        + (0.0, 0.2, -0.1, 0.1)[t % 4]
                        + 0.7 * period["z"]
                        + 3.0 * sa
                        + 2.0 * sb
                        + 1.5 * sa * sb
                    )
                    if design == "confounded":
                        mean += 0.8 * period["latent"]
                    self.assertAlmostEqual(
                        float(series[t + 2]["outcome"]), mean + period["ar_error"], places=13
                    )
                    previous_error = (
                        group["periods"][t - 1]["ar_error"] if t else group["initial_error"]
                    )
                    self.assertAlmostEqual(
                        period["ar_error"], 0.45 * previous_error + period["innovation"], places=14
                    )

    def test_heldout_joint_program_is_not_an_observed_training_trajectory(self):
        with (HERE / "randomized_interaction.csv").open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["time"]) >= 0]
        for group in range(4):
            series = [row for row in rows if int(row["group"]) == group]
            cells = {(float(row["channel_a"]), float(row["channel_b"])) for row in series}
            self.assertEqual(cells, {(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0)})
            self.assertNotEqual([float(row["channel_a"]) for row in series], [4.0] * 24)
        heldout = self.truth["variants"]["interaction"]["heldout_joint"]
        self.assertEqual(
            heldout["group_paths"], self.truth["variants"]["interaction"]["world_paths"]["joint"]
        )

    def test_confounded_design_has_no_off_diagonal_support_and_requires_withholding(self):
        with (HERE / "confounded_interaction.csv").open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["time"]) >= 0]
        self.assertTrue(all(row["channel_a"] == row["channel_b"] for row in rows))
        self.assertNotIn("latent", rows[0])
        design = self.truth["designs"]["confounded"]
        self.assertEqual(design["expected_causal_claim"], "withhold")
        self.assertEqual(design["missing_worlds"], ["a", "b"])
        self.assertEqual(len(design["obligations"]), 2)
        self.assertEqual(self.truth["evidence"]["calibration"], "unmeasured")

    def test_whole_program_experiment_retains_its_actual_independent_assignment(self):
        rng = random.Random(194)
        assigned = [rng.randrange(4) for _ in range(4)]
        design = self.truth["designs"]["whole_program_randomized"]
        self.assertEqual(design["assigned_program_indices"], assigned)
        self.assertEqual(design["probabilities"], [0.25] * 4)
        with (HERE / "whole_program_interaction.csv").open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["time"]) >= 0]
        for group in range(4):
            world = design["program_order"][assigned[group]]
            levels = ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0))[assigned[group]]
            source = [row for row in rows if int(row["group"]) == group]
            for row in source:
                time = int(row["time"])
                self.assertEqual((float(row["channel_a"]), float(row["channel_b"])), levels)
                truth = self.truth["variants"]["interaction"]
                expected = (
                    truth["world_paths"][world][group][time]
                    + truth["latent_truth"][group]["periods"][time]["ar_error"]
                )
                self.assertAlmostEqual(float(row["outcome"]), expected, places=13)

    def test_generator_has_no_production_or_external_numerical_imports(self):
        tree = ast.parse((HERE / "reference.py").read_text(encoding="utf-8"))
        allowed = {"__future__", "argparse", "csv", "io", "json", "math", "random", "pathlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(name.name in allowed for name in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)


if __name__ == "__main__":
    unittest.main()
