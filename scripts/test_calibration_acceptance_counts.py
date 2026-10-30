"""Frozen acceptance cases determine the first-pass replicate minima at each size."""

from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GATE = (ROOT / "scripts/gate_calibration.sh").read_text(encoding="utf-8")
SETUP = GATE[
    GATE.index("# Replicate count of the precision recheck") : GATE.index(
        'if [ "$RECHECK_NSIM" -le'
    )
]
PARENT = (
    "sample_split_parent_set_effect_calibration: "
    "parent_set_linear_10_variables_hub_confounded_multi_path_nominal_95_coverage"
)
WEIGHTED = "sample_split_weighted_effect_calibration: weighted_split_effect_interval_calibrated"


class AcceptanceReplicateCounts(unittest.TestCase):
    def counts(self, group: str, **overrides: str) -> list[int]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ANTECEDENT_CALIBRATION_")
            and key != "ANTECEDENT_PARENT_PROTOCOL_VERSION"
        }
        env.update(overrides)
        return [
            int(
                subprocess.check_output(
                    [
                        "bash",
                        "-c",
                        SETUP + '\nfirst_pass_nsim "$1"',
                        "acceptance-count-test",
                        f"{group} [grid point {point}]",
                    ],
                    cwd=ROOT,
                    env=env,
                    text=True,
                )
            )
            for point in range(3)
        ]

    def test_malformed_acceptance_bindings_stop_setup(self) -> None:
        cases = [
            '[[case]]\n[case.expected]\nkind = "calibrated_interval"\n[case.evidence]\nstatus = "bound"\n',
            '[[case]]\n[case.expected]\nkind = "calibrated_interval"\n[case.evidence]\nstatus = "bound"\ntest = "x::y"\n[case.configuration]\ncalibration_grid_point = 8\n',
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                parity = Path(directory) / "parity"
                parity.mkdir()
                (parity / "acceptance_manifest.toml").write_text(case, encoding="utf-8")
                result = subprocess.run(["bash", "-c", SETUP + '\necho SHOULD_NOT_RUN'], cwd=directory,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("SHOULD_NOT_RUN", result.stdout)

    def test_separately_frozen_sizes_each_get_the_acceptance_minimum(self) -> None:
        self.assertEqual(self.counts(PARENT), [2000, 2000, 2000])

    def test_a_base_only_case_keeps_the_ordinary_extension_counts(self) -> None:
        self.assertEqual(self.counts(WEIGHTED), [400, 2000, 400])

    def test_declared_nonbase_cases_receive_the_minimum_only_where_needed(self) -> None:
        for group in (
            "weighted_interval_calibration: weighted_aipw_interval",
            "static_calibration: mediation_contrasts_frequentist_nominal_95_coverage",
        ):
            self.assertEqual(self.counts(group), [400, 2000, 2000])
            historical = subprocess.check_output(
                ["git", "show", "70a71dd33:scripts/gate_calibration.sh"], cwd=ROOT, text=True
            )
            setup = historical[
                historical.index("# Replicate count of the precision recheck") : historical.index(
                    'if [ "$RECHECK_NSIM" -le'
                )
            ]
            with patch.dict(globals(), SETUP=setup):
                self.assertEqual(self.counts(group), [400, 2000, 400])

    def test_ranking_configured_budgets_receive_the_minimum_at_their_actual_point(self) -> None:
        for test in (
            "selection_corrected_three_candidates_l95",
            "selection_corrected_five_candidates_l95",
            "selection_corrected_sixteen_candidates_l95",
            "selection_corrected_thirty_two_candidates_l95",
            "selection_corrected_one_hundred_candidates_l95",
            "selection_corrected_sixteen_candidates_budget_32768_l95",
        ):
            self.assertEqual(
                self.counts(f"decision_reliability_interval_calibration: {test}"),
                [400, 2000, 2000],
            )
        self.assertEqual(
            self.counts(
                "decision_reliability_interval_calibration: selection_corrected_four_candidates_l95"
            ),
            [400, 2000, 400],
        )

    def test_grid_point_declaration_is_validated(self) -> None:
        import check_acceptance_manifest as manifest

        cases = tomllib.loads(
            (ROOT / "parity/acceptance_manifest.toml").read_text(encoding="utf-8")
        )["case"]
        case = next(c for c in cases if c["id"] == "weighted_aipw_interval_calibrated")
        self.assertFalse(manifest.case_problems(case))
        for bad in (True, -1, 3, "2", None):
            changed = copy.deepcopy(case)
            changed["configuration"]["calibration_grid_point"] = bad
            self.assertTrue(
                any("calibration_grid_point" in x for x in manifest.case_problems(changed))
            )

    def test_legacy_parent_replay_preserves_original_400_2000_400_plan(self) -> None:
        self.assertEqual(
            self.counts(PARENT, ANTECEDENT_PARENT_PROTOCOL_VERSION="1"), [400, 2000, 400]
        )

    def test_larger_explicit_counts_are_preserved(self) -> None:
        self.assertEqual(self.counts(PARENT, ANTECEDENT_CALIBRATION_NSIM="20000"), [20000] * 3)

    def test_smoke_runs_keep_the_reduced_count(self) -> None:
        self.assertEqual(
            self.counts(
                PARENT,
                ANTECEDENT_CALIBRATION_NSIM="8",
                ANTECEDENT_CALIBRATION_SMOKE="1",
            ),
            [8] * 3,
        )


if __name__ == "__main__":
    unittest.main()
