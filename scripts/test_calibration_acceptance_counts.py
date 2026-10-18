"""Frozen acceptance cases determine the first-pass replicate minima at each size."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

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
WEIGHTED = (
    "sample_split_weighted_effect_calibration: weighted_split_effect_interval_calibrated"
)


class AcceptanceReplicateCounts(unittest.TestCase):
    def counts(self, group: str, **overrides: str) -> list[int]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ANTECEDENT_CALIBRATION_") and key != "ANTECEDENT_PARENT_PROTOCOL_VERSION"
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

    def test_separately_frozen_sizes_each_get_the_acceptance_minimum(self) -> None:
        self.assertEqual(self.counts(PARENT), [2000, 2000, 2000])

    def test_a_base_only_case_keeps_the_ordinary_extension_counts(self) -> None:
        self.assertEqual(self.counts(WEIGHTED), [400, 2000, 400])

    def test_legacy_parent_replay_preserves_original_400_2000_400_plan(self) -> None:
        self.assertEqual(self.counts(PARENT, ANTECEDENT_PARENT_PROTOCOL_VERSION="1"), [400, 2000, 400])

    def test_larger_explicit_counts_are_preserved(self) -> None:
        self.assertEqual(
            self.counts(PARENT, ANTECEDENT_CALIBRATION_NSIM="20000"), [20000] * 3
        )

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
