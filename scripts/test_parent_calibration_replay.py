"""Legacy parent evidence selects a genuine production protocol, never new IDs."""
from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_facets as facets


class ParentReplayProtocolTest(unittest.TestCase):
    def record(self, functional: str) -> dict:
        return {
            "test": "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs::parent_set_linear_case",
            "functional": functional,
        }

    def test_original_records_select_real_version_one_and_its_original_budget(self) -> None:
        env = facets.parent_replay_environment([self.record("total_effect.variables_10.max_degree_4")])
        self.assertEqual(env, {
            "ANTECEDENT_PARENT_PROTOCOL_VERSION": "1",
            "ANTECEDENT_CALIBRATION_NSIM": "400",
            "ANTECEDENT_CALIBRATION_RECHECK_NSIM": "2000",
            "ANTECEDENT_CALIBRATION_ACCEPTANCE_NSIM": "2000",
        })

    def test_new_functional_selects_version_two_without_legacy_budget(self) -> None:
        env = facets.parent_replay_environment([self.record("total_effect.variables_10.max_degree_4.discovery_fraction_0_75")])
        self.assertEqual(env, {"ANTECEDENT_PARENT_PROTOCOL_VERSION": "2"})

    def test_mixed_protocols_or_unknown_new_fraction_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate replay groups"):
            facets.parent_replay_environment([
                self.record("total_effect.original"),
                self.record("total_effect.discovery_fraction_0_75"),
            ])
        with self.assertRaisesRegex(ValueError, "unknown parent discovery protocol"):
            facets.parent_replay_environment([self.record("total_effect.discovery_fraction_0_8")])

    def test_other_suites_do_not_acquire_parent_replay_overrides(self) -> None:
        self.assertEqual(facets.parent_replay_environment([{"test": "crates/other/tests/suite.rs::case"}]), {})


if __name__ == "__main__":
    unittest.main()
