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


class SynthesisReplayProtocolTest(unittest.TestCase):
    def record(self, method: str, role: str = "gated") -> dict:
        return {"test": "crates/antecedent/tests/evidence_synthesis_calibration.rs::pooled_random_effects_five_studies",
                "interval_method": method, "role": role}

    def test_stored_method_selects_actual_legacy_or_new_producer(self) -> None:
        self.assertEqual(facets.synthesis_replay_environment([self.record("hksj_t")]),
                         {"ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION": "1"})
        self.assertEqual(facets.synthesis_replay_environment([self.record("independent_gaussian_study_t", "conservative_point")]),
                         {"ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION": "2"})

    def test_unknown_or_mixed_identity_is_rejected(self) -> None:
        for records in ([self.record("analytic_se")], [self.record("independent_gaussian_study_t")],
                        [self.record("hksj_t"), self.record("independent_gaussian_study_t", "conservative_point")]):
            with self.assertRaises(ValueError):
                facets.synthesis_replay_environment(records)

    def test_fixed_effect_and_other_suites_do_not_acquire_an_override(self) -> None:
        fixed = {**self.record("analytic_se"), "test": "crates/antecedent/tests/evidence_synthesis_calibration.rs::pooled_fixed_effect_five_studies"}
        self.assertEqual(facets.synthesis_replay_environment([fixed]), {})
        self.assertEqual(facets.synthesis_replay_environment([{"test": "other::case"}]), {})


class WeightedSplitReplayProtocolTest(unittest.TestCase):
    def test_stored_versions_are_explicit_and_mixed_or_unknown_versions_refused(self):
        test = "crates/antecedent/tests/sample_split_weighted_effect_calibration.rs::weighted_split_effect_interval_calibrated"
        old = {"test": test, "functional": "weighted_population_total_effect"}
        new = {"test": test, "functional": "weighted_population_total_effect.discovery_fraction_0_75.conflict_resolution_treatment_invariant"}
        for record, version in ((old, "1"), (new, "2")):
            self.assertEqual(facets.weighted_split_replay_environment([record]),
                             {"ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL": version})
        for records in ([old,new], [{**old,"functional":"unknown"}]):
            with self.assertRaises(ValueError):
                facets.weighted_split_replay_environment(records)
        self.assertEqual(facets.weighted_split_replay_environment([{"test":"other::case"}]), {})


if __name__ == "__main__":
    unittest.main()
