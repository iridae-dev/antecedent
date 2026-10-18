"""Legacy parent evidence selects a genuine production protocol, never new IDs."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_facets as facets  # noqa: E402 (repository path bootstrap above)


class ParentReplayProtocolTest(unittest.TestCase):
    def record(self, functional: str) -> dict:
        return {
            "test": "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs::parent_set_linear_case",
            "functional": functional,
        }

    def test_original_records_select_real_version_one_and_its_original_budget(self) -> None:
        env = facets.parent_replay_environment(
            [self.record("total_effect.variables_10.max_degree_4")]
        )
        self.assertEqual(
            env,
            {
                "ANTECEDENT_PARENT_PROTOCOL_VERSION": "1",
                "ANTECEDENT_CALIBRATION_NSIM": "400",
                "ANTECEDENT_CALIBRATION_RECHECK_NSIM": "2000",
                "ANTECEDENT_CALIBRATION_ACCEPTANCE_NSIM": "2000",
            },
        )

    def test_new_functional_selects_version_two_without_legacy_budget(self) -> None:
        env = facets.parent_replay_environment(
            [self.record("total_effect.variables_10.max_degree_4.discovery_fraction_0_75")]
        )
        self.assertEqual(env, {"ANTECEDENT_PARENT_PROTOCOL_VERSION": "2"})

    def test_mixed_protocols_or_unknown_new_fraction_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate replay groups"):
            facets.parent_replay_environment(
                [
                    self.record("total_effect.original"),
                    self.record("total_effect.discovery_fraction_0_75"),
                ]
            )
        with self.assertRaisesRegex(ValueError, "unknown parent discovery protocol"):
            facets.parent_replay_environment([self.record("total_effect.discovery_fraction_0_8")])

    def test_other_suites_do_not_acquire_parent_replay_overrides(self) -> None:
        self.assertEqual(
            facets.parent_replay_environment([{"test": "crates/other/tests/suite.rs::case"}]), {}
        )


class SynthesisReplayProtocolTest(unittest.TestCase):
    def record(self, method: str, role: str = "gated") -> dict:
        return {
            "test": "crates/antecedent/tests/evidence_synthesis_calibration.rs::pooled_random_effects_five_studies",
            "interval_method": method,
            "role": role,
        }

    def test_stored_method_selects_actual_legacy_or_new_producer(self) -> None:
        self.assertEqual(
            facets.synthesis_replay_environment([self.record("hksj_t")]),
            {"ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION": "1"},
        )
        self.assertEqual(
            facets.synthesis_replay_environment(
                [self.record("independent_gaussian_study_t", "conservative_point")]
            ),
            {"ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION": "2"},
        )

    def test_unknown_or_mixed_identity_is_rejected(self) -> None:
        for records in (
            [self.record("analytic_se")],
            [self.record("independent_gaussian_study_t")],
            [
                self.record("hksj_t"),
                self.record("independent_gaussian_study_t", "conservative_point"),
            ],
        ):
            with self.assertRaises(ValueError):
                facets.synthesis_replay_environment(records)

    def test_fixed_effect_and_other_suites_do_not_acquire_an_override(self) -> None:
        fixed = {
            **self.record("analytic_se"),
            "test": "crates/antecedent/tests/evidence_synthesis_calibration.rs::pooled_fixed_effect_five_studies",
        }
        self.assertEqual(facets.synthesis_replay_environment([fixed]), {})
        self.assertEqual(facets.synthesis_replay_environment([{"test": "other::case"}]), {})


class WeightedSplitReplayProtocolTest(unittest.TestCase):
    def test_stored_versions_are_explicit_and_mixed_or_unknown_versions_refused(self):
        test = "crates/antecedent/tests/sample_split_weighted_effect_calibration.rs::weighted_split_effect_interval_calibrated"
        old = {"test": test, "functional": "weighted_population_total_effect"}
        new = {
            "test": test,
            "functional": "weighted_population_total_effect.discovery_fraction_0_75.conflict_resolution_treatment_invariant",
        }
        for record, version in ((old, "1"), (new, "2")):
            self.assertEqual(
                facets.weighted_split_replay_environment([record]),
                {"ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL": version},
            )
        for records in ([old, new], [{**old, "functional": "unknown"}]):
            with self.assertRaises(ValueError):
                facets.weighted_split_replay_environment(records)
        self.assertEqual(facets.weighted_split_replay_environment([{"test": "other::case"}]), {})


class WeightedClusterReplayProtocolTest(unittest.TestCase):
    def record(self, test: str, method: str, functional: str) -> dict:
        return {
            "test": f"crates/antecedent/tests/weighted_interval_calibration.rs::{test}",
            "interval_method": method,
            "functional": functional,
        }

    def test_each_actual_cluster_gate_selects_its_stored_numerical_protocol(
        self,
    ) -> None:
        for test in (
            "weighted_aipw_cluster_dml_interval",
            "weighted_dml_cluster_interval",
            "weighted_dr_learner_cluster_interval",
        ):
            with self.subTest(test=test):
                old = self.record(test, "analytic_se", "all_observed.mean")
                new = self.record(
                    test, "student_analytic", "all_observed.mean+reference=cluster_df"
                )
                self.assertEqual(
                    facets.weighted_cluster_replay_environment([old]),
                    {"ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL": "1"},
                )
                self.assertEqual(
                    facets.weighted_cluster_replay_environment([new]),
                    {"ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL": "2"},
                )
                with self.assertRaisesRegex(ValueError, "separate replay groups"):
                    facets.weighted_cluster_replay_environment([old, new])
                for altered in (
                    dict(old, interval_method="student_analytic"),
                    dict(new, functional="all_observed.mean"),
                ):
                    with self.assertRaisesRegex(ValueError, "unknown stored"):
                        facets.weighted_cluster_replay_environment([altered])

    def test_other_gate_does_not_acquire_a_cluster_override(self) -> None:
        self.assertEqual(
            facets.weighted_cluster_replay_environment(
                [self.record("weighted_aipw_ate_interval", "analytic_se", "all_observed.mean")]
            ),
            {},
        )


class ReplayWorkerBudgetTest(unittest.TestCase):
    def test_replay_defaults_to_one_worker_and_preserves_only_a_valid_cap(self) -> None:
        self.assertEqual(
            facets.replay_thread_environment(None), {"ANTECEDENT_CALIBRATION_THREADS": "1"}
        )
        self.assertEqual(
            facets.replay_thread_environment(" 2 "), {"ANTECEDENT_CALIBRATION_THREADS": "2"}
        )
        for value in ("0", "-1", "1.5", "ten", ""):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                facets.replay_thread_environment(value)


class LegacyPrefixReplayRecipeTest(unittest.TestCase):
    def fixture(self) -> tuple[dict, dict]:
        source = "e15c4b0c33daf638d11f64b6a7fff6e3bc216022"
        points = []
        grid = []
        for point, (budget, scored, covered) in enumerate(
            ((2000, 1930, 1830), (4000, 3800, 3610), (4000, 3870, 3670))
        ):
            grid.append({"point": point, "replicates": scored, "observed": covered / scored})
            cell = {
                "point": point,
                "planned_attempts": budget,
                "covered": covered,
                "scored": scored,
                "skipped": 0,
                "log_sha256": "a" * 64,
                "tally_sha256": "b" * 64,
            }
            if budget == 4000:
                cell.update(computed_indices=[2000, 4000], prefix_tally_sha256="c" * 64)
            else:
                cell["retained_original_point"] = True
            points.append(cell)
        record = {
            "id": "cov.parent",
            "calibration_sha": source,
            "test": "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs::parent_case",
            "functional": "effect.discovery_fraction_0_75",
            "nominal": 0.95,
            "grid": grid,
        }
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        entry = {
            **{k: record[k] for k in ("test", "functional", "nominal", "calibration_sha")},
            "record_id": record["id"],
            "measurement_payload_sha256": digest,
            "points": points,
        }
        history = {
            "schema": "calibration_validation_history.v1",
            "legacy_prefix_protocol": {
                "id": "original_source_prefix_completion.v1",
                "source_sha": source,
            },
            "legacy_prefix_completions": [entry],
        }
        return record, history

    def test_uses_declared_attempts_rather_than_conditional_scored_counts(self) -> None:
        record, history = self.fixture()
        result = facets.legacy_prefix_replay_environments([record], history)
        self.assertEqual(
            [int(env["ANTECEDENT_CALIBRATION_NSIM"]) for env in result], [2000, 4000, 4000]
        )
        self.assertEqual(
            [env["ANTECEDENT_CALIBRATION_GRID_POINTS"] for env in result], ["0", "1", "2"]
        )
        self.assertEqual(
            [env["ANTECEDENT_CALIBRATION_RECHECK_NSIM"] for env in result], ["2001", "4001", "4001"]
        )
        # Facet migrations do not alter the measured payload or its attempt recipe.
        record.update(facets=["core"], surface_list_blob="d" * 40)
        self.assertEqual(facets.legacy_prefix_replay_environments([record], history), result)

    def test_tampered_payload_counts_budgets_or_prefix_receipts_fail_closed(self) -> None:
        record, history = self.fixture()
        changed = copy.deepcopy(record)
        changed["grid"][0]["replicates"] -= 1
        with self.assertRaisesRegex(ValueError, "stored measurement payload"):
            facets.legacy_prefix_replay_environments([changed], history)
        for key, value in (
            ("planned_attempts", 20000),
            ("covered", 0),
            ("log_sha256", "bad"),
            ("computed_indices", [0, 4000]),
            ("prefix_tally_sha256", "bad"),
        ):
            altered = copy.deepcopy(history)
            altered["legacy_prefix_completions"][0]["points"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                facets.legacy_prefix_replay_environments([record], altered)

    def test_unrelated_or_later_source_ignores_historical_recipe_but_mixed_group_rejects(
        self,
    ) -> None:
        record, history = self.fixture()
        self.assertEqual(
            facets.legacy_prefix_replay_environments(
                [{**record, "calibration_sha": "f" * 40}], history
            ),
            [],
        )
        self.assertEqual(
            facets.legacy_prefix_replay_environments([{**record, "id": "other"}], history), []
        )
        with self.assertRaisesRegex(ValueError, "separate replay groups"):
            facets.legacy_prefix_replay_environments([record, {**record, "id": "other"}], history)


class CoverageFamilyReplayRecipeTest(unittest.TestCase):
    def fixture(self) -> tuple[dict, dict]:
        import calibration_decision as decision

        config = decision.config_for_manifest(
            {"groups": [{"index": i, "label": f"case{i}"} for i in range(1, 758)]}
        )
        grid = []
        for point, (planned, scored, covered) in enumerate(
            ((400, 400, 380), (2000, 1900, 1805), (20000, 19500, 18525))
        ):
            value = decision.evaluate(config, planned, scored, covered, 0.95, False)
            grid.append(
                {
                    "point": point,
                    "replicates": scored,
                    "observed": covered / scored,
                    "role": "gated",
                    "decision_protocol": value,
                }
            )
        record = {
            "id": "cov.family",
            "calibration_sha": "7" * 40,
            "nominal": 0.95,
            "role": "gated",
            "grid": grid,
            "decision_protocol": grid[0]["decision_protocol"],
        }
        return record, config

    def test_replays_the_original_family_domain_and_explicit_registered_attempts(self) -> None:
        import calibration_decision as decision

        record, config = self.fixture()
        with patch("calibration_family.config_at_sha", return_value=config) as historical:
            envs = facets.coverage_family_replay_environments([record])
        historical.assert_called_once_with(record["calibration_sha"])
        self.assertEqual([int(e["ANTECEDENT_CALIBRATION_NSIM"]) for e in envs], [400, 2000, 20000])
        self.assertEqual([e["ANTECEDENT_CALIBRATION_GRID_POINTS"] for e in envs], ["0", "1", "2"])
        for env in envs:
            self.assertEqual(json.loads(env[decision.ENV]), config)
            self.assertNotIn("ANTECEDENT_CALIBRATION_SHA", env)
            self.assertNotIn("ANTECEDENT_CALIBRATION_CONFIRMATION", env)
            self.assertNotIn("ANTECEDENT_CALIBRATION_PRIOR_TALLIES", env)

    def test_historical_family_binding_tampered_counts_and_mixed_sources_fail_closed(self) -> None:
        record, config = self.fixture()
        with (
            patch(
                "calibration_family.config_at_sha",
                return_value={**config, "family_digest": "f" * 64},
            ),
            self.assertRaisesRegex(ValueError, "frozen source roster"),
        ):
            facets.coverage_family_replay_environments([record])
        changed = copy.deepcopy(record)
        changed["grid"][1]["observed"] = 0.9
        foreign_domain = copy.deepcopy(record)
        foreign_domain["grid"][1]["decision_protocol"]["validation_domain"] = "other_domain"
        for changed_record in (changed, foreign_domain):
            with (
                patch("calibration_family.config_at_sha", return_value=config),
                self.assertRaises(ValueError),
            ):
                facets.coverage_family_replay_environments([changed_record])
        for mixed in (
            [record, {"id": "legacy"}],
            [record, {**record, "calibration_sha": "8" * 40}],
        ):
            with self.assertRaisesRegex(ValueError, "separate replay groups"):
                facets.coverage_family_replay_environments(mixed)

    def test_named_boundary_without_an_attempt_receipt_is_not_guessed(self) -> None:
        record, config = self.fixture()
        record["grid"][1].pop("decision_protocol")
        record["grid"][1]["role"] = "named_boundary"
        with (
            patch("calibration_family.config_at_sha", return_value=config),
            self.assertRaisesRegex(ValueError, "explicit attempt receipt"),
        ):
            facets.coverage_family_replay_environments([record])
        self.assertEqual(facets.coverage_family_replay_environments([{"id": "old"}]), [])


if __name__ == "__main__":
    unittest.main()
