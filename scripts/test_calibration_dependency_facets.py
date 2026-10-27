#!/usr/bin/env python3
"""Independent scope inventory for the reviewed discovery dependency split.

This checks declared current measurement callers, not arbitrary Rust reachability.
New procedures must extend the inventory before using these implementation paths.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_facets as facets
import collect_coverage_records as collector

ROOT = Path(__file__).resolve().parents[1]
DISCOVERED = "crates/antecedent/tests/discovered_transport_calibration.rs"


class DependencyFacets(unittest.TestCase):
    def setUp(self):
        self.surface = facets.load_surface()

    def test_model_backed_queries_cannot_omit_the_mechanism_dependency(self):
        # Observation table redraws use antecedent-model; sampled counterfactual
        # transport evaluates worlds through antecedent-counterfactual. Both
        # crates belong to mechanism, even when the facade estimator has its own name.
        for query in (
            "ObservationConditionalProbability",
            "CounterfactualTransport",
            "Counterfactual",
            "NestedCounterfactual",
        ):
            self.assertIn(
                "mechanism", facets.record_facets({"query": query}, self.surface)
            )
        self.assertNotIn(
            "mechanism",
            facets.record_facets(
                {"query": "AverageEffect", "estimator": "linear.adjustment"},
                self.surface,
            ),
        )
        for path in (
            "crates/antecedent/src/analysis/execute/attribution_path/contrast_set.rs",
            "crates/antecedent/src/gcm/counterfactual_contrasts.rs",
            "crates/antecedent-io-core/src/temporal_counterfactual_general_artifact.rs",
        ):
            self.assertEqual(self.surface.facet_of(path), "mechanism")

    def test_actual_procedure_depends_on_every_reclassified_path(self):
        current = {"query": "DiscoveredTransport", "test": DISCOVERED + "::any_gate"}
        depends = set(facets.record_facets(current, self.surface))
        for path in (
            "crates/antecedent-discovery/src/pcmci_plus.rs",
            "crates/antecedent/src/analysis/discovered_transport_estimate.rs",
            "crates/antecedent/src/analysis/discovered_transport_temporal.rs",
        ):
            self.assertIn(self.surface.facet_of(path), depends)
        self.assertEqual(
            self.surface.facet_of(
                "crates/antecedent-discovery/src/selective/contract.rs"
            ),
            facets.CORE,
        )

    def test_fixed_transport_does_not_invoke_discovery_adapter(self):
        fixed = {
            "query": "ClassicalTransport",
            "estimator": "transport.empirical_table_plugin",
        }
        depends = facets.record_facets(fixed, self.surface)
        self.assertNotIn("estimator.discovered_transport", depends)
        self.assertNotIn("estimator.pcmci_plus", depends)
        code = facets.rust_code(
            (
                ROOT / "crates/antecedent/tests/transport_statistical_calibration.rs"
            ).read_text(encoding="utf-8")
        )
        self.assertIn("identify_catalog_transport", code)
        self.assertNotIn("estimate_discovered_transport", code)
        self.assertNotIn("TemporalAlgorithm", code)

    def test_shared_plus_helpers_are_not_licensed_by_plain_pcmci_identity(self):
        for name in ("pcmci_plus", "jpcmci", "rpcmci", "orientation_stability"):
            for field in ("estimator", "functional"):
                self.assertIn(
                    "estimator.pcmci_plus",
                    facets.record_facets(
                        {field: "procedure." + name + ".v1"}, self.surface
                    ),
                )
        self.assertNotIn(
            "estimator.pcmci_plus",
            facets.record_facets({"estimator": "pcmci"}, self.surface),
        )

    def test_measured_direct_callers_match_reviewed_inventory(self):
        # Scan all Rust measurement files, including non-facade crates. Strings
        # remain visible because the facade also accepts algorithm names.
        users = set()
        names = (
            "PcmciPlus",
            "TemporalAlgorithm::PcmciPlus",
            "estimate_discovered_transport",
            "discovered_transport_estimate",
            "discovered_transport_temporal",
        )
        for path in (ROOT / "crates").rglob("*.rs"):
            text = path.read_text(encoding="utf-8")
            if "for_record(" not in text:
                continue
            code = facets.rust_code(text)
            if any(name in code for name in names) or '"pcmci_plus"' in text:
                users.add(path.relative_to(ROOT).as_posix())
        self.assertEqual(users, {DISCOVERED})
        self.assertIn(
            'query: "DiscoveredTransport"',
            (ROOT / DISCOVERED).read_text(encoding="utf-8"),
        )

    def test_frozen_source_records_can_be_retagged_without_restamping_evidence(self):
        frozen = "70a71dd33"
        sha = subprocess.check_output(
            ["git", "rev-parse", frozen], cwd=ROOT, text=True
        ).strip()
        old_text = subprocess.check_output(
            ["git", "show", f"{sha}:scripts/calibration_surface.list"],
            cwd=ROOT,
            text=True,
        )
        historical = facets.load_surface(text=old_text)
        record = {
            "query": "DiscoveredTransport",
            "test": DISCOVERED + "::gate",
            "calibration_sha": sha,
            "surface_list_blob": collector.surface_list_blob(sha),
        }
        record["facets"] = facets.record_facets(record, historical)
        self.assertNotIn("estimator.discovered_transport", record["facets"])
        source_identity = (record["calibration_sha"], record["surface_list_blob"])
        with (
            patch.object(collector.facets, "load_surface", return_value=self.surface),
            patch.object(collector.facets, "references", return_value=None),
        ):
            collector.tag_facets({"frozen": record})
        self.assertIn("estimator.discovered_transport", record["facets"])
        self.assertIn("estimator.pcmci_plus", record["facets"])
        self.assertEqual(
            source_identity, (record["calibration_sha"], record["surface_list_blob"])
        )
        self.assertEqual(record["facets"], facets.record_facets(record, self.surface))

    def test_catalog_and_exact_plan_users_carry_their_real_dependencies(self):
        for query in (
            "ClassicalTransport",
            "TemporalTransport",
            "CounterfactualTransport",
            "DiscoveredTransport",
            "TransportQuery",
        ):
            tagged = facets.record_facets({"query": query}, self.surface)
            self.assertIn("identity.catalog_transport", tagged)
            self.assertIn("estimator.exact_transport", tagged)
        for query in (
            "FiniteLaw",
            "ExactResponse",
            "Recalculation",
            "FunctionalDistribution",
            "CounterfactualEventQuery",
            "ObservationConditionalProbability",
        ):
            self.assertIn(
                "estimator.exact_transport",
                facets.record_facets({"query": query}, self.surface),
            )
        for estimator in (
            "functional.distribution",
            "finite_scm",
            "exact.evaluate",
            "recalc.temporal",
            "observation.count_fitted_tables.dirichlet",
        ):
            self.assertIn(
                "estimator.exact_transport",
                facets.record_facets({"estimator": estimator}, self.surface),
            )
        for query in (
            "ResponseCurve",
            "AverageEffect",
            "InterventionResponse",
            "DecisionReliability",
        ):
            tagged = facets.record_facets({"query": query}, self.surface)
            self.assertNotIn("identity.catalog_transport", tagged)
            self.assertNotIn("estimator.exact_transport", tagged)

    def test_measured_catalog_and_exact_api_callers_match_inventory(self):
        direct_catalog, direct_exact = set(), set()
        for path in (ROOT / "crates").rglob("*.rs"):
            text = path.read_text(encoding="utf-8")
            if "for_record(" not in text:
                continue
            code = facets.rust_code(text)
            rel = path.relative_to(ROOT).as_posix()
            if "identify_catalog_transport" in code:
                direct_catalog.add(rel)
            if any(
                name in code
                for name in (
                    "ExactEvaluation",
                    "ExactTransport",
                    "ExactDistribution",
                    "evaluate_statistical_transport",
                )
            ):
                direct_exact.add(rel)
        self.assertEqual(
            direct_catalog,
            {"crates/antecedent/tests/transport_statistical_calibration.rs"},
        )
        self.assertEqual(
            direct_exact,
            {
                "crates/antecedent/tests/joint_sensitivity_calibration.rs",
                "crates/antecedent/tests/mz_transport_calibration.rs",
                "crates/antecedent/tests/z_transport_surrogate_calibration.rs",
                "crates/antecedent/tests/transport_statistical_calibration.rs",
                "crates/antecedent-estimate/tests/counterfactual_transport_sampled.rs",
            },
        )

    def test_selection_separation_includes_transport_and_recovery_proof_routes(self):
        path = "crates/antecedent-identify/src/selection_separation.rs"
        self.assertEqual(self.surface.facet_of(path), "identity.transport_selection")
        for query in (
            "ClassicalTransport",
            "ConditionalTransport",
            "MetaTransport",
            "TemporalTransport",
            "DiscoveredTransport",
            "ZTransport",
            "MzTransport",
            "CounterfactualEventQuery",
            "ObservationConditionalProbability",
            "ObservationRecovery",
            "RecoveredEffect",
            "MixedSourceQuery",
        ):
            self.assertIn(
                "identity.transport_selection",
                facets.record_facets({"query": query}, self.surface),
                query,
            )
        for estimator in (
            "sampled_observation_recovery",
            "transport.smoothed_dose_transport_aipw",
        ):
            self.assertIn(
                "identity.transport_selection",
                facets.record_facets({"estimator": estimator}, self.surface),
            )
        for query in (
            "AverageEffect",
            "ResponseCurve",
            "SplitParentSetEffect",
            "PooledEffect",
        ):
            self.assertNotIn(
                "identity.transport_selection",
                facets.record_facets({"query": query}, self.surface),
            )
        for record in facets.load_records():
            if record["calibration_sha"].startswith("e15c4b0c3"):
                self.assertNotIn(
                    "identity.transport_selection",
                    facets.record_facets(record, self.surface),
                )

    def test_recordless_transport_consumers_keep_selection_dependency(self):
        gates = {row["id"]: row for row in facets.load_gate_rows()}
        for stem in (
            "smoothed_dose_calibration__smoothed_dose_transport_grid_points",
            "smoothed_dose_calibration__smoothed_dose_transport_weak_overlap_boundary",
            "learned_continuous_calibration__learned_trial_aipw_misspecified_nuisance_cases",
            "learned_continuous_calibration__learned_trial_aipw_weak_overlap_boundary",
        ):
            row = gates[facets.GATE_PREFIX + stem]
            self.assertIn(
                "identity.transport_selection", facets.record_facets(row, self.surface)
            )
        for stem in (
            "observational_effects_calibration__policy_held_out_n1000_l95",
            "observational_effects_calibration__survival_difference_covariate_censoring_n1000_l95",
        ):
            self.assertNotIn(
                "identity.transport_selection",
                facets.record_facets(gates[facets.GATE_PREFIX + stem], self.surface),
            )

    def test_selection_separation_direct_callers_match_reviewed_inventory(self):
        # These identifiers name the actual numerical kernels and independent checker.
        names = (
            "MutilatedSelection",
            "SubsetSearchEnd",
            "independently_separated",
            "for_each_admissible_subset",
            "for_each_admissible_subset_preferring",
        )
        users = set()
        source = ROOT / "crates/antecedent-identify/src"
        for path in source.rglob("*.rs"):
            if path.name == "selection_separation.rs":
                continue
            code = facets.rust_code(path.read_text(encoding="utf-8"))
            if any(name in code for name in names):
                users.add(path.relative_to(source).as_posix())
        self.assertEqual(
            users,
            {
                "sid.rs",
                "sid/catalog.rs",
                "sid/engine.rs",
                "sid/z_transport/derive.rs",
                "transport.rs",
            },
        )
        # Observation recovery is an indirect SID consumer and must not be omitted
        # simply because its public query is not spelled Transport.
        for rel in (
            "recovery.rs",
            "recovery/witness.rs",
            "sid/conditional.rs",
            "sid/cpdag_completion.rs",
            "sid/scenarios.rs",
            "sid/temporal_sequence.rs",
        ):
            code = facets.rust_code((source / rel).read_text(encoding="utf-8"))
            self.assertIn("identify_classical_transport_metered", code, rel)
        for rel in ("sid/meta.rs", "sid/mz_transport.rs", "sid/z_transport/search.rs"):
            code = facets.rust_code((source / rel).read_text(encoding="utf-8"))
            self.assertIn("Engine::new", code, rel)

    def test_selection_retag_keeps_historical_source_and_surface_blob(self):
        sha = subprocess.check_output(
            ["git", "rev-parse", "70a71dd33"], cwd=ROOT, text=True
        ).strip()
        old_surface = facets.load_surface(
            text=subprocess.check_output(
                ["git", "show", f"{sha}:scripts/calibration_surface.list"],
                cwd=ROOT,
                text=True,
            )
        )
        record = {
            "query": "ClassicalTransport",
            "calibration_sha": sha,
            "surface_list_blob": collector.surface_list_blob(sha),
        }
        record["facets"] = facets.record_facets(record, old_surface)
        self.assertNotIn("identity.transport_selection", record["facets"])
        identity = (record["calibration_sha"], record["surface_list_blob"])
        with (
            patch.object(collector.facets, "load_surface", return_value=self.surface),
            patch.object(collector.facets, "references", return_value=None),
        ):
            collector.tag_facets({"record": record})
        self.assertIn("identity.transport_selection", record["facets"])
        self.assertEqual(
            identity, (record["calibration_sha"], record["surface_list_blob"])
        )

    def test_reviewed_unit_modules_are_not_measurement_hosts(self):
        hosts = {
            str(row.get(field, "")).rsplit("::", 1)[0]
            for row in facets.load_records()
            for field in ("test", "dgp")
        }
        for path in (
            "crates/antecedent-discovery/src/pcmci_plus.rs",
            "crates/antecedent-estimate/src/population_aipw.rs",
            "crates/antecedent-estimate/src/propensity/matching.rs",
            "crates/antecedent-estimate/src/propensity/stratification.rs",
            "crates/antecedent/src/analysis/discovered_transport_trial.rs",
        ):
            self.assertIn(path, self.surface.testmods)
            self.assertNotIn(path, hosts)
            text = (ROOT / path).read_text(encoding="utf-8")
            self.assertNotIn("for_record(", facets.rust_code(text))
            self.assertNotIn("#[ignore", facets.rust_code(text))
            self.assertIn("mod tests", text)
            original = facets.scaffolding_view(path, text)
            unit = "\n#[cfg(test)] mod sentinel_unit { fn unit_only_marker() {} }\n"
            self.assertEqual(
                original.strip(), facets.scaffolding_view(path, text + unit).strip()
            )
            production = "\nfn production_change_must_remain_visible() {}\n"
            self.assertNotEqual(
                original, facets.scaffolding_view(path, text + production)
            )

    def test_recordless_pcmci_gates_execute_the_separate_plain_algorithm(self):
        # The old recordless ledger contains plain PCMCI, not PCMCI+/JPCMCI+.
        # Preserve that distinction instead of inventing a replay of PCMCI+.
        text = (ROOT / "crates/antecedent-discovery/src/pcmci.rs").read_text(
            encoding="utf-8"
        )
        code = facets.rust_code(text)
        self.assertIn("fn pcmci_null_fpr_near_alpha", code)
        self.assertIn("fn pcmci_planted_lag1_power", code)
        self.assertNotIn("PcmciPlus::", code)

    def test_recordless_knn_gate_retains_its_actual_numerical_dependencies(self):
        path = "crates/antecedent-estimate/tests/knn_matching_calibration.rs"
        code = facets.rust_code((ROOT / path).read_text(encoding="utf-8"))
        self.assertIn("KnnDistanceMatching::new(NEIGHBORS)", code)
        self.assertIn("with_population_variance(true)", code)
        expected = {
            "estimator.knn_distance_matching",
            "estimator.distance_matching",
            "estimator.propensity_matching",
            "suite.knn_matching_calibration",
        }
        for point in range(3):
            row = {
                "gate_group": f"knn_matching_calibration__knn_matching_interval_ci_coverage.p{point}"
            }
            depends = set(facets.record_facets(row, self.surface))
            self.assertLessEqual(expected, depends)
            for actual_path in (
                "crates/antecedent-estimate/src/propensity/knn.rs",
                "crates/antecedent-estimate/src/propensity/distance.rs",
                "crates/antecedent-estimate/src/propensity/matching.rs",
                path,
            ):
                self.assertTrue((ROOT / actual_path).is_file())
                self.assertIn(self.surface.facet_of(actual_path), depends)
        for group in (
            "observational_effects_calibration__policy_held_out_n1000_l95",
            "knn_matching_calibration__knn_matching_interval_ci_coverage.p9",
            "knn_matching_calibration__knn_matching_interval_ci_coverage.p0_backup",
        ):
            self.assertTrue(
                expected.isdisjoint(
                    facets.record_facets({"gate_group": group}, self.surface)
                )
            )
        for production in (
            "crates/antecedent-stats/src/ci/advanced.rs",
            "crates/antecedent-stats/src/ci/calibration.rs",
            "crates/antecedent-validate/src/sensitivity.rs",
        ):
            self.assertTrue((ROOT / production).is_file())
            self.assertEqual(self.surface.facet_of(production), facets.CORE)

    def test_recordless_sensitivity_retains_its_actual_linear_refit_dependency(self):
        sensitivity = (
            ROOT / "crates/antecedent-validate/src/sensitivity.rs"
        ).read_text(encoding="utf-8")
        common = (ROOT / "crates/antecedent-validate/src/common.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "NeighborhoodSensitivity::new().refute", facets.rust_code(sensitivity)
        )
        self.assertIn("linear_estimator_no_bootstrap()", facets.rust_code(sensitivity))
        self.assertIn(
            "fn linear_estimator_no_bootstrap() -> LinearAdjustmentAte", common
        )
        self.assertIn("LinearAdjustmentAte::new()", common)
        group = (
            "antecedent-validate__scalable_sensitivity_replacement_validated_separately"
        )
        row = {"gate_group": group}
        expected = "estimator.linear_adjustment"
        self.assertIn(expected, facets.record_facets(row, self.surface))
        self.assertEqual(
            self.surface.facet_of("crates/antecedent-estimate/src/adjustment.rs"),
            expected,
        )
        for other in (
            group + "_backup",
            "antecedent-stats__scalable_gpdc_replacement_validated_separately",
            "antecedent-validate__scalable_sensitivity_replacement_validated",
        ):
            self.assertNotIn(
                expected, facets.record_facets({"gate_group": other}, self.surface)
            )


if __name__ == "__main__":
    unittest.main()
