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
            ).read_text()
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
            text = path.read_text()
            if "for_record(" not in text:
                continue
            code = facets.rust_code(text)
            if any(name in code for name in names) or '"pcmci_plus"' in text:
                users.add(path.relative_to(ROOT).as_posix())
        self.assertEqual(users, {DISCOVERED})
        self.assertIn('query: "DiscoveredTransport"', (ROOT / DISCOVERED).read_text())

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
            text = path.read_text()
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

    def test_recordless_pcmci_gates_execute_the_separate_plain_algorithm(self):
        # The old recordless ledger contains plain PCMCI, not PCMCI+/JPCMCI+.
        # Preserve that distinction instead of inventing a replay of PCMCI+.
        text = (ROOT / "crates/antecedent-discovery/src/pcmci.rs").read_text()
        code = facets.rust_code(text)
        self.assertIn("fn pcmci_null_fpr_near_alpha", code)
        self.assertIn("fn pcmci_planted_lag1_power", code)
        self.assertNotIn("PcmciPlus::", code)


if __name__ == "__main__":
    unittest.main()
