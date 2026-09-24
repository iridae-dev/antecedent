from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compiler_migration", ROOT / "scripts/compiler_migration.py")
assert SPEC and SPEC.loader
compiler_migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler_migration)


class CompilerMigrationInventoryTests(unittest.TestCase):
    def test_all_licensed_coordinates_are_classified_and_current(self) -> None:
        support, routes = compiler_migration.load_registries()
        self.assertEqual(support.keys(), routes.keys())
        self.assertEqual(compiler_migration.validate(), [])
        manifest = compiler_migration.tomllib.loads(compiler_migration.INVENTORY.read_text())
        self.assertEqual(len(manifest["route"]), len(routes))
        for row in manifest["route"]:
            self.assertIn(row["kind"], compiler_migration.KINDS)
            self.assertEqual(row["migration_status"], "pending")
            self.assertEqual(row["checked_execution"], "unverified")
            self.assertEqual(row["builder_independent"], "unverified")
            self.assertEqual(row["execution_semantics"], "unverified")

    def test_release_gate_rejects_unverified_routes(self) -> None:
        issues = compiler_migration.validate(release_gate=True)
        self.assertEqual(len(issues), 472)
        self.assertTrue(all("2.1 release blocked" in issue for issue in issues))

    def test_route_classification_covers_each_execution_family(self) -> None:
        samples = [
            {"coordinate": "AverageEffect:Dag:explicit:Frequentist:none", "identifier": "general.id", "estimator": "functional.effect"},
            {"coordinate": "ResponseCurve:Dag:graph_posterior:Frequentist:none", "identifier": "response.backdoor", "estimator": "response.kennedy_dr"},
            {"coordinate": "Counterfactual:Dag:explicit:Bayesian:none", "identifier": "gcm.parametric", "estimator": "gcm.fit"},
            {"coordinate": "TransportQuery:Admg:explicit:Frequentist:none", "identifier": "transport.sid", "estimator": "transport.trial_ipw"},
            {"coordinate": "NestedCounterfactualEffect:Dag:explicit:Frequentist:none", "identifier": "path_specific.natural", "estimator": "mediation.linear"},
            {"coordinate": "InterventionResponse:Dag:explicit:Frequentist:none", "identifier": "general.id", "estimator": "functional.effect"},
        ]
        self.assertEqual(
            [compiler_migration.classify(route)[0] for route in samples],
            [
                "expression_evaluation", "composition", "model_operation", "composition",
                "model_operation", "specialized_estimation",
            ],
        )

    def test_existing_compiler_path_is_not_misrepresented_as_builder_discard_evidence(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import test_evidence

        path = ROOT / "crates/antecedent/tests/v110_licensed_compiler.rs"
        body = test_evidence.closure(path, "every_licensed_cell_completes_the_compiler_path")
        problems = compiler_migration.validate_evidence_body(body, "every_licensed_cell_completes_the_compiler_path")
        self.assertTrue(any("explicitly discard its builder" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
