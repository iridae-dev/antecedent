from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compiler_migration", ROOT / "scripts/compiler_migration.py")
assert SPEC and SPEC.loader
compiler_migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler_migration)


class CompilerMigrationInventoryTests(unittest.TestCase):
    def test_all_licensed_coordinates_are_classified_and_current(self) -> None:
        support, routes = compiler_migration.load_registries()
        optional_routes = compiler_migration.optional_estimator_routes(support, routes)
        stage_routes = compiler_migration.load_transport_stage_routes()
        self.assertEqual(support.keys(), routes.keys())
        self.assertEqual(compiler_migration.validate(), [])
        manifest = compiler_migration.tomllib.loads(compiler_migration.INVENTORY.read_text())
        self.assertEqual(len(manifest["route"]), len(routes) + len(optional_routes) + len(stage_routes))
        self.assertEqual(
            len(optional_routes),
            sum(len(cell["estimators"]) - 1 for cell in support.values()),
        )
        for row in manifest["route"]:
            self.assertIn(row["kind"], compiler_migration.KINDS)
            self.assertIn(row["migration_status"], compiler_migration.MIGRATION_STATES)
            if row["migration_status"] == "pending":
                self.assertEqual(row["checked_execution"], "unverified")
                self.assertEqual(row["builder_independent"], "unverified")
                self.assertEqual(row["execution_semantics"], "unverified")
            if row.get("source_registry") == "parity/transport_stages.toml":
                self.assertEqual(row["migration_status"], "pending")
        self.assertEqual(len(stage_routes), 21)
        self.assertIn(
            "AverageEffect:Dag:explicit:Frequentist:none::estimator=iv.2sls",
            {row["coordinate"] for row in manifest["route"]},
        )
        self.assertNotIn(
            "transport-stage:antecedent_estimate.evaluate_exact_z_transport",
            {row["coordinate"] for row in manifest["route"]},
            "low-level estimator entry points stay outside the high-level migration gate",
        )
        included = {row["coordinate"] for row in manifest["route"]}
        for low_level in ("transport.exact_table_evaluate", "transport.learned_categorical_plugin"):
            self.assertNotIn(f"transport-stage:{low_level}", included)

    def test_nested_counterfactual_is_a_model_operation(self) -> None:
        manifest = compiler_migration.tomllib.loads(compiler_migration.INVENTORY.read_text())
        row = next(row for row in manifest["route"] if row["query"] == "NestedCounterfactualEffect")
        self.assertEqual(row["kind"], "model_operation")

    def test_release_gate_rejects_unverified_routes(self) -> None:
        issues = compiler_migration.validate(release_gate=True)
        manifest = compiler_migration.tomllib.loads(compiler_migration.INVENTORY.read_text())
        pending = [row for row in manifest["route"] if row["migration_status"] != "verified"]
        self.assertGreater(len(pending), 0)
        blocked = {issue.split(": 2.1 release blocked", 1)[0] for issue in issues if "2.1 release blocked" in issue}
        self.assertTrue({row["coordinate"] for row in pending}.issubset(blocked))

    def test_progress_gate_runs_verified_evidence_before_full_release_closure(self) -> None:
        with patch.object(compiler_migration, "run_route_evidence") as run:
            self.assertEqual(compiler_migration.validate(verify_progress=True), [])
        run.assert_called_once()
        rows = list(run.call_args.args[0])
        self.assertTrue(any(row["migration_status"] == "verified" for row in rows))
        self.assertTrue(any(row["migration_status"] != "verified" for row in rows))

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

    def test_builder_discard_evidence_accepts_plain_and_prefixed_names(self) -> None:
        for discarded in ("drop(builder)", "drop(aipw_builder)", "del builder", "builder = None"):
            body = f"{{ let plan = prepared.checked_plan(); {discarded}; prepared.estimate(); }}"
            self.assertEqual(compiler_migration.validate_evidence_body(body, "route"), [])
        body = "{ let plan = prepared.checked_plan(); drop(result); prepared.estimate(); }"
        self.assertTrue(compiler_migration.validate_evidence_body(body, "route"))

    def test_status_cannot_claim_closure_without_all_three_execution_receipts(self) -> None:
        original = compiler_migration.INVENTORY
        row = original.read_text()
        manifest = compiler_migration.tomllib.loads(row)
        key = next(
            route['coordinate'] for route in manifest['route']
            if route['migration_status'] == 'verified'
        )
        start = row.index(f'coordinate = "{key}"')
        end = row.find('[[route]]', start)
        end = len(row) if end < 0 else end
        chunk = row[start:end].replace('checked_execution = "verified"', 'checked_execution = "unverified"')
        self.assertNotEqual(chunk, row[start:end])
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            temporary = Path(directory) / 'compiler_migration.toml'
            temporary.write_text(row[:start] + chunk + row[end:])
            compiler_migration.INVENTORY = temporary
            try:
                issues = compiler_migration.validate()
            finally:
                compiler_migration.INVENTORY = original
        self.assertTrue(any(key in issue and 'requires checked execution' in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
