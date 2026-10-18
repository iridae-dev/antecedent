"""Original recordless receipts close only their exact source-audited manual recipe."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_recipe_receipts as receipts  # noqa: E402
import check_release_closure as closure  # noqa: E402


class RecipeReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipes = json.loads((ROOT / receipts.REGISTRY).read_text(encoding="utf-8"))["recipe"]
        cls.cases = {
            c["id"]: c
            for c in tomllib.loads(
                (ROOT / "parity/acceptance_manifest.toml").read_text(encoding="utf-8")
            )["case"]
        }

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copytree(ROOT / "parity/recipe_evidence", self.root / "parity/recipe_evidence")
        proof = json.loads(
            (
                ROOT / "parity/recipe_evidence/v2.4.1/width_n1000/c30-numerical-source-audit.json"
            ).read_text(encoding="utf-8")
        )
        for item in proof["same_source_files_and_functions"]:
            path = item.partition("::")[0]
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
        shutil.copyfile(ROOT / "Cargo.lock", self.root / "Cargo.lock")
        path = "crates/antecedent-estimate/tests/knn_matching_calibration.rs"
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, self.root / path)
        shutil.copyfile(
            ROOT / "parity/calibration_gates.toml", self.root / "parity/calibration_gates.toml"
        )
        for path in (
            "crates/antecedent-stats/src/ci/calibration.rs",
            "crates/antecedent-validate/src/sensitivity.rs",
        ):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
        self.recipe = copy.deepcopy(
            next(
                r
                for r in self.recipes
                if r["case_id"] == "observational_policy_value_calibrated_1000"
            )
        )
        self.case = copy.deepcopy(self.cases[self.recipe["case_id"]])
        original = receipts.family.read_source
        self.source_patch = patch.object(
            receipts.family,
            "read_source",
            side_effect=lambda _, path, sha: original(ROOT, path, sha),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def validate(self):
        return receipts.validate_recipe(
            self.root, self.recipe, self.case, closure.case_size(self.case)
        )

    def replace_asset(self, descriptor, value) -> None:
        path = self.root / descriptor["path"]
        path.chmod(0o644)
        data = (json.dumps(value) + "\n").encode()
        path.write_bytes(data)
        descriptor["sha256"] = hashlib.sha256(data).hexdigest()

    def test_real_recipe_retains_refusal_outcomes_and_manual_api_boundary(self) -> None:
        value = self.validate()
        self.assertEqual(value["receipt_kind"], "recordless_gate")
        self.assertEqual(
            (value["outcomes"], value["reported"], value["covered"]), (2000, 1985, 1903)
        )
        self.assertAlmostEqual(value["coverage"], 1903 / 2000)
        self.assertAlmostEqual(value["reporting_rate"], 1985 / 2000)
        self.assertGreater(value["nominal_mcse"], 0)
        self.assertLess(value["median_width_over_effect_bound"], 1.5)

    def test_changed_current_numerical_implementation_cannot_borrow_old_receipt(self) -> None:
        path = self.root / "crates/antecedent-estimate/src/observational_policy.rs"
        path.write_bytes(path.read_bytes() + b"\n// changed numerical-source footprint\n")
        with self.assertRaisesRegex(ValueError, "numerical recipe dependency changed"):
            self.validate()

    def test_nearby_size_or_another_level_never_borrows_a_receipt(self) -> None:
        for field, value in [
            ("sample_size", 3000),
            ("nominal", 0.9),
            ("normal_critical_value", 1.96),
        ]:
            with self.subTest(field=field):
                broken = copy.deepcopy(self.recipe)
                broken[field] = value
                with self.assertRaisesRegex(ValueError, "exact case/construction/size"):
                    receipts.validate_recipe(self.root, broken, self.case, 1000)

    def test_changed_source_or_conditional_denominator_is_rejected(self) -> None:
        self.recipe["source_files"][receipts.SOURCE_PATH] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source audit mismatch"):
            self.validate()
        self.recipe = copy.deepcopy(
            next(r for r in self.recipes if r["case_id"] == self.case["id"])
        )
        self.recipe["coverage_semantics"] = "conditional_reported_intervals"
        with self.assertRaises(ValueError):
            self.validate()

    def test_raw_log_and_tally_cannot_hide_refusals(self) -> None:
        state = json.loads((self.root / self.recipe["tally"]["path"]).read_text(encoding="utf-8"))
        state.update(scored=2000, skipped=0, with_interval=2000)
        self.replace_asset(self.recipe["tally"], state)
        with self.assertRaisesRegex(ValueError, "legacy gate log.*counts disagree"):
            self.validate()

    def test_reporting_floor_is_applied_to_all_observed_outcomes(self) -> None:
        self.case["targets"]["min_reporting_rate"] = 0.999
        with self.assertRaisesRegex(ValueError, "reporting rate"):
            self.validate()

    def test_an_inconclusive_mean_width_bound_does_not_pass(self) -> None:
        self.recipe["width_evidence"] = {
            "kind": "saved_mean_positive_truth_bound",
            "truth_model": "normal_threshold_policy_value",
        }
        with self.assertRaisesRegex(ValueError, "does not prove"):
            self.validate()

    def test_width_evidence_cannot_use_another_truth_or_size(self) -> None:
        self.recipe["width_evidence"] = {
            "kind": "saved_mean_positive_truth_bound",
            "truth_model": "six_cell_exponential_survival_difference",
        }
        with self.assertRaisesRegex(ValueError, "another generating truth"):
            self.validate()

    def test_width_plan_changes_are_rejected_even_with_rehashed_metadata(self) -> None:
        spec = self.recipe["width_evidence"]
        plan = json.loads((self.root / spec["plan"]["path"]).read_text(encoding="utf-8"))
        plan["domain"] = "post_hoc_seed_domain"
        self.replace_asset(spec["plan"], plan)
        with self.assertRaisesRegex(ValueError, "predeclared plan"):
            self.validate()

    def test_raw_width_data_must_match_original_completed_receipt(self) -> None:
        spec = self.recipe["width_evidence"]
        path = self.root / spec["raw"]["path"]
        path.chmod(0o644)
        raw = path.read_bytes().replace(b"reported,", b"refused,", 1)
        path.write_bytes(raw)
        spec["raw"]["sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "original completed receipt"):
            self.validate()

    def test_no_synthetic_coverage_record_fields_are_accepted(self) -> None:
        self.recipe["cov_id"] = "cov.fabricated"
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            self.validate()

    def test_real_registry_accepts_nine_exact_receipts_including_fresh_family(self) -> None:
        valid, problems = receipts.validate_registry(
            ROOT, list(self.cases.values()), closure.case_size
        )
        self.assertEqual(problems, [])
        self.assertEqual(valid, {r["case_id"] for r in self.recipes})
        self.assertEqual(len(valid), 9)

    def test_knn_receipts_reuse_exact_original_counts_and_printed_precision(self) -> None:
        for n, expected_covered in [(10000, 1899), (100000, 1882)]:
            recipe = copy.deepcopy(
                next(r for r in self.recipes if r["case_id"] == f"knn_ci_path_{n}")
            )
            target = self.cases[recipe["case_id"]]
            value = receipts.validate_asserted_knn(self.root, recipe, target, n)
            self.assertEqual(
                (value["outcomes"], value["reported"], value["covered"]),
                (2000, 2000, expected_covered),
            )
            self.assertLess(value["median_width_over_effect_bound"], 1.5)
            recipe["execution_grid_point"] = 1
            with self.assertRaisesRegex(ValueError, "exact source construction/size"):
                receipts.validate_asserted_knn(self.root, recipe, target, n)

    def test_knn_passed_gate_is_not_a_generic_width_or_test_calibration_receipt(self) -> None:
        recipe = copy.deepcopy(next(r for r in self.recipes if r["case_id"] == "knn_ci_path_10000"))
        target = self.cases[recipe["case_id"]]
        path = self.root / recipe["coverage_log"]["path"]
        raw = path.read_bytes().replace(b"median width 0.1026", b"median width 0.9000")
        path.write_bytes(raw)
        recipe["coverage_log"]["sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "targets are not met"):
            receipts.validate_asserted_knn(self.root, recipe, target, 10000)
        target = self.cases["scalable_gpdc_replacement_validated_separately"]
        recipe["case_id"] = target["id"]
        recipe["test"] = target["evidence"]["test"]
        with self.assertRaisesRegex(ValueError, "exact source construction/size"):
            receipts.validate_asserted_knn(self.root, recipe, target, 8000)

    def count_recipe(self, case_id):
        recipe = copy.deepcopy(next(r for r in self.recipes if r["case_id"] == case_id))
        case = copy.deepcopy(self.cases[case_id])
        return recipe, case

    def test_genuine_count_witnesses_preserve_original_receipts_and_actual_conditional_n(
        self,
    ) -> None:
        for cid, covered in [
            ("scalable_gpdc_replacement_validated_separately", 1922),
            ("scalable_sensitivity_replacement_validated_separately", 2000),
        ]:
            recipe, case = self.count_recipe(cid)
            result = receipts.validate_replayed_assertion(
                self.root, recipe, case, closure.case_size(case)
            )
            self.assertEqual(
                (result["outcomes"], result["reported"], result["covered"]), (2000, 2000, covered)
            )
            self.assertEqual(result["receipt_kind"], "recordless_gate")
            self.assertEqual(
                result["coverage_semantics"], "conditional_reported_original_assertion"
            )
            case["targets"]["min_replicates"] = 2001
            with self.assertRaisesRegex(ValueError, "actual conditional reported count"):
                receipts.validate_replayed_assertion(
                    self.root, recipe, case, closure.case_size(case)
                )

    def test_statistical_test_width_exemption_cannot_license_a_sensitivity_interval(self) -> None:
        recipe, case = self.count_recipe("scalable_gpdc_replacement_validated_separately")
        case["expected"]["kind"] = "calibrated_interval"
        with self.assertRaisesRegex(ValueError, "exact source/size/role"):
            receipts.validate_replayed_assertion(self.root, recipe, case, 8000)
        recipe, case = self.count_recipe("scalable_sensitivity_replacement_validated_separately")
        case["targets"]["max_median_width_over_effect"] = 0.1
        with self.assertRaisesRegex(ValueError, "actual bracket width"):
            receipts.validate_replayed_assertion(self.root, recipe, case, 50000)

    def test_count_receipt_cannot_be_rebound_to_a_different_requested_seed_or_replicate_budget(
        self,
    ) -> None:
        for which in ("seed", "replicates"):
            recipe, case = self.count_recipe("scalable_gpdc_replacement_validated_separately")
            if which == "seed":
                case["dataset"]["seed"] += 1
            else:
                case["configuration"]["replicates"] += 1
            with self.assertRaisesRegex(ValueError, "original fixed source recipe"):
                receipts.validate_replayed_assertion(self.root, recipe, case, 8000)

    def test_count_witness_raw_counts_cannot_disagree_with_the_completed_receipt(self) -> None:
        recipe, case = self.count_recipe("scalable_gpdc_replacement_validated_separately")
        path = self.root / recipe["witness_log"]["path"]
        data = path.read_bytes().replace(b'"rejected":78', b'"rejected":0')
        path.write_bytes(data)
        recipe["witness_log"]["sha256"] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ValueError, "plan/raw execution"):
            receipts.validate_replayed_assertion(self.root, recipe, case, 8000)

    def test_count_recovery_requires_the_sealed_original_protocol_and_unchanged_current_recipe(
        self,
    ) -> None:
        recipe, case = self.count_recipe("scalable_sensitivity_replacement_validated_separately")
        plan = json.loads((self.root / recipe["witness_plan"]["path"]).read_text(encoding="utf-8"))
        plan["cases"][1]["attempted"] = 2200
        self.replace_asset(recipe["witness_plan"], plan)
        with self.assertRaisesRegex(ValueError, "sealed pre-outcome plan"):
            receipts.validate_replayed_assertion(self.root, recipe, case, 50000)
        recipe, case = self.count_recipe("scalable_gpdc_replacement_validated_separately")
        shutil.copyfile(
            ROOT / recipe["witness_plan"]["path"], self.root / recipe["witness_plan"]["path"]
        )
        source = self.root / recipe["source_path"]
        source.write_text(
            source.read_text(encoding="utf-8").replace("6_101_623", "6_101_624"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            receipts.validate_replayed_assertion(self.root, recipe, case, 8000)

    def test_validated_receipts_replace_only_the_missing_typed_requirement(self) -> None:
        data = {"case": [self.case]}
        self.assertTrue(closure.calibrated_record_problems(data, []))
        self.assertEqual(closure.calibrated_record_problems(data, [], {self.case["id"]}), [])
        self.assertTrue(closure.calibrated_record_problems(data, [], {"unrelated"}))


if __name__ == "__main__":
    unittest.main()
