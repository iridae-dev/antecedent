"""Fault controls for classification, reference integrity and local build reuse."""

from __future__ import annotations

import ast
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evidence_audit as audit
import native_build_identity as native
import run_dowhy_comparison as comparator
import test_evidence as evidence
import tomllib
import verify_provenance_citations as citations
from evidence_kinds import evidence_roles, inline_reference_problems, kind_problems
from reference_compare import values_close


class Classification(unittest.TestCase):
    def test_external_comparison_requires_each_named_case_and_rejects_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "result.xml"
            self.assertTrue(comparator.result_problems(report, {"truth"}))
            for contents, must_fail in (
                ('<testsuite><testcase name="truth"/></testsuite>', False),
                (
                    '<testsuite><testcase name="truth"><skipped/></testcase></testsuite>',
                    True,
                ),
                ('<testsuite><testcase name="other"/></testsuite>', True),
                ("<testsuite/>", True),
            ):
                report.write_text(contents, encoding="utf-8")
                problems = comparator.result_problems(report, {"truth"})
                self.assertEqual(bool(problems), must_fail)

    def test_inline_truth_is_bound_to_its_assertion_and_smoke_does_not_become_conformance(
        self,
    ):
        row = {
            "evidence_test": "test.rs",
            "evidence_assertion": "truth",
            "known_truth_reference": {
                "kind": "closed_form",
                "source": "test.rs::truth",
                "proposition": "effect=2",
                "tolerance": "absolute 1e-9",
                "limitations": "fixed linear model",
            },
        }
        self.assertEqual(inline_reference_problems(row), [])
        row["known_truth_reference"]["source"] = "other.rs::unrelated"
        self.assertTrue(inline_reference_problems(row))
        row["known_truth_reference"]["kind"] = "regression_pin"
        self.assertTrue(inline_reference_problems(row))
        self.assertEqual(
            evidence_roles("internal_known_truth", "statistical_smoke"),
            ["software", "statistical_smoke"],
        )

    def test_external_claim_cannot_use_enumeration_or_a_regression_pin(self):
        for kind in ("frozen_external_oracle", "behavioral_parity"):
            for origin in ("enumeration", "regression_pin", "closed_form"):
                self.assertTrue(kind_problems(kind, {"kind": origin}))
            self.assertEqual(kind_problems(kind, {"kind": "external_package"}), [])

    def test_consistency_does_not_acquire_a_scientific_role(self):
        for kind in ("internal_cross_check", "regression_pin", "implementation_exists"):
            self.assertEqual(evidence_roles(kind), ["software"])
        self.assertEqual(
            evidence_roles("internal_known_truth"), ["software", "scientific"]
        )
        self.assertTrue(kind_problems("invented", None))
        self.assertTrue(
            kind_problems("internal_known_truth", {"kind": "regression_pin"})
        )
        mixed = {"kind": "regression_pin", "independent_inferences": ["Frequentist"]}
        self.assertEqual(
            kind_problems("internal_known_truth", mixed, "Frequentist"), []
        )
        self.assertTrue(kind_problems("internal_known_truth", mixed, "Bayesian"))

    def test_empty_selection_fails_and_duplicate_assertions_execute_once(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            audit.scientific_rows([])
        row = {
            "key": "one",
            "roles": ["scientific"],
            "evidence_test": "test.py",
            "evidence_assertion": "test_truth",
        }
        selected = audit.scientific_rows([row, {**row, "key": "two"}])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["evidence_assertion"], "test_truth")
        with self.assertRaisesRegex(ValueError, "empty"):
            audit.scientific_rows(
                [{**row, "evidence_activity": "calibration_acceptance"}]
            )

    def test_calibration_marker_is_explicit_and_never_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_activity.py"
            path.write_text(
                "import pytest\n"
                "# @pytest.mark.requires_calibration\n"
                "def test_ordinary():\n    assert 1 == 1\n"
                "@pytest.mark.requires_calibration\n"
                "def test_measured():\n    assert 1 == 1\n",
                encoding="utf-8",
            )
            marks = evidence.python_test_markers(path)
            self.assertNotIn("requires_calibration", marks["test_ordinary"])
            self.assertIn("requires_calibration", marks["test_measured"])
            self.assertEqual(evidence.static_python_test(path, "test_measured"), [])
        runner = (Path(__file__).parents[1] / "scripts/test.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("python_tests --durations=30", runner)
        self.assertNotIn("pytest.skip", runner)

    def test_generator_presence_requires_an_explicit_retained_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "conformance/demo/truth"
            fixture.mkdir(parents=True)
            (fixture / "reference.py").write_text(
                "# a nearby file proves nothing\n", encoding="utf-8"
            )
            path = fixture / "expected.json"
            doc = {
                "oracle": {"kind": "external_package"},
                "reference": {
                    "command": "python /tmp/deleted.py",
                    "generator_retained": False,
                },
            }
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = audit.fixture_report(root, path)
            self.assertIn("external_generator_not_retained", result["gaps"])
            self.assertIn("historical_generator_deleted", result["gaps"])
            self.assertEqual(result["generators"], {})
            doc["reference"] = {"command": "python conformance/demo/truth/reference.py"}
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = audit.fixture_report(root, path)
            self.assertEqual(result["gaps"], [])
            self.assertEqual(len(result["generators"]), 1)
            self.assertNotEqual(result["independence_review"], "verified")


class ReferenceComparison(unittest.TestCase):
    def test_small_roundoff_passes_but_numpy_default_tolerance_drift_fails(self):
        self.assertTrue(values_close([2.0], [2.0 + 1e-12], rtol=1e-9, atol=0))
        self.assertFalse(values_close([2.0], [2.0 + 1e-6], rtol=1e-9, atol=0))

    def test_metadata_shapes_and_nonfinite_truth_cannot_pass(self):
        for left, right in (
            (True, 1),
            ({"n": 100000}, {"n": 100001}),
            ([1.0], [1.0, 2.0]),
            ({"a": 1}, {"b": 1}),
            (float("nan"), float("nan")),
            (float("inf"), float("inf")),
        ):
            self.assertFalse(values_close(left, right, rtol=1e-3, atol=1e-3))
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                values_close(1, 1, rtol=tolerance, atol=0)

    def test_actual_generators_reject_perturbed_expectations(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("rd_sharp", "path_specific_edge_gformula"):
            path = root / "conformance/estimate" / name / "reference.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            function = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "values_close"
            )
            # Load the actual comparison function without importing numerical
            # dependencies or recomputing fixture truth in this gate self-test.
            namespace = {"compare_values": values_close}
            exec(  # noqa: S102
                compile(
                    ast.Module(body=[function], type_ignores=[]), str(path), "exec"
                ),
                namespace,
            )
            compare = namespace["values_close"]
            self.assertTrue(compare({"truth": 2.0}, {"truth": 2.0 + 1e-12}))
            self.assertFalse(compare({"truth": 2.0}, {"truth": 2.0 + 1e-6}))


class NativeBuildIdentity(unittest.TestCase):
    def test_other_interpreter_abi_caches_do_not_replace_the_current_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python/antecedent"
            package.mkdir(parents=True)
            current = package / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
            current.write_bytes(b"current ABI")
            (package / "_native.cpython-000-unused.so").write_bytes(b"other ABI")
            self.assertEqual(native.extension(root), current)
            native.begin_build(root)
            native.finish_build(root)
            native.check(root)

    def test_sources_changed_during_a_build_cannot_receive_a_fresh_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python/antecedent"
            package.mkdir(parents=True)
            (
                package / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
            ).write_bytes(b"new build")
            source = root / "Cargo.toml"
            source.write_text("before build", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot missing"):
                native.finish_build(root)
            native.begin_build(root)
            source.write_text("edited during build", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed during"):
                native.finish_build(root)
            self.assertFalse((root / native.STAMP).exists())
            native.begin_build(root)
            native.finish_build(root)
            native.check(root)

    def test_dirty_sources_binary_replacement_and_removed_inputs_fail_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python/antecedent"
            package.mkdir(parents=True)
            binary = package / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
            binary.write_bytes(b"binary one")
            source = root / "crates/demo/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn original() {}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                native.check(root)
            native.begin_build(root)
            native.finish_build(root)
            native.check(root)
            source.write_text("fn changed() {}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                native.check(root)

            source.write_text("fn original() {}", encoding="utf-8")
            native.check(root)
            binary.write_bytes(b"binary two")
            with self.assertRaisesRegex(ValueError, "stale"):
                native.check(root)
            binary.write_bytes(b"binary one")
            source.unlink()
            with self.assertRaisesRegex(ValueError, "stale"):
                native.check(root)


class CitationIdentity(unittest.TestCase):
    def test_offline_gate_rejects_corrupted_or_missing_primary_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "provenance").mkdir()
            (root / "provenance/test.toml").write_text(
                'papers = [{title = "Independent causal inference", doi = "10.1000/test", sections = ["Theorem 1"]}]\n',
                encoding="utf-8",
            )
            cache = root / "cache.json"
            record = {
                "10.1000/test": {
                    "doi": "10.1000/test",
                    "titles": ["Independent causal inference"],
                }
            }
            cache.write_text(json.dumps(record), encoding="utf-8")
            command = [
                sys.executable,
                str(Path(citations.__file__)),
                "--root",
                str(root),
                "--cache",
                str(cache),
                "--check",
            ]
            self.assertEqual(
                subprocess.run(command, capture_output=True, check=False).returncode, 0
            )
            record["10.1000/test"]["titles"] = ["An unrelated paper"]
            cache.write_text(json.dumps(record), encoding="utf-8")
            failed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("title_mismatch", failed.stdout)
            cache.unlink()
            failed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("not_resolved", failed.stdout)

    def test_resolving_a_doi_to_an_unrelated_paper_never_verifies_it(self):
        row = {"doi": "10.1000/test", "title": "A causal identification algorithm"}
        cache = {
            row["doi"]: {"doi": row["doi"], "titles": ["Learning Lagrangian dynamics"]}
        }
        self.assertEqual(citations.verdict(row, cache), "title_mismatch")
        cache[row["doi"]]["titles"] = ["A Causal Identification Algorithm"]
        self.assertEqual(
            citations.verdict(row, cache), "bibliographic_identity_verified"
        )
        cache[row["doi"]]["doi"] = "10.1000/other"
        self.assertEqual(citations.verdict(row, cache), "doi_mismatch")
        self.assertEqual(citations.verdict(row, {}), "not_resolved")

    def test_arxiv_uses_datacite_and_publisher_uses_crossref(self):
        self.assertTrue(
            citations.registry_url("10.48550/arXiv.1234.5678").startswith(
                "https://api.datacite.org/dois/"
            )
        )
        self.assertTrue(
            citations.registry_url("10.2307/1234").startswith(
                "https://api.crossref.org/works/"
            )
        )


class TargetDiscovery(unittest.TestCase):
    def test_native_python_crate_closure_uses_its_own_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            package = root / "python"
            (package / "src").mkdir(parents=True)
            (package / "Cargo.toml").write_text(
                '[package]\nname = "antecedent-py"\n', encoding="utf-8"
            )
            path = package / "src/lib.rs"
            path.write_text(
                "#[cfg(test)] mod tests {\n"
                "#[test] fn native_boundary() { assert_eq!(reference_value(), 5); }\n"
                "fn reference_value() -> i32 { 2 + 3 }\n}\n",
                encoding="utf-8",
            )
            with patch.object(evidence, "ROOT", root):
                self.assertIn("2 + 3", evidence.rust_closure(path, "native_boundary"))

    def test_nested_include_has_its_actual_target_and_orphans_have_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            crate = root / "crates/demo"
            tests = crate / "tests"
            (tests / "sub").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "demo"\nautotests = false\n'
                '[[test]]\nname = "chain"\npath = "tests/outer.rs"\n',
                encoding="utf-8",
            )
            (tests / "outer.rs").write_text(
                '#[path = "sub/middle.rs"] mod middle;\n', encoding="utf-8"
            )
            (tests / "sub/middle.rs").write_text(
                '#[path = "inner.rs"] mod inner;\n', encoding="utf-8"
            )
            inner = tests / "sub/inner.rs"
            inner.write_text(
                "#[test] fn actually_compiled() { assert_eq!(2 + 2, 4); }",
                encoding="utf-8",
            )
            orphan = tests / "orphan.rs"
            orphan.write_text("#[test] fn never_compiled() {}", encoding="utf-8")
            with patch.object(evidence, "ROOT", root):
                resolved = evidence.target_root(inner)
                self.assertEqual(
                    resolved, (tests / "outer.rs", "demo", ["--test", "chain"])
                )
                self.assertIsNone(evidence.target_root(orphan))


class CalibrationBoundary(unittest.TestCase):
    def test_full_learner_measurement_is_explicit_and_point_conformance_remains_selected(
        self,
    ):
        root = Path(__file__).resolve().parents[1]
        path = root / "crates/antecedent-estimate/src/dml.rs"
        self.assertTrue(
            any(
                "ignore" in p
                for p in evidence.rust_test_declaration_problems(
                    path, "learner_ate_known_truth_coverage"
                )
            )
        )
        self.assertEqual(
            evidence.rust_test_declaration_problems(
                path, "learner_ate_known_truth_point_conformance"
            ),
            [],
        )
        rows = tomllib.loads(
            (root / "parity/estimate.toml").read_text(encoding="utf-8")
        )["capabilities"]
        for row in rows:
            if row["id"] in {
                "estimate.dml",
                "estimate.dr.learner",
                "estimate.causal.forest",
            }:
                self.assertEqual(
                    row["evidence_assertion"],
                    "learner_ate_known_truth_point_conformance",
                )
                self.assertEqual(
                    row["calibration_evidence"]["status"], "registered_unmeasured"
                )
        result = subprocess.run(
            ["bash", str(root / "scripts/gate_calibration.sh")],
            cwd=root,
            env={
                **os.environ,
                "ANTECEDENT_CALIBRATION_DRY_RUN": "1",
                "ANTECEDENT_JOINT_RESPONSE_STABLE_CANDIDATE": "",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("learner_ate_known_truth: coverage", result.stdout)


if __name__ == "__main__":
    unittest.main()
