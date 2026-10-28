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
import provenance_links
import run_dowhy_comparison as comparator
import test_evidence as evidence
import tomllib
import verify_provenance_citations as citations
from evidence_kinds import (
    evidence_roles,
    inline_reference_problems,
    kind_problems,
    reference_review_problems,
    software_only_roles,
)
from reference_compare import values_close


class Classification(unittest.TestCase):
    def test_reason_namespaces_keep_prose_and_device_fallbacks_out_of_scientific_codes(self):
        from parity_rows import reason_obligation, runtime_code_literals

        for name in ("counterparts.toml", "support_graphless.toml", "estimator_route_residual.toml", "test_refactor_dispositions.toml"):
            self.assertIsNone(reason_obligation(Path(name), "reason"))
            self.assertEqual(reason_obligation(Path(name), "calibration_reason"), "calibration")
            self.assertEqual(reason_obligation(Path(name), "reason_code"), "runtime_refusal")
        self.assertEqual(reason_obligation(Path("python_products.toml"), "reason"), "python_product")
        self.assertEqual(reason_obligation(Path("support_licensed.toml"), "calibration_reason"), "calibration")
        body = "pub const fn code(&self) -> &'static str {\n        match self { Self::Missing => \"no_device\" }\n    }"
        fallback = "impl FallbackReason {\n    " + body + "\n}"
        path = Path("crates/antecedent-learn-burn/src/policy.rs")
        self.assertEqual(runtime_code_literals(path, fallback), [])
        self.assertEqual(runtime_code_literals(path, fallback + "\nimpl ScientificError {\n    " + body + "\n}"), ["no_device"])
        self.assertEqual(runtime_code_literals(Path("crates/other/src/error.rs"), fallback), ["no_device"])
        self.assertEqual(runtime_code_literals(path, "// " + body.splitlines()[0] + "\n"), [])
        self.assertEqual(runtime_code_literals(path, 'const DECOY: &str = r#"' + body + '"#;'), [])
        self.assertEqual(runtime_code_literals(Path("crates/other/src/error.rs"), fallback.replace("fn code(&self)", "fn   code ( &self )")), ["no_device"])

    def test_registered_rust_acceptance_requires_exact_live_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / "crates/p"
            (crate / "src").mkdir(parents=True)
            (root / "parity").mkdir()
            (crate / "Cargo.toml").write_text('[package]\nname="p"\nversion="0.1.0"\nautotests=false\n', encoding="utf-8")
            source = crate / "src/lib.rs"
            text = '#[cfg(test)] mod tests { #[test] fn truth() { assert_eq!(actual(), 2); } }'
            source.write_text(text, encoding="utf-8")
            registry = root / "parity/v3_test_registry.toml"
            registry.write_text('[[test]]\npath="crates/p/src/lib.rs"\ncrate="p"\ncargo_target="lib"\ncalibration_acceptance_assertions=["truth"]\n', encoding="utf-8")
            rows = evidence.registered_calibration_acceptance(root)
            self.assertEqual(rows[0]["libtest_name"], "tests::truth")
            (root / "parity/claim.toml").write_text('[[claim]]\nevidence_kind="internal_known_truth"\nevidence_test="crates/p/src/lib.rs"\nevidence_assertion="truth"\n', encoding="utf-8")
            claim, = audit.claim_rows(root)
            self.assertEqual(claim["evidence_activity"], "calibration_acceptance")
            with self.assertRaisesRegex(ValueError, "empty"):
                audit.scientific_rows([claim])
            listing = {"rust-suites": {"p": {"testcases": {"tests::truth": {"ignored": False, "filter-match": {"status": "matches"}}}}}}
            self.assertEqual(evidence.calibration_discovery_problems(rows, listing), [])
            for mutation in ({}, {"rust-suites": {"wrong": listing["rust-suites"]["p"]}},
                             {"rust-suites": {"p": {"testcases": {"tests::truth": {"ignored": True}}}}},
                             {"rust-suites": {"p": {"testcases": {"tests::truth": {"ignored": False, "filter-match": {"status": "mismatch"}}}}}}):
                self.assertTrue(evidence.calibration_discovery_problems(rows, mutation))
            for changed in (text.replace("#[test]", "#[test] #[ignore]"), text.replace("fn truth", "fn removed")):
                source.write_text(changed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    evidence.registered_calibration_acceptance(root)

    def test_scientific_lane_resolves_exact_rust_assertions_and_rejects_unregistered_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / "crates/p"
            (crate / "src").mkdir(parents=True)
            (root / "parity").mkdir()
            (crate / "Cargo.toml").write_text('[package]\nname="p"\nversion="0.1.0"\nautotests=false\n', encoding="utf-8")
            source = crate / "src/lib.rs"
            text = '#[cfg(test)] mod tests { #[test] fn truth() { assert_eq!(actual(), 2); } }'
            source.write_text(text, encoding="utf-8")
            registry = root / "parity/v3_test_registry.toml"
            registry.write_text('[[test]]\npath="crates/p/src/lib.rs"\ncrate="p"\ncargo_target="lib"\n', encoding="utf-8")
            rows = evidence.registered_rust_assertions([("crates/p/src/lib.rs", "truth")] * 2, root)
            self.assertEqual([r["libtest_name"] for r in rows], ["tests::truth"])
            self.assertEqual(
                evidence.nextest_selection(rows),
                "(package(=p) and kind(lib) and test(=tests::truth))",
            )
            listing = {"rust-suites": {"p": {"testcases": {"tests::truth": {"ignored": False, "filter-match": {"status": "matches"}}}}}}
            self.assertEqual(evidence.calibration_discovery_problems(rows, listing), [])
            self.assertTrue(evidence.calibration_discovery_problems(rows, {"rust-suites": {}}))
            with self.assertRaisesRegex(ValueError, "not in parity/v3_test_registry"):
                evidence.registered_rust_assertions([("crates/p/src/other.rs", "truth")], root)
            for changed in (text.replace("#[test]", "#[test] #[ignore]"), text.replace("fn truth", "fn removed")):
                source.write_text(changed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    evidence.registered_rust_assertions([("crates/p/src/lib.rs", "truth")], root)

    def test_script_runner_rejects_failed_skipped_and_missing_required_assertions(self):
        import contextlib
        import io
        import run_script_tests as runner

        def passing():
            self.assertEqual(2 + 2, 4)

        def failing():
            self.fail("injected incorrect output")

        def skipped():
            raise unittest.SkipTest("injected required skip")

        class ExpectedFailure(unittest.TestCase):
            @unittest.expectedFailure
            def runTest(self):
                self.fail("a required assertion cannot be waived")

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "receipt.json"
            def run(functions, *extra):
                suite = unittest.TestSuite(fn if isinstance(fn, unittest.TestCase) else unittest.FunctionTestCase(fn) for fn in functions)
                with (
                    patch.object(sys, "argv", ["runner", "--report", str(report), *extra]),
                    patch.object(runner.unittest.defaultTestLoader, "discover", return_value=suite),
                    patch.object(runner, "checkout", return_value={"commit": "fixture"}),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    return runner.main()

            self.assertEqual(run([passing]), 0)
            self.assertEqual(run([failing]), 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["failures"], ["failing"])
            self.assertEqual(run([skipped]), 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["unexpected_skips"], ["skipped"])
            expected = ExpectedFailure()
            self.assertEqual(run([expected]), 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["expected_failures"], [expected.id()])
            self.assertEqual(run([]), 1)
            self.assertEqual(run([passing], "--exclude-module", "passing"), 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["explicit_deselections"], ["passing"])
            with self.assertRaises(SystemExit) as unknown:
                run([passing], "--exclude-module", "unknown")
            self.assertEqual(unknown.exception.code, 2)

    def test_cached_source_rejects_same_process_assertion_and_helper_changes(self):
        from _source_cache import source_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.rs"
            path.write_text("#[test] fn truth() { helper(); }\nfn helper() { assert_eq!(output(), 2); }", encoding="utf-8")
            self.assertIn("assert_eq!", evidence.source_masked(path).code)
            evidence.rust_items(path)
            path.write_text("#[ignore] #[test] fn truth() { helper(); }\nfn helper() { assert_eq!(output(), 3); }", encoding="utf-8")
            masked, items = evidence.rust_items(path)
            self.assertIn("output(), 3", masked.code)
            self.assertIn("#[ignore]", next(i for i in items if i.name == "truth").attrs)
            helper = Path(tmp) / "helper.rs"
            helper.write_text("first", encoding="utf-8")

            @source_cache
            def read(source):
                return source.read_text(encoding="utf-8")

            @source_cache
            def transitive(source):
                return read(source), read(helper)

            self.assertEqual(transitive(path)[1], "first")
            helper.write_text("other", encoding="utf-8")
            self.assertEqual(transitive(path)[1], "other")

    def test_script_inventory_names_exact_unittest_assertions_and_rejects_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_control.py"
            path.write_text(
                "import unittest\nclass Control(unittest.TestCase):\n"
                "    def test_actual(self):\n        self.assertEqual(run(), 2)\n"
                "    @unittest.skip('required assertion skipped')\n"
                "    def test_skipped(self):\n        self.assertTrue(run())\n"
                "class Helper:\n    def test_fake(self):\n        assert True\n",
                encoding="utf-8",
            )
            self.assertEqual(evidence.script_test_declaration_problems(path, "Control.test_actual"), [])
            for name in ["Helper.test_fake", "Control.test_skipped", "Control.missing"]:
                self.assertTrue(evidence.script_test_declaration_problems(path, name))
            rows = evidence.python_test_declarations(path)
            self.assertEqual(rows[0]["assertion_nodes"], 1)
            self.assertFalse(rows[0]["discovery_verified"])
            path.write_text(path.read_text(encoding="utf-8") +
                "\nclass Control(unittest.TestCase):\n    def test_actual(self):\n        self.assertTrue(run())\n", encoding="utf-8")
            self.assertTrue(evidence.script_test_declaration_problems(path, "Control.test_actual"))

    def test_primary_review_rejects_stale_code_and_altered_citation_sections(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for role, rel in [
                ("implementation", "method.rs"),
                ("assertion", "test.rs"),
            ]:
                path = root / rel
                path.write_text("// integrity control only\n", encoding="utf-8")
                artifacts.append(
                    {
                        "role": role,
                        "path": rel,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            record = {
                "feature_id": "control",
                "papers": [{"sections": ["reviewed section"]}],
            }
            record["method_review"] = {
                "status": "reviewed_with_limits",
                "record_sha256": provenance_links.method_review_digest(record),
                "artifacts": artifacts,
                **{
                    key: "synthetic integrity control"
                    for key in (
                        "scope",
                        "historical_exposure",
                        "redistribution",
                        "limitations",
                    )
                },
                **{
                    key: ["synthetic integrity control"]
                    for key in ("assumptions", "sources", "findings_resolved")
                },
            }
            self.assertEqual(provenance_links.method_review_problems(root, record), [])
            record["papers"][0]["sections"] = ["unreviewed section"]
            self.assertTrue(provenance_links.method_review_problems(root, record))
            record["papers"][0]["sections"] = ["reviewed section"]
            (root / "method.rs").write_text(
                "// changed implementation\n", encoding="utf-8"
            )
            self.assertTrue(provenance_links.method_review_problems(root, record))

    def test_non_algorithm_mapping_rejects_missing_targets_stale_digests_and_overreach(self):
        import hashlib

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m.rs").write_text("// integrity control only\n", encoding="utf-8")
            (root / "t.rs").write_text("// assertion control\n", encoding="utf-8")
            known = {"control", "other"}

            def build(kind="wrapper", **changes):
                record = {"feature_id": "control", "papers": [], "implementation_deviations": ["d"]}
                review = {
                    "status": "justified_non_algorithm",
                    "kind": kind,
                    "rationale": "synthetic control",
                    "no_independent_algorithm": "synthetic control",
                    "covered_by": ["other"],
                    "derivation": "synthetic control",
                    "artifacts": [
                        {"role": "implementation", "path": "m.rs", "sha256": digest(root / "m.rs")},
                        {"role": "assertion", "path": "t.rs", "sha256": digest(root / "t.rs")},
                    ],
                }
                review.update(changes)
                record["non_algorithm_review"] = review
                review["record_sha256"] = provenance_links.non_algorithm_digest(record)
                return record

            def problems(record):
                return provenance_links.non_algorithm_review_problems(root, record, known)

            self.assertEqual(problems(build()), [])
            self.assertEqual(problems(build("derivation")), [])
            # A record that carries no mapping is not a finding: absence is reported by the triage.
            self.assertEqual(problems({"feature_id": "control", "papers": []}), [])
            # Mapping to a record that does not exist, or to itself.
            self.assertTrue(problems(build(covered_by=["missing.record"])))
            self.assertTrue(problems(build(covered_by=["control"])))
            # A wrapper that names no covering record, an unknown kind, a missing rationale.
            self.assertTrue(problems(build(covered_by=[])))
            self.assertTrue(problems(build(kind="folklore")))
            self.assertTrue(problems(build(rationale="  ")))
            self.assertTrue(problems(build(no_independent_algorithm="")))
            # A derivation without its statement, or without an executing assertion.
            self.assertTrue(problems(build("derivation", derivation="")))
            record = build("closed_variant")
            record["non_algorithm_review"]["artifacts"].pop()
            record["non_algorithm_review"]["record_sha256"] = provenance_links.non_algorithm_digest(record)
            self.assertTrue(problems(record))
            # Mapping to a module that does not exist.
            record = build()
            record["non_algorithm_review"]["artifacts"][0]["path"] = "gone.rs"
            self.assertTrue(problems(record))
            # No mapped implementation at all.
            self.assertTrue(problems(build(artifacts=[])))
            # A record that lists a paper is for method_review, not for this mapping.
            record = build()
            record["papers"] = [{"title": "t", "sections": ["s"]}]
            self.assertTrue(problems(record))
            # Edited record text stales the mapping.
            record = build()
            record["implementation_deviations"] = ["edited"]
            self.assertTrue(problems(record))
            # Inline tests are checked, not trusted.
            self.assertTrue(problems(build("derivation", inline_tests=True, artifacts=[
                {"role": "implementation", "path": "m.rs", "sha256": digest(root / "m.rs")}])))
            # Changed mapped implementation stales the digest.
            record = build()
            (root / "m.rs").write_text("// changed implementation\n", encoding="utf-8")
            self.assertTrue(problems(record))

    def test_software_fixture_cannot_be_promoted_to_known_truth_by_an_origin_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "expected.json"
            path.write_text(
                json.dumps(
                    {
                        "evidence_roles": ["software"],
                        "oracle": {"kind": "external_package"},
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "evidence_kind": "frozen_external_oracle",
                "known_truth_fixture": "expected.json",
            }
            self.assertIn("software-only", evidence.fixture_kind_problems(root, row)[0])
            path.write_text(json.dumps({"evidence_roles": None}), encoding="utf-8")
            self.assertTrue(evidence.fixture_kind_problems(root, row))

    def test_unclassified_fixture_is_reported_and_software_declarations_need_a_live_assertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_live.py").write_text(
                "def test_live():\n    assert 1 == 1\n", encoding="utf-8"
            )
            path = root / "expected.json"
            declared = {
                "evidence_roles": ["software"],
                "evidence_kind": "regression_pin",
                "evidence_source": "test_live.py::test_live",
                "evidence_proposition": "A declared proposition.",
                "limitations": "A declared limit.",
            }

            def gaps(document):
                path.write_text(json.dumps(document), encoding="utf-8")
                return audit.fixture_report(root, path)["gaps"]

            self.assertIn("oracle_origin_unrecorded", gaps({"tolerance": 1.0}))
            self.assertEqual(gaps(declared), [])
            self.assertEqual(
                gaps({**declared, "evidence_roles": ["software", "statistical_smoke"]}), []
            )
            self.assertIn(
                "software_fixture_missing_exact_assertion",
                gaps({**declared, "evidence_source": "test_live.py"}),
            )
            self.assertTrue(
                gaps({**declared, "evidence_source": "test_live.py::test_removed"})
            )
            (root / "test_live.py").write_text(
                "import pytest\n@pytest.mark.skip\ndef test_live():\n    assert 1 == 1\n",
                encoding="utf-8",
            )
            self.assertTrue(gaps(declared))
            (root / "test_live.py").write_text(
                "def test_live():\n    assert 1 == 1\n", encoding="utf-8"
            )
            for field in ("evidence_proposition", "limitations"):
                self.assertIn(
                    f"software_fixture_missing_{field}", gaps({**declared, field: " "})
                )
            self.assertIn(
                "invalid_fixture_evidence_roles",
                gaps({**declared, "evidence_roles": ["software", "verified"]}),
            )
            # A scientific role is not a software-only declaration, so it needs an origin.
            self.assertIn(
                "oracle_origin_unrecorded",
                gaps({**declared, "evidence_roles": ["software", "scientific"]}),
            )

    def test_known_truth_claim_must_cite_a_test_that_names_its_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            test = root / "tests" / "it.rs"
            claim = {
                "evidence_kind": "internal_known_truth",
                "evidence_test": "tests/it.rs",
                "known_truth_fixture": "conformance/area/case",
            }
            test.write_text("#[test]\nfn other() {}\n", encoding="utf-8")
            self.assertEqual(
                audit.fixture_load_problems(root, claim),
                ["known_truth_fixture_not_named_by_cited_test"],
            )
            test.write_text('const F: &str = "conformance/area/case/expected.json";\n', encoding="utf-8")
            self.assertEqual(audit.fixture_load_problems(root, claim), [])
            (root / "tests" / "common").mkdir()
            test.write_text("#[test]\nfn other() {}\n", encoding="utf-8")
            (root / "tests" / "common" / "load.rs").write_text('fn f() { fixture_dir("case"); }\n', encoding="utf-8")
            self.assertEqual(audit.fixture_load_problems(root, claim), [])
            self.assertEqual(
                audit.fixture_load_problems(root, {**claim, "evidence_kind": "regression_pin"}), []
            )

    def test_reviews_that_contradict_their_independence_claim_are_rejected(self):
        import hashlib

        def build(root, generator_source, generator_path="reference.py", kind="closed_form",
                  assertion_path="test_truth.py"):
            artifacts = []
            for role, rel, text in (
                ("generator", generator_path, generator_source),
                ("environment_lock", "reference.py.lock", "v = 1\n"),
                ("assertion_source", assertion_path, "def test_t():\n    assert 1\n"),
            ):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_text(text, encoding="utf-8")
                artifacts.append(
                    {"role": role, "path": rel, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                )
            review = {
                key: "reviewed scope, with explicit limits"
                for key in ("scope", "independence_basis", "historical_exposure", "redistribution", "limitations")
            }
            return {"kind": kind, "reference_review": {**review, "artifacts": artifacts}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(reference_review_problems(root, build(root, "import numpy\n")), [])
        for name, source, kwargs, expected in (
            ("imports product", "import antecedent\n", {}, "depends on product code"),
            ("invokes cargo", "import subprocess\nsubprocess.run('cargo test', shell=True)\n", {}, "depends on product code"),
            ("regression pin", "import numpy\n", {"kind": "regression_pin"}, "regression_pin"),
            ("product source", "fn main() {}\n", {"generator_path": "crates/antecedent-stats/src/lib.rs"}, "product source"),
            ("test as generator", "import numpy\n", {"generator_path": "test_truth.py"}, "also the asserting test"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                problems = reference_review_problems(root, build(root, source, **kwargs))
                self.assertTrue(any(expected in p for p in problems), problems)

    def test_review_must_bind_the_generator_and_lock_the_oracle_declares(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for role, rel in (
                ("generator", "other_reference.py"),
                ("environment_lock", "other_reference.py.lock"),
                ("assertion_source", "test_truth.py"),
            ):
                (root / rel).write_text("x = 1\n", encoding="utf-8")
                artifacts.append(
                    {"role": role, "path": rel, "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
                )
            oracle = {
                "kind": "closed_form",
                "generator": "reference.py",
                "environment_lock": "reference.py.lock",
                "reference_review": {
                    **{
                        key: "reviewed scope, with explicit limits"
                        for key in ("scope", "independence_basis", "historical_exposure", "redistribution", "limitations")
                    },
                    "artifacts": artifacts,
                },
            }
            problems = reference_review_problems(root, oracle)
            self.assertTrue(any("declared generator" in p for p in problems), problems)
            self.assertTrue(any("declared environment_lock" in p for p in problems), problems)
            oracle["generator"] = "other_reference.py"
            oracle["environment_lock"] = "other_reference.py.lock"
            self.assertEqual(reference_review_problems(root, oracle), [])

    def test_software_only_claim_must_state_its_limits(self):
        claim = {"roles": ["software"], "evidence_kind": "implementation_exists"}
        self.assertEqual(
            audit.software_claim_problems(claim), ["software_claim_missing_limitations"]
        )
        self.assertTrue(audit.software_claim_problems({**claim, "limitations": "  "}))
        self.assertFalse(audit.software_claim_problems({**claim, "limitations": "Smoke only."}))
        self.assertFalse(
            audit.software_claim_problems({"roles": ["software", "scientific"], "evidence_kind": "internal_known_truth"})
        )

    def test_every_consumer_shares_one_software_only_predicate(self):
        self.assertTrue(software_only_roles(["software"]))
        self.assertTrue(software_only_roles(["software", "statistical_smoke"]))
        for roles in (None, [], ["scientific"], ["software", "scientific"], ["process"]):
            self.assertFalse(software_only_roles(roles))

    def test_known_truth_claim_cannot_rest_on_a_software_only_fixture(self):
        for roles in (["software"], ["software", "statistical_smoke"]):
            for kind in ("internal_known_truth", "frozen_external_oracle", "behavioral_parity"):
                self.assertTrue(
                    audit.software_fixture_claim_problems(
                        {"evidence_kind": kind}, {"evidence_roles": roles}
                    )
                )
        self.assertFalse(
            audit.software_fixture_claim_problems(
                {"evidence_kind": "regression_pin"}, {"evidence_roles": ["software"]}
            )
        )
        self.assertFalse(
            audit.software_fixture_claim_problems(
                {"evidence_kind": "internal_known_truth"},
                {"evidence_roles": ["software", "scientific"]},
            )
        )

    def test_altered_tolerance_or_false_independence_invalidates_a_retained_review(self):
        import hashlib

        from evidence_kinds import fixture_payload_digest, fixture_review_problems

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for role, rel in (
                ("generator", "reference.py"),
                ("environment_lock", "reference.py.lock"),
                ("assertion_source", "test_truth.py"),
            ):
                (root / rel).write_text("x = 1\n", encoding="utf-8")
                artifacts.append(
                    {
                        "role": role,
                        "path": rel,
                        "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(),
                    }
                )
            review = {
                key: "reviewed scope, with explicit limits"
                for key in (
                    "scope",
                    "independence_basis",
                    "historical_exposure",
                    "redistribution",
                    "limitations",
                )
            }
            review["artifacts"] = artifacts
            document = {
                "oracle": {"kind": "external_package", "reference_review": review},
                "tolerance": {"absolute": 1e-9},
            }
            review["fixture_payload_sha256"] = fixture_payload_digest(document)
            self.assertEqual(fixture_review_problems(root, document), [])
            loosened = {**document, "tolerance": {"absolute": 1e-3}}
            self.assertTrue(fixture_review_problems(root, loosened))
            # Re-declaring the origin cannot reuse a review recorded for another origin.
            relabelled = {
                **document,
                "oracle": {**document["oracle"], "kind": "closed_form"},
            }
            self.assertTrue(fixture_review_problems(root, relabelled))
            # A review that states no independence basis is not a review.
            blank = {**review, "independence_basis": " "}
            self.assertTrue(
                fixture_review_problems(
                    root, {"oracle": {"reference_review": blank}, "tolerance": 0}
                )
            )

    def test_retained_review_rejects_removed_assertions_and_changed_helpers(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for role, rel, source in [
                ("generator", "reference.py", "import numpy\n"),
                (
                    "comparison_helper",
                    "compare.py",
                    "def compare(x,y): return x == y\n",
                ),
                ("environment_lock", "reference.py.lock", "version = 1\n"),
                ("input_fixture", "data.csv", "x,y\n1,2\n"),
                (
                    "assertion_source",
                    "test_truth.py",
                    "def test_truth():\n    assert actual() == 2\n",
                ),
            ]:
                path = root / rel
                path.write_text(source, encoding="utf-8")
                artifacts.append(
                    {
                        "role": role,
                        "path": rel,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            review = {
                key: "reviewed scope, with explicit limits"
                for key in (
                    "scope",
                    "independence_basis",
                    "historical_exposure",
                    "redistribution",
                    "limitations",
                )
            }
            oracle = {"reference_review": {**review, "artifacts": artifacts}}
            self.assertEqual(reference_review_problems(root, oracle), [])
            from evidence_kinds import fixture_payload_digest, fixture_review_problems

            document = {"oracle": oracle, "expected": 2.0}
            oracle["reference_review"]["fixture_payload_sha256"] = (
                fixture_payload_digest(document)
            )
            self.assertEqual(fixture_review_problems(root, document), [])
            self.assertTrue(
                fixture_review_problems(root, {**document, "expected": 3.0})
            )
            for rel in ["test_truth.py", "compare.py", "data.csv"]:
                path = root / rel
                original = path.read_text(encoding="utf-8")
                path.write_text("# removed or changed\n", encoding="utf-8")
                self.assertTrue(reference_review_problems(root, oracle))
                path.write_text(original, encoding="utf-8")
            (root / "data.csv").unlink()
            self.assertIn(
                "reference_review missing artifact: data.csv",
                reference_review_problems(root, oracle),
            )
            oracle["reference_review"]["artifacts"] = []
            self.assertTrue(reference_review_problems(root, oracle))

    def test_mixed_fixture_cannot_relabel_regression_fields_as_independent_truth(self):
        mixed = {"kind": "external_package", "independent_fields": ["basis"]}
        self.assertTrue(kind_problems("frozen_external_oracle", mixed))
        self.assertTrue(
            kind_problems("frozen_external_oracle", mixed, fields=["fitted"])
        )
        self.assertEqual(
            kind_problems("frozen_external_oracle", mixed, fields=["basis"]), []
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "expected.json"
            path.write_text(json.dumps({"oracle": mixed}), encoding="utf-8")
            row = {
                "known_truth_fixture": "expected.json",
                "evidence_kind": "frozen_external_oracle",
                "evidence_fields": ["fitted"],
            }
            self.assertTrue(evidence.fixture_kind_problems(root, row))
            row["evidence_fields"] = ["basis"]
            self.assertEqual(evidence.fixture_kind_problems(root, row), [])
            row["known_truth_fixture"] = "missing.json"
            self.assertTrue(evidence.fixture_kind_problems(root, row))

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

    def test_claim_reader_classifies_calibration_for_every_report_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            (root / "parity/v3_test_registry.toml").write_text("test = []\n", encoding="utf-8")
            (root / "python/tests").mkdir(parents=True)
            (root / "parity/example.toml").write_text(
                '[[claim]]\nevidence_kind = "internal_known_truth"\n'
                'evidence_test = "python/tests/test_example.py"\n'
                'evidence_assertion = "test_acceptance"\n', encoding="utf-8"
            )
            (root / "python/tests/test_example.py").write_text(
                'import pytest\n@pytest.mark.requires_calibration\n'
                'def test_acceptance():\n    assert measured() == expected\n',
                encoding="utf-8",
            )
            row, = audit.claim_rows(root)
            self.assertEqual(row["evidence_activity"], "calibration_acceptance")
            self.assertEqual(row["roles"], ["software", "scientific"])
            with self.assertRaisesRegex(ValueError, "empty"):
                audit.scientific_rows([row])

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

    def test_a_removed_or_foreign_extension_fails_even_with_a_fresh_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python/antecedent"
            package.mkdir(parents=True)
            binary = package / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
            binary.write_bytes(b"binary one")
            (root / "Cargo.toml").write_text("input", encoding="utf-8")
            native.begin_build(root)
            native.finish_build(root)
            native.check(root)
            # The stamp is fresh but the extension for this interpreter is gone.
            binary.unlink()
            with self.assertRaisesRegex(ValueError, "no native extension matches"):
                native.check(root)
            # An extension built for another interpreter ABI is not a substitute.
            (package / "_native.cpython-000-unused.so").write_bytes(b"other ABI")
            with self.assertRaisesRegex(ValueError, "no native extension matches"):
                native.check(root)


class PythonSkipGuard(unittest.TestCase):
    def run_session(self, tests: str, *args: str) -> subprocess.CompletedProcess:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "conftest.py").write_text(
                (repo / "python/tests/conftest.py").read_text(encoding="utf-8"), encoding="utf-8"
            )
            (root / "_repo_text.py").write_text(f"from pathlib import Path\nREPO_ROOT = Path({str(repo)!r})\n", encoding="utf-8")
            (root / "test_session.py").write_text(tests, encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
                cwd=root, capture_output=True, text=True, check=False,
            )

    def test_a_session_that_mostly_skips_or_selects_nothing_fails(self):
        def session(skipped: int) -> str:
            lines = ["import pytest"]
            for i in range(30):
                action = "pytest.skip('injected')" if i < skipped else "assert 1 + 1 == 2"
                lines.append(f"def test_{i}():\n    {action}")
            return "\n".join(lines) + "\n"

        body = session(3)
        failed = self.run_session(body)
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertIn("collected tests skipped", failed.stdout)
        few = session(1)
        self.assertEqual(self.run_session(few).returncode, 0)
        self.assertEqual(self.run_session(few, "-k", "no_such_test").returncode, 5)


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


class ScenarioMatrix(unittest.TestCase):
    def test_only_exact_live_tests_count_and_the_live_matrix_is_consistent(self):
        import scenario_matrix as matrix

        root = Path(__file__).resolve().parents[1]
        file_only = matrix._check_token(root, "crates/antecedent/tests/missing_policy_support.rs")
        self.assertFalse(file_only[0])
        smoke = matrix._check_token(root, "crates/antecedent/tests/x.rs::a_smoke_test")
        self.assertFalse(smoke[0])
        self.assertFalse(matrix._check_token(root, "crates/antecedent/tests/x.rs::limit_case")[0])
        missing = matrix._check_token(root, "crates/antecedent/tests/none.rs::t")
        self.assertFalse(missing[0])
        live = matrix._check_token(
            root,
            "crates/antecedent/tests/missing_policy_support.rs::"
            "every_route_cell_matches_the_declared_support",
        )
        self.assertTrue(live[0], live[1])
        built = matrix.build(root)
        self.assertEqual(built["findings"], [])
        self.assertTrue(all(c["evidence"] for c in built["cells"] if c["status"] == "delivered"))
        self.assertTrue(all(c.get("gap_kind") for c in built["cells"] if c["status"] == "gap"))
        self.assertEqual(sum(built["counts"].values()), len(built["cells"]))


if __name__ == "__main__":
    unittest.main()
