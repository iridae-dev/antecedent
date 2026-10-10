"""Strict finite refusal and shared-owner proof regressions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomllib
from promotion_source import (
    defensive_carrier_route_problems,
    finite_refusal_helpers,
    input_exception_evidence_problems,
    nontest_pair,
    refusal_stage_literals,
    shared_namespace_problems,
    typed_input_exception_problems,
)


class FiniteRefusalTests(unittest.TestCase):
    def prove(self, caller, *, helper_prefix="", body=None, extra=""):
        body = (
            body
            or 'EstimationError::refused(antecedent_core::reason_code!("invalid_argument"), format!("example.{detail}: {message}"),)'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.rs"
            path.write_text(
                f"{helper_prefix}fn invalid(detail: &str, message: &str) -> EstimationError {{{body}}}\npub fn run(k: &str) {{ {caller} }}\n{extra}",
                encoding="utf-8",
            )
            return finite_refusal_helpers(path, in_crate=False)

    def test_literal_calls_are_finite(self):
        literals, lines = self.prove('invalid("bad_a", "a"); invalid("bad_b", "b");')
        self.assertEqual(
            [literal.value for literal in literals], ["example.bad_a", "example.bad_b"]
        )
        self.assertEqual(lines, {1})
        self.assertEqual(
            {literal.reason_code for literal in literals}, {"invalid_argument"}
        )

    def test_nonliteral_or_escaping_calls_remain_dynamic(self):
        for caller in [
            'invalid(k, "a");',
            "let callback = invalid;",
            'invalid(concat!("bad", "_a"), "a");',
            'wrap!(invalid("bad_a", "a"));',
            'super::invalid("bad_a", "a");',
        ]:
            with self.subTest(caller=caller):
                self.assertEqual(self.prove(caller), ([], set()))

    def test_open_or_unbounded_helpers_remain_dynamic(self):
        for args in [
            {"helper_prefix": "pub "},
            {"extra": "mod child;"},
            {"body": "other(detail, message)"},
            {
                "body": 'format!("example.{detail}: {message}"); invalid(detail, message)'
            },
        ]:
            with self.subTest(args=args):
                self.assertEqual(
                    self.prove('invalid("bad_a", "a");', **args), ([], set())
                )

    def test_actual_private_nested_helpers_have_finite_literal_calls(self):
        root = Path(__file__).resolve().parents[1]
        literals, lines = finite_refusal_helpers(
            root / "crates/antecedent-estimate/src/nested_markov_binary.rs"
        )
        self.assertEqual(
            {literal.value for literal in literals},
            {
                "nested_markov.invalid_counts",
                "nested_markov.iteration_bound",
                "nested_markov.invalid_options",
                "nested_markov.constraint_violated",
                "nested_markov.fit_not_converged",
                "nested_markov.not_normalized",
                "nested_markov.fitted_boundary",
            },
        )
        self.assertEqual(len(lines), 2)


class SharedNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/strict-proof")
        self.source = "crates/core/src/helper.rs"
        self.descriptor = {
            "namespace": "shared",
            "owner_record": "owner",
            "sources": [self.source],
        }
        self.pair = {"detail": "shared.outside_scope", "code": "cell_not_licensed"}
        self.owner = {
            "id": "owner",
            "status": "promoted",
            "shared_refusal_namespaces": [self.descriptor],
            "refusals": [self.pair],
        }
        self.record = {
            "id": "consumer",
            "shared_refusal_namespaces": [self.descriptor],
            "refusals": [self.pair, {"detail": "own.bad", "code": "invalid_argument"}],
        }
        self.emissions = {
            "shared": {"shared.outside_scope": [(self.root / self.source, True, None)]}
        }

    def check(self):
        return shared_namespace_problems(
            self.record, [self.owner, self.record], self.emissions, self.root
        )

    def test_explicit_finite_owner_is_accepted(self):
        self.assertEqual(self.check(), [])

    def test_unknown_namespace_unowned_source_and_code_mismatch_are_rejected(self):
        for mutate in [
            lambda: self.record["refusals"].append(
                {"detail": "other.bad", "code": "invalid_argument"}
            ),
            lambda: self.descriptor["sources"].append("crates/core/src/unowned.rs"),
            lambda: self.owner.update(shared_refusal_namespaces=[]),
            lambda: self.owner.update(
                refusals=[
                    {"detail": "shared.outside_scope", "code": "invalid_argument"}
                ]
            ),
        ]:
            self.setUp()
            mutate()
            self.assertTrue(self.check())

    def test_class_only_scientific_wrong_class_or_helper_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        valid = {
            "evidence_test": "python/tests/test_measured_inference.py",
            "evidence_assertion": "test_native_input_exceptions_preserve_class_without_scientific_reason",
            "detail": "measured_inference.invalid_expectation",
            "exception_class": "CausalValueError",
            "helper": "crate::value_err",
            "source": "python/src/measured_inference_api.rs",
            "reason_code_absent": True,
        }
        self.assertEqual(typed_input_exception_problems(valid, root), [])
        for mutation in [
            {"detail": "nested_markov.measured_scope"},
            {"exception_class": "CausalResourceError"},
            {"helper": "arbitrary::value_err"},
            {"reason_code_absent": False},
            {"code": "invalid_argument"},
            {"unproven": True},
            {"defensive_source_only": True},
        ]:
            self.assertTrue(typed_input_exception_problems(valid | mutation, root))

    def test_defensive_route_proof_rejects_factory_and_dispatch_drift(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(defensive_carrier_route_problems(root), [])
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            for name in [
                "python/src/measured_inference_api.rs",
                "crates/antecedent-io/src/measured_inference.rs",
            ]:
                (fake / name).parent.mkdir(parents=True, exist_ok=True)
                (fake / name).write_text(
                    (root / name).read_text(encoding="utf-8"), encoding="utf-8"
                )
            path = fake / "python/src/measured_inference_api.rs"
            path.write_text(
                path.read_text().replace('"joint_bayesian" =>', '"seventh_route" =>'),
                encoding="utf-8",
            )
            self.assertTrue(defensive_carrier_route_problems(fake))

    def test_class_only_constructor_drift_and_unproven_evidence_fail(self):
        root = Path(__file__).resolve().parents[1]
        valid = {
            "evidence_test": "python/tests/test_measured_inference.py",
            "evidence_assertion": "test_native_input_exceptions_preserve_class_without_scientific_reason",
            "detail": "measured_inference.memory_budget_exceeded",
            "exception_class": "CausalResourceError",
            "helper": "crate::CausalResourceError::new_err",
            "source": "python/src/measured_inference_api.rs",
            "reason_code_absent": True,
        }
        original_pair = nontest_pair

        def mutated_pair(path, **kwargs):
            code, skel = original_pair(path, **kwargs)
            if path == root / valid["source"]:
                code = code.replace(
                    "return Err(crate::CausalResourceError::new_err(",
                    "return Err(crate::with_reason_code(crate::CausalResourceError::new_err(",
                )
                skel = code
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=mutated_pair):
            self.assertTrue(typed_input_exception_problems(valid, root))
        self.assertEqual(
            input_exception_evidence_problems(
                valid, root / valid["evidence_test"], valid["evidence_assertion"]
            ),
            [],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_false.py"
            path.write_text(
                'def test_false():\n    assert "CausalResourceError"\n    assert getattr(error, "reason_code", None) is None\n',
                encoding="utf-8",
            )
            self.assertTrue(
                input_exception_evidence_problems(valid, path, "test_false")
            )

    def test_stage_metadata_is_distinct_from_message_and_same_line_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.rs"
            path.write_text(
                'pub fn run() { let fields = RefusalFields { stage: Some("example.fit".to_owned()), };\n let message = "example.fit"; }',
                encoding="utf-8",
            )
            self.assertEqual(
                refusal_stage_literals(path, in_crate=False), {(1, "example.fit")}
            )
            other = Path(directory) / "other.rs"
            other.write_text(
                'pub fn run() { let fields = RefusalFields { stage: Some("example.fit".to_owned()), }; let message = "example.fit"; }',
                encoding="utf-8",
            )
            self.assertEqual(refusal_stage_literals(other, in_crate=False), set())


class UntrustedIntervalInputTests(unittest.TestCase):
    def test_original_report_input_passes_and_authority_or_numeric_drift_refuses(self):
        from check_interval_coordinates import untrusted_interval_input_problems

        root = Path(__file__).resolve().parents[1]
        records = tomllib.loads(
            (root / "parity/promotion_2_3.toml").read_text(encoding="utf-8")
        )["record"]
        record = next(
            r for r in records if r["id"] == "2.3A.F15.measured_scalar_inference"
        )
        item = record["untrusted_interval_inputs"][0]
        self.assertEqual(untrusted_interval_input_problems(root, record, item), [])
        for mutation in [
            {"class": "MeasuredScalarReport"},
            {"authority_type": "ScalarExecution"},
            {"unknown": True},
            {"authority_fixture": "unproven"},
        ]:
            self.assertTrue(
                untrusted_interval_input_problems(root, record, item | mutation)
            )
        source = root / item["source"]
        original_pair = nontest_pair
        drifts = [
            ("for execution in executions {", "for mut execution in executions {"),
            ("lower: execution.interval.0,", "lower: execution.interval.0 + 1.0,"),
            (
                "Result<MeasuredInferenceReport, IoError>",
                "Result<MeasuredInference, IoError>",
            ),
            ("pub(crate) fn authorize(", "pub fn authorize("),
            (
                "    report: MeasuredInferenceReport,",
                "    pub report: MeasuredInferenceReport,",
            ),
            (
                "#[derive(Clone, Debug)]\npub struct MeasuredInference",
                "#[derive(Clone, Debug, Deserialize)]\npub struct MeasuredInference",
            ),
        ]
        for before, after in drifts:

            def changed(path, before=before, after=after, **kwargs):
                code, skel = original_pair(path, **kwargs)
                return (
                    (code.replace(before, after), skel)
                    if path == source
                    else (code, skel)
                )

            with (
                self.subTest(drift=after),
                patch("check_interval_coordinates.nontest_pair", side_effect=changed),
            ):
                self.assertTrue(untrusted_interval_input_problems(root, record, item))


if __name__ == "__main__":
    unittest.main()
