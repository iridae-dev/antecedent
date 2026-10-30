"""Strict finite refusal and shared-owner proof regressions."""

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from _repo import REPO_ROOT
from promotion_source import (
    defensive_carrier_route_problems,
    _core_refusal_helpers,
    Lit,
    Symbol,
    external_surface_ownership,
    _RECOVERY_EXTERNAL_SYMBOLS,
    _multiple_imputation_error_helpers,
    finite_refusal_helpers,
    input_exception_evidence_problems,
    nontest_pair,
    python_literals,
    rust_literals,
    refusal_stage_literals,
    rust_pub_items,
    sectioned_artifact_refusals,
    shared_namespace_problems,
    typed_input_exception_problems,
    uncalibrated_draws_allowed,
    _OBSERVATION_DRAWS_PROOF_FILES,
    _OBSERVATION_DRAWS_ROUTES,
    duplicate_refusal_pairs,
    refusal_code_pair_problems,
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
        self.assertEqual({literal.reason_code for literal in literals}, {"invalid_argument"})

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
            {"body": 'format!("example.{detail}: {message}"); invalid(detail, message)'},
        ]:
            with self.subTest(args=args):
                self.assertEqual(self.prove('invalid("bad_a", "a");', **args), ([], set()))

    def test_actual_private_nested_helpers_have_finite_literal_calls(self):
        root = REPO_ROOT
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

    def test_generated_artifact_refusals_are_a_closed_typed_macro_set(self):
        root = REPO_ROOT
        path = root / "crates/antecedent-io/src/nonlinear_mediation_artifact.rs"
        self.assertEqual(
            {(item.value, item.reason_code) for item in sectioned_artifact_refusals(path)},
            {
                ("nonlinear_mediation.wrong_contract", "route_not_supported"),
                ("nonlinear_mediation.report_replay_mismatch", "invalid_argument"),
            },
        )
        original_pair = nontest_pair

        def changed(path_arg, **kwargs):
            code, skel = original_pair(path_arg, **kwargs)
            if path_arg.resolve() == path.resolve():
                code = code.replace('"nonlinear_mediation",', '"unregistered_namespace",')
                skel = skel.replace('"nonlinear_mediation",', " " * len('"nonlinear_mediation",'))
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertEqual(sectioned_artifact_refusals(path), [])

        def comment_only(path_arg, **kwargs):
            code, skel = original_pair(path_arg, **kwargs)
            if path_arg.resolve() == path.resolve():
                code = code.replace("sectioned_artifact_error!(", "// sectioned_artifact_error!(")
                skel = skel.replace(
                    "sectioned_artifact_error!(",
                    " " * len("sectioned_artifact_error!("),
                )
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=comment_only):
            self.assertEqual(sectioned_artifact_refusals(path), [])

        def cfg_attr_call(path_arg, **kwargs):
            code, skel = original_pair(path_arg, **kwargs)
            if path_arg.resolve() == path.resolve():
                marker = "crate::sectioned_artifact_error!("
                replacement = "#[cfg_attr(all(), cfg(test))]\n" + marker
                code = code.replace(marker, replacement, 1)
                skel = skel.replace(marker, replacement, 1)
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=cfg_attr_call):
            self.assertEqual(sectioned_artifact_refusals(path), [])

        def changed_macro_detail_with_equal_length(path_arg, **kwargs):
            code, skel = original_pair(path_arg, **kwargs)
            if path_arg.name == "artifact_codec.rs":
                old_detail = '"{contract}.wrong_contract"'
                changed_detail = '"{contract}.wrong _contract"'
                old_spacing = "contract = $contract"
                changed_spacing = "contract= $contract"
                self.assertIn(old_spacing, code)
                self.assertEqual(len(changed_detail), len(old_detail) + 1)
                self.assertEqual(len(changed_spacing), len(old_spacing) - 1)
                code = code.replace(old_detail, changed_detail, 1).replace(
                    old_spacing, changed_spacing, 1
                )
                self.assertEqual(len(code), len(original_pair(path_arg, **kwargs)[0]))
            return code, skel

        with patch(
            "promotion_source.nontest_pair",
            side_effect=changed_macro_detail_with_equal_length,
        ):
            self.assertEqual(sectioned_artifact_refusals(path), [])

        for before, after in [
            ('reason_code!("route_not_supported")', 'reason_code!("invalid_argument")'),
            ('"{contract}.wrong_contract"', '"{contract}.wrong_namespace"'),
            ('"{contract}.report_replay_mismatch"', '"{contract}.other_replay"'),
            ("if $report {", "if true {"),
            (
                'reason_code!("route_not_supported")',
                'let decoy = r#"reason_code!(\\"route_not_supported\\")"#;',
            ),
        ]:

            def changed_definition(path_arg, before=before, after=after, **kwargs):
                code, skel = original_pair(path_arg, **kwargs)
                if path_arg.name == ("refusal.rs" if "reason_code!" in before else "artifact_codec.rs"):
                    self.assertIn(before, code)
                    code = code.replace(before, after)
                return code, skel

            with (
                self.subTest(before=before),
                patch("promotion_source.nontest_pair", side_effect=changed_definition),
            ):
                self.assertEqual(sectioned_artifact_refusals(path), [])

        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake.rs"
            fake.write_text(
                'pub fn text() { let value = "sectioned_artifact_error!(Fake, '
                'fake, wrong, result, true)"; }\n'
                "pub fn comment() { /* sectioned_artifact_error!(Fake, fake, wrong, result, true) */ }",
                encoding="utf-8",
            )
            self.assertEqual(sectioned_artifact_refusals(fake, in_crate=False), [])

    def test_sectioned_enum_expansion_rejects_noncode_and_conditional_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.rs"
            path.write_text(
                'pub fn strings() { let call = "crate::sectioned_artifact_enum!{Fake}"; }\n'
                "pub fn comments() { /* crate::sectioned_artifact_enum!{Fake} */ }\n"
                '#[cfg(test)] crate::sectioned_artifact_enum! { TestOnly { noun: "n", identity: "i", variants {}, display(f) {} } }\n'
                'other::sectioned_artifact_enum! { Unknown { noun: "n", identity: "i", variants {}, display(f) {} } }',
                encoding="utf-8",
            )
            names = {item.name for item in rust_pub_items(path, in_crate=False)}
            self.assertFalse({"Fake", "TestOnly", "Unknown"} & names)

    def test_enum_macro_and_public_aliases_are_inventory_symbols(self):
        root = REPO_ROOT
        receipt = rust_pub_items(root / "crates/antecedent-io-core/src/recalc_receipt_artifact.rs")
        self.assertIn("RecalcReceiptArtifactError", {item.name for item in receipt})
        aliases = rust_pub_items(root / "crates/antecedent-io-core/src/frozen_scores_artifact.rs")
        self.assertLessEqual(
            {"ScoreColumnPlain", "ScoreTablePlain"}, {item.name for item in aliases}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.rs"
            path.write_text(
                "pub type PublicAlias = u8; pub(crate) type CrateAlias = u8; "
                "mod private { pub type PrivateAlias = u8; }\n"
                "pub fn hidden() { struct Local; }\n",
                encoding="utf-8",
            )
            names = {item.name for item in rust_pub_items(path, in_crate=False)}
            self.assertEqual(names, {"PublicAlias", "hidden"})


    def test_reviewed_index_macro_expands_owner_types_and_mutation_fails_closed(self):
        path = REPO_ROOT / "crates/antecedent-identify/src/sid/counterfactual_transport.rs"
        items = rust_pub_items(path)
        names = ("CounterfactualVariableIndex", "CounterfactualLevelIndex", "CounterfactualSymbolId")
        for name in names:
            self.assertTrue(any(item.name == name and item.kind == "type" for item in items))
            self.assertTrue(any(item.name == "new" and item.owner == name for item in items))
            self.assertTrue(any(item.name == "get" and item.owner == name for item in items))
        self.assertFalse(any(item.name == "get" and item.owner is None for item in items))
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.resolve() == path.resolve():
                code = code.replace("Self(index)", "Self(0    )", 1)
                skel = skel.replace("Self(index)", "Self(0    )", 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            altered = rust_pub_items(path)
            self.assertFalse(any(item.name in names for item in altered))
            self.assertTrue(any(item.name == "get" and item.owner is None for item in altered))

    def test_generated_macro_requires_exact_transitive_core_mapping(self):
        original = nontest_pair
        def changed(path, **kwargs):
            code, skeleton = original(path, **kwargs)
            if path.name == "refusal.rs":
                code = code.replace('reason_code!("route_not_supported")', 'reason_code!("invalid_argument")')
            return code, skeleton
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertEqual(sectioned_artifact_refusals(
                REPO_ROOT / "crates/antecedent-io/src/nonlinear_mediation_artifact.rs"
            ), [])


class MetadataLiteralTests(unittest.TestCase):
    def test_artifact_proof_is_excluded_but_same_refusal_is_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "consumer.py"
            path.write_text(
                '@artifact_consumer(proof="example.consume: checked")\n'
                'def consume():\n'
                '    raise ValueError("example.consume: checked")\n'
                '@artifact_consumer(proof=make("example.dynamic: error"))\n'
                'def other(): pass\n', encoding="utf-8")
            values = [(lit.line, lit.value) for lit in python_literals(path)]
            self.assertNotIn((1, "example.consume: checked"), values)
            self.assertIn((3, "example.consume: checked"), values)
            self.assertIn((4, "example.dynamic: error"), values)

    def test_recalc_classification_is_not_an_emitter_and_mutation_fails_closed(self):
        original = Path.read_text
        for name in ("recalc.py", "recalc_cell.py"):
            path = REPO_ROOT / "python/antecedent" / name
            with self.subTest(name=name):
                self.assertNotIn("recalc.no_live_state", [x.value for x in python_literals(path)])
                def changed(path_arg, *args, **kwargs):
                    text = original(path_arg, *args, **kwargs)
                    if path_arg.resolve() == path.resolve():
                        return text.replace("return RecalcNoLiveState", "return RecalcRefusal", 1)
                    return text
                with patch.object(Path, "read_text", changed):
                    self.assertIn("recalc.no_live_state", [x.value for x in python_literals(path)])
                def emitted(path_arg, *args, **kwargs):
                    text = original(path_arg, *args, **kwargs)
                    if path_arg.resolve() == path.resolve():
                        return text + '\ndef new_refusal():\n    raise ValueError("recalc.no_live_state")\n'
                    return text
                with patch.object(Path, "read_text", emitted):
                    self.assertIn("recalc.no_live_state", [x.value for x in python_literals(path)])

    def test_provider_metadata_and_report_prose_do_not_hide_real_error_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.rs"
            path.write_text(
                'fn f() { let p = DistributionProvenance { provider_id: "example.redraw".to_owned(), };\n'
                'limits.push(format!("{} directions are left unresolved by the recovery.", count));\n'
                'refuse("example.redraw");\n'
                'refuse(format!("{} directions are left unresolved by the recovery.", count));\n'
                'refuse(format!("example.{detail}: {message}")); }\n', encoding="utf-8")
            values = [(lit.line, lit.value) for lit in rust_literals(path, in_crate=False)]
            self.assertNotIn((1, "example.redraw"), values)
            self.assertNotIn((2, "{} directions are left unresolved by the recovery."), values)
            self.assertIn((3, "example.redraw"), values)
            self.assertIn((4, "{} directions are left unresolved by the recovery."), values)
            self.assertIn((5, "example.{detail}: {message}"), values)


class CrossFileHelperTests(unittest.TestCase):
    def test_core_helpers_prove_only_literal_unshadowed_calls(self):
        self.assertEqual(_core_refusal_helpers(), {
            "changed_identity": "route_not_supported", "replay_mismatch": "invalid_argument"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.rs"
            path.write_text(
                'use antecedent_core::refusal::{changed_identity, replay_mismatch};\n'
                'fn run() { changed_identity("example.identity", field); '
                'replay_mismatch("example.replay", explanation); '
                'unknown("example.identity"); }\n', encoding="utf-8")
            entries = [(lit.value, lit.reason_code) for lit in rust_literals(path, in_crate=False)]
            self.assertIn(("example.identity", "route_not_supported"), entries)
            self.assertIn(("example.identity", None), entries)
            self.assertIn(("example.replay", "invalid_argument"), entries)
            path.write_text(path.read_text() + '\nfn changed_identity(x: &str, y: &str) {}\n')
            entries = [(lit.value, lit.reason_code) for lit in rust_literals(path, in_crate=False)]
            self.assertNotIn(("example.identity", "route_not_supported"), entries)

    def test_enum_metadata_does_not_mask_an_unproven_call_on_the_same_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.rs"
            path.write_text(
                'use antecedent_core::refusal::{changed_identity, replay_mismatch};\n'
                'crate::sectioned_artifact_enum! { E { identity: "example.identity", variants {}, '
                'display(f) { Self::X(what) => write!(f, "example.replay: {what}") } } }\n'
                'fn run() { changed_identity("example.identity", field); '
                'unknown("example.identity"); replay_mismatch("example.replay", text); '
                'let remedy = match detail.as_str() { "example.replay" => unknown("example.replay"), _ => () }; }\n',
                encoding="utf-8")
            entries = [(lit.value, lit.reason_code) for lit in rust_literals(path, in_crate=False)]
            self.assertEqual(entries.count(("example.identity", "route_not_supported")), 1)
            self.assertEqual(entries.count(("example.identity", None)), 1)
            self.assertEqual(entries.count(("example.replay", "invalid_argument")), 1)
            self.assertEqual(entries.count(("example.replay", None)), 1)

    def test_reason_code_match_patterns_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.rs"
            path.write_text('fn f(detail: &str) { match detail { "example.reason" => reason_code!("invalid_argument"), _ => "" }; }', encoding="utf-8")
            self.assertIn("example.reason", [lit.value for lit in rust_literals(path, in_crate=False)])

    def test_changed_core_body_cannot_prove_original_code(self):
        original = nontest_pair
        def changed(path, **kwargs):
            code, skel = original(path, **kwargs)
            if path.name == "refusal.rs" and "antecedent-core" in str(path):
                code = code.replace('reason_code!("route_not_supported")',
                                    'reason_code!("cancelled_no_claim")', 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertNotIn("changed_identity", _core_refusal_helpers())

    def test_sibling_error_helpers_require_exact_forwarding(self):
        self.assertEqual(_multiple_imputation_error_helpers(), {
            "invalid": "invalid_argument", "not_applicable": "missing_policy_not_applicable"})
        path = REPO_ROOT / "crates/antecedent-estimate/src/multiple_imputation/pool.rs"
        entries = [(lit.value, lit.reason_code) for lit in rust_literals(path)]
        self.assertIn(("multiple_imputation.pool_shape", "invalid_argument"), entries)
        original = nontest_pair
        def changed(path, **kwargs):
            code, skel = original(path, **kwargs)
            if path.name == "mod.rs" and path.parent.name == "multiple_imputation":
                code = code.replace('reason_code!("invalid_argument")',
                                    'reason_code!("cancelled_no_claim")', 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertNotIn("invalid", _multiple_imputation_error_helpers())

    def test_adaptive_premise_requires_the_verified_facade_and_default_mapping(self):
        path = REPO_ROOT / "crates/antecedent-stats/src/adaptive_inference/repeated_measures_posterior.rs"
        details = {"adaptive_inference.arm_not_observed", "adaptive_inference.selection_not_ignorable",
                   "adaptive_inference.structure_incomplete"}
        self.assertEqual({lit.value for lit in rust_literals(path) if lit.value in details and
                          lit.reason_code == "cell_not_licensed"}, details)
        original = nontest_pair
        for filename, before, after in [
            ("adaptive_procedure_report.rs", "adaptive_inference.borrowing_frozen", "adaptive_inference.arm_not_observed"),
            ("procedure_uncertainty.rs", 'reason_code!("cell_not_licensed")', 'reason_code!("invalid_argument")'),
            ("repeated_measures_posterior_report.rs", "refusal_of(&error)", "invalid(&error)"),
        ]:
            def changed(path_arg, filename=filename, before=before, after=after, **kwargs):
                code, skel = original(path_arg, **kwargs)
                if path_arg.name == filename:
                    code = code.replace(before, after, 1)
                return code, skel
            with self.subTest(filename=filename), patch("promotion_source.nontest_pair", side_effect=changed):
                self.assertFalse(any(lit.value in details and lit.reason_code == "cell_not_licensed"
                                     for lit in rust_literals(path)))

    def test_adaptive_verified_remedy_patterns_are_not_emitters(self):
        producer = REPO_ROOT / "crates/antecedent-stats/src/adaptive_inference/repeated_measures_posterior.rs"
        report = REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs"
        details = {"adaptive_inference.arm_not_observed", "adaptive_inference.selection_not_ignorable",
                   "adaptive_inference.structure_incomplete"}
        self.assertFalse(any(x.value in details for x in rust_literals(report)))
        self.assertEqual({x.value for x in rust_literals(producer)
                          if x.value in details and x.reason_code == "cell_not_licensed"}, details)
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.resolve() == report.resolve():
                code = code.replace("adaptive_inference.borrowing_frozen",
                                    "adaptive_inference.borrowing_freeze", 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertTrue(details <= {x.value for x in rust_literals(report)})
            self.assertFalse(any(x.value in details and x.reason_code == "cell_not_licensed"
                                 for x in rust_literals(producer)))

    def test_adaptive_rule_preserves_two_codes_and_fails_on_mapping_mutation(self):
        gap = REPO_ROOT / "crates/antecedent-stats/src/adaptive_inference/procedure_contract.rs"
        premise = REPO_ROOT / "crates/antecedent-stats/src/adaptive_inference/procedure_uncertainty.rs"
        detail = "adaptive_inference.rule_mismatch"
        self.assertTrue(any(x.value == detail and x.reason_code == "invalid_argument" for x in rust_literals(gap)))
        self.assertTrue(any(x.value == detail and x.reason_code == "cell_not_licensed" for x in rust_literals(premise)))
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.name == "procedure_contract.rs":
                code = code.replace('reason_code!("invalid_argument")', 'reason_code!("cell_not_licensed")', 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertFalse(any(x.value == detail and x.reason_code == "invalid_argument" for x in rust_literals(gap)))

    def test_decision_builder_pairs_require_unchanged_transitive_builder(self):
        path = REPO_ROOT / "crates/antecedent-design/src/decision_reliability.rs"
        expected = {("decision_reliability.search_record_required", "candidate_search_record_required"),
                    ("decision_reliability.validation_not_independent", "validation_stream_not_independent"),
                    ("decision_reliability.dependence_unknown", "sampling_dependence_unknown")}
        self.assertTrue(expected <= {(x.value, x.reason_code) for x in rust_literals(path)})
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.name == "signal.rs":
                code = code.replace("ExternalRefusal::new(self.code, self.stage, self.detail)",
                                    "ExternalRefusal::new(self.stage, self.code, self.detail)", 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertFalse(expected & {(x.value, x.reason_code) for x in rust_literals(path)})

    def test_no_live_state_requires_exact_bridge_forwarding(self):
        path = REPO_ROOT / "python/src/recalc_cell_api.rs"
        detail = "recalc.no_live_state"
        self.assertTrue(any(x.value == detail and x.reason_code == "score_table_unavailable"
                            for x in rust_literals(path)))
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.name == "bridge.rs":
                code = code.replace('map.insert("code".into(), Text(code.to_owned()))',
                                    'map.insert("code".into(), Text(stage.to_owned()))', 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            self.assertFalse(any(x.value == detail and x.reason_code == "score_table_unavailable"
                                 for x in rust_literals(path)))

    def test_transfer_helper_requires_closed_literal_calls(self):
        declaration = (
            "fn transfer_refusal(code: &'static str, scope: &str, message: &str) -> CausalError {"
            'antecedent_io::IoError::Refused { code, message: format!("mapped_posterior_transfer.{scope}: {message}"), }.into() }\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            for caller, expected in [
                ('transfer_refusal(antecedent_core::reason_code!("invalid_argument"), "wrong", message);', True),
                ('transfer_refusal(code, "wrong", message);', False),
                ('transfer_refusal(antecedent_core::reason_code!("invalid_argument"), scope, message);', False),
                ('let f = transfer_refusal;', False),
                ('other::transfer_refusal(antecedent_core::reason_code!("invalid_argument"), "wrong", message);', False),
            ]:
                with self.subTest(caller=caller):
                    path = Path(directory) / ("case" + str(len(caller)) + ".rs")
                    path.write_text(declaration + "fn run(message: &str) { " + caller + " }", encoding="utf-8")
                    found, lines = finite_refusal_helpers(path, in_crate=False)
                    self.assertEqual(bool(lines), expected)
                    if expected:
                        self.assertIn(("mapped_posterior_transfer.wrong", "invalid_argument"),
                                      [(lit.value, lit.reason_code) for lit in found])


class UncalibratedDrawQualification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for relative in _OBSERVATION_DRAWS_PROOF_FILES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO_ROOT / relative).read_bytes())
        self.record_id = "observation_query.finite_law_conditioning"
        self.draws = {
            "name": "antecedent_model.observation_uncertainty.redraw_conditional",
            "stage": "uncertainty", "status": "licensed", "claim": "point_only",
            "uncertainty_output": "uncalibrated_parameter_draws",
            "interval_route": "antecedent.observation_query.ObservationClaim.parameter_interval",
        }
        self.interval = {
            "name": self.draws["interval_route"], "stage": "uncertainty", "status": "closed",
            "reason_code": "cell_not_licensed",
            "refusal_test": "crates/antecedent/tests/observation_query_facade.rs",
            "refusal_assertion": "dirichlet_table_redraw_agrees_with_the_count_fitted_posterior_and_labels_its_spread",
        }

    def allow(self, draws=None, intervals=None, valid=lambda _: True, record=None):
        draws = self.draws if draws is None else draws
        intervals = [self.interval] if intervals is None else intervals
        return uncalibrated_draws_allowed(
            self.record_id if record is None else record, draws, [draws, *intervals], valid, root=self.root
        )

    def test_only_reviewed_producers_and_exercised_interval_refusals_pass(self):
        self.assertTrue(self.allow())
        self.assertFalse(self.allow(valid=lambda _: False))
        for producer, (interval_name, path, assertion) in _OBSERVATION_DRAWS_ROUTES.items():
            with self.subTest(producer=producer):
                draws = {**self.draws, "name": producer, "interval_route": interval_name}
                interval = {**self.interval, "name": interval_name,
                            "refusal_test": path, "refusal_assertion": assertion}
                self.assertTrue(self.allow(draws, [interval]))

    def test_wrong_or_missing_qualifiers_never_license(self):
        for key, value in [("name", "some_other_producer"), ("claim", "none"),
                           ("claim", "calibrated"), ("uncertainty_output", "posterior_interval"),
                           ("uncertainty_output", None), ("interval_route", None),
                           ("interval_route", "other_record.interval"), ("stage", "evaluate"),
                           ("status", "closed")]:
            with self.subTest(key=key, value=value):
                self.assertFalse(self.allow({**self.draws, key: value}))
        self.assertFalse(self.allow(record="another.record"))

    def test_unrelated_interval_or_test_cannot_satisfy_the_exception(self):
        for key, value in [("name", "another.interval"), ("status", "licensed"),
                           ("stage", "evaluate"), ("reason_code", "invalid_argument"),
                           ("refusal_test", "other.rs"), ("refusal_assertion", "other_test")]:
            with self.subTest(key=key, value=value):
                self.assertFalse(self.allow(intervals=[{**self.interval, key: value}]))
        self.assertFalse(self.allow(intervals=[self.interval, dict(self.interval)]))
        self.assertFalse(self.allow(intervals=[]))
        # Even another real closed interval and its real refusal test is unrelated.
        python_name, python_path, python_test = _OBSERVATION_DRAWS_ROUTES[
            "antecedent.python.observation_query.parameter_uncertainty"
        ]
        unrelated = {**self.interval, "name": python_name, "refusal_test": python_path,
                     "refusal_assertion": python_test}
        self.assertFalse(self.allow(intervals=[unrelated]))

    def test_producer_or_refusal_test_mutation_fails_closed(self):
        mutations = [
            ("crates/antecedent-model/src/observation_uncertainty.rs",
             b"pub fn redraw_conditional(", b"pub fn changed_redraw_conditional("),
            ("crates/antecedent/tests/observation_query_facade.rs",
             b'assert_eq!(refusal.code(), "cell_not_licensed");', b'assert!(true);'),
            ("python/tests/test_observation_query.py",
             b'assert caught.value.reason_code == "cell_not_licensed"', b'assert True'),
        ]
        for relative, before, after in mutations:
            with self.subTest(file=relative):
                path = self.root / relative
                original = path.read_bytes()
                self.assertIn(before, original)
                path.write_bytes(original.replace(before, after, 1))
                self.assertFalse(self.allow())
                path.write_bytes(original)
                self.assertTrue(self.allow())


class MultiCodeRefusalTests(unittest.TestCase):
    def test_two_codes_for_one_detail_are_distinct_but_identical_pairs_duplicate(self):
        rows = [{"code": "invalid_argument", "detail": "joint.example"},
                {"code": "option_not_applicable", "detail": "joint.example"}]
        self.assertEqual(duplicate_refusal_pairs(rows), set())
        self.assertEqual(duplicate_refusal_pairs(rows + [dict(rows[0])]),
                         {("invalid_argument", "joint.example")})
        self.assertEqual(refusal_code_pair_problems(
            {"invalid_argument", "option_not_applicable"},
            {"invalid_argument", "option_not_applicable"},
            {"invalid_argument", "option_not_applicable"}), (set(), set()))

    def test_missing_emitted_pair_and_unregistered_runtime_code_fail(self):
        codes = {"invalid_argument", "option_not_applicable"}
        self.assertEqual(refusal_code_pair_problems(codes, codes, {"invalid_argument"}),
                         ({"option_not_applicable"}, set()))
        self.assertEqual(refusal_code_pair_problems({"invalid_argument", "unproven"},
                         codes | {"unproven"}, codes), ({"unproven"}, set()))
        self.assertEqual(refusal_code_pair_problems({"invalid_argument"}, {"invalid_argument"}, codes),
                         (set(), {"option_not_applicable"}))

    def test_direct_refuse_pairs_prove_only_exact_code_detail_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "horizons.rs"
            path.write_text(
                'fn f() { refuse(antecedent_core::reason_code!("invalid_argument"), "joint.example", a); '
                'refuse(antecedent_core::reason_code!("option_not_applicable"), "joint.example", b); '
                'other("joint.example"); }', encoding="utf-8")
            entries = [(lit.value, lit.reason_code) for lit in rust_literals(path, in_crate=False)]
            self.assertEqual(entries.count(("joint.example", None)), 3)
        horizon = REPO_ROOT / "crates/antecedent/src/analysis/horizon_composition.rs"
        entries = {(lit.value, lit.reason_code) for lit in rust_literals(horizon)}
        self.assertIn(("joint_law_propagation.horizon_composition", "invalid_argument"), entries)
        self.assertIn(("joint_law_propagation.horizon_composition", "option_not_applicable"), entries)
        original = nontest_pair
        def changed(path_arg, **kwargs):
            code, skel = original(path_arg, **kwargs)
            if path_arg.name == "external_refusal.rs":
                code = code.replace("detail: detail.into()", "detail: String::new()", 1)
            return code, skel
        with patch("promotion_source.nontest_pair", side_effect=changed):
            entries = {(lit.value, lit.reason_code) for lit in rust_literals(horizon)}
            self.assertNotIn(("joint_law_propagation.horizon_composition", "invalid_argument"), entries)


class ExternalSurfaceOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = "crates/antecedent-estimate/src/recovery.rs"
        self.test = "crates/antecedent-estimate/tests/recovery_held_out_regimes.rs"
        self.fixture = "parity/cyclic/recovered-asymptotic-normal"
        for name in (self.source, self.test):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fixture", encoding="utf-8")
        (self.root / self.fixture).mkdir(parents=True)
        self.entry = {"registry": "parity/cyclic/acceptance.toml", "owner": "integrated_families",
                      "source": self.source, "symbols": sorted(_RECOVERY_EXTERNAL_SYMBOLS)}
        self.record = {"id": "recovery.binary_observation", "surface_rust": [self.source],
                       "surface_external_owners": [self.entry]}
        self.checked = []
        for name in sorted(_RECOVERY_EXTERNAL_SYMBOLS):
            owner, sep, method = name.partition(".")
            symbol = Symbol(method, "function", False, "method", owner) if sep else Symbol(name, "type", False, "top")
            self.checked.append((self.source, symbol))
        self.write_registry()

    def write_registry(self, *, owner="integrated_families", source=None, test=None, fixture=None):
        registry = self.root / self.entry["registry"]
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(
            f'[[capability]]\nid = "{owner}"\nworkflow = "prepare_experimental_recovery"\n'
            f'rust = ["{source or self.source}"]\ntests = ["{test or self.test}"]\n'
            f'fixtures = ["{fixture or self.fixture}"]\n', encoding="utf-8")

    def check(self):
        return external_surface_ownership(self.root, self.record, self.checked)

    def test_reviewed_exact_ownership_does_not_cover_future_symbol(self):
        owned, problems = self.check()
        self.assertEqual(problems, [])
        self.assertEqual(len(owned), len(_RECOVERY_EXTERNAL_SYMBOLS))
        self.assertNotIn((self.source, "future_public_operation"), owned)

    def test_wrong_registry_owner_source_and_extra_symbol_fail(self):
        for key, value in [("registry", "parity/other.toml"), ("owner", "other"),
                           ("source", "crates/other/src/recovery.rs"),
                           ("symbols", sorted(_RECOVERY_EXTERNAL_SYMBOLS | {"future_public_operation"}))]:
            with self.subTest(key=key):
                saved = self.entry[key]
                self.entry[key] = value
                self.assertTrue(self.check()[1])
                self.entry[key] = saved

    def test_missing_or_ambiguous_symbol_and_missing_evidence_fail(self):
        removed = self.checked.pop()
        self.assertTrue(self.check()[1])
        self.checked.append(removed)
        self.checked.append(removed)
        self.assertTrue(self.check()[1])
        self.checked.pop()
        (self.root / self.test).unlink()
        self.assertTrue(self.check()[1])

    def test_capability_must_be_unique_and_match_workflow_and_source(self):
        self.write_registry(owner="other")
        self.assertTrue(self.check()[1])
        self.write_registry(source="other.rs")
        self.assertTrue(self.check()[1])
        self.write_registry()
        registry = self.root / self.entry["registry"]
        registry.write_text(registry.read_text().replace("prepare_experimental_recovery", "other_workflow"), encoding="utf-8")
        self.assertTrue(self.check()[1])
        self.write_registry()
        (self.root / self.fixture).rmdir()
        self.assertTrue(self.check()[1])
        (self.root / self.fixture).mkdir()
        self.write_registry()
        registry.write_text(registry.read_text() + registry.read_text(), encoding="utf-8")
        self.assertTrue(self.check()[1])



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

    def test_shared_owner_must_register_each_proven_code_for_same_detail(self):
        self.owner["refusals"].append({"detail": "shared.outside_scope", "code": "invalid_argument"})
        self.emissions["shared"]["shared.outside_scope"] = [
            (self.root / self.source, True, Lit(1, "shared.outside_scope", None, "cell_not_licensed")),
            (self.root / self.source, True, Lit(2, "shared.outside_scope", None, "invalid_argument")),
        ]
        self.assertEqual(self.check(), [])
        self.owner["refusals"].pop()
        self.assertTrue(self.check())

    def test_unknown_namespace_unowned_source_and_code_mismatch_are_rejected(self):
        for mutate in [
            lambda: self.record["refusals"].append(
                {"detail": "other.bad", "code": "invalid_argument"}
            ),
            lambda: self.descriptor["sources"].append("crates/core/src/unowned.rs"),
            lambda: self.owner.update(shared_refusal_namespaces=[]),
            lambda: self.owner.update(
                refusals=[{"detail": "shared.outside_scope", "code": "invalid_argument"}]
            ),
        ]:
            self.setUp()
            mutate()
            self.assertTrue(self.check())

    def test_class_only_scientific_wrong_class_or_helper_are_rejected(self):
        root = REPO_ROOT
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
        original_pair = nontest_pair

        def uncoded_value_error(path, **kwargs):
            code, skel = original_pair(path, **kwargs)
            if path == root / "python/src/errors.rs":
                code = code.replace(
                    "attach_reason_from_message(value_error_of(message), message)",
                    "value_error_of(message)",
                )
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=uncoded_value_error):
            self.assertTrue(typed_input_exception_problems(valid, root))

        def forced_code_value_error(path, **kwargs):
            code, skel = original_pair(path, **kwargs)
            if path == root / "python/src/errors.rs":
                code = code.replace(
                    "attach_reason_from_message(value_error_of(message), message)",
                    'with_reason_code(attach_reason_from_message(value_error_of(message), message), "invalid_argument")',
                )
            return code, skel

        with patch("promotion_source.nontest_pair", side_effect=forced_code_value_error):
            self.assertTrue(typed_input_exception_problems(valid, root))

        def changed_prefix_with_old_text_decoy(path, **kwargs):
            code, skel = original_pair(path, **kwargs)
            if path == root / "crates/antecedent-core/src/reason_code.rs":
                code = code.replace(
                    'pub const PREFIX: &str = concat!("reason", "=");',
                    'pub const PREFIX: &str = concat!("other", "=");',
                )
                code += '\nlet decoy = r#"pub const PREFIX: &str = concat!("reason", "=");"#;'
            return code, skel

        with patch(
            "promotion_source.nontest_pair",
            side_effect=changed_prefix_with_old_text_decoy,
        ):
            self.assertTrue(typed_input_exception_problems(valid, root))

        def changed_prefix_literal_with_equal_length(path, **kwargs):
            code, skel = original_pair(path, **kwargs)
            if path == root / "crates/antecedent-core/src/reason_code.rs":
                old_literal = 'concat!("reason", "=")'
                changed_literal = 'concat!("rea son", "=")'
                old_spacing = "PREFIX: &str"
                changed_spacing = "PREFIX:&str"
                self.assertEqual(len(changed_literal), len(old_literal) + 1)
                self.assertEqual(len(changed_spacing), len(old_spacing) - 1)
                code = code.replace(old_literal, changed_literal, 1).replace(
                    old_spacing, changed_spacing, 1
                )
                self.assertEqual(len(code), len(original_pair(path, **kwargs)[0]))
            return code, skel

        with patch(
            "promotion_source.nontest_pair",
            side_effect=changed_prefix_literal_with_equal_length,
        ):
            self.assertTrue(typed_input_exception_problems(valid, root))
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
        root = REPO_ROOT
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
                path.read_text(encoding="utf-8").replace(
                    '"joint_bayesian" =>', '"seventh_route" =>'
                ),
                encoding="utf-8",
            )
            self.assertTrue(defensive_carrier_route_problems(fake))

    def test_class_only_constructor_drift_and_unproven_evidence_fail(self):
        root = REPO_ROOT
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
            self.assertTrue(input_exception_evidence_problems(valid, path, "test_false"))

    def test_stage_metadata_is_distinct_from_message_and_same_line_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.rs"
            path.write_text(
                'pub fn run() { let fields = RefusalFields { stage: Some("example.fit".to_owned()), };\n let message = "example.fit"; }',
                encoding="utf-8",
            )
            self.assertEqual(refusal_stage_literals(path, in_crate=False), {(1, "example.fit")})
            other = Path(directory) / "other.rs"
            other.write_text(
                'pub fn run() { let fields = RefusalFields { stage: Some("example.fit".to_owned()), }; let message = "example.fit"; }',
                encoding="utf-8",
            )
            self.assertEqual(refusal_stage_literals(other, in_crate=False), set())


class UntrustedIntervalInputTests(unittest.TestCase):
    def test_original_report_input_passes_and_authority_or_numeric_drift_refuses(self):
        from check_interval_coordinates import untrusted_interval_input_problems

        root = REPO_ROOT
        records = tomllib.loads((root / "parity/promotion.toml").read_text(encoding="utf-8"))[
            "record"
        ]
        record = next(r for r in records if r["id"] == "inference.measured_scalar")
        item = record["untrusted_interval_inputs"][0]
        self.assertEqual(untrusted_interval_input_problems(root, record, item), [])
        for mutation in [
            {"class": "MeasuredScalarReport"},
            {"authority_type": "ScalarExecution"},
            {"unknown": True},
            {"authority_fixture": "unproven"},
        ]:
            self.assertTrue(untrusted_interval_input_problems(root, record, item | mutation))
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
                return (code.replace(before, after), skel) if path == source else (code, skel)

            with (
                self.subTest(drift=after),
                patch("check_interval_coordinates.nontest_pair", side_effect=changed),
            ):
                self.assertTrue(untrusted_interval_input_problems(root, record, item))


if __name__ == "__main__":
    unittest.main()
