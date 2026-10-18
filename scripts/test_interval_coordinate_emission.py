"""Exact emission/registration ownership boundaries for interval-coordinate audit."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_interval_coordinates as audit


class EmissionTests(unittest.TestCase):
    def test_labelled_id_keeps_actual_test_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "crates/example/tests/measure.rs"
            path.parent.mkdir(parents=True)
            path.write_text('#[test]\n#[ignore]\nfn measure() { run("measure"); }\nfn run(name: &str) { labelled("population_linear"); }\n', encoding="utf-8")
            with patch.object(audit, "ROOT", root):
                self.assertTrue(audit.emitted_by_a_test("cov.effect.dag.frequentist.analytic_se.l95.measure.population_linear"))
                self.assertFalse(audit.emitted_by_a_test("cov.effect.dag.frequentist.analytic_se.l95.other.population_linear"))
                self.assertFalse(audit.emitted_by_a_test("cov.effect.dag.frequentist.analytic_se.l95.measure.bad.extra"))
                self.assertFalse(audit.emitted_by_a_test("cov.effect.dag.frequentist.analytic_se.l95.measure.unused_label"))
            path.write_text('// #[ignore] fn measure() { "measure"; }\n', encoding="utf-8")
            audit.rust_items.cache_clear()
            with patch.object(audit, "ROOT", root):
                self.assertFalse(audit.emitted_by_a_test("cov.effect.dag.frequentist.analytic_se.l95.measure.population_linear"))

    def test_dynamic_wrapper_requires_positional_target_contract(self):
        wrapper = ('run_suite() {\n  local file="$1"\n  local filter="$2"\n'
                   '  cargo test --test "$file" "$filter" -- --ignored --exact\n}\n'
                   'run_suite real_target measure\n# run_suite fake ignored\n')
        self.assertEqual(audit.registrations(wrapper), [
            {"header": "", "target": "real_target", "name": "measure", "fn": "run_suite"},
        ])
        for changed in (wrapper.replace('file="$1"', 'file="$2"'),
                        wrapper.replace('--test "$file"', '--test "$other"'),
                        wrapper.replace('filter="$2"', 'filter="$1"'),
                        wrapper.replace('  local file=', '  # local file=')):
            self.assertEqual(audit.registrations(changed), [])

    def test_reviewed_mixed_suite_diagnostic_never_emits(self):
        relative = "crates/antecedent/tests/sample_split_effect_calibration.rs"
        original = (audit.ROOT / relative).read_text(encoding="utf-8")
        name = "sample_split_cancelling_edge_boundary_95"
        registration = {"target": "sample_split_effect_calibration", "name": name}
        rows = [row for row in audit.load("parity/calibration_gates.toml")["gate"]
                if row["group"].startswith(f"sample_split_effect_calibration__{name}.p")]
        ledger = "".join('[[gate]]\ngroup="' + row["group"] + '"\npassed=1\ncalibration_sha="'
                         + row["calibration_sha"] + '"\n' for row in rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / relative
            path.parent.mkdir(parents=True)
            (root / "parity").mkdir()
            receipt = root / "parity/calibration_gates.toml"
            path.write_text(original, encoding="utf-8")
            receipt.write_text(ledger, encoding="utf-8")
            with patch.object(audit, "ROOT", root):
                self.assertTrue(audit.reviewed_nonemitting_boundary(registration, path))
                self.assertFalse(audit.reviewed_nonemitting_boundary({**registration, "name": "other"}, path))
                for changed in (original.replace("        return;", "        tally.emit_named_boundary(); return;", 1),
                                original.replace("reported.persist();", "tally.assert(); reported.persist();")):
                    path.write_text(changed, encoding="utf-8")
                    audit.rust_items.cache_clear()
                    self.assertFalse(audit.reviewed_nonemitting_boundary(registration, path))
                path.write_text(original, encoding="utf-8")
                audit.rust_items.cache_clear()
                for changed in (ledger.replace("passed=1", "passed=0", 1),
                                ledger.split("[[gate]]", 2)[0] + "[[gate]]" + ledger.split("[[gate]]", 2)[1],
                                ledger.replace("c30e10304f4bd85a01ad81dd133a360ddc95c2f1", "a" * 40)):
                    receipt.write_text(changed, encoding="utf-8")
                    self.assertFalse(audit.reviewed_nonemitting_boundary(registration, path))

    def test_recordless_requires_exact_passed_ledger_and_untyped_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "measure.rs"
            parity = root / "parity"
            parity.mkdir()
            ledger = parity / "calibration_gates.toml"
            text = ('fn helper() { let t = CoverageTally::new("measure", 0.95); }\n'
                    '#[test]\n#[ignore]\nfn measure() { helper("measure"); }\n')
            path.write_text(text, encoding="utf-8")
            row = ('[[gate]]\ngroup="target__measure"\npassed=1\ncalibration_sha="' + 'a' * 40 + '"\n')
            registration = {"target": "target", "name": "measure"}
            with patch.object(audit, "ROOT", root):
                for changed in (row.replace('target__measure', 'other__measure'),
                                row.replace('passed=1', 'passed=0'),
                                row.replace('a' * 40, 'unknown')):
                    ledger.write_text(changed, encoding="utf-8")
                    self.assertFalse(audit.measured_recordless_registration(registration, path))
                ledger.write_text(row, encoding="utf-8")
                self.assertTrue(audit.measured_recordless_registration(registration, path))
                for changed in (text + '\nfn typed() { RecordKey::new(); }',
                                text + '\nfn typed() { CoverageTally::for_record(key, 0.95); }',
                                text.replace('#[ignore]', ''),
                                text.replace('helper("measure")', 'helper("other")')):
                    path.write_text(changed, encoding="utf-8")
                    audit.rust_items.cache_clear()
                    self.assertFalse(audit.measured_recordless_registration(registration, path))


if __name__ == "__main__":
    unittest.main()
