"""Explicit missing test paths and a vacuous citation scan fail closed."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("gate_coverage_citations.py").resolve()


class CoverageModuleResolution(unittest.TestCase):
    def run_gate(self, limitations, *, source=None, lane=None, target=True, module="existing", suite=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            (root / "crates/antecedent/tests").mkdir(parents=True)
            (root / f"crates/antecedent/tests/{module}.rs").write_text(
                source if source is not None else "#[test]\nfn present() {}\n", encoding="utf-8")
            if suite is not None:
                (root / "crates/antecedent/tests/suite.rs").write_text(suite, encoding="utf-8")
            (root / "crates/antecedent/Cargo.toml").write_text(
                '[package]\nname="antecedent"\nversion="0.0.0"\nautotests=false\n'
                + (f'[[test]]\nname="{module}"\npath="tests/{module}.rs"\n' if target else '')
                + ('[[test]]\nname="suite"\npath="tests/suite.rs"\n' if suite is not None else ''), encoding="utf-8")
            if lane is not None:
                (root / "scripts").mkdir()
                (root / "scripts/gate_calibration.sh").write_text(lane, encoding="utf-8")
            (root / "parity/support_licensed.toml").write_text(
                '[[cell]]\nquery="q"\ngraph_class="g"\nstructure="s"\ninference="i"\nvalidation="v"\nlimitations=' + repr(limitations) + '\n', encoding="utf-8")
            return subprocess.run([sys.executable, str(GATE)], cwd=root, capture_output=True, text=True, check=False)

    def test_deleted_explicit_module_and_vacuous_scan(self):
        missing = self.run_gate("coverage test crates/antecedent/tests/deleted.rs::case")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("deleted.rs does not exist", missing.stdout)
        none = self.run_gate("documentation mentions std::vec")
        self.assertNotEqual(none.returncode, 0)
        self.assertIn("no coverage test citations were checked", none.stdout)
        present = self.run_gate("coverage test existing::present")
        self.assertEqual(present.returncode, 0)

    def test_helpers_commented_tests_ignored_tests_and_match_arms_fail(self):
        for source in (
            'fn cited() {}\n',
            '// #[test] fn cited() {}\n',
            '#[test] #[ignore] fn cited() {}\n',
            'fn dispatch() { match 0 { cited => 1, _ => 0 }; }\n',
            '#[test] #[cfg(any())] fn cited() {}\n',
        ):
            with self.subTest(source=source):
                result = self.run_gate("existing::cited", source=source)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.run_gate("existing::cited", source='#[test] fn cited() {}\n').returncode, 0)
        self.assertNotEqual(self.run_gate("existing::cited", source='#[test] fn cited() {}\n', target=False).returncode, 0)

    def test_calibration_ignore_requires_exact_target_and_registered_test(self):
        source = '#[test] #[ignore = "calibration: run via scripts/gate_calibration.sh"] fn cited() {}\n'
        template = 'run_cal() {\n  local filter="$1"\n  cargo test -p antecedent --test existing "$filter" -- --ignored --nocapture\n}\n'
        valid = template + 'run_cal cited\n'
        self.assertEqual(self.run_gate("existing::cited", source=source, lane=valid).returncode, 0)
        for lane in (template, template + '# run_cal cited\n',
                     valid.replace('--test existing', '--test wrong'),
                     valid.replace('-- --ignored', '-- --nocapture'),
                     valid.replace('-- --ignored', '-- --ignored --skip cited'),
                     valid.replace('-- --ignored', '-- --ignored unrelated_filter'),
                     valid.replace('  cargo test', '  echo cargo test'),
                     valid.replace('  cargo test', '  if false; then cargo test'),
                     valid.replace('run_cal cited', '  run_cal cited')):
            with self.subTest(lane=lane):
                self.assertNotEqual(self.run_gate("existing::cited", source=source, lane=lane).returncode, 0)
        arbitrary = source.replace('calibration: run via scripts/gate_calibration.sh', 'slow: ignored')
        self.assertNotEqual(self.run_gate("existing::cited", source=arbitrary, lane=valid).returncode, 0)

    def test_registered_suite_module_and_continued_calibration_commands(self):
        source = '#[test] fn cited() {}\n'
        self.assertEqual(self.run_gate("existing::cited", source=source, target=False,
                                       suite="mod existing;\n").returncode, 0)
        self.assertNotEqual(self.run_gate("existing::cited", source=source, target=False,
                                          suite="// mod existing;\n").returncode, 0)
        ignored = '#[test] #[ignore = "calibration: run via scripts/gate_calibration.sh"] fn cited() {}\n'
        lane = ('run_suite() {\n  local file="$1"\n  local filter="$2"\n'
                '  check "label" \\\n    cargo test -p antecedent --test "$file" "$filter" \\\n    -- --ignored --exact\n}\nrun_suite existing cited\n')
        self.assertEqual(self.run_gate("existing::cited", source=ignored, lane=lane).returncode, 0)
        self.assertNotEqual(self.run_gate("existing::cited", source=ignored, lane=lane.replace("run_suite existing cited", "run_suite wrong cited")).returncode, 0)
        whole = 'check "all" \\\n  cargo test -p antecedent --test existing -- --ignored\n'
        self.assertEqual(self.run_gate("existing::cited", source=ignored, lane=whole).returncode, 0)
        for suffix in (" unrelated_filter", " --skip cited", " --unknown"):
            with self.subTest(suffix=suffix):
                self.assertNotEqual(self.run_gate("existing::cited", source=ignored,
                                                  lane=whole.rstrip() + suffix + "\n").returncode, 0)

    def test_reviewed_macro_requires_real_template_invocation_and_ignored_lane(self):
        from test_evidence import rust_items
        actual = GATE.parent.parent / "crates/antecedent/tests/bayesian_temporal.rs"
        masked, items = rust_items(actual.resolve())
        item = next(it for it in items if it.kind == "macro" and it.name == "dag_coverage")
        template = masked.code[item.start:item.body[1]]
        source = template + '\ndag_coverage! { cited => (Cell::Pulse, Regime::IID); }\n'
        lane = ('run_cal() {\n  local filter="$1"\n  cargo test -p antecedent --test bayesian_temporal "$filter" -- --ignored --nocapture\n}\nrun_cal cited\n')
        self.assertEqual(self.run_gate("bayesian_temporal::cited", source=source, lane=lane, module="bayesian_temporal").returncode, 0)
        mutated = source.replace("#[test]", "#[ignore]", 1)
        self.assertNotEqual(self.run_gate("bayesian_temporal::cited", source=mutated, lane=lane, module="bayesian_temporal").returncode, 0)
        fake = 'fn dispatch() { match 0 { cited => 1, _ => 0 }; }\n'
        self.assertNotEqual(self.run_gate("bayesian_temporal::cited", source=fake, lane=lane, module="bayesian_temporal").returncode, 0)


if __name__ == "__main__":
    unittest.main()
