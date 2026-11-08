"""Opt-in compiler integration: disposable engineering fixtures, no product gates."""
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scientific_records as sr


class PipelineTest(unittest.TestCase):
    def setUp(self):
        # These disposable builds test the earning pipeline independently of
        # ambient host settings; the preflight has its own refusal tests.
        ambient = patch('evidence_earning.noncanonical_compile_environment', return_value=[])
        ambient.start()
        self.addCleanup(ambient.stop)

    def test_compile_time_data_requires_a_boundary_and_invalidates_the_pass(self):
        with tempfile.TemporaryDirectory(prefix='antecedent-compile-input-') as tmp:
            root = Path(tmp)
            original = Path(__file__).resolve().parents[1]
            for crate in ('reader', 'toy'):
                (root / f'crates/{crate}/src').mkdir(parents=True)
            (root / 'crates/toy/tests').mkdir()
            (root / 'tools/evidence-analyzer').mkdir(parents=True)
            shutil.copyfile(original / 'rust-toolchain.toml', root / 'rust-toolchain.toml')
            shutil.copyfile(original / 'tools/evidence-analyzer/pin.toml', root / 'tools/evidence-analyzer/pin.toml')
            (root / 'Cargo.toml').write_text('[workspace]\nmembers=["crates/reader","crates/toy"]\nresolver="2"\n', encoding="utf-8")
            (root / 'crates/reader/Cargo.toml').write_text('[package]\nname="reader"\nversion="0.1.0"\nedition="2021"\n[lib]\nproc-macro=true\n', encoding="utf-8")
            (root / 'crates/reader/src/lib.rs').write_text('''extern crate proc_macro;
#[proc_macro]
pub fn measured(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let input = std::fs::read_to_string(format!("{dir}/input.txt")).unwrap();
    let value: i32 = input.trim().parse().unwrap();
    format!("pub const VALUE:i32={value};").parse().unwrap()
}
''', encoding="utf-8")
            (root / 'crates/toy/Cargo.toml').write_text('[package]\nname="toy"\nversion="0.1.0"\nedition="2021"\n[dependencies]\nreader={path="../reader"}\n', encoding="utf-8")
            (root / 'crates/toy/src/lib.rs').write_text('reader::measured!();\npub fn used()->i32 { VALUE }\n', encoding="utf-8")
            data = root / 'crates/toy/input.txt'
            data.write_text('7', encoding="utf-8")
            (root / 'crates/toy/tests/check.rs').write_text('#[test] fn check() { assert_eq!(toy::used(),7); }\n', encoding="utf-8")
            subprocess.run(['cargo','generate-lockfile','--offline'], cwd=root, check=True, capture_output=True)
            row = dict(evidence_test='crates/toy/tests/check.rs', evidence_assertion='check', fixtures=[], roles=['scientific'])
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'undeclared compile-time'):
                sr.earn_rust(root, [row])
            self.assertFalse(sr.load_records(root))
            (root / 'parity').mkdir(exist_ok=True)
            (root / 'parity/fingerprint_inputs.toml').write_text('''[[input]]
source="crates/toy/Cargo.toml"
compile_time=["reader"]
inputs=["crates/toy/input.txt"]
env=[]
why="reader macro reads its consumer's input.txt"
''', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(sr.earn_rust(root, [row]), 0)
            self.assertEqual(len(sr.classify(root, [row])[1]), 1)
            data.write_text('8', encoding="utf-8")
            self.assertEqual(len(sr.classify(root, [row])[0]), 1)
            rebuilt = subprocess.run(['cargo','test','--offline','-p','toy','--test','check'],
                                     cwd=root, capture_output=True)
            self.assertNotEqual(rebuilt.returncode, 0)

    def test_capture_earn_and_check_with_pinned_compiler(self):
        with tempfile.TemporaryDirectory(prefix='antecedent-manifest-engineering-') as tmp:
            root = Path(tmp)
            (root/'crates/toy/src').mkdir(parents=True)
            (root/'crates/toy/tests').mkdir()
            (root/'tools/evidence-analyzer').mkdir(parents=True)
            original = Path(__file__).resolve().parents[1]
            shutil.copyfile(original/'rust-toolchain.toml', root/'rust-toolchain.toml')
            for name in ('driver.rs','pin.toml'):
                shutil.copyfile(original/'tools/evidence-analyzer'/name,root/'tools/evidence-analyzer'/name)
            (root/'Cargo.toml').write_text('[workspace]\nmembers=["crates/toy"]\nresolver="2"\n', encoding="utf-8")
            (root/'crates/toy/Cargo.toml').write_text('[package]\nname="toy"\nversion="0.1.0"\nedition="2021"\n', encoding="utf-8")
            source = 'pub fn used()->i32 { 2 }\npub fn unused()->i32 { 7 }\npub const fn konst()->i32 { 3 }\n'
            (root/'crates/toy/src/lib.rs').write_text(source, encoding="utf-8")
            (root/'crates/toy/tests/check.rs').write_text('#[test] fn check() { assert_eq!(toy::used(), 2); }\n', encoding="utf-8")
            subprocess.run(['cargo','generate-lockfile','--offline'],cwd=root,check=True,capture_output=True)
            row = dict(evidence_test='crates/toy/tests/check.rs',evidence_assertion='check',fixtures=[],roles=['scientific'])
            # Only the selection provider is replaced: compilation, instrumentation,
            # exact assertion execution, publication and production checking are real.
            # An inventory from an earlier compilation that called `konst` an ordinary body
            # must not make it a hole: a constant's body is compile-time input.
            stale = root/'target/evidence-compiler-inventory'
            stale.mkdir(parents=True)
            lib = (root/'crates/toy/src/lib.rs').resolve()
            lo = source.encode().index(b'{ 3 }')
            (stale/'old.tsv').write_text('\t'.join(['body','toy::konst',str(lib).encode().hex(),str(lo),str(lo+5),b'{ 3 }'.hex()])+'\n', encoding="utf-8")
            with patch.object(sr,'all_scientific_rows',return_value=[row]), redirect_stdout(io.StringIO()):
                self.assertEqual(sr.earn_rust(root,[row]),0)
            records = sr.load_records(root)
            record = records['toy::check::check']
            self.assertIn('dependency_manifest',record)
            with patch.object(sr,'Fingerprinter',side_effect=AssertionError('compiler used by current checker')):
                owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(0,1), owed)
            (root/'crates/toy/src/lib.rs').write_text(source.replace('{ 7 }','{ 8 }'), encoding="utf-8")
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(0,1), [owed, json.loads((root/'parity/evidence_manifests'/f"{record['dependency_manifest']}.json").read_text(encoding="utf-8"))])
            (root/'crates/toy/src/lib.rs').write_text(source.replace('{ 3 }','{ 4 }'), encoding="utf-8")
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(1,0))
            (root/'crates/toy/src/lib.rs').write_text(source.replace('unused()','unused(x:i32)'), encoding="utf-8")
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(1,0))

            # Cargo auto-discovers a new build.rs without a Cargo.toml edit.
            (root/'crates/toy/src/lib.rs').write_text('pub fn used()->i32 { option_env!("REVIEW_VALUE").unwrap_or("2").parse().unwrap() }\n', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(sr.earn_rust(root, [row]), 0)
            self.assertEqual(len(sr.classify(root, [row])[1]), 1)
            (root/'crates/toy/build.rs').write_text('fn main() { println!("cargo:rustc-env=REVIEW_VALUE=3"); }', encoding="utf-8")
            self.assertEqual(len(sr.classify(root, [row])[0]), 1)


if __name__ == '__main__':
    unittest.main()
