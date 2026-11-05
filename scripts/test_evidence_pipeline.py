"""Opt-in compiler integration: disposable engineering fixtures, no product gates."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import scientific_records as sr


class PipelineTest(unittest.TestCase):
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
            (root/'Cargo.toml').write_text('[workspace]\nmembers=["crates/toy"]\nresolver="2"\n')
            (root/'crates/toy/Cargo.toml').write_text('[package]\nname="toy"\nversion="0.1.0"\nedition="2021"\n')
            source = 'pub fn used()->i32 { 2 }\npub fn unused()->i32 { 7 }\npub const fn konst()->i32 { 3 }\n'
            (root/'crates/toy/src/lib.rs').write_text(source)
            (root/'crates/toy/tests/check.rs').write_text('#[test] fn check() { assert_eq!(toy::used(), 2); }\n')
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
            (stale/'old.tsv').write_text('\t'.join(['body','toy::konst',str(lib).encode().hex(),str(lo),str(lo+5),b'{ 3 }'.hex()])+'\n')
            with patch.object(sr,'all_scientific_rows',return_value=[row]), redirect_stdout(io.StringIO()):
                self.assertEqual(sr.earn_rust(root,[row]),0)
            records = sr.load_records(root)
            record = records['toy::check::check']
            self.assertIn('dependency_manifest',record)
            with patch.object(sr,'Fingerprinter',side_effect=AssertionError('compiler used by current checker')):
                owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(0,1), owed)
            (root/'crates/toy/src/lib.rs').write_text(source.replace('{ 7 }','{ 8 }'))
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(0,1), [owed, json.loads((root/'parity/evidence_manifests'/f"{record['dependency_manifest']}.json").read_text())])
            (root/'crates/toy/src/lib.rs').write_text(source.replace('{ 3 }','{ 4 }'))
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(1,0))
            (root/'crates/toy/src/lib.rs').write_text(source.replace('unused()','unused(x:i32)'))
            owed,current = sr.classify(root,[row])
            self.assertEqual((len(owed),len(current)),(1,0))


if __name__ == '__main__': unittest.main()
