"""Python runtime identity unit checks; no scientific tests or evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nix_python_runtime_identity as identity


class PythonRuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = self.root / 'store'
        self.store.mkdir()
        self.source = self.store / ('a' * 32 + '-source')
        self.output = self.store / ('b' * 32 + '-runtime')
        self.wheel = self.store / ('c' * 32 + '-wheel')
        self.drv = str(self.store / ('d' * 32 + '-runtime.drv'))
        self.python = self.store / ('e' * 32 + '-python') / 'bin/python3.13'
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'python')
        self.wheel.mkdir()
        (self.source / 'python/antecedent').mkdir(parents=True)
        (self.source / 'python/antecedent/__init__.py').write_bytes(b'fixture')
        (self.source / 'python/uv.lock').write_text('[[package]]\nname="example"\nversion="1.2.3"\nwheels=[{url="https://example.test/example-1.2.3-py3-none-any.whl",hash="sha256:' + 'a' * 64 + '"}]\n', encoding="utf-8")
        (self.source / 'python/pyproject.toml').write_text('[project]\nname="antecedent"\nversion="3.0.0"\n', encoding="utf-8")
        (self.output / 'site/antecedent').mkdir(parents=True)
        (self.output / 'site/antecedent/__init__.py').write_bytes(b'fixture')
        native = self.output / 'site/antecedent/_native.cpython-313-darwin.so'
        native.write_bytes(b'extension')
        for name,version in (('example','1.2.3'),('antecedent','3.0.0')):
            dist = self.output / f'site/{name}-{version}.dist-info'
            dist.mkdir()
            (dist / 'METADATA').write_text(f'Name: {name}\nVersion: {version}\n', encoding="utf-8")
        (self.output / 'bin').mkdir()
        (self.output / 'bin/python3.13').symlink_to(self.python)
        (self.output / 'source-root').write_text(str(self.source), encoding="utf-8")
        (self.output / 'wheel-output').write_text(str(self.wheel), encoding="utf-8")
        self.row = {'name': 'example', 'version': '1.2.3', 'file': 'example-1.2.3-py3-none-any.whl', 'sha': 'a' * 64}
        self.package_file = self.output / 'locked-packages.json'
        self.package_file.write_text(json.dumps([self.row]), encoding="utf-8")
        recipe = self.root / 'nix/evidence-python-runtime.nix'
        recipe.parent.mkdir()
        recipe.write_text('fixture', encoding="utf-8")
        self.nar = 'sha256-' + 'A' * 43 + '='
        self.derivation = {'outputs': {'out': {'path': str(self.output)}}, 'inputSrcs': [str(self.source)]}
        self.metadata = {'deriver': self.drv, 'references': [str(self.source),str(self.wheel)], 'narHash': self.nar}
        import hashlib
        wheel = {'source_nar_hash': 'source-nar', 'wheel_sha256': 'wheel-sha',
                 'extension_sha256': hashlib.sha256(b'extension').hexdigest(),
                 'extension_filename': 'antecedent/_native.cpython-313-darwin.so',
                 'python': {'executable': str(self.python)}}
        for replacement in (patch.object(identity, '_nix', side_effect=self.nix),
                            patch.object(identity, 'inspect_python_wheel', return_value=wheel)):
            replacement.start()
            self.addCleanup(replacement.stop)

    def nix(self, *args):
        if args[0] == 'nix-instantiate': return self.drv
        if 'derivation' in args: return json.dumps({self.drv: self.derivation})
        if 'path-info' in args: return json.dumps({str(self.output): self.metadata})
        if 'hash' in args: return self.nar
        raise AssertionError(args)

    def inspect(self):
        return identity.inspect_python_runtime(self.root, str(self.output), str(self.source), str(self.wheel), store=self.store)

    def test_locked_runtime_is_identified_without_authorizing_evidence(self):
        result = self.inspect()
        self.assertEqual(result['locked_packages'], [self.row])
        self.assertFalse(result['sandbox_verified'])
        self.assertFalse(result['evidence_authorized'])

    def test_changed_locked_wheel_or_missing_distribution_is_rejected(self):
        self.package_file.write_text(json.dumps([self.row | {'sha': 'b' * 64}]), encoding="utf-8")
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'differs from lock'): self.inspect()
        self.package_file.write_text(json.dumps([self.row]), encoding="utf-8")
        (self.output / 'site/example-1.2.3.dist-info/METADATA').write_text('Name: example\nVersion: 9.0\n', encoding="utf-8")
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'distributions differ'): self.inspect()

    def test_changed_native_or_source_bytes_is_rejected(self):
        (self.output / 'site/antecedent/_native.cpython-313-darwin.so').write_bytes(b'changed')
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'extension differs'): self.inspect()
        (self.output / 'site/antecedent/_native.cpython-313-darwin.so').write_bytes(b'extension')
        (self.output / 'site/antecedent/__init__.py').write_bytes(b'changed')
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'source differs'): self.inspect()


if __name__ == '__main__': unittest.main()
