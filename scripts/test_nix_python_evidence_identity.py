"""Unit checks for Python wheel identity; no evidence or project tests run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nix_python_evidence_identity as identity


class PythonWheelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = self.root / 'store'
        self.store.mkdir()
        self.source = self.store / ('a' * 32 + '-source')
        self.output = self.store / ('b' * 32 + '-output')
        self.python = self.store / ('d' * 32 + '-python') / 'bin/python3.13'
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'python')
        package = self.source / 'python/antecedent'
        package.mkdir(parents=True)
        (package / '__init__.py').write_bytes(b'from . import _native\n')
        (package / 'py.typed').write_bytes(b'')
        (self.output / 'wheels').mkdir(parents=True)
        (self.output / 'bin').mkdir()
        (self.output / 'bin/python3.13').symlink_to(self.python)
        self.wheel = self.output / 'wheels/antecedent-3.0.0-py3-none-any.whl'
        self.write_wheel()
        (self.output / 'source-root').write_text(str(self.source))
        (self.output / 'python-config.json').write_text(json.dumps({'executable': str(self.python), 'implementation': 'cpython', 'version': '3.13.14'}))
        (self.output / 'rustc-version').write_text('release: 1.85.1\ncommit-hash: abc\n')
        (self.output / 'cargo-version').write_text('cargo 1.97.0 (fixture)\n')
        (self.output / 'maturin-version').write_text('maturin 1.14.1\n')
        for name in ('cargo-metadata.json', 'cargo-unit-graph.json'):
            (self.output / name).write_text('{}')
        for name in ('nix/evidence-python-wheel.nix', 'nix/evidence-toolchain.nix', 'flake.lock'):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(name)
        pin = self.root / 'tools/evidence-analyzer/pin.toml'
        pin.parent.mkdir(parents=True)
        pin.write_text('release="1.85.1"\ncommit="abc"\n')
        self.drv = str(self.store / ('c' * 32 + '-target.drv'))
        self.nar = 'sha256-' + 'A' * 43 + '='
        self.source_nar = 'sha256-' + 'B' * 43 + '='
        self.derivation = {'outputs': {'out': {'path': str(self.output)}}, 'inputSrcs': [str(self.source)]}
        self.metadata = {'deriver': self.drv, 'references': [str(self.source)], 'narHash': self.nar}
        graph = {'roots': [0], 'units': [{'name': 'antecedent-py', 'mode': 'build', 'target_kind': ['cdylib']}]}
        for replacement in (patch.object(identity, '_nix', side_effect=self.nix),
                            patch.object(identity, 'compiler_units', return_value=graph)):
            replacement.start()
            self.addCleanup(replacement.stop)

    def write_wheel(self, source=b'from . import _native\n', extra=None):
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr('antecedent/__init__.py', source)
            archive.writestr('antecedent/py.typed', b'')
            archive.writestr('antecedent/_native.cpython-313-darwin.so', b'fixture extension')
            archive.writestr('antecedent-3.0.0.dist-info/METADATA', b'fixture')
            if extra:
                archive.writestr(extra, b'unexpected')

    def nix(self, *args):
        if args[0] == 'nix-instantiate': return self.drv
        if 'derivation' in args: return json.dumps({self.drv: self.derivation})
        if 'path-info' in args: return json.dumps({str(self.output): self.metadata})
        if 'hash' in args: return self.nar if args[-1] == str(self.output) else self.source_nar
        raise AssertionError(args)

    def inspect(self):
        return identity.inspect_python_wheel(self.root, str(self.output), str(self.source), store=self.store)

    def test_full_wheel_and_interpreter_are_identified_without_attesting_evidence(self):
        result = self.inspect()
        self.assertEqual(result['extension_filename'], 'antecedent/_native.cpython-313-darwin.so')
        self.assertFalse(result['runtime_requirements_verified'])
        self.assertFalse(result['sandbox_verified'])
        self.assertFalse(result['evidence_authorized'])

    def test_changed_package_source_or_extra_package_file_is_rejected(self):
        self.write_wheel(source=b'changed')
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'source differs'):
            self.inspect()
        self.write_wheel(extra='antecedent/injected.py')
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'membership'):
            self.inspect()

    def test_wrong_deriver_or_interpreter_is_rejected(self):
        self.metadata['deriver'] = 'forged'
        with self.assertRaises(identity.ArtifactIdentityError): self.inspect()
        self.metadata['deriver'] = self.drv
        (self.output / 'python-config.json').write_text(json.dumps({'executable': '/usr/bin/python3', 'implementation': 'cpython', 'version': '3.13.14'}))
        with self.assertRaises(identity.ArtifactIdentityError): self.inspect()


if __name__ == '__main__': unittest.main()
