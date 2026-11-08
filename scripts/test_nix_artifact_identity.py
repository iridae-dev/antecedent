"""Engineering checks for Nix artifact identity; no build or test is run."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_artifact_identity as identity


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = self.root / 'store'
        self.store.mkdir()
        self.source = self.store / ('a' * 32 + '-source')
        self.output = self.store / ('b' * 32 + '-output')
        self.drv = str(self.store / ('c' * 32 + '-target.drv'))
        self.source.mkdir()
        self.binary = self.output / 'targets/fixture/test/basic/bin/test'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'fixture binary')
        (self.output / 'source-root').write_text(str(self.source) + '\n', encoding="utf-8")
        for name in ('nix/evidence-target.nix', 'nix/evidence-toolchain.nix', 'flake.lock'):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(name, encoding="utf-8")
        self.nar = 'sha256-' + 'A' * 43 + '='
        self.source_nar = 'sha256-' + 'B' * 43 + '='
        self.derivation = {'outputs': {'out': {'path': str(self.output)}}, 'inputSrcs': [str(self.source)]}
        self.metadata = {'deriver': self.drv, 'references': [str(self.source)], 'narHash': self.nar}
        self.calls = []
        mock = patch.object(identity, '_nix', side_effect=self.nix)
        mock.start()
        self.addCleanup(mock.stop)

    def nix(self, *args):
        self.calls.append(args)
        if args[0] == 'nix-instantiate':
            return self.drv
        if 'derivation' in args:
            return json.dumps({self.drv: self.derivation})
        if 'path-info' in args:
            return json.dumps({str(self.output): self.metadata})
        if 'hash' in args:
            return self.nar if args[-1] == str(self.output) else self.source_nar
        raise AssertionError(args)

    def inspect(self, **overrides):
        args = dict(root=self.root, output=str(self.output), source=str(self.source), crate='fixture',
                    target_kind='test', target_name='basic', features=['z', 'a', 'a'], store=self.store)
        args.update(overrides)
        return identity.inspect_artifact(**args)

    def test_recipe_is_instantiated_with_exact_normalized_arguments(self):
        result = self.inspect()
        command = self.calls[0]
        self.assertEqual(json.loads(command[command.index('targetsJson') + 1]),
                         [{'crate': 'fixture', 'features': ['a', 'z'], 'kind': 'test', 'name': 'basic'}])
        self.assertIn(str(self.source), command[command.index('src') + 1])
        self.assertFalse(result['sandbox_verified'])
        self.assertEqual(result['features'], ['a', 'z'])
        self.assertEqual(result['binary'], 'targets/fixture/test/basic/bin/test')

    def test_a_shared_build_names_each_target_and_refuses_one_it_did_not_compile(self):
        both = [{'crate': 'other', 'kind': 'lib', 'name': '', 'features': []},
                {'crate': 'fixture', 'kind': 'test', 'name': 'basic', 'features': ['a', 'z']}]
        result = self.inspect(build_targets=both)
        command = self.calls[0]
        self.assertEqual([t['crate'] for t in json.loads(command[command.index('targetsJson') + 1])],
                         ['fixture', 'other'])
        self.assertEqual(identity.target_binary(result), self.binary)
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'did not compile|not one'):
            self.inspect(build_targets=both[:1])

    def test_wrong_source_or_deriver_or_output_hash_is_refused(self):
        self.derivation['inputSrcs'] = []
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect()
        self.derivation['inputSrcs'] = [str(self.source)]
        self.metadata['deriver'] = str(self.store / ('d' * 32 + '-other.drv'))
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect()
        self.metadata['deriver'] = self.drv
        self.metadata['narHash'] = self.source_nar
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect()

    def test_binary_recipe_and_source_root_mutations_are_detected(self):
        recorded = self.inspect()
        self.binary.write_bytes(b'changed binary')
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'identity changed'):
            identity.recheck_artifact(self.root, recorded, store=self.store)
        self.binary.write_bytes(b'fixture binary')
        (self.root / 'flake.lock').write_text('changed', encoding="utf-8")
        with self.assertRaisesRegex(identity.ArtifactIdentityError, 'identity changed'):
            identity.recheck_artifact(self.root, recorded, store=self.store)
        (self.output / 'source-root').write_text('wrong', encoding="utf-8")
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect()

    def test_bad_target_and_linked_binary_are_refused(self):
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect(crate='--bad')
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect(features=['--bad'])
        binary = self.binary
        binary.unlink()
        binary.symlink_to(self.root / 'flake.lock')
        with self.assertRaises(identity.ArtifactIdentityError):
            self.inspect()


if __name__ == '__main__':
    unittest.main()
