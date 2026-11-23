"""Draft Nix receipt checks; no project assertions or evidence are run."""
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_manifest_binding as binding


class DraftReceiptTests(unittest.TestCase):
    def setUp(self):
        self.root, self.snapshot = Path('/tmp/repo'), Path('/tmp/snapshot')
        self.build = {
            'source': {'commit': 'a' * 40, 'store_path': '/nix/store/source', 'nar_hash': 'source-nar'},
            'artifact': {'schema': 1, 'source': '/nix/store/source', 'source_nar_hash': 'source-nar',
                         'derivation': '/nix/store/test.drv', 'output': '/nix/store/output',
                         'binary': 'targets/toy/test/fixture/bin/test',
                         'output_nar_hash': 'output-nar', 'binary_sha256': 'b' * 64,
                         'crate': 'toy', 'target_kind': 'test', 'target_name': 'fixture',
                         'features': ['feature'], 'recipe_sha256': {'flake.lock': 'c' * 64},
                         'sandbox_verified': False},
            'tools': {'rustc': '1.85.1', 'rustc_commit': 'd' * 40, 'cargo': '1.97.0'},
            'compiler_graph': {'units': [{'name': 'toy'}], 'roots': [0]},
            'sandbox_verified': False, 'evidence_authorized': False,
        }
        self.assumptions = {key: 'not verified by this fixture' for key in binding.ASSUMPTIONS}

    def test_draft_binds_all_components_but_never_authorizes_evidence(self):
        draft = binding.draft_receipt(self.build, self.assumptions)
        self.assertEqual(draft['status'], 'unverified')
        self.assertIs(draft['publication_authorized'], False)
        self.assertEqual(draft['source']['commit'], 'a' * 40)
        self.assertEqual(draft['artifact']['target_name'], 'fixture')
        self.assertEqual(draft['assumptions'], self.assumptions)
        self.build['artifact']['binary_sha256'] = 'changed'
        self.assertNotEqual(draft['artifact']['binary_sha256'], 'changed')

    def test_fabricated_success_or_unmatched_source_is_refused(self):
        self.build['sandbox_verified'] = True
        with self.assertRaisesRegex(binding.InvalidNixReceipt, 'unverified'):
            binding.draft_receipt(self.build, self.assumptions)
        self.build['sandbox_verified'] = False
        self.build['artifact']['source_nar_hash'] = 'different'
        with self.assertRaisesRegex(binding.InvalidNixReceipt, 'identity'):
            binding.draft_receipt(self.build, self.assumptions)

    def test_every_trust_assumption_must_be_recorded(self):
        del self.assumptions['private_tmp']
        with self.assertRaisesRegex(binding.InvalidNixReceipt, 'assumptions'):
            binding.draft_receipt(self.build, self.assumptions)

    def test_source_and_artifact_are_rechecked_on_both_sides_of_fixture(self):
        calls = []
        with patch.object(binding, 'verify_committed_snapshot', side_effect=lambda *a: calls.append('source')), \
             patch.object(binding, 'recheck_artifact', side_effect=lambda *a: calls.append('artifact')):
            draft = binding.exercise_fixture(self.root, self.root, self.snapshot, self.build, self.assumptions,
                                             lambda binary: calls.append(str(binary)))
        self.assertEqual(calls, ['source', 'artifact', '/nix/store/output/targets/toy/test/fixture/bin/test', 'source', 'artifact'])
        self.assertIs(draft['publication_authorized'], False)

    def test_changed_artifact_after_fixture_is_refused(self):
        with (
            patch.object(binding, 'verify_committed_snapshot'),
            patch.object(binding, 'recheck_artifact', side_effect=[None, ValueError('changed artifact')]),
            self.assertRaisesRegex(ValueError, 'changed artifact'),
        ):
            binding.exercise_fixture(self.root, self.root, self.snapshot, self.build, self.assumptions, lambda binary: None)

    def test_fixture_failure_still_rechecks_source_and_artifact(self):
        def fail(binary):
            raise RuntimeError('fixture failure')
        with (
            patch.object(binding, 'verify_committed_snapshot') as source,
            patch.object(binding, 'recheck_artifact') as artifact,
            self.assertRaisesRegex(RuntimeError, 'fixture failure'),
        ):
            binding.exercise_fixture(self.root, self.root, self.snapshot, self.build, self.assumptions, fail)
        self.assertEqual(source.call_count, 2)
        self.assertEqual(artifact.call_count, 2)


if __name__ == '__main__':
    unittest.main()
