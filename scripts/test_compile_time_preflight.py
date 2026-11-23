"""Engineering checks that earning refuses unassessed ambient inputs early."""
import argparse
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import evidence_earning
import scientific_records


class PreflightTests(unittest.TestCase):
    def test_actual_override_is_refused_without_leaking_its_value(self):
        with (
            patch.dict(os.environ, {'HOME': '/tmp', 'CC': 'private compiler'}, clear=True),
            self.assertRaises(RuntimeError) as error,
        ):
            evidence_earning.require_compile_environment(Path('/tmp/repo'))
        self.assertIn('environment override CC', str(error.exception))
        self.assertNotIn('private compiler', str(error.exception))

    def test_noncanonical_build_flags_are_refused_before_metadata(self):
        with (
            patch.dict(os.environ, {'HOME': '/tmp', 'RUSTFLAGS': '--cfg secret'}, clear=True),
            self.assertRaises(RuntimeError) as error,
        ):
            evidence_earning.require_compile_environment(Path('/tmp/repo'))
        self.assertIn('environment override RUSTFLAGS', str(error.exception))
        self.assertNotIn('secret', str(error.exception))

    def test_snapshot_refuses_before_fingerprint_or_metadata(self):
        with (
            patch.object(evidence_earning, 'noncanonical_compile_environment',
                          return_value=['environment override CC']),
            patch.object(evidence_earning, 'Fingerprinter') as fingerprint,
            self.assertRaisesRegex(RuntimeError, 'unset overrides.*environment override CC'),
        ):
            evidence_earning.Snapshot(Path('/tmp/repo'), {}, {'kind': 'scientific'})
        fingerprint.assert_not_called()

    def test_snapshot_refuses_changed_environment_before_publication(self):
        snapshot = object.__new__(evidence_earning.Snapshot)
        snapshot.root = Path('/tmp/repo')
        with (
            patch.object(evidence_earning, 'noncanonical_compile_environment',
                          return_value=['environment override SDKROOT']),
            self.assertRaisesRegex(RuntimeError, 'environment override SDKROOT'),
        ):
            snapshot.unchanged()

    def test_scientific_entrypoint_refuses_before_row_discovery(self):
        with (
            patch.object(evidence_earning, 'require_compile_environment',
                          side_effect=RuntimeError('ambient override')),
            patch.object(scientific_records, 'all_scientific_rows') as rows,
            self.assertRaisesRegex(RuntimeError, 'ambient override'),
        ):
            scientific_records.cmd_run(Path('/tmp/repo'), argparse.Namespace())
        rows.assert_not_called()

    def test_direct_rust_earning_refuses_before_target_discovery(self):
        with (
            patch.object(evidence_earning, 'require_compile_environment',
                          side_effect=RuntimeError('ambient override')),
            patch.object(scientific_records, 'locate') as locate,
            self.assertRaisesRegex(RuntimeError, 'ambient override'),
        ):
            scientific_records.earn_rust(Path('/tmp/repo'), [{}])
        locate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
