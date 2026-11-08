"""Build-only Python evidence checks; no project tests or evidence run."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_python_evidence_build as build


class PythonWheelBuildTests(unittest.TestCase):
    def test_nix_argv_is_offline_local_rebuild_and_never_executes_test(self):
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, '/nix/store/output\n', '')
        with patch.object(build.subprocess, 'run', side_effect=run):
            self.assertEqual(build._build(Path('/project'), '/nix/store/source'), '/nix/store/output')
        self.assertEqual(len(calls), 2)
        self.assertIn('--offline', calls[0])
        self.assertEqual(calls[0][calls[0].index('substitute') + 1], 'false')
        self.assertEqual(calls[0][calls[0].index('builders') + 1], '')
        self.assertIn('--rebuild', calls[1])
        self.assertTrue(all('test' not in argv for argv in calls))

    def test_changed_wheel_cannot_return_as_verified_build(self):
        root = Path('/project')
        source = {'store_path': '/nix/store/source', 'nar_hash': 'nar'}
        artifact = {'source_nar_hash': 'nar', 'sandbox_verified': False,
                    'runtime_requirements_verified': False, 'evidence_authorized': False}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(build.tempfile, 'TemporaryDirectory', return_value=tempfile.TemporaryDirectory(dir=tmp)),
            patch.object(build, 'materialize_committed_source', return_value={'commit': 'abc'}),
            patch.object(build, 'import_committed_snapshot', return_value=source),
            patch.object(build, 'verify_committed_snapshot'),
            patch.object(build, '_build', return_value='/nix/store/output'),
            patch.object(build, 'inspect_python_wheel', side_effect=[artifact, artifact | {'wheel_sha256': 'changed'}]),
            self.assertRaisesRegex(ValueError, 'changed while inspecting'),
        ):
            build.build_committed_python_wheel(root)


if __name__ == '__main__':
    unittest.main()
