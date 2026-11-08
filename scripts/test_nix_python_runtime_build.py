"""Build-only Python runtime checks; no evidence or project tests run."""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import nix_python_runtime_build as build


class PythonRuntimeBuildTests(unittest.TestCase):
    def test_nix_argv_is_offline_local_rebuild_without_test_execution(self):
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, '/nix/store/output\n', '')
        with patch.object(build.subprocess, 'run', side_effect=run):
            self.assertEqual(build._build(Path('/project'), '/nix/store/source', '/nix/store/wheel'), '/nix/store/output')
        self.assertEqual(len(calls), 2)
        self.assertIn('--offline', calls[0])
        self.assertEqual(calls[0][calls[0].index('substitute') + 1], 'false')
        self.assertEqual(calls[0][calls[0].index('builders') + 1], '')
        self.assertIn('--rebuild', calls[1])
        self.assertTrue(all('test' not in argv for argv in calls))

    def test_changed_runtime_cannot_return_as_verified_build(self):
        source = {'commit': 'abc', 'store_path': '/nix/store/source', 'nar_hash': 'nar'}
        wheel = {'output': '/nix/store/wheel'}
        runtime = {'source_nar_hash': 'nar', 'sandbox_verified': False, 'evidence_authorized': False}
        with (
            patch.object(build, 'build_committed_python_wheel', return_value={'source': source, 'artifact': wheel}),
            patch.object(build, '_clean_head', return_value=('abc', 'sha1')),
            patch.object(build, '_build', return_value='/nix/store/runtime'),
            patch.object(build, 'inspect_python_runtime', side_effect=[runtime, runtime | {'output_nar_hash': 'changed'}]),
            self.assertRaisesRegex(ValueError, 'changed while inspecting'),
        ):
            build.build_committed_python_runtime(Path('/project'))


if __name__ == '__main__':
    unittest.main()
