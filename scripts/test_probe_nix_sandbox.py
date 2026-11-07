"""Check the disposable Nix sandbox probe invocation without running Nix."""
import unittest
from pathlib import Path

from probe_nix_sandbox import probe_command


class ProbeCommandTests(unittest.TestCase):
    def test_invocation_passes_both_sentinels_and_live_port_as_arguments(self):
        args = probe_command(Path('/repo'), Path('/private/tmp/probe'), Path('/repo/probe'), '1234')
        self.assertEqual(args[args.index('hostProbe') + 1], '/private/tmp/probe')
        self.assertEqual(args[args.index('workspaceProbe') + 1], '/repo/probe')
        self.assertEqual(args[args.index('loopbackPort') + 1], '1234')
        self.assertEqual(args[args.index('substitute') + 1], 'false')

    def test_override_is_explicit_and_single_argument(self):
        args = probe_command(Path('/repo'), Path('/tmp/probe'), Path('/repo/probe'), '9', '/usr/lib /bin/sh')
        self.assertEqual(args[args.index('sandbox-paths') + 1], '/usr/lib /bin/sh')

    def test_invalid_paths_or_ports_are_rejected(self):
        for file, workspace, port in [
            (Path('relative'), Path('/repo/file'), '10'),
            (Path('/tmp/file'), Path('relative'), '10'),
            (Path('/tmp/file'), Path('/repo/file'), '0'),
            (Path('/tmp/file'), Path('/repo/file'), '65536'),
            (Path('/tmp/file'), Path('/repo/file'), 'abc'),
        ]:
            with self.subTest(file=file, workspace=workspace, port=port), self.assertRaises(ValueError):
                probe_command(Path('/repo'), file, workspace, port)


if __name__ == '__main__':
    unittest.main()
