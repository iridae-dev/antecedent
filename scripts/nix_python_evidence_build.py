"""Build and inspect a committed Python wheel, without running or earning tests.

The result remains diagnostic: sandbox policy, Python runtime requirements,
and the executable/test binding are not attested by this helper.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from nix_evidence_source import (
    import_committed_snapshot,
    materialize_committed_source,
    verify_committed_snapshot,
)
from nix_python_evidence_identity import inspect_python_wheel


def _build(root: Path, source: str) -> str:
    command = ['nix', '--extra-experimental-features', 'nix-command flakes', '--offline',
               '--option', 'substitute', 'false', '--option', 'builders', '',
               'build', '--file', str(root / 'nix/evidence-python-wheel.nix'), '--no-link',
               '--print-out-paths', '--arg', 'src', 'builtins.storePath ' + json.dumps(source)]
    def invoke(argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=True)
        if 'ignoring untrusted' in result.stderr.lower() or 'not a trusted user' in result.stderr.lower():
            raise ValueError('Nix ignored a requested build setting')
        return result.stdout.strip()
    output = invoke(command)
    rebuild = command.copy()
    rebuild.insert(rebuild.index('build') + 1, '--rebuild')
    if invoke(rebuild) != output:
        raise ValueError('Nix Python wheel rebuild returned a different output')
    return output


def build_committed_python_wheel(root: Path) -> dict:
    """Build a wheel from clean HEAD; never run tests or authorize evidence."""
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix='antecedent-nix-python-source-') as tmp:
        snapshot = Path(tmp) / 'snapshot'
        commit = materialize_committed_source(root, snapshot)['commit']
        source = import_committed_snapshot(root, snapshot, commit)
        verify_committed_snapshot(root, snapshot, commit)
        output = _build(root, source['store_path'])
        artifact = inspect_python_wheel(root, output, source['store_path'])
        if artifact['source_nar_hash'] != source['nar_hash']:
            raise ValueError('Python wheel source differs from committed snapshot')
        if (artifact['sandbox_verified'] is not False or artifact['evidence_authorized'] is not False
                or artifact['runtime_requirements_verified'] is not False):
            raise ValueError('Python wheel cannot authorize evidence')
        if inspect_python_wheel(root, output, source['store_path']) != artifact:
            raise ValueError('Python wheel changed while inspecting')
        verify_committed_snapshot(root, snapshot, commit)
    return {'source': source, 'artifact': artifact, 'sandbox_verified': False,
            'runtime_requirements_verified': False, 'evidence_authorized': False}
