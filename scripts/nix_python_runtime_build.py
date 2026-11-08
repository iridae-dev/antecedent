"""Build and inspect a locked Nix Python runtime; never run or record tests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nix_evidence_source import _clean_head
from nix_python_evidence_build import build_committed_python_wheel
from nix_python_runtime_identity import inspect_python_runtime


def _build(root: Path, source: str, wheel: str) -> str:
    command = ['nix', '--extra-experimental-features', 'nix-command flakes', '--offline',
               '--option', 'substitute', 'false', '--option', 'builders', '',
               'build', '--file', str(root / 'nix/evidence-python-runtime.nix'), '--no-link',
               '--print-out-paths', '--arg', 'src', 'builtins.storePath ' + json.dumps(source),
               '--arg', 'wheelOutput', 'builtins.storePath ' + json.dumps(wheel)]
    def invoke(argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=True)
        if 'ignoring untrusted' in result.stderr.lower() or 'not a trusted user' in result.stderr.lower():
            raise ValueError('Nix ignored a requested runtime build setting')
        return result.stdout.strip()
    output = invoke(command)
    rebuild = command.copy()
    rebuild.insert(rebuild.index('build') + 1, '--rebuild')
    if invoke(rebuild) != output:
        raise ValueError('Nix Python runtime rebuild returned a different output')
    return output


def build_committed_python_runtime(root: Path) -> dict:
    """Build from a verified committed wheel; result is never an attestation."""
    root = root.resolve()
    build = build_committed_python_wheel(root)
    source = build['source']
    if _clean_head(root)[0] != source['commit']:
        raise ValueError('checkout changed before Python runtime build')
    output = _build(root, source['store_path'], build['artifact']['output'])
    artifact = inspect_python_runtime(root, output, source['store_path'], build['artifact']['output'])
    if artifact['source_nar_hash'] != source['nar_hash']:
        raise ValueError('Python runtime source differs from committed snapshot')
    if artifact['sandbox_verified'] is not False or artifact['evidence_authorized'] is not False:
        raise ValueError('Python runtime cannot authorize evidence')
    if inspect_python_runtime(root, output, source['store_path'], build['artifact']['output']) != artifact:
        raise ValueError('Python runtime changed while inspecting')
    if _clean_head(root)[0] != source['commit']:
        raise ValueError('checkout changed during Python runtime build')
    return {'source': source, 'wheel': build['artifact'], 'runtime': artifact,
            'sandbox_verified': False, 'evidence_authorized': False}
