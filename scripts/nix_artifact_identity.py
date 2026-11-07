"""Identify a Nix test artifact without claiming it was built in a sandbox.

This diagnostic is not an evidence attestation. Its caller must separately
verify the source against Git, prove the builder policy, and bind the actual
executed bytes to any eventual evidence manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


class ArtifactIdentityError(ValueError):
    """The Nix result does not match the requested source or build."""


_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_FEATURE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./+-]*\Z")
_STORE_ITEM = re.compile(r"[0-9abcdfghijklmnpqrsvwxyz]{32}-[^/]+\Z")
_SRI = re.compile(r"sha256-[A-Za-z0-9+/]{43}=\Z")


def _store_path(value: str, store: Path, *, derivation: bool = False) -> str:
    if not isinstance(value, str):
        raise ArtifactIdentityError("invalid Nix store path")
    path = Path(value)
    if path.parent != store or not _STORE_ITEM.fullmatch(path.name) or path.name.endswith('.drv') != derivation:
        raise ArtifactIdentityError("invalid Nix store path")
    return value


def _nix(*args: str) -> str:
    try:
        return subprocess.check_output(list(args), text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactIdentityError("Nix identity query failed") from error


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIdentityError(f"missing or linked artifact file: {path}")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_artifact(root: Path, output: str, source: str, crate: str,
                     target_kind: str, target_name: str, features: list[str],
                     *, store: Path = Path('/nix/store')) -> dict:
    """Bind requested inputs to an existing store result and its executable.

    The derivation is evaluated independently from the tracked recipe and the
    exact arguments, then cross-checked against store metadata and actual NAR
    hashes. The source's relationship to the checkout and sandbox enforcement
    are deliberately outside this identity check.
    """
    root, store = root.resolve(), store.resolve()
    output = _store_path(output, store)
    source = _store_path(source, store)
    if any(Path(p).is_symlink() or not Path(p).is_dir() for p in (output, source)):
        raise ArtifactIdentityError('missing or linked Nix store directory')
    if not _NAME.fullmatch(crate) or target_kind not in ('lib', 'test'):
        raise ArtifactIdentityError('invalid Cargo target')
    if (target_kind == 'lib' and target_name) or (target_kind == 'test' and not _NAME.fullmatch(target_name)):
        raise ArtifactIdentityError('invalid Cargo target name')
    if not isinstance(features, list) or any(not isinstance(f, str) or not _FEATURE.fullmatch(f) for f in features):
        raise ArtifactIdentityError('invalid Cargo features')
    features = sorted(set(features))
    recipe, lock, toolchain = (root / 'nix/evidence-target.nix', root / 'flake.lock', root / 'nix/evidence-toolchain.nix')
    recipe_hashes = {str(p.relative_to(root)): _file_hash(p) for p in (recipe, lock, toolchain)}
    expr = f'builtins.storePath {json.dumps(source)}'
    drv = _nix('nix-instantiate', '--extra-experimental-features', 'nix-command flakes', str(recipe),
               '--arg', 'src', expr, '--argstr', 'crate', crate, '--argstr', 'targetKind', target_kind,
               '--argstr', 'targetName', target_name, '--argstr', 'featuresJson',
               json.dumps(features, separators=(',', ':')))
    _store_path(drv, store, derivation=True)
    try:
        derivations = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'derivation', 'show', drv))
        info = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'path-info', '--json', output))
        definition, metadata = derivations[drv], info[output]
        if definition['outputs']['out']['path'] != output or source not in definition['inputSrcs']:
            raise ArtifactIdentityError('derivation output or source does not match the request')
        if metadata['deriver'] != drv or source not in metadata['references']:
            raise ArtifactIdentityError('Nix output metadata does not match the request')
        output_nar = metadata['narHash']
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIdentityError('invalid Nix derivation or output metadata') from error
    if not isinstance(output_nar, str) or not _SRI.fullmatch(output_nar):
        raise ArtifactIdentityError('invalid Nix output NAR hash')
    if _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', output) != output_nar:
        raise ArtifactIdentityError('Nix output content differs from store metadata')
    source_nar = _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', source)
    if not _SRI.fullmatch(source_nar):
        raise ArtifactIdentityError('invalid source NAR hash')
    binary = Path(output) / 'bin/test'
    recorded_source = Path(output) / 'source-root'
    if recorded_source.is_symlink() or not recorded_source.is_file() or recorded_source.read_text().strip() != source:
        raise ArtifactIdentityError('artifact source-root does not match')
    if any(_file_hash(p) != recipe_hashes[str(p.relative_to(root))] for p in (recipe, lock, toolchain)):
        raise ArtifactIdentityError('Nix recipe changed during inspection')
    return {
        'schema': 1, 'derivation': drv, 'output': output, 'output_nar_hash': output_nar,
        'source': source, 'source_nar_hash': source_nar,
        'crate': crate, 'target_kind': target_kind, 'target_name': target_name, 'features': features,
        'binary_sha256': _file_hash(binary), 'recipe_sha256': recipe_hashes,
        'sandbox_verified': False,
    }


def recheck_artifact(root: Path, identity: dict, *, store: Path = Path('/nix/store')) -> None:
    """Refuse a change to the recorded inputs, metadata, or executable bytes."""
    try:
        current = inspect_artifact(root, identity['output'], identity['source'], identity['crate'],
                                   identity['target_kind'], identity['target_name'], identity['features'], store=store)
    except (KeyError, TypeError) as error:
        raise ArtifactIdentityError('invalid recorded artifact identity') from error
    if current != identity:
        raise ArtifactIdentityError('Nix artifact identity changed')
