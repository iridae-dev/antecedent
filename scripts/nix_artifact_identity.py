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


# Every file that defines the evidence derivations; each is bound into an artifact identity.
RECIPE_FILES = ('nix/evidence-target.nix', 'nix/evidence-deps.nix', 'nix/evidence-common.nix',
                'nix/evidence-driver.nix',
                'nix/evidence-toolchain.nix', 'flake.lock')
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


_SOURCE_NARS: dict[str, str] = {}
_STORE_INFO: dict[tuple[str, str], tuple[dict, dict]] = {}


def _source_nar(source: str) -> str:
    if source not in _SOURCE_NARS:
        _SOURCE_NARS[source] = _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', source)
    return _SOURCE_NARS[source]


_FILE_HASHES: dict[tuple, str] = {}


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIdentityError(f"missing or linked artifact file: {path}")
    # A test binary is hundreds of megabytes and is checked once per assertion: hash it
    # once per file state (device, inode, size, modification and change times).
    info = path.stat()
    key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if key not in _FILE_HASHES:
        _FILE_HASHES[key] = _hash_bytes(path)
    return _FILE_HASHES[key]


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_dir(crate: str, target_kind: str, target_name: str) -> str:
    """The output-relative directory of one target in a shared evidence build."""
    return f"targets/{crate}/{target_kind}/{target_name or 'lib'}"


def target_binary(artifact: dict) -> Path:
    """The executable a recorded artifact identity names."""
    return Path(artifact['output']) / artifact['binary']


def canonical_targets(build_targets: list[dict]) -> list[dict]:
    """The sorted, validated target list a shared evidence derivation is built from."""
    out = []
    for item in build_targets:
        crate, kind, name, features = item['crate'], item['kind'], item['name'], item['features']
        if not isinstance(crate, str) or not _NAME.fullmatch(crate) or kind not in ('lib', 'test'):
            raise ArtifactIdentityError('invalid Cargo target')
        if (kind == 'lib' and name) or (kind == 'test' and (not isinstance(name, str) or not _NAME.fullmatch(name))):
            raise ArtifactIdentityError('invalid Cargo target name')
        if not isinstance(features, list) or any(not isinstance(f, str) or not _FEATURE.fullmatch(f) for f in features):
            raise ArtifactIdentityError('invalid Cargo features')
        out.append({'crate': crate, 'kind': kind, 'name': name, 'features': sorted(set(features))})
    out.sort(key=lambda t: (t['crate'], t['kind'], t['name']))
    if not out or len({(t['crate'], t['kind'], t['name']) for t in out}) != len(out):
        raise ArtifactIdentityError('empty or duplicate evidence build targets')
    return out


def targets_json(build_targets: list[dict]) -> str:
    return json.dumps(canonical_targets(build_targets), separators=(',', ':'), sort_keys=True)


# Within one process, a derivation evaluated from unchanged recipe bytes and arguments,
# and an output whose content hash was verified, are not evaluated or hashed again.
# Store paths are immutable; rechecks still compare store metadata and executable bytes.
_DERIVATIONS: dict[tuple, str] = {}
_VERIFIED_OUTPUTS: set[tuple[str, str]] = set()


def inspect_artifact(root: Path, output: str, source: str, crate: str,
                     target_kind: str, target_name: str, features: list[str],
                     *, build_targets: list[dict] | None = None, deps_source: str,
                     deps_targets: list[dict] | None = None,
                     store: Path = Path('/nix/store')) -> dict:
    """Bind requested inputs to an existing store result and its executable.

    The derivation is evaluated independently from the tracked recipe and the
    exact arguments, then cross-checked against store metadata and actual NAR
    hashes. The source's relationship to the checkout and sandbox enforcement
    are deliberately outside this identity check.
    """
    root, store = root.resolve(), store.resolve()
    output = _store_path(output, store)
    source = _store_path(source, store)
    deps_source = _store_path(deps_source, store)
    if any(Path(p).is_symlink() or not Path(p).is_dir() for p in (output, source)):
        raise ArtifactIdentityError('missing or linked Nix store directory')
    if not _NAME.fullmatch(crate) or target_kind not in ('lib', 'test'):
        raise ArtifactIdentityError('invalid Cargo target')
    if (target_kind == 'lib' and target_name) or (target_kind == 'test' and not _NAME.fullmatch(target_name)):
        raise ArtifactIdentityError('invalid Cargo target name')
    if not isinstance(features, list) or any(not isinstance(f, str) or not _FEATURE.fullmatch(f) for f in features):
        raise ArtifactIdentityError('invalid Cargo features')
    features = sorted(set(features))
    build_targets = canonical_targets(build_targets if build_targets is not None else [
        {'crate': crate, 'kind': target_kind, 'name': target_name, 'features': features}])
    if {'crate': crate, 'kind': target_kind, 'name': target_name, 'features': features} not in build_targets:
        raise ArtifactIdentityError('the target is not one the evidence build compiled')
    deps_targets = canonical_targets(deps_targets if deps_targets is not None else build_targets)
    recipe = root / 'nix/evidence-target.nix'
    recipe_files = [root / rel for rel in RECIPE_FILES]
    recipe_hashes = {str(p.relative_to(root)): _file_hash(p) for p in recipe_files}
    expr = f'builtins.storePath {json.dumps(source)}'
    arguments, deps_arguments = targets_json(build_targets), targets_json(deps_targets)
    key = (tuple(sorted(recipe_hashes.items())), source, arguments, deps_source, deps_arguments)
    if key not in _DERIVATIONS:
        _DERIVATIONS[key] = _nix('nix-instantiate', '--extra-experimental-features', 'nix-command flakes',
                                 str(recipe), '--arg', 'src', expr, '--argstr', 'targetsJson', arguments,
                                 '--arg', 'depsSrc', f'builtins.storePath {json.dumps(deps_source)}',
                                 '--argstr', 'depsTargetsJson', deps_arguments)
    drv = _DERIVATIONS[key]
    _store_path(drv, store, derivation=True)
    try:
        # Store derivations and valid outputs are immutable: query each once per process.
        if (drv, output) not in _STORE_INFO:
            _STORE_INFO[(drv, output)] = (
                json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'derivation', 'show', drv)),
                json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'path-info', '--json', output)),
            )
        derivations, info = _STORE_INFO[(drv, output)]
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
    if (output, output_nar) not in _VERIFIED_OUTPUTS:
        if _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', output) != output_nar:
            raise ArtifactIdentityError('Nix output content differs from store metadata')
        _VERIFIED_OUTPUTS.add((output, output_nar))
    source_nar = _source_nar(source)
    if not _SRI.fullmatch(source_nar):
        raise ArtifactIdentityError('invalid source NAR hash')
    relative = target_dir(crate, target_kind, target_name) + '/bin/test'
    binary = Path(output) / relative
    recorded_source = Path(output) / 'source-root'
    if recorded_source.is_symlink() or not recorded_source.is_file() or recorded_source.read_text(encoding="utf-8").strip() != source:
        raise ArtifactIdentityError('artifact source-root does not match')
    if any(_file_hash(p) != recipe_hashes[str(p.relative_to(root))] for p in recipe_files):
        raise ArtifactIdentityError('Nix recipe changed during inspection')
    return {
        'schema': 2, 'derivation': drv, 'output': output, 'output_nar_hash': output_nar,
        'source': source, 'source_nar_hash': source_nar,
        'crate': crate, 'target_kind': target_kind, 'target_name': target_name, 'features': features,
        'build_targets': build_targets, 'deps_source': deps_source, 'deps_targets': deps_targets,
        'binary': relative,
        'binary_sha256': _file_hash(binary), 'recipe_sha256': recipe_hashes,
        'sandbox_verified': False,
    }


def recheck_artifact(root: Path, identity: dict, *, store: Path = Path('/nix/store')) -> None:
    """Refuse a change to the recorded inputs, metadata, or executable bytes."""
    try:
        current = inspect_artifact(root, identity['output'], identity['source'], identity['crate'],
                                   identity['target_kind'], identity['target_name'], identity['features'],
                                   build_targets=identity['build_targets'], deps_source=identity['deps_source'],
                                   deps_targets=identity['deps_targets'], store=store)
    except (KeyError, TypeError) as error:
        raise ArtifactIdentityError('invalid recorded artifact identity') from error
    if current != identity:
        raise ArtifactIdentityError('Nix artifact identity changed')
