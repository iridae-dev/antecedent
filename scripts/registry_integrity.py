"""Verify resolved Cargo registry sources against the locked package archives.

This establishes package-source integrity only. It does not assess build-script
reads, the native compiler, SDK, interpreter, Cargo configuration, or the
environment in which those programs execute.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tomllib
from pathlib import Path, PurePosixPath


class RegistryIntegrityError(ValueError):
    """A resolved source does not match its locked registry archive."""


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _safe_member(name: str, root: str) -> str:
    path = PurePosixPath(name)
    if (not name or name.startswith('/') or '\\' in name
            or any(p in ('', '.', '..') for p in name.split('/'))
            or path.parts[0] != root):
        raise RegistryIntegrityError(f'unsafe archive member: {name}')
    return '/'.join(path.parts[1:])


def verify_package(directory: Path, archive: Path, checksum: str) -> None:
    """Check the archive, all extracted files, and exact tree membership."""
    archive_data = archive.read_bytes()
    if hashlib.sha256(archive_data).hexdigest() != checksum:
        raise RegistryIntegrityError(f'archive checksum mismatch: {archive.name}')
    expected_files: dict[str, str] = {}
    expected_dirs: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode='r:gz', encoding="utf-8") as tar:
        for member in tar:
            name = member.name.rstrip('/') if member.isdir() else member.name
            rel = _safe_member(name, directory.name)
            if not rel:
                if not member.isdir():
                    raise RegistryIntegrityError('archive root is not a directory')
                continue
            ancestors = list(PurePosixPath(rel).parents)[:-1]
            expected_dirs.update(str(p) for p in ancestors if str(p) != '.')
            if member.isdir():
                expected_dirs.add(rel)
            elif member.isfile():
                if rel in expected_files or rel in expected_dirs:
                    raise RegistryIntegrityError(f'duplicate archive entry: {rel}')
                f = tar.extractfile(member)
                if f is None:
                    raise RegistryIntegrityError(f'unreadable archive entry: {rel}')
                h = hashlib.sha256()
                for chunk in iter(lambda f=f: f.read(1024 * 1024), b''):
                    h.update(chunk)
                expected_files[rel] = h.hexdigest()
            else:
                raise RegistryIntegrityError(f'unsupported archive entry: {rel}')
    if directory.is_symlink() or not directory.is_dir():
        raise RegistryIntegrityError(f'missing or linked package source: {directory}')
    files: set[str] = set()
    dirs: set[str] = set()
    def failed_walk(error: OSError) -> None:
        raise RegistryIntegrityError(f'cannot read package source: {directory}') from error

    for parent, children, names in os.walk(directory, followlinks=False, onerror=failed_walk):
        for child in children:
            path = Path(parent) / child
            if path.is_symlink():
                raise RegistryIntegrityError(f'linked package directory: {path}')
            dirs.add(path.relative_to(directory).as_posix())
        for name in names:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                raise RegistryIntegrityError(f'unsupported package entry: {path}')
            files.add(path.relative_to(directory).as_posix())
    # Cargo creates these metadata files on unpack; they are not crate inputs.
    if '.cargo-ok' in files and '.cargo-ok' not in expected_files:
        try:
            if json.loads((directory / '.cargo-ok').read_text(encoding="utf-8")) != {'v': 1}:
                raise RegistryIntegrityError('unexpected Cargo unpack marker')
        except (ValueError, OSError) as error:
            raise RegistryIntegrityError('invalid Cargo unpack marker') from error
        files.remove('.cargo-ok')
    if '.cargo-checksum.json' in files and '.cargo-checksum.json' not in expected_files:
        data = json.loads((directory / '.cargo-checksum.json').read_text(encoding="utf-8"))
        if data.get('package') != checksum or data.get('files') != expected_files:
            raise RegistryIntegrityError('invalid Cargo checksum metadata')
        files.remove('.cargo-checksum.json')
    if files != set(expected_files) or dirs != expected_dirs:
        raise RegistryIntegrityError(f'extracted package membership mismatch: {directory.name}')
    for rel, expected in expected_files.items():
        if _hash_file(directory / rel) != expected:
            raise RegistryIntegrityError(f'extracted package file mismatch: {directory.name}/{rel}')


def verify_registry(root: Path, metadata: dict) -> None:
    """Verify every resolved registry package against its Cargo.lock checksum."""
    lock = tomllib.loads((root / 'Cargo.lock').read_text(encoding='utf-8'))
    rows = {(p['name'], p['version'], p.get('source')): p.get('checksum')
            for p in lock['package']}
    for package in metadata['packages']:
        source = package.get('source')
        if source is None:
            continue
        if not source.startswith('registry+'):
            raise RegistryIntegrityError(f'unsupported non-registry dependency: {package["name"]}')
        checksum = rows.get((package['name'], package['version'], source))
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise RegistryIntegrityError(f'missing locked registry checksum: {package["name"]}')
        directory = Path(package['manifest_path']).parent
        if directory.name != f'{package["name"]}-{package["version"]}':
            raise RegistryIntegrityError(f'unexpected registry source path: {package["name"]}')
        archive = directory.parent.parent.parent / 'cache' / directory.parent.name / (directory.name + '.crate')
        verify_package(directory, archive, checksum)


def main() -> int:
    from _repo import REPO_ROOT
    from compile_time_environment import noncanonical_compile_environment
    issues = noncanonical_compile_environment(REPO_ROOT, os.environ)
    if issues:
        raise SystemExit('unassessed compile-time environment: ' + ', '.join(issues))
    result = subprocess.run(
        ['cargo', 'metadata', '--format-version', '1', '--locked', '--offline'],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit('cannot verify registry inputs: locked offline Cargo metadata failed')
    try:
        verify_registry(REPO_ROOT, json.loads(result.stdout))
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'cannot verify registry inputs: {error}') from error
    print('Cargo registry archives and extracted sources match Cargo.lock')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
