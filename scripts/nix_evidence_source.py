"""Materialize and check committed inputs for a prospective Nix build.

This is an engineering helper, not proof that a Nix build was sandboxed or an
authorization to issue evidence. Evidence earning does not use it yet.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout


def _clean_head(root: Path) -> tuple[str, str]:
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("source checkout must have no tracked or untracked changes")
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    algorithm = _git(root, "rev-parse", "--show-object-format").decode().strip()
    if algorithm not in ("sha1", "sha256"):
        raise ValueError("unsupported Git object format")
    return head, algorithm


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"unsafe source path: {name!r}")
    return path


def _expected_tree(root: Path, head: str) -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for entry in _git(root, "ls-tree", "-r", "-z", "--full-tree", head).split(b"\0"):
        if not entry:
            continue
        info, filename = entry.split(b"\t", 1)
        mode, kind, oid = info.decode("ascii").split(" ")
        name = os.fsdecode(filename)
        _safe_path(name)
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"unsupported source tree entry: {name}")
        expected[name] = (mode, oid)
    return expected


def _blob_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_committed_snapshot(root: Path, destination: Path, commit: str) -> None:
    """Revalidate source immediately before or after an external import.

    Rechecking both the checkout and snapshot narrows races but does not make
    the Nix import atomic or establish that any build used this snapshot.
    """
    root = root.resolve()
    head, algorithm = _clean_head(root)
    if head != commit:
        raise ValueError("checkout HEAD changed since source snapshot")
    expected = _expected_tree(root, head)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("source snapshot is missing or is a symlink")
    found: set[str] = set()
    expected_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    found_dirs: set[str] = set()
    for parent, directories, files in os.walk(destination, followlinks=False):
        for name in directories:
            directory = Path(parent) / name
            if directory.is_symlink():
                raise ValueError("source snapshot contains a symlink")
            found_dirs.add(directory.relative_to(destination).as_posix())
        for name in files:
            path = Path(parent) / name
            relative = path.relative_to(destination).as_posix()
            if path.is_symlink() or not path.is_file() or relative not in expected:
                raise ValueError(f"unexpected source snapshot entry: {relative}")
            mode, oid = expected[relative]
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (mode == "100755") or _blob_hash(path, algorithm) != oid:
                raise ValueError(f"source snapshot differs from committed blob: {relative}")
            found.add(relative)
    if found != set(expected) or found_dirs != expected_dirs:
        raise ValueError("source snapshot membership differs from committed tree")
    if _clean_head(root)[0] != head:
        raise ValueError("source checkout changed while checking the snapshot")


_STORE_SOURCE = re.compile(r"/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-antecedent-source\Z")
_NAR_HASH = re.compile(r"sha256-[A-Za-z0-9+/]{43}=\Z")


def import_committed_snapshot(root: Path, snapshot: Path, commit: str, *, run=subprocess.run) -> dict[str, str]:
    """Import a verified snapshot and compare its NAR with the store copy.

    The returned identity does not attest Nix daemon policy or a build. Callers
    must independently establish those facts before using it for any evidence.
    """
    verify_committed_snapshot(root, snapshot, commit)
    snapshot = snapshot.resolve()

    def nix(*args: str) -> str:
        result = run(["nix", "--extra-experimental-features", "nix-command", "--offline", *args],
                     check=True, capture_output=True, text=True)
        return result.stdout.strip()

    def nar(path: Path) -> str:
        value = nix("hash", "path", "--algo", "sha256", "--format", "sri", str(path))
        if not _NAR_HASH.fullmatch(value):
            raise ValueError("invalid Nix NAR hash")
        return value

    before = nar(snapshot)
    store = nix("store", "add-path", "--name", "antecedent-source", str(snapshot))
    if not _STORE_SOURCE.fullmatch(store):
        raise ValueError("invalid imported Nix store path")
    stored = nar(Path(store))
    verify_committed_snapshot(root, snapshot, commit)
    after = nar(snapshot)
    if not before == stored == after:
        raise ValueError("Nix store source differs from the verified snapshot")
    verify_committed_snapshot(root, snapshot, commit)
    return {"commit": commit, "store_path": store, "nar_hash": stored}


def materialize_committed_source(root: Path, destination: Path) -> dict[str, str]:
    """Copy the exact clean HEAD tree, checking every archived Git blob.

    `destination` must not exist. Symlinks and submodules are refused rather
    than allowing a snapshot to depend on a host path or another checkout.
    No change to Git or the source checkout is made.
    """
    root = root.resolve()
    head, algorithm = _clean_head(root)
    expected = _expected_tree(root, head)
    destination.mkdir(parents=True, exist_ok=False)
    found: set[str] = set()
    try:
        with tempfile.TemporaryFile() as archive:
            subprocess.run(["git", "archive", "--format=tar", head], cwd=root,
                           stdout=archive, stderr=subprocess.PIPE, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:") as tar:
                _extract_archive(tar, destination, expected, algorithm, found)
        if found != set(expected):
            raise ValueError("archive membership differs from committed tree")
        verify_committed_snapshot(root, destination, head)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {"commit": head, "object_format": algorithm}


def _extract_archive(tar: tarfile.TarFile, destination: Path,
                     expected: dict[str, tuple[str, str]], algorithm: str,
                     found: set[str]) -> None:
    for member in tar:
        name = member.name.rstrip("/") if member.isdir() else member.name
        path = _safe_path(name)
        if member.isdir():
            (destination / path).mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile() or name not in expected or name in found:
            raise ValueError(f"unexpected source archive entry: {name}")
        data_file = tar.extractfile(member)
        if data_file is None:
            raise ValueError(f"unreadable source archive entry: {name}")
        mode, oid = expected[name]
        output = destination / path
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            shutil.copyfileobj(data_file, stream, 1024 * 1024)
        if _blob_hash(output, algorithm) != oid:
            raise ValueError(f"archive content differs from committed blob: {name}")
        output.chmod(0o755 if mode == "100755" else 0o644)
        found.add(name)


def map_coverage_sources(coverage: dict[str, set[int]], source: Path, checkout: Path) -> dict[str, set[int]]:
    """Map fixture coverage from an immutable source snapshot to matching files.

    Only source paths inside `source` are mapped; external dependency paths are
    not treated as checkout coverage. Each mapped file must still match bytes.
    This translation does not prove build provenance or sandbox enforcement.
    """
    source, checkout = source.resolve(), checkout.resolve()
    mapped: dict[str, set[int]] = {}
    for filename, lines in coverage.items():
        if Path(filename).is_symlink():
            raise ValueError(f"symlink coverage source: {filename}")
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(source)
        except ValueError:
            continue
        if not relative.parts or ".." in relative.parts:
            raise ValueError("invalid coverage source path")
        target = checkout / relative
        if target.resolve() != target:
            raise ValueError(f"symlink or non-canonical coverage path: {filename}")
        if not path.is_file() or not target.is_file() or path.read_bytes() != target.read_bytes():
            raise ValueError(f"coverage source differs from checkout: {relative}")
        mapped.setdefault(str(target), set()).update(lines)
    return mapped
