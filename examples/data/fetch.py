#!/usr/bin/env python3
"""Fetch and verify the datasets described by the manifests beside this file.

Standard library only. Nothing is fetched on import; a download happens only when
``fetch`` runs with ``offline=False`` and a file is missing or fails its checksum.

    python examples/data/fetch.py nsw_psid            # download and verify
    python examples/data/fetch.py nsw_psid --offline  # verify what is already cached
    python examples/data/fetch.py nsw_psid --print-sha256   # maintainers: print digests

The cache directory is ``$ANTECEDENT_DATA_DIR`` or ``~/.cache/antecedent/data``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


class DatasetUnavailable(Exception):
    """The dataset is not usable offline: a file is absent, unchecked or fails its checksum."""


def cache_dir() -> Path:
    override = os.environ.get("ANTECEDENT_DATA_DIR")
    return (
        Path(override).expanduser()
        if override
        else Path.home() / ".cache" / "antecedent" / "data"
    )


def load_manifest(dataset: str) -> dict:
    with (HERE / f"{dataset}.toml").open("rb") as handle:
        return tomllib.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(dataset: str, name: str) -> Path:
    return cache_dir() / dataset / name


def verified_paths(dataset: str) -> dict[str, Path]:
    """Cached files whose SHA-256 matches the manifest, keyed by file name.

    Raises ``DatasetUnavailable`` naming the first problem: a missing file, a manifest
    whose checksum is not yet recorded (so nothing can be verified), or a mismatch.
    """
    manifest = load_manifest(dataset)
    paths: dict[str, Path] = {}
    for entry in manifest["file"]:
        path = _target(dataset, entry["name"])
        expected = entry.get("sha256", "")
        if not expected:
            raise DatasetUnavailable(
                f"{entry['name']}: no checksum recorded in the manifest, so it cannot be verified"
            )
        if not path.is_file():
            raise DatasetUnavailable(
                f"{entry['name']}: not in the cache ({path.parent})"
            )
        actual = _sha256(path)
        if actual != expected:
            raise DatasetUnavailable(
                f"{entry['name']}: checksum {actual} does not match the manifest"
            )
        paths[entry["name"]] = path
    return paths


def fetch(dataset: str, *, offline: bool = False) -> dict[str, Path]:
    """Return verified local paths, downloading missing or corrupt files unless ``offline``."""
    manifest = load_manifest(dataset)
    if not offline:
        for entry in manifest["file"]:
            if not entry.get("sha256"):
                raise DatasetUnavailable(
                    f"{entry['name']}: no checksum recorded in the manifest; refusing an unverified download"
                )
            path = _target(dataset, entry["name"])
            if path.is_file() and _sha256(path) == entry["sha256"]:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(entry["url"], timeout=60) as response:  # noqa: S310
                partial.write_bytes(response.read())
            if _sha256(partial) != entry["sha256"]:
                partial.unlink()
                raise DatasetUnavailable(
                    f"{entry['name']}: downloaded bytes do not match the manifest checksum"
                )
            partial.replace(path)
    return verified_paths(dataset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument(
        "--offline", action="store_true", help="verify the cache; never use the network"
    )
    parser.add_argument(
        "--print-sha256",
        action="store_true",
        help="download each file and print its digest (maintainers)",
    )
    args = parser.parse_args()
    if args.print_sha256:
        for entry in load_manifest(args.dataset)["file"]:
            with urllib.request.urlopen(entry["url"], timeout=60) as response:  # noqa: S310
                print(
                    f'{entry["name"]}  sha256 = "{hashlib.sha256(response.read()).hexdigest()}"'
                )
        return 0
    try:
        paths = fetch(args.dataset, offline=args.offline)
    except DatasetUnavailable as problem:
        print(f"unavailable: {problem}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"verified {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
