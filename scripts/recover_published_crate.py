"""Finish a partially published Rust release without moving its published tag."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = "iridae-dev/antecedent"
NEW_DEPENDENCY = 'antecedent = { path = ".", features = ["calibration-internal"] }'


@dataclass(frozen=True)
class Release:
    """The published release being completed and the evidence that identifies it."""

    tag: str
    source_sha: str
    release_run: str
    ci_run: str

    @property
    def version(self) -> str:
        return self.tag.removeprefix("v")


def old_dependency(version: str) -> str:
    """The exact self dependency line the release was originally published with."""
    return (
        f'antecedent = {{ version = "{version}", path = ".", '
        'features = ["calibration-internal"] }'
    )


def run(
    source: Path, *command: str, capture: bool = False, env: dict | None = None
) -> str:
    result = subprocess.run(
        command, cwd=source, check=True, text=True, capture_output=capture, env=env
    )
    return result.stdout.strip() if capture else ""


def repair_manifest(text: str, version: str) -> str:
    """Remove only the version of the original local test-only self dependency."""
    original = tomllib.loads(text)
    expected = {"version": version, "path": ".", "features": ["calibration-internal"]}
    if original.get("dev-dependencies", {}).get("antecedent") != expected:
        raise ValueError("unexpected original test-only self dependency")
    old = old_dependency(version)
    if text.count(old) != 1:
        raise ValueError("the exact original dependency line must occur once")
    repaired = text.replace(old, NEW_DEPENDENCY)
    expected_manifest = tomllib.loads(text)
    del expected_manifest["dev-dependencies"]["antecedent"]["version"]
    if tomllib.loads(repaired) != expected_manifest:
        raise ValueError(
            "repair changed more than the test-only self dependency version"
        )
    return repaired


def verify_release(source: Path, release: Release) -> None:
    if run(source, "git", "status", "--porcelain", capture=True):
        raise ValueError("release checkout must be clean before the packaging repair")
    for ref in ("HEAD", f"refs/tags/{release.tag}^{{commit}}"):
        if run(source, "git", "rev-parse", ref, capture=True) != release.source_sha:
            raise ValueError("recovery must use the original immutable release source")
    published = json.loads(
        run(
            source,
            "gh",
            "release",
            "view",
            release.tag,
            "--repo",
            REPOSITORY,
            "--json",
            "tagName,isDraft",
            capture=True,
        )
    )
    if published != {"tagName": release.tag, "isDraft": False}:
        raise ValueError("the original GitHub release must already be published")
    workflow = json.loads(
        run(
            source,
            "gh",
            "run",
            "view",
            release.release_run,
            "--repo",
            REPOSITORY,
            "--json",
            "headSha,workflowName,jobs",
            capture=True,
        )
    )
    if (
        workflow["headSha"] != release.source_sha
        or workflow["workflowName"] != "publish-release"
    ):
        raise ValueError("original release workflow identity does not match")
    jobs = {job["name"]: job["conclusion"] for job in workflow["jobs"]}
    required = (
        "Prepare version",
        "crates.io dry-run",
        "Publish GitHub Release",
        "Publish wheels to PyPI",
    )
    if any(jobs.get(name) != "success" for name in required):
        raise ValueError(
            "the original source verification and Python release must have succeeded"
        )
    run(source, "bash", "scripts/gate_calibration_attestation.sh")
    # This manual workflow runs on main; verification operates on the actual
    # release-tag checkout, whose exact identity was checked above.
    env = {**os.environ, "GH_REPO": REPOSITORY, "GITHUB_REF": f"refs/tags/{release.tag}"}
    run(source, "bash", "scripts/verify_release.sh", "tag", release.version, release.ci_run, env=env)


def recover(source: Path, release: Release, execute: bool) -> None:
    if execute and not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise ValueError("CARGO_REGISTRY_TOKEN is required for publication")
    verify_release(source, release)
    manifest = source / "crates/antecedent/Cargo.toml"
    original = manifest.read_bytes()
    manifest.write_bytes(repair_manifest(original.decode("utf-8"), release.version).encode("utf-8"))
    try:
        if (
            run(source, "git", "diff", "--name-only", capture=True)
            != "crates/antecedent/Cargo.toml"
        ):
            raise ValueError("only the reviewed packaging manifest may change")
        print(
            f"Packaging original release source {release.source_sha}; only the test-only version is omitted.",
            flush=True,
        )
        command = ("cargo", "publish", "-p", "antecedent", "--locked", "--allow-dirty")
        run(source, *command, "--dry-run")
        if execute:
            run(source, *command)
            print(
                f"Published antecedent {release.version}; existing supporting crates and Python artifacts were preserved."
            )
    finally:
        manifest.write_bytes(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="clean full-history checkout of the original release tag",
    )
    parser.add_argument(
        "--execute", action="store_true", help="publish after verification and dry-run"
    )
    parser.add_argument("--tag", required=True, help="published release tag, e.g. v1.2.3")
    parser.add_argument("--source-sha", required=True, help="commit the tag points at")
    parser.add_argument("--release-run", required=True, help="publish-release run id")
    parser.add_argument("--ci-run", required=True, help="CI run id of the tagged commit")
    args = parser.parse_args()
    release = Release(args.tag, args.source_sha, args.release_run, args.ci_run)
    recover(args.source.resolve(), release, args.execute)


if __name__ == "__main__":
    main()
