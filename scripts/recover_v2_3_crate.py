"""Finish the original 2.3.0 Rust release without moving its published tag."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path

REPOSITORY = "iridae-dev/antecedent"
TAG = "v2.3.0"
SOURCE_SHA = "61c2ab9e0a15003eb205b69d25fdcb33be1efa62"
RELEASE_RUN = "38129776722"
CI_RUN = "38094397825"
OLD_DEPENDENCY = 'antecedent = { version = "2.3.0", path = ".", features = ["calibration-internal"] }'
NEW_DEPENDENCY = 'antecedent = { path = ".", features = ["calibration-internal"] }'


def run(
    source: Path, *command: str, capture: bool = False, env: dict | None = None
) -> str:
    result = subprocess.run(
        command, cwd=source, check=True, text=True, capture_output=capture, env=env
    )
    return result.stdout.strip() if capture else ""


def repair_manifest(text: str) -> str:
    """Remove only the version of the original local test-only self dependency."""
    original = tomllib.loads(text)
    expected = {"version": "2.3.0", "path": ".", "features": ["calibration-internal"]}
    if original.get("dev-dependencies", {}).get("antecedent") != expected:
        raise ValueError("unexpected original test-only self dependency")
    if text.count(OLD_DEPENDENCY) != 1:
        raise ValueError("the exact original dependency line must occur once")
    repaired = text.replace(OLD_DEPENDENCY, NEW_DEPENDENCY)
    expected_manifest = tomllib.loads(text)
    del expected_manifest["dev-dependencies"]["antecedent"]["version"]
    if tomllib.loads(repaired) != expected_manifest:
        raise ValueError(
            "repair changed more than the test-only self dependency version"
        )
    return repaired


def verify_release(source: Path) -> None:
    if run(source, "git", "status", "--porcelain", capture=True):
        raise ValueError("release checkout must be clean before the packaging repair")
    for ref in ("HEAD", f"refs/tags/{TAG}^{{commit}}"):
        if run(source, "git", "rev-parse", ref, capture=True) != SOURCE_SHA:
            raise ValueError("recovery must use the original immutable 2.3.0 source")
    release = json.loads(
        run(
            source,
            "gh",
            "release",
            "view",
            TAG,
            "--repo",
            REPOSITORY,
            "--json",
            "tagName,isDraft",
            capture=True,
        )
    )
    if release != {"tagName": TAG, "isDraft": False}:
        raise ValueError("the original GitHub release must already be published")
    workflow = json.loads(
        run(
            source,
            "gh",
            "run",
            "view",
            RELEASE_RUN,
            "--repo",
            REPOSITORY,
            "--json",
            "headSha,workflowName,jobs",
            capture=True,
        )
    )
    if (
        workflow["headSha"] != SOURCE_SHA
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
    env = {**os.environ, "GH_REPO": REPOSITORY, "GITHUB_REF": f"refs/tags/{TAG}"}
    run(source, "bash", "scripts/verify_release.sh", "tag", "2.3.0", CI_RUN, env=env)


def recover(source: Path, execute: bool) -> None:
    if execute and not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise ValueError("CARGO_REGISTRY_TOKEN is required for publication")
    verify_release(source)
    manifest = source / "crates/antecedent/Cargo.toml"
    original = manifest.read_bytes()
    manifest.write_bytes(repair_manifest(original.decode("utf-8")).encode("utf-8"))
    try:
        if (
            run(source, "git", "diff", "--name-only", capture=True)
            != "crates/antecedent/Cargo.toml"
        ):
            raise ValueError("only the reviewed packaging manifest may change")
        print(
            f"Packaging original release source {SOURCE_SHA}; only the test-only version is omitted.",
            flush=True,
        )
        command = ("cargo", "publish", "-p", "antecedent", "--locked", "--allow-dirty")
        run(source, *command, "--dry-run")
        if execute:
            run(source, *command)
            print(
                "Published antecedent 2.3.0; existing supporting crates and Python artifacts were preserved."
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
    args = parser.parse_args()
    recover(args.source.resolve(), args.execute)


if __name__ == "__main__":
    main()
