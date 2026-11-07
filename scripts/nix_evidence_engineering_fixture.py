"""Exercise the Nix target prototype with a disposable engineering fixture.

This command accepts no repository target or user-supplied code. It cannot
publish evidence and does not establish that the Nix daemon used a sandbox.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

from nix_artifact_identity import inspect_artifact
from nix_evidence_build import _compiler_graph
from nix_evidence_source import (import_committed_snapshot, map_coverage_sources,
                                 materialize_committed_source, verify_committed_snapshot)
from nix_manifest_binding import exercise_fixture
from scientific_records import line_coverage

ROOT = Path(__file__).resolve().parent.parent
CRATE = "nix_boundary_fixture"
ASSERTION = "engineering_fixture"
FEATURES = ["guarded"]
SOURCE = """#[cfg(not(feature = "guarded"))]
compile_error!("the selected target feature must be enabled");
pub fn value() -> u32 { 4 }
#[test]
fn engineering_fixture() { assert_eq!(value(), 4); }
"""


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if command[0] == "nix":
        for line in result.stderr.splitlines():
            if "warning:" in line.lower() and ("ignor" in line.lower() or "untrusted" in line.lower()):
                print(line, file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"engineering fixture command failed: {command[0]}: {result.stderr[-4000:]}")
    return result.stdout.strip()


def _prepare_fixture(root: Path) -> None:
    root.mkdir()
    selected = root / "crates" / CRATE
    default = root / "crates/default_fixture"
    (selected / "src").mkdir(parents=True)
    (default / "src").mkdir(parents=True)
    (selected / "src/lib.rs").write_text(SOURCE, encoding="utf-8")
    (default / "src/lib.rs").write_text("pub fn unrelated_default() {}\n", encoding="utf-8")
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["crates/default_fixture", "crates/nix_boundary_fixture"]\n'
        'default-members = ["crates/default_fixture"]\nresolver = "2"\n',
        encoding="utf-8",
    )
    (selected / "Cargo.toml").write_text(
        '[package]\nname = "nix_boundary_fixture"\nversion = "0.1.0"\nedition = "2021"\n'
        '[features]\nguarded = []\n',
        encoding="utf-8",
    )
    (default / "Cargo.toml").write_text(
        '[package]\nname = "default_fixture"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (root / "Cargo.lock").write_text(
        'version = 4\n\n[[package]]\nname = "default_fixture"\nversion = "0.1.0"\n'
        '\n[[package]]\nname = "nix_boundary_fixture"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    # Avoid the user's global signing, hooks, and Git configuration; this is an
    # isolated synthetic repository and not a commit in the project checkout.
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.test",
           "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.test"}
    _run(["git", "init", "-q"], cwd=root, env=env)
    _run(["git", "add", "Cargo.toml", "Cargo.lock", "crates"], cwd=root, env=env)
    _run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
          "commit", "-qm", "disposable engineering fixture"], cwd=root, env=env)


def _build(source: str) -> str:
    recipe = ROOT / "nix/evidence-target.nix"
    command = [
        "nix", "--extra-experimental-features", "nix-command flakes", "--offline",
        "--option", "substitute", "false", "--option", "builders", "",
        "build", "--file", str(recipe), "--no-link", "--print-out-paths",
        "--arg", "src", "builtins.storePath " + json.dumps(source),
        "--argstr", "crate", CRATE, "--argstr", "targetKind", "lib",
        "--argstr", "targetName", "", "--argstr", "featuresJson", json.dumps(FEATURES),
    ]
    output = _run(command, cwd=ROOT)
    rebuild = command.copy()
    rebuild.insert(rebuild.index("build") + 1, "--rebuild")
    if _run(rebuild, cwd=ROOT) != output:
        raise ValueError("fixture rebuild returned a different store output")
    return output


def _execute_and_measure(binary: Path, source: Path, checkout: Path, scratch: Path) -> None:
    output = binary.parent.parent
    raw, profile = scratch / "fixture.profraw", scratch / "fixture.profdata"
    env = {**os.environ, "LLVM_PROFILE_FILE": str(raw)}
    result = _run([str(binary), "--exact", ASSERTION, "--test-threads", "1"],
                  cwd=checkout, env=env)
    if "test result: ok. 1 passed" not in result:
        raise ValueError("the disposable engineering assertion was not executed")
    _run([str(output / "bin/llvm-profdata"), "merge", "-sparse", str(raw), "-o", str(profile)], cwd=checkout)
    exported = _run([str(output / "bin/llvm-cov"), "export", f"-instr-profile={profile}",
                     str(binary)], cwd=checkout)
    coverage = map_coverage_sources(line_coverage(json.loads(exported)), source, checkout)
    lines = coverage.get(str((checkout / "crates" / CRATE / "src/lib.rs").resolve()), set())
    if not lines:
        raise ValueError("the disposable fixture's source coverage was not mapped")


def _tool_versions(output: Path) -> dict[str, str]:
    """Confirm the compiler pin and report the separately pinned Cargo tool."""
    pin = tomllib.loads((ROOT / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
    rust = (output / "rustc-version").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in rust.splitlines() if ": " in line)
    if fields.get("release") != pin["release"] or fields.get("commit-hash") != pin["commit"]:
        raise ValueError("fixture compiler does not match the evidence compiler pin")
    cargo = (output / "cargo-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"cargo \d+\.\d+\.\d+(?: \([^\n]*\))?", cargo):
        raise ValueError("fixture Cargo version is missing or malformed")
    return {"rustc": fields["release"], "rustc_commit": fields["commit-hash"], "cargo": cargo}


def _check_metadata_features(output: Path) -> None:
    metadata = json.loads((output / "cargo-metadata.json").read_text(encoding="utf-8"))
    matches = [p for p in metadata["packages"] if p["name"] == CRATE]
    if len(matches) != 1:
        raise ValueError("fixture metadata does not identify the selected package")
    package_id = matches[0]["id"]
    if package_id in metadata["workspace_default_members"]:
        raise ValueError("fixture target unexpectedly became a default workspace member")
    nodes = [node for node in metadata["resolve"]["nodes"] if node["id"] == package_id]
    if len(nodes) != 1 or not set(FEATURES).issubset(nodes[0]["features"]):
        raise ValueError("fixture metadata did not activate the selected package's features")


def run_fixture() -> dict:
    """Run only the fixed engineering fixture; never create an evidence record."""
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-engineering-") as directory:
        scratch = Path(directory)
        checkout, snapshot = scratch / "fixture", scratch / "snapshot"
        _prepare_fixture(checkout)
        commit = materialize_committed_source(checkout, snapshot)["commit"]
        source = import_committed_snapshot(checkout, snapshot, commit)
        verify_committed_snapshot(checkout, snapshot, commit)
        output = _build(source["store_path"])
        artifact = inspect_artifact(ROOT, output, source["store_path"], CRATE, "lib", "", FEATURES)
        if artifact["source_nar_hash"] != source["nar_hash"] or artifact["output"] != output:
            raise ValueError("artifact source differs from the committed fixture")
        if artifact["sandbox_verified"] is not False:
            raise ValueError("an engineering fixture must not attest a sandbox")
        versions = _tool_versions(Path(output))
        graph = _compiler_graph(Path(output), CRATE, "lib", "", FEATURES)
        _check_metadata_features(Path(output))
        build = {"source": source, "artifact": artifact, "tools": versions, "compiler_graph": graph,
                 "sandbox_verified": False, "evidence_authorized": False}
        assumptions = {
            "daemon": "local Nix daemon is trusted; not attested by this fixture",
            "sandbox": "Nix sandbox enforcement is not attested by this fixture",
            "private_tmp": "Darwin host temporary paths remain subject to source audit",
            "host_runtime": "Darwin system runtime is a host dependency",
            "clock_entropy": "clock and OS entropy are not isolated by this fixture",
            "package_sources": "pinned Nix inputs and configured signed cache are trusted",
        }
        receipt = exercise_fixture(ROOT, checkout, snapshot, build, assumptions,
                                   lambda binary: _execute_and_measure(binary, Path(source["store_path"]),
                                                                       checkout, scratch))
        if receipt["publication_authorized"] is not False:
            raise ValueError("a draft receipt cannot authorize evidence")
        return {"fixture": "passed", "source_nar_hash": source["nar_hash"],
                "binary_sha256": artifact["binary_sha256"], "sandbox_verified": False,
                "evidence_published": False, "tools": versions,
                "compiler_units": len(graph["units"])}


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2))
