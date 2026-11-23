"""Build and import only the fixed disposable Nix/PyO3 engineering fixture.

The fixture does not accept a project module or issue evidence. A passing
result does not attest sandbox policy, external runtime identity, or the
Python extension used by Antecedent's scientific workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from nix_compile_time_inventory import compiler_units
from nix_evidence_source import (
    import_committed_snapshot,
    materialize_committed_source,
    verify_committed_snapshot,
)

ROOT = Path(__file__).resolve().parent.parent
MODULE = "nix_python_boundary"
STORE_ITEM = re.compile(r"/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[^/]+\Z")
HASH = re.compile(r"sha256-[A-Za-z0-9+/]{43}=\Z")


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Python engineering fixture command failed: {command[0]}: {result.stderr[-4000:]}")
    return result.stdout.strip()


def _prepare_fixture(root: Path) -> None:
    shutil.copytree(ROOT / "nix/fixtures/python", root)
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.test",
           "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.test"}
    _run(["git", "init", "-q"], cwd=root, env=env)
    _run(["git", "add", "Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs"], cwd=root, env=env)
    _run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
          "commit", "-qm", "disposable Python engineering fixture"], cwd=root, env=env)


def _build(source: str) -> str:
    command = ["nix", "--extra-experimental-features", "nix-command flakes", "--offline",
               "--option", "substitute", "false", "--option", "builders", "",
               "build", "--file", str(ROOT / "nix/evidence-python-fixture.nix"),
               "--no-link", "--print-out-paths", "--arg", "src", "builtins.storePath " + json.dumps(source)]
    output = _run(command, cwd=ROOT)
    if not STORE_ITEM.fullmatch(output):
        raise ValueError("invalid Nix Python fixture output path")
    rebuild = command.copy()
    rebuild.insert(rebuild.index("build") + 1, "--rebuild")
    if _run(rebuild, cwd=ROOT) != output:
        raise ValueError("Nix Python fixture rebuild produced a different output")
    return output


def inspect_output(output: Path, source: str, source_nar: str) -> dict:
    """Check observable fixture content; this is not a sandbox attestation."""
    if not STORE_ITEM.fullmatch(str(output)) or not STORE_ITEM.fullmatch(source) or not HASH.fullmatch(source_nar):
        raise ValueError("invalid fixture source or output identity")
    actual_source_nar = _run(["nix", "--extra-experimental-features", "nix-command", "hash", "path", source], cwd=ROOT)
    if actual_source_nar != source_nar:
        raise ValueError("Python fixture imported source changed")
    if (output / "source-root").read_text(encoding="utf-8").strip() != source:
        raise ValueError("Python fixture source does not match the imported snapshot")
    pin = tomllib.loads((ROOT / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
    rust = (output / "rustc-version").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in rust.splitlines() if ": " in line)
    if fields.get("release") != pin["release"] or fields.get("commit-hash") != pin["commit"]:
        raise ValueError("Python fixture compiler differs from the evidence compiler pin")
    cargo = (output / "cargo-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"cargo \d+\.\d+\.\d+(?: \([^\n]*\))?", cargo):
        raise ValueError("Python fixture Cargo version is invalid")
    python = json.loads((output / "python-config.json").read_text(encoding="utf-8"))
    executable = Path(python["executable"])
    if (python.get("implementation") != "cpython" or not python.get("version", "").startswith("3.13.")
            or not str(executable).startswith("/nix/store/")
            or (output / "bin/python3.13").resolve() != executable.resolve()):
        raise ValueError("Python fixture does not bind the pinned Python 3.13 interpreter")
    extension = output / "lib" / f"{MODULE}.so"
    if extension.is_symlink() or not extension.is_file():
        raise ValueError("Python fixture extension is absent or linked")
    metadata = json.loads((output / "cargo-metadata.json").read_text(encoding="utf-8"))
    unit_graph = json.loads((output / "cargo-unit-graph.json").read_text(encoding="utf-8"))
    graph = compiler_units(metadata, unit_graph)
    if len(graph["roots"]) != 1:
        raise ValueError("Python fixture graph has multiple roots")
    unit = graph["units"][graph["roots"][0]]
    if unit["name"] != MODULE or "cdylib" not in unit["target_kind"] or unit["mode"] != "build":
        raise ValueError("Python fixture graph does not identify its extension build")
    nar = _run(["nix", "--extra-experimental-features", "nix-command", "hash", "path", str(output)], cwd=ROOT)
    if not HASH.fullmatch(nar):
        raise ValueError("invalid Python fixture output NAR hash")
    return {"output_nar_hash": nar, "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "python": python, "rustc": fields["release"], "rustc_commit": fields["commit-hash"],
            "cargo": cargo, "compiler_units": len(graph["units"])}


def run_fixture() -> dict:
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-python-engineering-") as directory:
        scratch = Path(directory)
        checkout, snapshot = scratch / "fixture", scratch / "snapshot"
        _prepare_fixture(checkout)
        commit = materialize_committed_source(checkout, snapshot)["commit"]
        source = import_committed_snapshot(checkout, snapshot, commit)
        verify_committed_snapshot(checkout, snapshot, commit)
        output = Path(_build(source["store_path"]))
        identity = inspect_output(output, source["store_path"], source["nar_hash"])
        # Python -I disables user and environment module search. Only this
        # fixed toy extension is imported; no project Python package is loaded.
        snippet = ('import sys; sys.path.insert(0, sys.argv[1]); '
                   'import nix_python_boundary; assert nix_python_boundary.add(19, 23) == 42')
        _run([str(output / "bin/python3.13"), "-I", "-c", snippet, str(output / "lib")], cwd=scratch)
        if inspect_output(output, source["store_path"], source["nar_hash"]) != identity:
            raise ValueError("Python fixture output changed during import")
        verify_committed_snapshot(checkout, snapshot, commit)
        return {"fixture": "passed", "source_nar_hash": source["nar_hash"], **identity,
                "sandbox_verified": False, "evidence_published": False}


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2))
