"""Unpublished diagnostic capture for a Nix-built Rust target.

This module is not connected to the scientific earning CLI. It neither creates
nor withdraws records and does not establish sandbox, runtime or receipt trust.
The caller must have explicit authorization before executing selected tests.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import types
from pathlib import Path

import evidence_capture

from nix_artifact_identity import recheck_artifact, target_binary
from nix_evidence_build import build_committed_target
from nix_evidence_source import (
    import_committed_snapshot,
    map_coverage_sources,
    materialize_committed_source,
    verify_committed_snapshot,
)


def diagnose_target(root: Path, rows: list[dict], run=subprocess.run) -> dict:
    """Capture exactly selected assertions without creating evidence records.

    All rows must name the same Rust target. The returned coverage is a
    diagnostic only: Nix evidence earning remains disabled until its portable
    receipt, policy and runtime requirements have been implemented and audited.
    """
    from scientific_records import exact_test_name, locate

    if not rows:
        raise ValueError("at least one Rust assertion is required")
    root = root.resolve()
    infos = [{**locate(root, row)} for row in rows]
    for row, info in zip(rows, infos, strict=True):
        if info["lang"] == "rust":
            info["libtest"] = exact_test_name(root, row, info)
    if any(info["lang"] != "rust" for info in infos):
        raise ValueError("Nix Rust diagnostics do not accept Python assertions")
    crate = infos[0]["crate"]
    target = tuple(infos[0]["target"])
    if any(info["crate"] != crate or tuple(info["target"]) != target for info in infos):
        raise ValueError("Nix Rust diagnostics require one Cargo target")
    if target == ("--lib",):
        kind, name = "lib", ""
    elif len(target) == 2 and target[0] == "--test":
        kind, name = "test", target[1]
    else:
        raise ValueError("unsupported Cargo target")

    built = build_committed_target(root, crate, kind, name)
    return capture_built_target(root, rows, built, run)


def _capture(root: Path, binary: str, info: dict, env: dict[str, str], run,
             tools: dict[str, str]) -> dict[str, set[int]] | None:
    from scientific_records import capture_one
    if run is subprocess.run:
        return capture_one(root, binary, info, env, llvm_tools=tools)
    return capture_one(root, binary, info, env, run, llvm_tools=tools)


# Held while the checkout is read against its committed snapshot, and by a caller while
# it writes and allows run outputs: a check never sees a written output not yet allowed.
CHECKOUT_LOCK = threading.RLock()

# The verified committed snapshot of each commit, materialized and imported once per
# process: every target of one run is checked against the same commit.
_SNAPSHOTS: dict[tuple[str, str], tuple[tempfile.TemporaryDirectory, Path, str]] = {}


def _verified_snapshot(root: Path, source: dict, allowed_outputs: dict[str, str] | None) -> tuple[Path, str]:
    key = (str(root), source["commit"])
    with CHECKOUT_LOCK:
        return _snapshot_locked(root, source, allowed_outputs, key)


def _snapshot_locked(root: Path, source: dict, allowed_outputs: dict[str, str] | None,
                     key: tuple[str, str]) -> tuple[Path, str]:
    if key not in _SNAPSHOTS:
        directory = tempfile.TemporaryDirectory(prefix="antecedent-nix-capture-")
        snapshot = Path(directory.name) / "snapshot"
        commit = materialize_committed_source(root, snapshot, allowed_outputs=allowed_outputs)["commit"]
        if commit != source["commit"]:
            directory.cleanup()
            raise ValueError("checkout changed after the Nix build")
        imported = import_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
        if imported != source:
            directory.cleanup()
            raise ValueError("verified source import differs from the Nix build")
        _SNAPSHOTS[key] = (directory, snapshot, commit)
    _, snapshot, commit = _SNAPSHOTS[key]
    return snapshot, commit


def capture_built_target(root: Path, rows: list[dict], built: dict, run=subprocess.run,
                         *, continue_on_failure: bool = False,
                         allowed_outputs: dict[str, str] | None = None) -> dict:
    """Capture a previously built target, rechecking its committed source and executable.

    The caller is responsible for creating snapshots before execution and for
    validating the build's audited policy before publishing any records.
    """
    from scientific_records import capture_env, capture_one, exact_test_name, locate

    if not rows:
        raise ValueError("at least one Rust assertion is required")
    root = root.resolve()
    infos = [{**locate(root, row)} for row in rows]
    for row, info in zip(rows, infos, strict=True):
        if info["lang"] == "rust":
            info["libtest"] = exact_test_name(root, row, info)
    if any(info["lang"] != "rust" for info in infos):
        raise ValueError("Nix Rust capture does not accept Python assertions")
    crate, target = infos[0]["crate"], tuple(infos[0]["target"])
    if any(info["crate"] != crate or tuple(info["target"]) != target for info in infos):
        raise ValueError("Nix Rust capture requires one Cargo target")
    artifact = built["artifact"]
    expected = ("lib", "") if target == ("--lib",) else (
        ("test", target[1]) if len(target) == 2 and target[0] == "--test" else None
    )
    if expected is None or (artifact.get("crate"), artifact.get("target_kind"), artifact.get("target_name")) != (crate, *expected):
        raise ValueError("Nix build target differs from the requested assertions")
    if built.get("sandbox_verified") is not False or built.get("evidence_authorized") is not False:
        raise ValueError("unpublished Nix capture cannot attest or authorize evidence")
    source, artifact = built["source"], built["artifact"]
    if artifact["source"] != source["store_path"] or artifact["source_nar_hash"] != source["nar_hash"]:
        raise ValueError("Nix build source and artifact identity disagree")
    output = Path(artifact["output"])
    tools = {name: str(output / "bin" / name) for name in ("llvm-profdata", "llvm-cov")}
    results: dict[str, dict[str, set[int]]] = {}
    failures: list[str] = []
    snapshot, commit = _verified_snapshot(root, source, allowed_outputs)
    env = capture_env(root)
    binary = str(target_binary(artifact))
    with CHECKOUT_LOCK:
        verify_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
        recheck_artifact(root, artifact)
    try:
        # Rows are independent test processes, queued behind every earlier submission
        # (`evidence_capture`). The source and artifact are checked before and after.
        from scientific_records import capture_one
        local = run is not subprocess.run or not isinstance(capture_one, types.FunctionType)
        futures = [evidence_capture.submit(_capture, root, binary, info, env, run, tools, local=local)
                   for info in infos]
        captured = [future.result() for future in futures]
    finally:
        # Recheck even on a test or coverage failure; no result is returned if
        # the source or artifact changed while the batch ran.
        with CHECKOUT_LOCK:
            recheck_artifact(root, artifact)
            verify_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
    for info, coverage in zip(infos, captured, strict=True):
        if coverage is None:
            if continue_on_failure:
                failures.append(info["selector"])
                continue
            raise RuntimeError(f"diagnostic assertion did not pass: {info['selector']}")
        results[info["selector"]] = map_coverage_sources(coverage, Path(source["store_path"]), root)
    return {"coverage": results, "failures": failures, "build": built, "sandbox_verified": False,
            "evidence_authorized": False}
