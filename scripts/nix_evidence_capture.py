"""Unpublished diagnostic capture for a Nix-built Rust target.

This module is not connected to the scientific earning CLI. It neither creates
nor withdraws records and does not establish sandbox, runtime or receipt trust.
The caller must have explicit authorization before executing selected tests.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from nix_artifact_identity import recheck_artifact
from nix_evidence_build import build_committed_target
from nix_evidence_source import (import_committed_snapshot, map_coverage_sources,
                                 materialize_committed_source, verify_committed_snapshot)


def diagnose_target(root: Path, rows: list[dict], run=subprocess.run) -> dict:
    """Capture exactly selected assertions without creating evidence records.

    All rows must name the same Rust target. The returned coverage is a
    diagnostic only: Nix evidence earning remains disabled until its portable
    receipt, policy and runtime requirements have been implemented and audited.
    """
    from scientific_records import capture_env, capture_one, locate

    if not rows:
        raise ValueError("at least one Rust assertion is required")
    root = root.resolve()
    infos = [locate(root, row) for row in rows]
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
    if built.get("sandbox_verified") is not False or built.get("evidence_authorized") is not False:
        raise ValueError("unpublished Nix capture cannot attest or authorize evidence")
    source, artifact = built["source"], built["artifact"]
    if artifact["source"] != source["store_path"] or artifact["source_nar_hash"] != source["nar_hash"]:
        raise ValueError("Nix build source and artifact identity disagree")
    output = Path(artifact["output"])
    tools = {name: str(output / "bin" / name) for name in ("llvm-profdata", "llvm-cov")}
    results: dict[str, dict[str, set[int]]] = {}
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-capture-") as directory:
        snapshot = Path(directory) / "snapshot"
        commit = materialize_committed_source(root, snapshot)["commit"]
        if commit != source["commit"]:
            raise ValueError("checkout changed after the Nix build")
        imported = import_committed_snapshot(root, snapshot, commit)
        if imported != source:
            raise ValueError("verified source import differs from the Nix build")
        env = capture_env(root)
        for info in infos:
            verify_committed_snapshot(root, snapshot, commit)
            recheck_artifact(root, artifact)
            try:
                coverage = capture_one(root, str(output / "bin/test"), info, env, run, llvm_tools=tools)
            finally:
                # Recheck even on a test or coverage failure; no result is
                # returned if the source or artifact changed while it ran.
                recheck_artifact(root, artifact)
                verify_committed_snapshot(root, snapshot, commit)
            if coverage is None:
                raise RuntimeError(f"diagnostic assertion did not pass: {info['selector']}")
            results[info["selector"]] = map_coverage_sources(coverage, Path(source["store_path"]), root)
    return {"coverage": results, "build": built, "sandbox_verified": False,
            "evidence_authorized": False}
