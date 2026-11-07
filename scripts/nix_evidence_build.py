"""Build a committed Rust target with the experimental Nix recipe.

This build-only diagnostic cannot execute a target or publish evidence. Nix
daemon policy, host runtime, and portable manifest binding remain unverified;
callers must not interpret the result as authorization to earn a record.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

from _repo import scientific_target_feature_args
from nix_artifact_identity import inspect_artifact, recheck_artifact
from nix_compile_time_inventory import compiler_units
from nix_evidence_source import (import_committed_snapshot, materialize_committed_source,
                                 verify_committed_snapshot)

_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_FEATURE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./+-]*\Z")


def _target_features(root: Path, crate: str, target_kind: str, target_name: str) -> list[str]:
    if not _NAME.fullmatch(crate) or target_kind not in ("lib", "test"):
        raise ValueError("invalid Rust target")
    if (target_kind == "lib" and target_name) or (target_kind == "test" and not _NAME.fullmatch(target_name)):
        raise ValueError("invalid Rust target name")
    target = ["--lib"] if target_kind == "lib" else ["--test", target_name]
    flags = scientific_target_feature_args(root / "crates" / crate / "Cargo.toml", target)
    features = sorted(set(flags[1].split(","))) if flags else []
    if any(not _FEATURE.fullmatch(feature) for feature in features):
        raise ValueError("invalid Rust target features")
    return features


def _build(root: Path, source: str, crate: str, target_kind: str,
           target_name: str, features: list[str]) -> str:
    command = [
        "nix", "--extra-experimental-features", "nix-command flakes", "--offline",
        "--option", "substitute", "false", "--option", "builders", "",
        "build", "--file", str(root / "nix/evidence-target.nix"),
        "--no-link", "--print-out-paths", "--arg", "src", "builtins.storePath " + json.dumps(source),
        "--argstr", "crate", crate, "--argstr", "targetKind", target_kind,
        "--argstr", "targetName", target_name, "--argstr", "featuresJson", json.dumps(features),
    ]

    def build(argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=True)
        warnings = result.stderr.lower()
        if "ignoring untrusted" in warnings or "not a trusted user" in warnings:
            raise ValueError("Nix ignored an untrusted build setting")
        return result.stdout.strip()

    output = build(command)
    rebuild = command.copy()
    rebuild.insert(rebuild.index("build") + 1, "--rebuild")
    if build(rebuild) != output:
        raise ValueError("Nix rebuild returned a different output")
    return output


def _versions(root: Path, output: Path) -> dict[str, str]:
    pin = tomllib.loads((root / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
    text = (output / "rustc-version").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    if fields.get("release") != pin["release"] or fields.get("commit-hash") != pin["commit"]:
        raise ValueError("Nix build compiler does not match the evidence compiler pin")
    cargo = (output / "cargo-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"cargo \d+\.\d+\.\d+(?: \([^\n]*\))?", cargo):
        raise ValueError("Nix build Cargo version is missing or malformed")
    return {"rustc": fields["release"], "rustc_commit": fields["commit-hash"], "cargo": cargo}


def _compiler_graph(output: Path, crate: str, target_kind: str,
                    target_name: str, features: list[str]) -> dict:
    metadata = json.loads((output / "cargo-metadata.json").read_text(encoding="utf-8"))
    graph = json.loads((output / "cargo-unit-graph.json").read_text(encoding="utf-8"))
    normalized = compiler_units(metadata, graph)
    if len(normalized["roots"]) != 1:
        raise ValueError("Nix Cargo graph must have exactly one selected root")
    unit = normalized["units"][normalized["roots"][0]]
    if (unit["name"] != crate or unit["mode"] != "test"
            or target_kind not in unit["target_kind"]
            or (target_kind == "test" and unit["target_name"] != target_name)
            or not set(features).issubset(unit["features"])):
        raise ValueError("Nix Cargo graph root does not match the selected target and features")
    return normalized


def build_committed_target(root: Path, crate: str, target_kind: str, target_name: str = "") -> dict:
    """Build but never run a target from clean HEAD; the result is not attestation."""
    root = root.resolve()
    features = _target_features(root, crate, target_kind, target_name)
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-source-") as tmp:
        snapshot = Path(tmp) / "snapshot"
        commit = materialize_committed_source(root, snapshot)["commit"]
        source = import_committed_snapshot(root, snapshot, commit)
        verify_committed_snapshot(root, snapshot, commit)
        output = _build(root, source["store_path"], crate, target_kind, target_name, features)
        artifact = inspect_artifact(root, output, source["store_path"], crate, target_kind, target_name, features)
        if artifact["source_nar_hash"] != source["nar_hash"] or artifact["sandbox_verified"] is not False:
            raise ValueError("Nix artifact source or sandbox status is inconsistent")
        tools = _versions(root, Path(output))
        graph = _compiler_graph(Path(output), crate, target_kind, target_name, features)
        recheck_artifact(root, artifact)
        verify_committed_snapshot(root, snapshot, commit)
    return {"source": source, "artifact": artifact, "tools": tools, "compiler_graph": graph,
            "sandbox_verified": False, "evidence_authorized": False}
