"""Build committed Rust test targets with the Nix evidence recipe.

One derivation builds every requested target, each with its own Cargo invocation
over a shared target directory. This build-only step cannot execute a target or
publish evidence. Nix daemon policy, host runtime, and portable manifest binding
are verified by the caller; the result is never authorization to earn a record.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from _repo import scientific_target_feature_args
from nix_artifact_identity import (
    RECIPE_FILES,
    inspect_artifact,
    recheck_artifact,
    target_dir,
    targets_json,
)
from nix_compile_time_inventory import compiler_units
from nix_evidence_deps import import_deps_source, write_deps_source
from nix_evidence_source import (
    import_committed_snapshot,
    materialize_committed_source,
    verify_committed_snapshot,
)

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


# Recipe and lock combinations whose builds a `--rebuild` reproduced, on this machine.
# Reproducibility is a property of the recipe: once checked for given recipe bytes
# and lock, rebuilding every run proves nothing new. The record is local state.
REPRODUCED = "target/nix-evidence/reproduced.json"


def _reproducibility_key(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in (*RECIPE_FILES, "Cargo.lock"):
        digest.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def _reproduced(root: Path) -> set[str]:
    path = root / REPRODUCED
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _record_reproduced(root: Path, key: str) -> None:
    path = root / REPRODUCED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(_reproduced(root) | {key})) + "\n", encoding="utf-8")


def _build(root: Path, source: str, build_targets: list[dict], deps_source: str | None = None,
           deps_targets: list[dict] | None = None) -> str:
    deps_source = deps_source or source
    deps_arguments = ["--arg", "depsSrc", "builtins.storePath " + json.dumps(deps_source),
                      "--argstr", "depsTargetsJson", targets_json(deps_targets or build_targets)]
    base = ["nix", "--extra-experimental-features", "nix-command flakes", "--offline",
            "--option", "substitute", "false", "--option", "builders", "", "build"]
    command = [
        *base, "--file", str(root / "nix/evidence-target.nix"),
        "--no-link", "--print-out-paths", "--arg", "src", "builtins.storePath " + json.dumps(source),
        "--argstr", "targetsJson", targets_json(build_targets), *deps_arguments,
    ]

    def build(argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=True)
        warnings = result.stderr.lower()
        if "ignoring untrusted" in warnings or "not a trusted user" in warnings:
            raise ValueError("Nix ignored an untrusted build setting")
        return result.stdout.strip()

    output = build(command)
    key = _reproducibility_key(root)
    if key not in _reproduced(root):
        # A new recipe or lock: check that both derivations rebuild bit for bit.
        deps = [*base, "--file", str(root / "nix/evidence-deps.nix"), "--no-link", "--print-out-paths",
                *deps_arguments]
        for argv in (deps, command):
            first = build(argv)
            rebuild = argv.copy()
            rebuild.insert(rebuild.index("build") + 1, "--rebuild")
            if build(rebuild) != first:
                raise ValueError("Nix rebuild returned a different output")
        _record_reproduced(root, key)
    return output


def _versions(root: Path, output: Path) -> dict[str, str]:
    pin = tomllib.loads((root / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
    text = (output / "rustc-version").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    if (fields.get("release") != pin["release"] or fields.get("commit-hash") != pin["commit"]
            or fields.get("host") != "aarch64-apple-darwin"):
        raise ValueError("Nix build compiler does not match the evidence compiler pin")
    cargo = (output / "cargo-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"cargo \d+\.\d+\.\d+(?: \([^\n]*\))?", cargo):
        raise ValueError("Nix build Cargo version is missing or malformed")
    return {"rustc": fields["release"], "rustc_commit": fields["commit-hash"],
            "host": fields["host"], "cargo": cargo}


def _compiler_graph(output: Path, crate: str, target_kind: str,
                    target_name: str, features: list[str]) -> dict:
    directory = output / target_dir(crate, target_kind, target_name)
    metadata = json.loads((directory / "cargo-metadata.json").read_text(encoding="utf-8"))
    graph = json.loads((directory / "cargo-unit-graph.json").read_text(encoding="utf-8"))
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


def build_committed_targets(root: Path, targets: list[tuple[str, str, str]], *,
                            deps_targets: list[tuple[str, str, str]] | None = None,
                            allowed_outputs: dict[str, str] | None = None) -> dict[tuple[str, str, str], dict]:
    """Build but never run targets from clean HEAD in one derivation; never attestation.

    `targets` are (crate, kind, name) with kind "lib" (name "") or "test". The result
    maps each target to its own build record: shared source and tools, its own
    artifact identity, executable and compile-time unit graph."""
    root = root.resolve()
    wanted = sorted(set(targets))
    if not wanted:
        raise ValueError("no Rust targets to build")
    def described(selection):
        return [{"crate": crate, "kind": kind, "name": name,
                 "features": _target_features(root, crate, kind, name)} for crate, kind, name in selection]
    build_targets = described(wanted)
    # The dependency build covers every scientific target, not just the owed ones,
    # so it is the same derivation whichever targets a run builds.
    all_targets = described(sorted(set(deps_targets or ()) | set(wanted)))
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-source-") as tmp:
        snapshot = Path(tmp) / "snapshot"
        commit = materialize_committed_source(root, snapshot, allowed_outputs=allowed_outputs)["commit"]
        source = import_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
        verify_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
        write_deps_source(snapshot, Path(tmp) / "deps")
        deps_source = import_deps_source(Path(tmp) / "deps")
        output = _build(root, source["store_path"], build_targets, deps_source, all_targets)
        tools = _versions(root, Path(output))
        built = {}
        for target in build_targets:
            crate, kind, name, features = target["crate"], target["kind"], target["name"], target["features"]
            artifact = inspect_artifact(root, output, source["store_path"], crate, kind, name, features,
                                        build_targets=build_targets, deps_source=deps_source,
                                        deps_targets=all_targets)
            if artifact["source_nar_hash"] != source["nar_hash"] or artifact["sandbox_verified"] is not False:
                raise ValueError("Nix artifact source or sandbox status is inconsistent")
            graph = _compiler_graph(Path(output), crate, kind, name, features)
            recheck_artifact(root, artifact)
            built[(crate, kind, name)] = {"source": source, "artifact": artifact, "tools": tools,
                                          "compiler_graph": graph, "sandbox_verified": False,
                                          "evidence_authorized": False}
        verify_committed_snapshot(root, snapshot, commit, allowed_outputs=allowed_outputs)
    return built


def build_committed_target(root: Path, crate: str, target_kind: str, target_name: str = "", *,
                           allowed_outputs: dict[str, str] | None = None) -> dict:
    """Build but never run one target from clean HEAD; the result is not attestation."""
    return build_committed_targets(root, [(crate, target_kind, target_name)],
                                   allowed_outputs=allowed_outputs)[(crate, target_kind, target_name)]
