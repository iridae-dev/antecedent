"""Fail-closed policy for the reviewed Darwin Nix compile-time graph.

The reviewed scope is ambient /private/tmp reads, not a claim that the host OS
or compiler is adversarially isolated. The local daemon and pinned Nix/cache
inputs remain explicitly trusted. The build adapter independently verifies its
source, derivation, executable, local rebuild and no-substitution flags.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

POLICY = "nix-darwin-private-tmp-audit-v1"
REGISTRY = "nix/compile-time-reviewed.json"
SYSTEM_CONFIG = Path("/etc/nix/nix.conf")
ALLOWED_SANDBOX_PATHS = frozenset({
    "/System/Library/Frameworks", "/System/Library/PrivateFrameworks",
    "/bin/bash", "/bin/sh", "/private/tmp", "/private/var/tmp", "/usr/lib",
})
NIX_OVERRIDES = frozenset({"NIX_CONFIG", "NIX_USER_CONF_FILES", "NIX_REMOTE", "NIX_PATH", "NIX_CONF_DIR"})
# reviewed-safe: reads nothing that selects repository text or positions.
# complete-source: assessed host inputs (for example an interpreter); evidence
#   keeps complete source. unresolved: not yet assessed for projection.
PROJECTION_STATES = frozenset({"reviewed-safe", "complete-source", "unresolved"})
UNIT_KEYS = ("name", "version", "mode", "target_kind", "target_path", "features", "platform")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _signature(unit: dict) -> tuple:
    try:
        name, version, mode, path = (unit[k] for k in ("name", "version", "mode", "target_path"))
        kinds, features, platform = unit["target_kind"], unit["features"], unit["platform"]
        if not all(isinstance(x, str) and x for x in (name, version, mode, path)):
            raise ValueError("invalid compiler unit identity")
        if (not isinstance(kinds, list) or not all(isinstance(x, str) for x in kinds)
                or not isinstance(features, list) or not all(isinstance(x, str) for x in features)
                or (platform is not None and not isinstance(platform, str))):
            raise ValueError("invalid compiler unit kind, feature or platform")
        return name, version, mode, tuple(sorted(kinds)), path, tuple(sorted(features)), platform
    except (KeyError, TypeError) as error:
        raise ValueError("invalid compiler unit identity") from error


def _verify_sandbox() -> None:
    """Check local trusted daemon configuration; client flags alone are insufficient."""
    overrides = sorted(NIX_OVERRIDES.intersection(os.environ))
    if overrides:
        raise ValueError("unreviewed Nix configuration override: " + ", ".join(overrides))
    try:
        if SYSTEM_CONFIG.is_symlink() or not SYSTEM_CONFIG.is_file():
            raise ValueError("Nix system configuration is missing or linked")
        stat = SYSTEM_CONFIG.stat()
        if stat.st_uid != 0 or stat.st_mode & 0o002:
            raise ValueError("Nix system configuration is not root-owned or is world-writable")
        settings: dict[str, list[str]] = {}
        for raw in SYSTEM_CONFIG.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if re.match(r"!?include(?:\s|$)", line):
                raise ValueError("included Nix system configuration is not reviewed")
            if "=" in line:
                key, value = line.split("=", 1)
                settings.setdefault(key.strip(), []).append(value.strip())
        for key, expected in (("sandbox", "true"), ("sandbox-fallback", "false")):
            if settings.get(key) != [expected]:
                raise ValueError(f"Nix daemon configuration must explicitly set {key} = {expected}")
        process = subprocess.run(
            ["nix", "--extra-experimental-features", "nix-command", "config", "show", "--json"],
            capture_output=True, text=True, check=True,
        )
        config = json.loads(process.stdout)
        if config["sandbox"]["value"] is not True or config["sandbox-fallback"]["value"] is not False:
            raise ValueError("Nix client sandbox settings disagree with the reviewed daemon policy")
        paths = config["sandbox-paths"]["value"]
        if not isinstance(paths, list) or any(not isinstance(p, str) or p not in ALLOWED_SANDBOX_PATHS for p in paths):
            raise ValueError("Nix exposes an unreviewed sandbox path")
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cannot verify local Nix sandbox configuration") from error


def validate_build(root: Path, built: dict) -> dict:
    """Validate actual emitted compiler units against the lock-bound source audit.

    This checks the scoped audit and local policy only; callers must independently
    validate the source, target, build/rebuild provenance and executed artifact.
    """
    _verify_sandbox()
    return validate_units(root, built["compiler_graph"])


def host_compiler_graph(root: Path, crate: str, target: list[str]) -> dict:
    """The compiler-unit graph Cargo would build for a host evidence target.

    `target` is Cargo's target selection (`--lib`, `--test NAME`) for a Rust row,
    or `["python"]` for the native extension as maturin builds it. Only metadata
    is produced; nothing is compiled or run.
    """
    from _repo import scientific_target_feature_args
    from nix_compile_time_inventory import compiler_units
    root = root.resolve()
    if target == ["python"]:
        maturin = tomllib.loads((root / "python/pyproject.toml").read_text(encoding="utf-8"))["tool"]["maturin"]
        features = maturin.get("features", [])
        command = ["build", "-p", "antecedent-py", "--lib"] + (["--features", ",".join(features)] if features else [])
    else:
        features = scientific_target_feature_args(root / "crates" / crate / "Cargo.toml", target)
        command = ["test", "--no-run", "-p", crate, *target, *features]
    def cargo(*args: str, env: dict | None = None) -> dict:
        out = subprocess.run(["cargo", *args], cwd=root, capture_output=True, text=True, check=True, env=env)
        return json.loads(out.stdout)
    metadata = cargo("metadata", "--offline", "--locked", "--format-version", "1")
    # The unstable flag only prints the graph; RUSTC_BOOTSTRAP is confined to it.
    graph = cargo("-Z", "unstable-options", *command[:1], "--unit-graph", "--offline", "--locked",
                  *command[1:], env={**os.environ, "RUSTC_BOOTSTRAP": "1"})
    return compiler_units(metadata, graph)


def validate_units(root: Path, graph: dict, *, require_compile_time: bool = True) -> dict:
    """Check every compile-time unit (and its dependencies) against the audit registry."""
    root = root.resolve()
    try:
        registry_bytes = (root / REGISTRY).read_bytes()
        registry = json.loads(registry_bytes)
        lock_bytes = (root / "Cargo.lock").read_bytes()
        if (registry.get("schema") != 1 or registry.get("policy") != POLICY
                or registry.get("target") != "aarch64-apple-darwin"
                or registry.get("cargo_lock_sha256") != _sha(lock_bytes)):
            raise ValueError("Nix compile-time audit is stale or unrecognized")
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        checksums: dict[tuple[str, str], str] = {}
        for package in lock["package"]:
            if not package.get("source", "").startswith("registry+"):
                continue
            key = package["name"], package["version"]
            if key in checksums or not re.fullmatch(r"[0-9a-f]{64}", package.get("checksum", "")):
                raise ValueError("ambiguous or unlocked registry package")
            checksums[key] = package["checksum"]
        approved: dict[tuple, list[dict]] = {}
        for row in registry["units"]:
            signature = _signature(row)
            approved.setdefault(signature, []).append(row)
        units, roots = graph["units"], graph["roots"]
        if (not isinstance(units, list) or not units or not isinstance(roots, list) or not roots
                or any(type(i) is not int or i < 0 or i >= len(units) for i in roots)):
            raise ValueError("invalid Nix compiler graph")
        seen: set[int] = set()
        selected: dict[str, dict] = {}
        todo = []
        for i, unit in enumerate(units):
            kinds, mode = unit.get("target_kind"), unit.get("mode")
            if not isinstance(kinds, list):
                raise ValueError("invalid Nix compiler unit kinds")
            if mode == "run-custom-build" or "proc-macro" in kinds:
                todo.append(i)
        if not todo and require_compile_time:
            raise ValueError("Nix compiler graph has no reviewed compile-time units")
        while todo:
            i = todo.pop()
            if i in seen:
                continue
            seen.add(i)
            unit = units[i]
            signature = _signature(unit)
            rows = approved.get(signature, [])
            if len(rows) != 1 or checksums.get(signature[:2]) != rows[0].get("checksum"):
                raise ValueError(f"unreviewed Nix compile-time unit: {signature[0]} {signature[1]} {signature[2]}")
            selected[_sha(_canonical(signature))] = rows[0]
            dependencies = unit.get("dependencies")
            if not isinstance(dependencies, list):
                raise ValueError("invalid Nix compiler unit dependencies")
            for dependency in dependencies:
                index = dependency.get("index") if isinstance(dependency, dict) else None
                if type(index) is not int or index < 0 or index >= len(units):
                    raise ValueError("invalid Nix compiler dependency index")
                todo.append(index)
        projection_states = {row.get("source_projection") for row in selected.values()}
        if not projection_states <= PROJECTION_STATES:
            raise ValueError("invalid Nix source projection review status")
        receipt = {"id": POLICY, "cargo_lock_sha256": _sha(lock_bytes),
                   "target": registry["target"], "covered_units": len(seen),
                   "source_projection": _overall(projection_states),
                   "source_projection_allowed": projection_states == {"reviewed-safe"},
                   "selected_unit_ids": sorted(selected)}
        receipt["audit_selection_sha256"] = _selection_digest(registry, selected)
        return receipt
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("invalid or incomplete Nix build policy input") from error


def _overall(states: set) -> str:
    """The weakest review state among the selected units."""
    for state in ("unresolved", "complete-source"):
        if state in states:
            return state
    return "reviewed-safe"


def _selection_digest(registry: dict, selected: dict[str, dict]) -> str:
    return _sha(_canonical({"policy": registry["policy"], "target": registry["target"],
                            "scope": registry["scope"], "units": selected}))


def recheck_selection(root: Path, receipt: dict) -> None:
    """Portably recheck the selected audit rows without depending on other rows."""
    try:
        registry = json.loads((root / REGISTRY).read_bytes())
        lock = (root / "Cargo.lock").read_bytes()
        if (registry.get("schema") != 1 or registry.get("policy") != POLICY
                or receipt.get("id") != POLICY or registry.get("target") != receipt.get("target")
                or registry.get("cargo_lock_sha256") != _sha(lock)
                or receipt.get("cargo_lock_sha256") != _sha(lock)):
            raise ValueError("Nix audit policy or lock changed")
        ids = receipt["selected_unit_ids"]
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or not all(isinstance(i, str) and re.fullmatch(r"[0-9a-f]{64}", i) for i in ids)):
            raise ValueError("invalid selected Nix audit identity")
        wanted = set(ids)
        selected: dict[str, dict] = {}
        for row in registry["units"]:
            identity = _sha(_canonical(_signature(row)))
            if identity in wanted:
                if identity in selected:
                    raise ValueError("duplicate selected Nix audit unit")
                selected[identity] = row
        if set(selected) != wanted or receipt.get("audit_selection_sha256") != _selection_digest(registry, selected):
            raise ValueError("selected Nix compile-time audit changed")
        states = {row.get("source_projection") for row in selected.values()}
        if (not states <= PROJECTION_STATES
                or receipt.get("source_projection") != _overall(states)
                or receipt.get("source_projection_allowed") is not (states == {"reviewed-safe"})):
            raise ValueError("Nix source projection review status changed")
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid or incomplete selected Nix audit") from error
