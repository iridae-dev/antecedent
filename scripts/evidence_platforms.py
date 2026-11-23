#!/usr/bin/env python3
"""The platforms evidence is earned on, and the Nix policy reviewed for each.

A record is earned on the machine of the developer who implemented its feature and
names that platform in its receipt (`nix_build.policy.target`, a Rust host triple,
consistent with `policy.id` and `tools.host`). Any machine can check that a record
is current: that reads the repository only. Re-running a record's replicate digests
is possible only on the platform that earned it, since they are bit-exact and
depend on its math library.

    python3 scripts/evidence_platforms.py status    # this host's policy and whether it can earn
    python3 scripts/evidence_platforms.py prepare   # check the host and realise the build inputs
    python3 scripts/evidence_platforms.py draft-registry [--platform TARGET | --all]
                                                    # rebind a registry to Cargo.lock, from any host

Each platform has its own policy id, Nix system, build recipes, sandbox rule and
reviewed compile-time registry. The Apple-silicon policy is `nix_evidence_policy`,
unchanged; the others are checked here against the same registry format.

Evidence builds run offline and without substitutes, so a host first realises what
they read besides the source (`prepare`: the pinned toolchain, the build tools and
the crates of Cargo.lock, from `nix/evidence-inputs.nix`). On Linux the first earning
run does this itself.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import platform as host_info
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import nix_evidence_policy as darwin_policy
from nix_evidence_policy import (
    NIX_OVERRIDES,
    PROJECTION_STATES,
    SYSTEM_CONFIG,
    UNIT_KEYS,
    _canonical,
    _overall,
    _selection_digest,
    _sha,
    _signature,
)


class UnreviewedPlatform(ValueError):
    """This host has no reviewed evidence policy, or has not completed its bring-up."""


@dataclass(frozen=True)
class Platform:
    target: str  # Rust host triple: the platform's identity in a receipt
    policy: str
    nix_system: str
    registry: str  # reviewed compile-time registry
    recipes: tuple[str, ...]  # the evidence build, its recipe first
    release_recipes: tuple[str, ...]  # the uninstrumented calibration build, its recipe first
    toolchain_pins: str | None  # archive hashes a recipe reads; None when the recipe holds them
    policy_sources: tuple[str, ...]  # policy code a record binds
    # Host paths the daemon may expose inside a build. A build must also be
    # sandboxed (`sandbox = true`) with no unsandboxed fallback.
    sandbox_paths: frozenset[str]
    scope: str  # what the registry review assumes
    # Earning realises the build inputs (`prepare`) and rebuilds a new recipe or lock
    # once to check that it reproduces, without being asked.
    prepares: bool = False

    @property
    def build_recipe(self) -> str:
        return self.recipes[0]

    @property
    def release_recipe(self) -> str:
        return self.release_recipes[0]


DARWIN = Platform(
    target="aarch64-apple-darwin",
    policy=darwin_policy.POLICY,
    nix_system="aarch64-darwin",
    registry=darwin_policy.REGISTRY,
    recipes=("nix/evidence-target.nix", "nix/evidence-deps.nix", "nix/evidence-common.nix",
             "nix/evidence-driver.nix", "nix/evidence-toolchain.nix", "flake.lock"),
    release_recipes=("nix/calibration-release.nix", "nix/evidence-common.nix",
                     "nix/evidence-toolchain.nix", "flake.lock"),
    toolchain_pins=None,
    policy_sources=("scripts/nix_evidence_policy.py", "scripts/nix_evidence_deps.py"),
    sandbox_paths=darwin_policy.ALLOWED_SANDBOX_PATHS,
    scope="",  # stated by its registry
)

LINUX_PINS = "nix/linux/evidence-toolchain.json"
_LINUX_RECIPES = ("nix/linux/evidence-target.nix", "nix/linux/evidence-deps.nix",
                  "nix/linux/evidence-common.nix", "nix/linux/evidence-driver.nix",
                  "nix/linux/evidence-toolchain.nix", LINUX_PINS, "flake.lock")
_LINUX_RELEASE = ("nix/linux/calibration-release.nix", "nix/linux/evidence-common.nix",
                  "nix/linux/evidence-toolchain.nix", LINUX_PINS, "flake.lock")
_LINUX_SCOPE = ("Scoped source audit of compile-time units built in the Linux Nix sandbox (private "
                "mount, network and process namespaces; no host path but the store shell); trusted "
                "local Nix daemon, pinned Nix toolchain and signed-cache dependencies, and the "
                "reviewed helper code are explicit assumptions. New units/features fail closed.")


def _linux(arch: str) -> Platform:
    target = f"{arch}-unknown-linux-gnu"
    return Platform(
        target=target, policy="nix-linux-sandbox-audit-v1", nix_system=f"{arch}-linux",
        registry=f"nix/compile-time-reviewed.{target}.json",
        recipes=_LINUX_RECIPES, release_recipes=_LINUX_RELEASE, toolchain_pins=LINUX_PINS,
        policy_sources=(*DARWIN.policy_sources, "scripts/evidence_platforms.py"),
        sandbox_paths=frozenset({"/bin/sh"}), scope=_LINUX_SCOPE, prepares=True,
    )


PLATFORMS: dict[str, Platform] = {p.target: p for p in (DARWIN, _linux("x86_64"), _linux("aarch64"))}


def host_target() -> str:
    """This machine's Rust host triple (the name a receipt's `tools.host` carries)."""
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(host_info.machine().lower(), host_info.machine().lower())
    if sys.platform == "darwin":
        return f"{machine}-apple-darwin"
    if sys.platform.startswith("linux"):
        libc = "gnu" if host_info.libc_ver()[0] == "glibc" else "musl"
        return f"{machine}-unknown-linux-{libc}"
    return f"{machine}-{sys.platform}"


def host_platform() -> Platform:
    """The reviewed policy of this host; refuses a host without one."""
    target = host_target()
    if target not in PLATFORMS:
        raise UnreviewedPlatform(
            f"evidence cannot be earned on {target}: no Nix evidence policy is reviewed for it "
            f"(reviewed: {', '.join(sorted(PLATFORMS))}). Records earned elsewhere can still be checked here.")
    return PLATFORMS[target]


def bring_up_gaps(root: Path, platform: Platform) -> list[str]:
    """What the repository lacks before a record can be earned on `platform`; empty when ready.

    Reads the repository only, so it gives the same answer on every host."""
    gaps = [f"{rel} is missing" for rel in dict.fromkeys((*platform.recipes, *platform.release_recipes))
            if not (root / rel).is_file()]
    if platform.toolchain_pins and (root / platform.toolchain_pins).is_file():
        try:
            table = json.loads((root / platform.toolchain_pins).read_bytes())
            pin = tomllib.loads((root / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
            if table["release"] != pin["release"]:
                gaps.append(f"{platform.toolchain_pins}: release is not the evidence compiler pin's")
            pins = table["targets"][platform.target]
            unverified = sorted(name for name, value in pins.items()
                                if not (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)))
            if not pins or unverified:
                gaps.append(f"{platform.toolchain_pins}: no verified SHA-256 for {', '.join(unverified) or 'any archive'}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            gaps.append(f"{platform.toolchain_pins}: no pins for {platform.target}")
    if not (root / platform.registry).is_file():
        gaps.append(f"{platform.registry} (the reviewed compile-time registry) is missing")
    return gaps


def earning_platform(root: Path, *, prepare: bool = False) -> Platform:
    """The policy records are earned under on this host; refuses a host or tree that cannot earn.

    With `prepare`, a platform that prepares itself (`Platform.prepares`) checks the
    host and realises the build inputs first, once per recipe set and lock."""
    platform = host_platform()
    gaps = bring_up_gaps(root, platform)
    if gaps:
        raise UnreviewedPlatform(
            f"evidence cannot be earned on {platform.target}: the tree lacks part of its policy "
            "(docs/development.md, \"Earning platforms\"):\n  " + "\n  ".join(gaps))
    if prepare and platform.prepares and not prepared(root, platform):
        prepare_host(root)
    return platform


# --------------------------------------------------------------------------
# Preparing a host: the sandbox rule, the build inputs and the registry's cover.
# --------------------------------------------------------------------------

# What an offline evidence build reads besides the committed source.
INPUTS_RECIPE = "nix/evidence-inputs.nix"
# Recipe sets and locks whose inputs this machine has realised. Local state, as the
# reproducibility record is: a build that lacks an input fails; nothing trusts this file.
PREPARED = "target/nix-evidence/prepared.json"
PREPARE = "python3 scripts/evidence_platforms.py prepare"
# The daemon's configuration is root-owned so that no unprivileged process, earning
# included, can change the sandbox a build ran in; only root can make it comply.
SANDBOX_REMEDY = ("printf 'sandbox = true\\nsandbox-fallback = false\\n' | sudo tee -a /etc/nix/nix.conf "
                  "&& sudo systemctl restart nix-daemon")


def _inputs_key(root: Path, platform: Platform) -> str:
    digest = hashlib.sha256(platform.target.encode() + b"\0")
    for rel in dict.fromkeys((*platform.recipes, *platform.release_recipes, INPUTS_RECIPE, platform.registry,
                              "Cargo.lock")):
        digest.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def prepared(root: Path, platform: Platform) -> bool:
    """Whether this machine has prepared `platform`'s current recipes, registry and lock."""
    try:
        return _inputs_key(root, platform) in json.loads((root / PREPARED).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False


def verify_sandbox(platform: Platform) -> None:
    """Check the local daemon against `platform`'s sandbox rule."""
    if platform is DARWIN:
        darwin_policy._verify_sandbox()
    else:
        _verify_sandbox(platform)


def sandbox_refusal(platform: Platform, error: ValueError) -> str:
    """Why this host's daemon cannot earn, and what resolves it."""
    cause = f" ({error.__cause__})" if error.__cause__ else ""
    # Two missing settings are one appended line each; anything else is the host's to undo.
    fix = f":\n  {SANDBOX_REMEDY}\nthen" if "must explicitly set" in str(error) else "; then"
    return (f"evidence cannot be earned on this host: {error}{cause}. /etc/nix/nix.conf must be a root-owned "
            "file with no include line that sets `sandbox = true` and `sandbox-fallback = false`, and the "
            f"daemon may expose no host path but {', '.join(sorted(platform.sandbox_paths))}. Only root can "
            f"change the daemon's configuration{fix} run `{PREPARE}`.")


def uncovered_units(root: Path, platform: Platform, graphs: list[dict]) -> list[str]:
    """The compile-time units of `graphs` the platform's registry does not license, sorted.

    A unit with no row of its exact identity and package checksum, and a row still
    `unresolved`, are both unlicensed: a build that selects either earns nothing."""
    registry, lock = _registry(root, platform)
    checksums = _checksums(lock)
    rows = {_signature(row): row for row in registry["units"]}
    found = set()
    for graph in graphs:
        for unit in _compile_time_units(graph, require_compile_time=False):
            signature = _signature(unit)
            row = rows.get(signature)
            if row is None or row.get("checksum") != checksums.get(signature[:2]):
                found.add(f"{signature[0]} {signature[1]} {signature[2]} (not in the registry)")
            elif row.get("source_projection") not in PROJECTION_STATES - {"unresolved"}:
                found.add(f"{signature[0]} {signature[1]} {signature[2]} (unresolved)")
    return sorted(found)


def prepare_host(root: Path, *, run=subprocess.run, graphs=None) -> Platform:
    """Make this host ready to earn, or refuse naming the command that resolves it.

    Checks the tree's policy for the host, the daemon's sandbox rule and that the
    reviewed registry licenses every compile-time unit this host's Cargo resolves,
    then realises the build inputs, online and with the daemon's trusted
    substituters. Nothing here is an input of a record: an evidence build evaluates
    its own derivations offline and validates its own emitted unit graph."""
    platform = earning_platform(root)
    try:
        verify_sandbox(platform)
    except ValueError as error:
        raise UnreviewedPlatform(sandbox_refusal(platform, error)) from error
    try:
        if graphs is None:
            graphs = _evidence_graphs(root, platform)
    except subprocess.CalledProcessError as error:
        tail = "\n".join((error.stderr or "").strip().splitlines()[-5:])
        raise UnreviewedPlatform(
            f"Cargo could not resolve this host's unit graphs offline:\n{tail}\n`cargo fetch --locked` downloads "
            f"the locked crates; then run `{PREPARE}`.") from error
    try:
        uncovered = uncovered_units(root, platform, graphs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UnreviewedPlatform(
            f"{platform.registry} is not bound to the current Cargo.lock ({error}); "
            "`python3 scripts/evidence_platforms.py draft-registry --all` rebinds every platform's registry "
            f"from any host. Commit it, then run `{PREPARE}`.") from error
    if uncovered:
        raise UnreviewedPlatform(
            f"{len(uncovered)} compile-time unit(s) of this host's build are not reviewed in {platform.registry}:\n  "
            + "\n  ".join(uncovered) + "\nWhat a build script or procedural macro reads is a reviewer's "
            "finding, not something a build can measure. `python3 scripts/evidence_platforms.py draft-registry` "
            "adds each as `unresolved`; set its `source_projection` once its source is read "
            "(nix/linux/compile-time-audit.md), commit the registry, then run "
            f"`{PREPARE}`.")
    command = ["nix", "--extra-experimental-features", "nix-command flakes", "build", "--no-link",
               "--file", str(root / INPUTS_RECIPE)]
    print(f"{platform.target}: realising the evidence build inputs ({shlex.join(command[3:])})", flush=True)
    if run(command, cwd=root).returncode != 0:
        raise UnreviewedPlatform(
            f"the evidence build inputs could not be realised on {platform.target} (the build's output is "
            f"above); once `nix build --file {INPUTS_RECIPE}` succeeds, run `{PREPARE}`.")
    path = root / PREPARED
    try:
        done = set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        done = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done | {_inputs_key(root, platform)})) + "\n", encoding="utf-8")
    return platform


def receipt_platform(receipt: dict) -> Platform:
    """The platform a Nix build receipt was earned on.

    `policy.target` identifies it; the policy id and the compiler's host triple
    must be that platform's."""
    policy, tools = receipt["policy"], receipt["tools"]
    platform = PLATFORMS.get(policy["target"])
    if platform is None or policy["id"] != platform.policy or tools["host"] != platform.target:
        raise ValueError("receipt policy, target and compiler host are not one reviewed platform")
    return platform


def manifest_platform(manifest: object) -> str | None:
    """The host triple a stored manifest was earned on; `None` when it does not say."""
    try:
        return receipt_platform(manifest["nix_build"]).target
    except (KeyError, TypeError, ValueError):
        return None


@functools.cache
def stored_platform(root: Path, ref: object) -> str | None:
    """`manifest_platform` of the manifest stored under `ref` (content-addressed)."""
    if not isinstance(ref, str) or not re.fullmatch(r"[0-9a-f]{64}", ref):
        return None
    try:
        return manifest_platform(json.loads((root / "parity/evidence_manifests" / f"{ref}.json").read_bytes()))
    except (OSError, ValueError):
        return None


def replay_refusal(earned: str | None, host: str | None = None) -> str | None:
    """Why this host cannot replay the digests of a record earned on `earned`; `None` when it can.

    Replicate digests are bit-exact and depend on the platform math library, so a
    different result on another platform says nothing about the code."""
    host = host or host_target()
    if earned is None or earned == host:
        return None
    return f"earned on {earned}; this host ({host}) cannot replay its digests"


def recipes_named(names: object) -> tuple[str, ...]:
    """The reviewed build recipe set with exactly these files, its recipe first."""
    for platform in PLATFORMS.values():
        if set(platform.recipes) == set(names):
            return platform.recipes
    raise ValueError("recipe files are not one reviewed platform's evidence build")


# --------------------------------------------------------------------------
# Policy checks. The Apple-silicon platform delegates to `nix_evidence_policy`.
# --------------------------------------------------------------------------


def _verify_sandbox(platform: Platform) -> None:
    """Check the local daemon builds in a sandbox that exposes only the reviewed host paths."""
    overrides = sorted(NIX_OVERRIDES.intersection(os.environ))
    if overrides:
        raise ValueError("unreviewed Nix configuration override: " + ", ".join(overrides))
    try:
        # NixOS links the system configuration into the store; resolve it, then
        # hold the file itself to the same ownership rule.
        config_file = SYSTEM_CONFIG.resolve(strict=True)
        if SYSTEM_CONFIG.is_symlink() and Path("/nix/store") not in config_file.parents:
            raise ValueError("Nix system configuration links outside the store")
        stat = config_file.stat()
        if not config_file.is_file() or stat.st_uid != 0 or stat.st_mode & 0o002:
            raise ValueError("Nix system configuration is not a root-owned, protected file")
        settings: dict[str, list[str]] = {}
        for raw in config_file.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
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
        # `true` only: `relaxed` lets a derivation opt out of the sandbox.
        if config["sandbox"]["value"] is not True or config["sandbox-fallback"]["value"] is not False:
            raise ValueError("Nix client sandbox settings disagree with the reviewed daemon policy")
        paths = config["sandbox-paths"]["value"]
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("Nix exposes an unreviewed sandbox path")
        for entry in paths:
            # `<path in the build>=<host path>`; a reviewed path may only map to the store.
            inside, _, outside = entry.rstrip("?").partition("=")
            if inside not in platform.sandbox_paths or (outside and not outside.startswith("/nix/store/")):
                raise ValueError("Nix exposes an unreviewed sandbox path")
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cannot verify local Nix sandbox configuration") from error


def _registry(root: Path, platform: Platform) -> tuple[dict, bytes]:
    """The platform's registry, checked against its policy, target and the current lock."""
    registry = json.loads((root / platform.registry).read_bytes())
    lock = (root / "Cargo.lock").read_bytes()
    if (registry.get("schema") != 1 or registry.get("policy") != platform.policy
            or registry.get("target") != platform.target or registry.get("cargo_lock_sha256") != _sha(lock)):
        raise ValueError("Nix compile-time audit is stale or unrecognized")
    return registry, lock


def _checksums(lock: bytes) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for package in tomllib.loads(lock.decode("utf-8"))["package"]:
        if not package.get("source", "").startswith("registry+"):
            continue
        key = package["name"], package["version"]
        if key in found or not re.fullmatch(r"[0-9a-f]{64}", package.get("checksum", "")):
            raise ValueError("ambiguous or unlocked registry package")
        found[key] = package["checksum"]
    return found


def _compile_time_units(graph: dict, *, require_compile_time: bool = True) -> list[dict]:
    """Every build-script and procedural-macro unit of `graph`, and all they depend on."""
    units, roots = graph["units"], graph["roots"]
    if (not isinstance(units, list) or not units or not isinstance(roots, list) or not roots
            or any(type(i) is not int or i < 0 or i >= len(units) for i in roots)):
        raise ValueError("invalid Nix compiler graph")
    todo = []
    for i, unit in enumerate(units):
        kinds = unit.get("target_kind")
        if not isinstance(kinds, list):
            raise ValueError("invalid Nix compiler unit kinds")
        if unit.get("mode") == "run-custom-build" or "proc-macro" in kinds:
            todo.append(i)
    if not todo and require_compile_time:
        raise ValueError("Nix compiler graph has no reviewed compile-time units")
    seen: set[int] = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        dependencies = units[i].get("dependencies")
        if not isinstance(dependencies, list):
            raise ValueError("invalid Nix compiler unit dependencies")
        for dependency in dependencies:
            index = dependency.get("index") if isinstance(dependency, dict) else None
            if type(index) is not int or index < 0 or index >= len(units):
                raise ValueError("invalid Nix compiler dependency index")
            todo.append(index)
    return [units[i] for i in sorted(seen)]


def validate_units(root: Path, graph: dict, *, require_compile_time: bool = True,
                   platform: Platform | None = None) -> dict:
    """Check every compile-time unit of `graph` against the platform's reviewed registry.

    The result is the receipt's `policy`; its `id` and `target` name the platform."""
    platform = platform or host_platform()
    if platform is DARWIN:
        return darwin_policy.validate_units(root, graph, require_compile_time=require_compile_time)
    root = root.resolve()
    try:
        registry, lock = _registry(root, platform)
        checksums = _checksums(lock)
        approved: dict[tuple, list[dict]] = {}
        for row in registry["units"]:
            approved.setdefault(_signature(row), []).append(row)
        selected: dict[str, dict] = {}
        covered = _compile_time_units(graph, require_compile_time=require_compile_time)
        for unit in covered:
            signature = _signature(unit)
            rows = approved.get(signature, [])
            if len(rows) != 1 or checksums.get(signature[:2]) != rows[0].get("checksum"):
                raise ValueError(f"unreviewed Nix compile-time unit: {signature[0]} {signature[1]} {signature[2]}")
            selected[_sha(_canonical(signature))] = rows[0]
        states = {row.get("source_projection") for row in selected.values()}
        if not states <= PROJECTION_STATES:
            raise ValueError("invalid Nix source projection review status")
        return {"id": platform.policy, "cargo_lock_sha256": _sha(lock), "target": platform.target,
                "covered_units": len(covered), "source_projection": _overall(states),
                "source_projection_allowed": states == {"reviewed-safe"},
                "selected_unit_ids": sorted(selected),
                "audit_selection_sha256": _selection_digest(registry, selected)}
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("invalid or incomplete Nix build policy input") from error


def validate_build(root: Path, built: dict) -> dict:
    """Validate a build made on this host against this host's policy.

    As `nix_evidence_policy.validate_build`: the caller independently validates the
    source, target, build provenance and executed artifact."""
    platform = host_platform()
    if built["tools"]["host"] != platform.target:
        raise ValueError(f"the build's compiler host is not this host ({platform.target})")
    if platform is DARWIN:
        return darwin_policy.validate_build(root, built)
    _verify_sandbox(platform)
    return validate_units(root, built["compiler_graph"], platform=platform)


def recheck_selection(root: Path, receipt: dict) -> None:
    """Recheck a receipt's selected audit rows against the registry of the platform it names.

    Reads the repository only, so it gives the same answer on every host."""
    platform = PLATFORMS.get(receipt.get("target")) if isinstance(receipt, dict) else None
    if platform is None or platform is DARWIN:
        # An unknown target is refused there, as before.
        return darwin_policy.recheck_selection(root, receipt)
    try:
        registry, _ = _registry(root, platform)
        ids = receipt["selected_unit_ids"]
        if (receipt.get("id") != platform.policy or not isinstance(ids, list) or not ids
                or len(ids) != len(set(ids))
                or not all(isinstance(i, str) and re.fullmatch(r"[0-9a-f]{64}", i) for i in ids)):
            raise ValueError("invalid selected Nix audit identity")
        selected: dict[str, dict] = {}
        for row in registry["units"]:
            identity = _sha(_canonical(_signature(row)))
            if identity in ids:
                if identity in selected:
                    raise ValueError("duplicate selected Nix audit unit")
                selected[identity] = row
        if set(selected) != set(ids) or receipt.get("audit_selection_sha256") != _selection_digest(registry, selected):
            raise ValueError("selected Nix compile-time audit changed")
        states = {row.get("source_projection") for row in selected.values()}
        if (not states <= PROJECTION_STATES or receipt.get("source_projection") != _overall(states)
                or receipt.get("source_projection_allowed") is not (states == {"reviewed-safe"})):
            raise ValueError("Nix source projection review status changed")
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid or incomplete selected Nix audit") from error


# --------------------------------------------------------------------------
# Keeping a registry bound to Cargo.lock.
# --------------------------------------------------------------------------


def draft_registry(platform: Platform, lock: bytes, graphs: list[dict], existing: dict | None = None) -> dict:
    """`existing` rebound to `lock`, with a row for every compile-time unit of `graphs`.

    A row of `existing` whose unit and package checksum are unchanged keeps its
    review, and a reviewed row no graph selects is kept as it is: records that
    selected it are rechecked against it. Every new unit is `unresolved`, which
    earning refuses until a reviewer has read what the unit reads on this platform
    and set its state."""
    checksums = _checksums(lock)
    kept = {}
    for row in (existing or {}).get("units", []):
        kept[_signature(row)] = row
    rows: dict[tuple, dict] = {signature: row for signature, row in kept.items()
                               if row.get("source_projection") != "unresolved"}
    for graph in graphs:
        for unit in _compile_time_units(graph, require_compile_time=False):
            signature = _signature(unit)
            checksum = checksums.get(signature[:2])  # None for a workspace package
            previous = kept.get(signature)
            if previous is not None and previous.get("checksum") == checksum:
                rows[signature] = previous
            else:
                rows[signature] = {**{k: unit[k] for k in UNIT_KEYS}, "checksum": checksum,
                                   "source_projection": "unresolved"}
    return {"schema": 1, "policy": platform.policy, "target": platform.target,
            "cargo_lock_sha256": _sha(lock), "scope": (existing or {}).get("scope", platform.scope),
            "audit_documents": (existing or {}).get("audit_documents", []),
            "units": [rows[key] for key in sorted(rows, key=lambda s: json.dumps(s))]}


@contextlib.contextmanager
def _resolving_as(target: str):
    """Have Cargo resolve unit graphs as a `target` host would.

    Cargo takes the host triple and its cfg from `rustc`, so a wrapper that reports
    `target` as the host and answers every query for it yields the graph Cargo
    builds natively there. Only metadata is asked for; nothing is compiled, so the
    target's standard library need not be installed."""
    if target == host_target():
        yield
        return
    rustc = shutil.which("rustc")
    if rustc is None:
        raise ValueError("rustc is not on PATH")
    with tempfile.TemporaryDirectory(prefix="antecedent-host-as-") as directory:
        wrapper = Path(directory) / "rustc"
        wrapper.write_text(
            "#!/bin/sh\n"
            f'for a; do [ "$a" = "-vV" ] && {{ {shlex.quote(rustc)} -vV | sed "s/^host: .*/host: {target}/"; exit; }}; done\n'
            f'exec {shlex.quote(rustc)} "$@" --target {target}\n', encoding="utf-8")
        wrapper.chmod(0o755)
        previous = os.environ.get("RUSTC")
        os.environ["RUSTC"] = str(wrapper)
        try:
            yield
        finally:
            if previous is None:
                del os.environ["RUSTC"]
            else:
                os.environ["RUSTC"] = previous


def _evidence_graphs(root: Path, platform: Platform | None = None) -> list[dict]:
    """The compiler graph, as `platform`'s host resolves it (this host's by default), of
    every scientific and calibration target and of the extension."""
    import calibration_runner
    import scientific_records
    targets = {(info["crate"], tuple(info["target"]))
               for info in (scientific_records.locate(root, row) for row in scientific_records.all_scientific_rows(root))
               if info["lang"] == "rust"}
    for group in calibration_runner.load_groups(root):
        crate, kind, name = group.build_target
        targets.add((crate, ("--lib",) if kind == "lib" else ("--test", name)))
    with _resolving_as(platform.target if platform else host_target()):
        graphs = [darwin_policy.host_compiler_graph(root, crate, list(target)) for crate, target in sorted(targets)]
        return [*graphs, darwin_policy.host_compiler_graph(root, "", ["python"])]


def _status(root: Path, platform: Platform) -> int:
    gaps = bring_up_gaps(root, platform)
    print(f"{platform.target}: policy {platform.policy}, Nix system {platform.nix_system}, "
          f"registry {platform.registry}")
    for gap in gaps:
        print(f"  tree: {gap}")
    try:
        verify_sandbox(platform)
    except ValueError as error:
        gaps.append(str(error))
        print("  " + sandbox_refusal(platform, error).replace("\n", "\n  "))
    if gaps:
        print("records cannot be earned here")
    elif platform.prepares and not prepared(root, platform):
        print(f"records can be earned here; the first earning run realises the build inputs (as `{PREPARE}` does)")
    else:
        print("records can be earned here")
    return 1 if gaps else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from _repo import REPO_ROOT as root
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "prepare", "draft-registry"))
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--platform", choices=sorted(PLATFORMS), help="draft-registry: the platform (default: this host's)")
    which.add_argument("--all", action="store_true", help="draft-registry: every platform")
    args = parser.parse_args(argv)
    if args.command == "draft-registry":
        try:
            chosen = (list(PLATFORMS.values()) if args.all
                      else [PLATFORMS[args.platform] if args.platform else host_platform()])
        except UnreviewedPlatform as error:
            print(error)
            return 1
        lock = (root / "Cargo.lock").read_bytes()
        unresolved = 0
        for platform in chosen:
            path = root / platform.registry
            existing = json.loads(path.read_bytes()) if path.is_file() else None
            draft = draft_registry(platform, lock, _evidence_graphs(root, platform), existing)
            if draft != existing:
                path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
            pending = sum(1 for row in draft["units"] if row["source_projection"] == "unresolved")
            unresolved += pending
            print(f"{platform.registry}: {len(draft['units'])} unit(s), {pending} to review"
                  + ("" if draft != existing else " (unchanged)"))
        return 1 if unresolved else 0
    try:
        if args.command == "prepare":
            platform = prepare_host(root)
            print(f"{platform.target}: records can be earned here")
            return 0
        return _status(root, host_platform())
    except UnreviewedPlatform as error:
        print(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
