#!/usr/bin/env python3
"""Currency checker for content-addressed evidence dependency manifests.

Scientific passes, calibration measurements and probe attestations reference a
manifest in `parity/evidence_manifests/<sha256>.json`. A record is current when
its manifest binds exactly that record, was written under the current analyzer
policy and compiler pin, and every input still matches. Inputs are hashed once
per invocation. Directory membership is every file git does not ignore, and
environment inputs are compared with the canonical build environment, never
the environment of the process running the check. No compiler, Cargo metadata,
network or persistent cache is used.

    python3 scripts/evidence_manifests.py [--kind all|scientific|calibration|probe]

prints each record's status and reasons, the input-to-record index and timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

import _repo

ROOT = Path(__file__).resolve().parents[1]
STORE = "parity/evidence_manifests"
SCHEMA = 1
NORMALIZATION = "raw-bytes-v1"
# The dependency policy that writes manifests. Changing what earning records,
# or how, renames it, which owes every manifest written under the old policy.
ANALYZER = "conservative-source-v2"
HEX = re.compile(r"[0-9a-f]{64}\Z")
KINDS = ("file", "tree", "env", "guarded_file", "membership", "evidence_file", "glob", "build_env")
ROLES = ("execution", "compile", "selection", "external", "identity")
# The build configuration evidence is earned and released under. Earning refuses
# any other value; lint levels are not configuration and are ignored.
BUILD_ENV = {
    "RUSTFLAGS": "",
    "CARGO_ENCODED_RUSTFLAGS": "",
    "CARGO_BUILD_TARGET": "",
    "RUSTC": "rustc",
    "RUSTC_BOOTSTRAP": "",
    "CARGO_PROFILE_TEST_OPT_LEVEL": "2",
    "CARGO_PROFILE_TEST_DEBUG": "line-tables-only",
}
LINT_FLAG = re.compile(r"-[DWAF]\S+|--(?:deny|warn|allow|forbid|force-warn)=\S+")
LINT_OPTION = {"-D", "-W", "-A", "-F", "--deny", "--warn", "--allow", "--forbid", "--force-warn"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_id(manifest: dict) -> str:
    return digest(canonical(manifest))


def without_lints(flags: str, separator: str | None = None) -> str:
    tokens = flags.split(separator) if separator else flags.split()
    kept, skip = [], False
    for token in tokens:
        if skip:
            skip = False
        elif token in LINT_OPTION:
            skip = True
        elif token and not LINT_FLAG.fullmatch(token):
            kept.append(token)
    return (separator or " ").join(kept)


def effective_build_env(environ) -> dict[str, str]:
    values = {name: environ.get(name, default) for name, default in BUILD_ENV.items()}
    values["RUSTFLAGS"] = without_lints(values["RUSTFLAGS"])
    values["CARGO_ENCODED_RUSTFLAGS"] = without_lints(values["CARGO_ENCODED_RUSTFLAGS"], "\x1f")
    return values


def env_digest(environ, name: str) -> str:
    return digest(canonical({"present": name in environ, "value": environ.get(name)}))


def noncanonical_environment(environ, names) -> list[str]:
    """Variables whose value in `environ` differs from the canonical environment."""
    build = effective_build_env(environ)
    wrong = [f"{n}={build[n]!r}" for n, default in BUILD_ENV.items() if build[n] != default]
    wrong += [f"{n}={environ[n]!r}" for n in sorted(set(names)) if n in environ]
    return wrong


class InvalidManifest(ValueError):
    pass


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise InvalidManifest("invalid repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("..", ".") for p in value.split("/")):
        raise InvalidManifest("path must be normalized and inside the repository")
    if str(path) != value:
        raise InvalidManifest("path must be canonical")
    return value


def validate(manifest: dict, binding: dict) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema")) is not int
            or manifest["schema"] != SCHEMA):
        raise InvalidManifest("unsupported manifest schema")
    if manifest.get("normalization") != NORMALIZATION:
        raise InvalidManifest("unsupported normalization")
    if manifest.get("record") != binding:
        raise InvalidManifest("manifest does not bind this exact evidence record")
    identity = manifest.get("compiler")
    if not isinstance(identity, dict) or not all(
        isinstance(identity.get(k), str) and identity[k] for k in ("release", "commit")
    ):
        raise InvalidManifest("missing compiler identity")
    if not isinstance(manifest.get("analyzer"), str):
        raise InvalidManifest("missing analyzer identity")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise InvalidManifest("empty dependency manifest")
    seen = set()
    for entry in inputs:
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            raise InvalidManifest("unsupported dependency kind")
        kind = entry["kind"]
        key = entry.get("name") if kind in ("env", "build_env") else relative_path(entry.get("path"))
        if not isinstance(key, str) or not key:
            raise InvalidManifest("missing input name")
        if (kind, key) in seen:
            raise InvalidManifest("duplicate dependency")
        seen.add((kind, key))
        if not isinstance(entry.get("digest"), str) or not HEX.fullmatch(entry["digest"]):
            raise InvalidManifest("invalid input digest")
        if kind == "guarded_file":
            from evidence_projection import fingerprint
            if not isinstance(entry.get("projection"), dict) or fingerprint(entry["projection"]) != entry["digest"]:
                raise InvalidManifest("invalid guarded source projection")
        if kind == "build_env" and key not in BUILD_ENV:
            raise InvalidManifest("unknown build environment input")
        if entry.get("role") not in ROLES:
            raise InvalidManifest("missing dependency role")
    fallbacks = manifest.get("fallbacks")
    if not isinstance(fallbacks, list):
        raise InvalidManifest("missing fallback declarations")
    for fallback in fallbacks:
        if not isinstance(fallback, dict) or not isinstance(fallback.get("reason"), str) or not fallback["reason"]:
            raise InvalidManifest("invalid fallback explanation")
        path = relative_path(fallback.get("path"))
        if ("tree", path) not in seen and ("membership", path) not in seen:
            raise InvalidManifest("fallback boundary must have a tree dependency")
    if "nix_build" in manifest:
        validate_nix_receipt(manifest["nix_build"], binding, inputs)


def validate_nix_receipt(receipt: object, binding: dict, inputs: list[dict]) -> None:
    """Validate portable Nix build identity; execution is attested by the earning runner."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "nix-build-v1":
        raise InvalidManifest("Nix build provenance is not yet supported")
    try:
        artifact, source, policy, tools = (receipt[k] for k in ("artifact", "source", "policy", "tools"))
        if binding.get("kind") != "scientific" or receipt.get("execution") != "exact-binary":
            raise ValueError("invalid execution binding")
        crate, target, _ = binding["key"].split("::", 2)
        expected = "lib" if artifact["target_kind"] == "lib" else artifact["target_name"]
        if artifact["crate"] != crate or target != expected or artifact["target_kind"] not in ("lib", "test"):
            raise ValueError("target mismatch")
        # One shared derivation builds every owed target; each artifact names its own executable.
        name = artifact["target_name"] if artifact["target_kind"] == "test" else "lib"
        if (artifact.get("schema") != 2
                or artifact["binary"] != f"targets/{crate}/{artifact['target_kind']}/{name}/bin/test"
                or not any(isinstance(t, dict) and (t.get("crate"), t.get("kind"), t.get("name"))
                           == (crate, artifact["target_kind"], artifact["target_name"])
                           for t in artifact["build_targets"])):
            raise ValueError("target outside its evidence build")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["commit"]):
            raise ValueError("invalid source commit")
        if artifact["source"] != source["store_path"] or artifact["source_nar_hash"] != source["nar_hash"]:
            raise ValueError("source mismatch")
        for key in ("source_nar_hash", "output_nar_hash"):
            if not re.fullmatch(r"sha256-[A-Za-z0-9+/]{43}=", artifact[key]):
                raise ValueError("invalid NAR identity")
        if not HEX.fullmatch(artifact["binary_sha256"]) or not HEX.fullmatch(receipt["compiler_graph_sha256"]):
            raise ValueError("invalid content identity")
        if not all(isinstance(tools[k], str) and tools[k] for k in ("rustc", "rustc_commit", "cargo")):
            raise ValueError("missing tools")
        if (policy["id"] != "nix-darwin-private-tmp-audit-v1"
                or policy.get("source_projection_allowed") is not True
                or not HEX.fullmatch(policy["audit_selection_sha256"])):
            raise ValueError("invalid policy")
        tracked = {i.get("path"): i.get("digest") for i in inputs if i.get("kind") == "evidence_file"}
        required = dict(artifact["recipe_sha256"])
        if set(required) != {"nix/evidence-target.nix", "nix/evidence-toolchain.nix", "flake.lock"}:
            raise ValueError("incomplete recipe identity")
        if not required or any(tracked.get(path) != value for path, value in required.items()):
            raise ValueError("unbound recipe or audit registry")
    except (AttributeError, KeyError, TypeError, ValueError):
        raise InvalidManifest("invalid Nix build provenance") from None


def tracked_files(root: Path) -> list[str] | None:
    """Files git does not ignore (tracked, and untracked but not ignored) present in the work
    tree, or `None` outside a git work tree. Ignored build products and host files are never
    inputs; a new untracked file in an input directory is a change. Earning refuses untracked
    inputs, so a manifest only ever records tracked files."""
    if not (root / ".git").exists():
        return None
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=root, capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(p for p in set(listed.split("\0")) if p and (root / p).is_file())


class Checker:
    """One read and digest per input, shared by every evidence kind."""

    def __init__(self, root: Path, environ: dict[str, str] | None = None):
        self.root = root.resolve()
        # Currency is judged against the canonical environment, not this process's.
        self.environ = {} if environ is None else environ
        self.cache: dict[tuple[str, str], str] = {}
        self.manifests: dict[str, dict] = {}
        self.reverse: dict[str, set[str]] = defaultdict(set)
        self.started = time.perf_counter()
        self.parsing_seconds = 0.0
        self.hashing_seconds = 0.0
        self.bytes_hashed = 0
        self.files_hashed = 0
        self.source_cache: dict[str, tuple[str, str]] = {}
        self.projection_cache: dict[tuple[str, str], bool] = {}
        self._tracked: list[str] | None = None
        self._tracked_loaded = False

    @property
    def pin(self) -> dict:
        if not hasattr(self, "_pin"):
            self._pin = _repo.load_toml(self.root / "tools/evidence-analyzer/pin.toml")
        return self._pin

    def path(self, rel: str) -> Path:
        path = self.root / relative_path(rel)
        # Symlinks are rejected, including in parents: a tree guard must not
        # silently omit a linked directory or depend on external host content.
        current = self.root
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                raise InvalidManifest(f"symlink input is unsupported: {rel}")
        return path

    def tracked(self) -> list[str] | None:
        if not self._tracked_loaded:
            self._tracked, self._tracked_loaded = tracked_files(self.root), True
        return self._tracked

    def members(self, rel: str) -> list[str]:
        """Files under a directory: those git does not ignore in a git work tree, else every file."""
        from evidence_fingerprint import SKIP_DIRS
        directory = self.path(rel)
        if not directory.is_dir():
            raise InvalidManifest(f"missing source directory: {rel}")
        tracked = self.tracked()
        if tracked is not None:
            prefix = rel + "/"
            found = [p for p in tracked if p.startswith(prefix)]
        else:
            found = []
            for parent, dirs, files in os.walk(directory, followlinks=False):
                dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
                found += [(Path(parent) / name).relative_to(self.root).as_posix() for name in files]
        out = []
        for p in sorted(found):
            if set(PurePosixPath(p).parts) & SKIP_DIRS or p.endswith((".pyc", ".pyo")):
                continue
            self.path(p)
            out.append(p)
        return out

    def input_digest(self, kind: str, key: str) -> str:
        cache_key = kind, key
        if cache_key in self.cache:
            return self.cache[cache_key]
        start = time.perf_counter()
        if kind == "env":
            value = env_digest(self.environ, key)
        elif kind == "build_env":
            value = digest(canonical(effective_build_env(self.environ)[key]))
        elif kind == "file":
            data = self.path(key).read_bytes()
            value = digest(data)
            self.bytes_hashed += len(data)
            self.files_hashed += 1
        elif kind == "evidence_file":
            from evidence_fingerprint import DERIVED_STATUS_FIELDS, file_digest
            path = self.path(key)
            if path.is_dir():
                value = digest(b"directory-input")
            elif not path.is_file():
                value = digest(b"missing-input")
            elif key in DERIVED_STATUS_FIELDS:
                value = file_digest(path, self.root)
            else:
                value = self.input_digest("file", key)
        elif kind == "membership":
            value = digest(canonical(self.members(key)))
        elif kind == "glob":
            relative_path(key)
            tracked = self.tracked()
            if tracked is not None:
                listed = set(tracked) | {str(parent) for p in tracked for parent in PurePosixPath(p).parents}
            members = []
            for path in self.root.glob(key):
                rel = path.relative_to(self.root).as_posix()
                if tracked is not None and rel not in listed:
                    continue
                self.path(rel)
                members.append(rel)
            value = digest(canonical(sorted(members)))
        elif kind == "tree":
            # Sorted membership detects added, removed and renamed inputs.
            value = digest(canonical([(rel, self.input_digest("file", rel)) for rel in self.members(key)]))
        else:
            raise InvalidManifest("unsupported dependency kind")
        # Tree timing includes children; account only leaf time here.
        if kind != "tree":
            self.hashing_seconds += time.perf_counter() - start
        self.cache[cache_key] = value
        return value

    def load(self, ref: object) -> dict:
        if not isinstance(ref, str) or not HEX.fullmatch(ref):
            raise InvalidManifest("missing or invalid dependency_manifest reference")
        if ref not in self.manifests:
            start = time.perf_counter()
            manifest = json.loads(self.path(f"{STORE}/{ref}.json").read_bytes())
            if manifest_id(manifest) != ref:
                raise InvalidManifest("manifest content does not match its address")
            self.manifests[ref] = manifest
            self.parsing_seconds += time.perf_counter() - start
        return self.manifests[ref]

    def check(self, binding: dict, ref: object) -> dict:
        label = f"{binding['kind']}:{binding['key']}"
        result = {"record": label, "current": False, "reasons": []}
        try:
            manifest = self.load(ref)
            validate(manifest, binding)
            reasons = result["reasons"]
            if manifest["compiler"] != {k: self.pin[k] for k in ("release", "commit")}:
                reasons.append("compiler_identity_changed")
            if manifest["analyzer"] != ANALYZER:
                reasons.append("analyzer_policy_changed")
            if "nix_build" in manifest:
                from nix_evidence_policy import recheck_selection
                recheck_selection(self.root, manifest["nix_build"]["policy"])
            for entry in manifest["inputs"]:
                kind = entry["kind"]
                key = entry["name"] if kind in ("env", "build_env") else entry["path"]
                self.reverse[f"{kind}:{key}"].add(label)
                if kind == "guarded_file":
                    from _rust_source import lex
                    from evidence_projection import matches
                    cache_key = key, entry["digest"]
                    if cache_key not in self.projection_cache:
                        start = time.perf_counter()
                        if key not in self.source_cache:
                            source = self.path(key).read_bytes().decode("utf-8")
                            self.source_cache[key] = source, lex(source).skeleton
                        source, skeleton = self.source_cache[key]
                        self.projection_cache[cache_key] = matches(source, entry["projection"], skeleton)
                        self.parsing_seconds += time.perf_counter() - start
                    unchanged = self.projection_cache[cache_key]
                else:
                    unchanged = self.input_digest(kind, key) == entry["digest"]
                if not unchanged:
                    reasons.append(f"input_changed:{kind}:{key}")
            result["current"] = not reasons
        except (OSError, ValueError, KeyError, TypeError, RecursionError, subprocess.CalledProcessError) as error:
            result["reasons"].append(f"manifest_owed:{error}")
        return result

    def report(self, records: list[tuple[dict, object]]) -> dict:
        results = [self.check(binding, ref) for binding, ref in records]
        elapsed = time.perf_counter() - self.started
        return {
            "records": results,
            "counts": {"total": len(results), "current": sum(r["current"] for r in results)},
            "inputs_to_records": {k: sorted(v) for k, v in sorted(self.reverse.items())},
            "timings": {"total_seconds": elapsed, "parsing_seconds": self.parsing_seconds,
                        "hashing_seconds": self.hashing_seconds,
                        "other_seconds": max(0, elapsed - self.parsing_seconds - self.hashing_seconds),
                        "bytes_hashed": self.bytes_hashed, "files_hashed": self.files_hashed},
        }


def inventory(root: Path, kind: str = "all") -> list[tuple[dict, object]]:
    from evidence_earning import binding, measurement_binding, probe_binding
    records = []
    for category, filename, table in (
        ("scientific", "scientific_records.toml", "record"),
        ("calibration", "coverage_records.toml", "record"),
        ("probe", "calibration_probe_attestations.toml", "attestation"),
    ):
        if kind != "all" and category != kind:
            continue
        path = root / "parity" / filename
        rows = _repo.load_toml(path).get(table, []) if path.exists() else []
        for row in rows:
            if category == "scientific":
                bound = binding("scientific", row.get("test"), row.get("fingerprint"))
            elif category == "calibration":
                bound = measurement_binding(row)
            else:
                bound = probe_binding(row)
            records.append((bound, row.get("dependency_manifest")))
    return records


def report(root: Path = ROOT, kind: str = "all") -> dict:
    return Checker(root).report(inventory(root, kind))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("all", "scientific", "calibration", "probe"), default="all")
    args = parser.parse_args()
    print(json.dumps(report(kind=args.kind), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
