#!/usr/bin/env python3
"""Currency checker for content-addressed evidence dependency manifests.

Scientific passes, calibration measurements and probe attestations reference a
standing pass (`claim_passes`) or a content-addressed manifest in the object store
(`evidence_objects`). A record is current when
its manifest binds exactly that record, was written under an accepted analyzer
policy and the compiler pin, and every input still matches. A build recipe is an
input by what it says (`evidence_recipes`), not by its bytes. Inputs are hashed once
per invocation. Directory membership is every file git does not ignore, and
environment inputs are compared with the canonical build environment, never
the environment of the process running the check. No compiler, Cargo metadata,
network or persistent cache is used.

    python3 scripts/evidence_manifests.py [--kind all|scientific|calibration|probe]

prints each record's status and reasons, the input-to-record index and timings.
"""
from __future__ import annotations

import argparse
import functools
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
from evidence_objects import OBJECTS, CorruptObject, Decoded, Store, decode, encode, encode_object, reachable

class ShallowClone(RuntimeError):
    """Currency cannot be judged in a clone that lacks the commits records were built at."""


def is_shallow_clone(root: Path) -> bool:
    """Whether `root` is the top of a shallow clone: its common git directory holds `shallow`.
    Read from the filesystem, so constructing a checker starts no subprocess."""
    dot = root / ".git"
    if dot.is_dir():
        gitdir = dot
    elif dot.is_file():  # a linked worktree or submodule: `gitdir: <dir>`
        text = dot.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"):
            return False
        gitdir = (root / text.removeprefix("gitdir:").strip()).resolve()
        if (gitdir / "commondir").is_file():
            gitdir = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    else:
        return False
    return (gitdir / "shallow").is_file()


def input_group_key(entry: dict) -> str:
    """The group of an input: its directory, or its kind for named inputs."""
    path = entry.get("path")
    if isinstance(path, str):
        return path.rsplit("/", 1)[0] if "/" in path else "."
    return f":{entry.get('kind')}"


def stored_form(manifest: dict) -> tuple[dict, dict[str, bytes]]:
    """The stored form of a manifest and the objects it names, by address.

    Each field is encoded as shared objects (`evidence_objects.encode`); the inputs are a list
    of the inputs of each directory. Passes of one build, policy or review, or with mostly the
    same inputs, store each shared part once."""
    groups: dict[str, list[dict]] = {}
    for entry in manifest["inputs"]:
        groups.setdefault(input_group_key(entry), []).append(entry)
    objects: dict[str, bytes] = {}
    stored = {key: encode(value, objects) for key, value in manifest.items() if key != "inputs"}
    stored["inputs"] = encode(
        [[_encoded_entry(e, objects)
          for e in sorted(groups[key], key=lambda e: (e["kind"], e.get("path", e.get("name", ""))))]
         for key in sorted(groups)], objects, encoded_parts=True)
    return stored, objects


# Encoded inputs by canonical bytes: the passes of one build share most of their inputs. Each
# input is its own object, so a directory's inputs are a short list of references.
_ENCODED: dict[bytes, tuple[object, dict[str, bytes]]] = {}


def _encoded_entry(entry: dict, objects: dict[str, bytes]):
    key = canonical(entry)
    if key not in _ENCODED:
        if len(_ENCODED) > 100_000:
            _ENCODED.clear()
        parts: dict[str, bytes] = {}
        _ENCODED[key] = (encode_object(entry, parts), parts)
    encoded, parts = _ENCODED[key]
    objects.update(parts)
    return encoded


def expand(manifest: dict, read) -> dict:
    """A stored manifest with its objects restored; `read` returns an object's decoded value
    (`evidence_objects.Decoded`)."""
    if not isinstance(manifest, dict) or "inputs" not in manifest:
        raise InvalidManifest("a stored manifest names its inputs")
    try:
        restored = {key: decode(value, read) for key, value in manifest.items()}
    except CorruptObject as error:
        raise InvalidManifest(str(error)) from error
    groups = restored["inputs"]
    if not isinstance(groups, list) or not all(
            isinstance(g, list) and all(isinstance(e, dict) for e in g) for g in groups):
        raise InvalidManifest("a stored manifest lists its inputs by directory")
    restored["inputs"] = [entry for group in groups for entry in group]
    return restored


def store_manifest(root: Path, manifest: dict) -> tuple[str, dict, list[str]]:
    """Store a manifest's objects and its stored form; its address, stored form and the
    shards written."""
    import evidence_objects
    stored, objects = stored_form(manifest)
    data = canonical(stored)
    return digest(data), stored, evidence_objects.write(root, {**objects, digest(data): data})


SCHEMA = 1
NORMALIZATION = "raw-bytes-v1"
# The dependency policy that writes manifests. Changing what earning records,
# or how, renames it. A manifest of an earlier policy stays current only while that
# policy is listed here: `compiler-items-v1` bound a superset of what the current
# policy binds (every function the executed code names, and recipes by their bytes),
# so what it attests still holds.
ANALYZER = "compiler-items-v2"
ANALYZERS = ("compiler-items-v1", ANALYZER)
HEX = re.compile(r"[0-9a-f]{64}\Z")
KINDS = ("file", "tree", "env", "membership", "evidence_file", "glob", "build_env",
         "rust_items", "rust_names", "rust_guards", "declarations", "locked_package", "build_recipe")
# Inputs keyed by a name rather than a repository path.
NAMED_KINDS = ("env", "build_env", "locked_package")
# A locked registry package, `<name> <version>`; its digest is the Cargo.lock checksum.
LOCKED_PACKAGE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]* [0-9A-Za-z.+-]+\Z")
DECLARATIONS = "parity/fingerprint_inputs.toml"


def locked_checksums(lock_bytes: bytes) -> dict[str, str]:
    """`<name> <version>` to checksum of each registry package in a Cargo.lock."""
    import tomllib
    found: dict[str, str] = {}
    for package in tomllib.loads(lock_bytes.decode("utf-8")).get("package", []):
        checksum = package.get("checksum")
        if package.get("source", "").startswith("registry+") and isinstance(checksum, str) \
                and HEX.fullmatch(checksum):
            found[f"{package['name']} {package['version']}"] = checksum
    return found


def declarations_digest(root: Path, sources: list[str]) -> str:
    """Digest of the `parity/fingerprint_inputs.toml` rows of `sources`, in file order: what
    a record's dynamic reads were resolved by. Rows of other sources are not its inputs."""
    path = root / DECLARATIONS
    rows = _repo.load_toml(path).get("input", []) if path.is_file() else []
    wanted = set(sources)
    return digest(canonical([row for row in rows if row.get("source") in wanted]))
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


@functools.cache
def rust_text_digest(text: str) -> str:
    """Digest of a Rust source as code: comments (doc comments included) blanked, literals
    kept, blank lines dropped. A comment or layout edit leaves it unchanged."""
    from evidence_fingerprint import normalized_rust
    return digest(b"rust-text\0" + normalized_rust(text).encode("utf-8"))


# Cargo target declarations that are separate executables: building one target never
# reads another's table, so another target's declaration is not an input of a binary.
OTHER_TARGET_TABLES = ("test", "bench", "example")


def build_manifest_digest(data: bytes, target: str | None, implicit_file: bool = False) -> str | None:
    """Digest of a Cargo.toml as the build of one executable reads it: every `[[test]]`,
    `[[bench]]` and `[[example]]` declaration removed except the one named `target` (the
    record's own test target; `None` for a library unit-test binary or a dependency's
    manifest). Package, library, binary, feature, dependency, lint and profile tables are
    kept whole. `None` when the bytes are not a TOML manifest of that shape."""
    import tomllib
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
        # Autodiscovery only selects executable targets. Compare the effective
        # declaration of the recorded target and retain every build option.
        auto = parsed["package"].pop("autotests", True)
        if target is not None:
            tests = parsed.get("test", [])
            own_test = next((entry for entry in tests if entry.get("name") == target), None)
            if own_test is None and auto and implicit_file:
                parsed.setdefault("test", []).append({"name": target, "path": f"tests/{target}.rs"})
            elif own_test is not None and implicit_file:
                own_test.setdefault("path", f"tests/{target}.rs")
            elif own_test is None:
                # Preserve the distinction when a source path cannot be established.
                parsed["package"]["autotests"] = auto
        for table in OTHER_TARGET_TABLES:
            entries = parsed.pop(table, None)
            if entries is None:
                continue
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                return None
            own = [e for e in entries if target is not None and table == "test" and e.get("name") == target]
            if own:
                parsed[table] = own
        return digest(b"cargo-build-manifest\0" + canonical(parsed))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, AttributeError, TypeError, ValueError):
        return None


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


_VALID_PATHS: set[str] = set()


def relative_path(value: object) -> str:
    if isinstance(value, str) and value in _VALID_PATHS:
        return value
    if not isinstance(value, str) or not value or "\\" in value:
        raise InvalidManifest("invalid repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("..", ".") for p in value.split("/")):
        raise InvalidManifest("path must be normalized and inside the repository")
    if str(path) != value:
        raise InvalidManifest("path must be canonical")
    _VALID_PATHS.add(value)
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
        key = entry.get("name") if kind in NAMED_KINDS else relative_path(entry.get("path"))
        if not isinstance(key, str) or not key:
            raise InvalidManifest("missing input name")
        if kind == "locked_package" and not LOCKED_PACKAGE.fullmatch(key):
            raise InvalidManifest("invalid locked package")
        if (kind, key) in seen:
            raise InvalidManifest("duplicate dependency")
        seen.add((kind, key))
        if not isinstance(entry.get("digest"), str) or not HEX.fullmatch(entry["digest"]):
            raise InvalidManifest("invalid input digest")
        if kind == "rust_items" and not (
                isinstance(entry.get("file"), str) and HEX.fullmatch(entry["file"])
                and isinstance(entry.get("table"), str) and HEX.fullmatch(entry["table"])
                and isinstance(entry.get("select"), dict) and isinstance(entry.get("scopes"), list)):
            raise InvalidManifest("invalid item selection")
        if kind == "rust_names" and not (isinstance(entry.get("bits"), int) and isinstance(entry.get("filter"), str)):
            raise InvalidManifest("invalid name filter")
        if kind == "rust_guards" and not (
                isinstance(entry.get("dirs"), list) and isinstance(entry.get("spec"), dict)
                and isinstance(entry.get("tree"), str) and HEX.fullmatch(entry["tree"])):
            raise InvalidManifest("invalid resolution guards")
        if kind == "declarations" and not (
                key == DECLARATIONS and isinstance(entry.get("sources"), list)
                and all(isinstance(s, str) for s in entry["sources"]) and entry["sources"] == sorted(set(entry["sources"]))):
            raise InvalidManifest("invalid input declarations")
        if kind == "evidence_file" and entry.get("text") not in (None, "rust"):
            raise InvalidManifest("unknown evidence text normalization")
        if kind == "build_recipe":
            from evidence_recipes import FORMS
            nodes = entry.get("nodes")
            if (entry.get("form") not in FORMS or not isinstance(entry.get("raw"), str)
                    or not HEX.fullmatch(entry["raw"])
                    or (entry["form"] == "flake-lock") != isinstance(nodes, list)
                    or (nodes is not None and (not nodes or nodes != sorted(set(nodes))
                                               or not all(isinstance(n, str) for n in nodes)))):
                raise InvalidManifest("invalid build recipe binding")
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
        if binding.get("kind") not in ("scientific", "calibration", "probe") or receipt.get("execution") != "exact-binary":
            raise ValueError("invalid execution binding")
        crate = artifact["crate"]
        if artifact["target_kind"] not in ("lib", "test"):
            raise ValueError("target mismatch")
        if binding["kind"] == "scientific":
            # A scientific record names its test target; a calibration record names its
            # measurement, whose binary the collector binds from the run that emitted it.
            key_crate, target, _ = binding["key"].split("::", 2)
            expected = "lib" if artifact["target_kind"] == "lib" else artifact["target_name"]
            if key_crate != crate or target != expected:
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
        # The receipt names the platform that earned it; its recipes are that platform's.
        from evidence_platforms import receipt_platform
        platform = receipt_platform(receipt)
        if (policy.get("source_projection_allowed") is not True
                or not HEX.fullmatch(policy["audit_selection_sha256"])):
            raise ValueError("invalid policy")
        # A recipe is bound by its bytes, or by its content with the bytes it was read from.
        tracked = {i.get("path"): i.get("digest") for i in inputs if i.get("kind") == "evidence_file"}
        tracked.update({i.get("path"): i.get("raw") for i in inputs if i.get("kind") == "build_recipe"})
        required = dict(artifact["recipe_sha256"])
        if set(required) != set(platform.recipes):
            raise ValueError("incomplete recipe identity")
        if not required or any(tracked.get(path) != value for path, value in required.items()):
            raise ValueError("unbound recipe or audit registry")
        if "release" in receipt:
            # Replicates ran on the uninstrumented build of the same targets; only a
            # calibration measurement does that, and its recipe is bound like the other.
            release = receipt["release"]
            if binding["kind"] not in ("calibration", "probe"):
                raise ValueError("release binary outside a calibration measurement")
            if release["binary"] != artifact["binary"] or not HEX.fullmatch(release["binary_sha256"]):
                raise ValueError("invalid release binary")
            if set(release["recipe_sha256"]) != set(platform.release_recipes):
                raise ValueError("incomplete release recipe identity")
            if any(tracked.get(path) != value for path, value in release["recipe_sha256"].items()):
                raise ValueError("unbound release recipe")
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


# File digests keyed by (path, device, inode, size, mtime_ns, ctime_ns), shared by every
# Checker in the process: earning checks the same inputs several times per row, and a
# file whose stat signature is unchanged is not read again.
_FILE_DIGESTS: dict[tuple, str] = {}
# Input paths already found free of symlinks in this process. A symlink appearing later
# is a change to the tree, which the clean-checkout check refuses.
_LINK_FREE: set[tuple[Path, str]] = set()


def _stat_signature(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class Checker:
    """One read and digest per input, shared by every evidence kind."""

    def __init__(self, root: Path, environ: dict[str, str] | None = None):
        self.root = root.resolve()
        # `same_code_as_bound` recovers bound bytes from the record's build commit. A shallow
        # clone lacks those commits, which would mark current records owed without an error
        # (and a release build would embed them as not_supported).
        if is_shallow_clone(self.root):
            raise ShallowClone(
                f"{self.root} is a shallow clone; evidence currency needs full history "
                "(actions/checkout fetch-depth: 0, or git fetch --unshallow)"
            )
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
        self.objects = Store(self.root)
        self.decoded = Decoded(self.objects.get)
        self.parsed: dict[str, object] = {}
        from evidence_items import ItemChecker
        self.items = ItemChecker(self)
        self._tracked: list[str] | None = None
        self._tracked_loaded = False

    @property
    def pin(self) -> dict:
        if not hasattr(self, "_pin"):
            self._pin = _repo.load_toml(self.root / "tools/evidence-analyzer/pin.toml")
        return self._pin

    def path(self, rel: str) -> Path:
        path = self.root / relative_path(rel)
        if (self.root, rel) in _LINK_FREE:
            return path
        # Symlinks are rejected, including in parents: a tree guard must not
        # silently omit a linked directory or depend on external host content.
        current = self.root
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                raise InvalidManifest(f"symlink input is unsupported: {rel}")
        _LINK_FREE.add((self.root, rel))
        return path

    def tracked(self) -> list[str] | None:
        if not self._tracked_loaded:
            self._tracked, self._tracked_loaded = tracked_files(self.root), True
        return self._tracked

    def members(self, rel: str) -> list[str]:
        """Files under a directory: those git does not ignore in a git work tree, else every file.

        A checker reads one state of the tree, so each directory is listed once."""
        cache = self.__dict__.setdefault("_members", {})
        if rel not in cache:
            cache[rel] = self._members_uncached(rel)
        return list(cache[rel])

    def _members_uncached(self, rel: str) -> list[str]:
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

    @functools.cached_property
    def pass_index(self) -> dict[str, str]:
        """The pass index of the tree this checker reads."""
        from claim_passes import read_index
        return read_index(self.root)

    def object(self, address: str):
        """The parsed object at `address`; `InvalidManifest` if it is missing or corrupt."""
        if address not in self.parsed:
            try:
                self.parsed[address] = json.loads(self.objects.get(address))
            except (CorruptObject, ValueError) as error:
                raise InvalidManifest(str(error)) from error
        return self.parsed[address]

    def expanded(self, manifest: dict) -> dict:
        """A stored manifest with its objects restored."""
        return expand(manifest, self.decoded)

    def locked_checksums(self) -> dict[str, str]:
        """`<name> <version>` to Cargo.lock checksum, for every registry package."""
        path = self.path("Cargo.lock")
        signature = _stat_signature(path)
        if getattr(self, "_locked", (None, None))[0] != signature or signature is None:
            self._locked = (signature, locked_checksums(path.read_bytes()))
        return self._locked[1]

    def rust_text(self, key: str) -> str:
        """`rust_text_digest` of a checkout file ("missing-input" when absent)."""
        path = self.path(key)
        signature = _stat_signature(path)
        if signature is None:
            return digest(b"missing-input")
        entry = ("rust_text", key, signature)
        if entry not in self.cache:
            self.cache[entry] = rust_text_digest(path.read_text(encoding="utf-8", errors="replace"))
        return self.cache[entry]

    def recipe_current(self, entry: dict) -> bool:
        """Whether a build recipe still says what it said when bound (`evidence_recipes`)."""
        import evidence_recipes
        path = self.path(entry["path"])
        if self.input_digest("file", entry["path"]) == entry["raw"] if path.is_file() else False:
            return True
        key = ("recipe", entry["path"], entry["form"], tuple(entry.get("nodes") or ()))
        if key not in self.cache:
            try:
                self.cache[key] = evidence_recipes.digest(entry["form"], path.read_bytes(), nodes=entry.get("nodes"))
            except (OSError, ValueError):
                self.cache[key] = digest(b"missing-input")
        return self.cache[key] == entry["digest"]

    def same_code_as_bound(self, key: str, bound: str, manifest: dict) -> bool:
        """Whether a Rust file bound by raw bytes differs from them only in comments or layout.

        The bound bytes are read from the commit the record's Nix build used and must hash
        to the recorded digest; anything else (no such commit in this clone, other bytes)
        leaves the input changed."""
        commit = (manifest.get("nix_build") or {}).get("source", {}).get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            return False
        cache = self.__dict__.setdefault("_bound_code", {})
        if (commit, key) not in cache:
            shown = subprocess.run(["git", "show", f"{commit}:{key}"], cwd=self.root, capture_output=True)
            data = shown.stdout if shown.returncode == 0 else None
            cache[(commit, key)] = (digest(data), rust_text_digest(data.decode("utf-8", "replace"))) if data is not None else (None, None)
        raw_digest, code_digest = cache[(commit, key)]
        return bound == raw_digest and code_digest == self.rust_text(key)

    def same_build_manifest_as_bound(self, key: str, bound: str, manifest: dict) -> bool:
        """Whether a Cargo.toml bound by raw bytes differs from them only in the
        declarations of other test, bench or example targets, which the build of the
        record's executable does not read (`build_manifest_digest`).

        As for Rust code, the bound bytes are recovered from version control and
        must hash to the recorded digest; the record's own target comes from that
        build's artifact. Anything else leaves the input changed."""
        build = manifest.get("nix_build") or {}
        commit = build.get("source", {}).get("commit")
        artifact = build.get("artifact") or {}
        if (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
                or artifact.get("target_kind") not in ("lib", "test")):
            return False
        target = artifact.get("target_name") if artifact["target_kind"] == "test" else None
        if target is not None and not isinstance(target, str):
            return False
        path = self.path(key)
        if not path.is_file():
            return False
        if artifact.get("crate") and path.parent.name != artifact["crate"] and key.startswith("crates/"):
            target = None
        source = f"tests/{target}.rs" if target else None
        old_files = self.__dict__.setdefault("_bound_cargo_files", {})
        if (commit, key) not in old_files:
            shown = subprocess.run(["git", "show", f"{commit}:{key}"], cwd=self.root, capture_output=True)
            old_files[commit, key] = shown.stdout if shown.returncode == 0 else None
        data = old_files[commit, key]
        if data is None:
            return False
        if digest(data) != bound:
            # A reviewed carry may have rebound the raw file at a later commit.
            # Recover the exact bytes named by the recorded SHA-256. Never
            # substitute the current tree or an approximate historical match.
            history = self.__dict__.setdefault("_cargo_history", {})
            if key not in history:
                versions = subprocess.run(["git", "log", "--format=%H", "HEAD", "--", key],
                                          cwd=self.root, capture_output=True, text=True)
                history[key] = iter(versions.stdout.splitlines() if versions.returncode == 0 else [])
            recovered = self.__dict__.setdefault("_cargo_versions", {})
            if (key, bound) not in recovered:
                for revision in history[key]:
                    shown = subprocess.run(["git", "show", f"{revision}:{key}"], cwd=self.root, capture_output=True)
                    if shown.returncode == 0:
                        identity = digest(shown.stdout)
                        recovered[key, identity] = revision, shown.stdout
                        if identity == bound:
                            break
            if (key, bound) not in recovered:
                return False
            commit, data = recovered[key, bound]
        cache = self.__dict__.setdefault("_bound_build_manifest", {})
        if (commit, key, target) not in cache:
            old_source = bool(source and subprocess.run(
                ["git", "cat-file", "-e", f"{commit}:{PurePosixPath(key).parent}/{source}"],
                cwd=self.root, capture_output=True,
            ).returncode == 0)
            cache[commit, key, target] = build_manifest_digest(data, target, old_source)
        signature = _stat_signature(path)
        current = self.__dict__.setdefault("_current_build_manifest", {})
        exists = bool(source and (path.parent / source).is_file())
        entry = key, signature, target, exists
        if entry not in current or signature is None:
            current[entry] = build_manifest_digest(path.read_bytes(), target, exists)
        return bool(cache[commit, key, target]) and cache[commit, key, target] == current[entry]

    def input_digest(self, kind: str, key: str) -> str:
        cache_key = kind, key
        if cache_key in self.cache:
            return self.cache[cache_key]
        start = time.perf_counter()
        if kind == "env":
            value = env_digest(self.environ, key)
        elif kind == "build_env":
            value = digest(canonical(effective_build_env(self.environ)[key]))
        elif kind == "locked_package":
            value = self.locked_checksums().get(key, digest(b"missing-input"))
        elif kind == "file":
            path = self.path(key)
            signature = _stat_signature(path)
            value = _FILE_DIGESTS.get(signature) if signature is not None else None
            if value is None:
                data = path.read_bytes()
                value = digest(data)
                self.bytes_hashed += len(data)
                self.files_hashed += 1
                if signature is not None:
                    _FILE_DIGESTS[signature] = value
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
        if not isinstance(ref, str):
            raise InvalidManifest("missing or invalid dependency_manifest reference")
        if ref.startswith("pass:"):
            if ref not in self.manifests:
                address = self.pass_index.get(ref)
                if address is None:
                    raise InvalidManifest(f"no pass {ref}")
                self.manifests[ref] = self.expanded(self.object(address))
            return self.manifests[ref]
        if not HEX.fullmatch(ref):
            raise InvalidManifest("missing or invalid dependency_manifest reference")
        if ref not in self.manifests:
            start = time.perf_counter()
            manifest = self.object(ref)
            if not isinstance(manifest, dict) or manifest_id(manifest) != ref:
                raise InvalidManifest("manifest content does not match its address")
            self.manifests[ref] = self.expanded(manifest)
            self.parsing_seconds += time.perf_counter() - start
        return self.manifests[ref]

    def recheck_nix(self, policy: dict) -> None:
        """`recheck_selection` once per distinct policy: records of one build share it."""
        from evidence_platforms import recheck_selection
        cache = self.__dict__.setdefault("_nix_rechecks", {})
        key = canonical(policy)
        if key not in cache:
            try:
                recheck_selection(self.root, policy)
                cache[key] = None
            except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                cache[key] = error
        if cache[key] is not None:
            raise cache[key]

    def prior_receipt(self, binding: dict, ref: object) -> dict:
        """Validate a stored pass even when the checkout has moved past its inputs."""
        manifest = self.load(ref)
        validate(manifest, binding)
        if manifest["analyzer"] not in ANALYZERS:
            raise InvalidManifest("unrecognized prior analyzer policy")
        for entry in manifest["inputs"]:
            if entry["kind"] == "rust_items":
                self.items.table(entry["table"])
        return manifest

    def judgment_fingerprint(self, binding: dict, ref: object, *, context=None) -> str:
        """Bind a judgment to the original receipt and a conservative view of its inputs now."""
        from evidence_items import guard_tree
        from evidence_platforms import receipt_platform

        manifest = self.prior_receipt(binding, ref)
        files = {"Cargo.toml", "Cargo.lock", ".cargo/config", ".cargo/config.toml",
                 "rust-toolchain", "rust-toolchain.toml", "tools/evidence-analyzer/pin.toml"}
        if receipt := manifest.get("nix_build"):
            platform = receipt_platform(receipt)
            files.update((platform.registry, *platform.recipes, *platform.policy_sources))

        current = []
        for entry in manifest["inputs"]:
            kind = entry["kind"]
            key = entry.get("path", entry.get("name"))
            if kind in ("file", "evidence_file", "build_recipe", "rust_items"):
                files.add(key)
            elif kind == "declarations":
                current.append((kind, key, declarations_digest(self.root, entry["sources"])))
            elif kind == "rust_guards":
                current.append((kind, key, guard_tree(self, entry["dirs"])))
            elif kind != "rust_names":
                current.append((kind, key, self.input_digest(kind, key)))

        current.extend(("file", key, self.input_digest("evidence_file", key)) for key in sorted(files))
        address = self.pass_index[ref] if ref.startswith("pass:") else ref
        return digest(canonical(["reviewed-evidence-inputs-v1", address, sorted(current), context]))

    def check(self, binding: dict, ref: object, *, first_failure: bool = False) -> dict:
        label = f"{binding['kind']}:{binding['key']}"
        result = {"record": label, "current": False, "reasons": []}
        try:
            manifest = self.load(ref)
            if first_failure and "nix_build" in manifest:
                # A stale shared build policy invalidates every artifact built under it. Avoid
                # traversing thousands of inputs per record when only the verdict is requested.
                self.recheck_nix(manifest["nix_build"]["policy"])
            validate(manifest, binding)
            reasons = result["reasons"]
            if manifest["compiler"] != {k: self.pin[k] for k in ("release", "commit")}:
                reasons.append("compiler_identity_changed")
            if manifest["analyzer"] not in ANALYZERS:
                reasons.append("analyzer_policy_changed")
            if "nix_build" in manifest:
                self.recheck_nix(manifest["nix_build"]["policy"])
            names = next((e for e in manifest["inputs"] if e["kind"] == "rust_names"), None)
            for entry in manifest["inputs"]:
                kind = entry["kind"]
                key = entry["name"] if kind in NAMED_KINDS else entry["path"]
                if not first_failure:
                    self.reverse[f"{kind}:{key}"].add(label)
                if kind == "rust_items":
                    unchanged = self.items.items_current(entry, names)
                elif kind == "rust_guards":
                    unchanged = self.items.guards_current(entry)
                elif kind == "rust_names":
                    unchanged = True
                elif kind == "declarations":
                    unchanged = declarations_digest(self.root, entry["sources"]) == entry["digest"]
                elif kind == "build_recipe":
                    unchanged = self.recipe_current(entry)
                elif kind == "evidence_file" and entry.get("text") == "rust":
                    unchanged = self.rust_text(key) == entry["digest"]
                else:
                    unchanged = self.input_digest(kind, key) == entry["digest"]
                    if not unchanged and kind == "evidence_file" and key.endswith(".rs"):
                        # Bound by raw bytes before code text was recorded: recover the bound
                        # bytes from the commit the record was built from and compare as code.
                        unchanged = self.same_code_as_bound(key, entry["digest"], manifest)
                    elif (not unchanged and kind == "evidence_file"
                          and PurePosixPath(key).name == "Cargo.toml"):
                        # A new target declared beside the record's own is not an input of
                        # its binary: compare the manifest as that build reads it.
                        unchanged = self.same_build_manifest_as_bound(key, entry["digest"], manifest)
                if not unchanged:
                    reasons.append(f"input_changed:{kind}:{key}")
                    if first_failure:
                        break
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


def _stored(root: Path) -> set[str]:
    from claim_passes import PASS_INDEX, read_index
    return {f"{OBJECTS}/{address}" for address in Store(root).addresses()} \
        | {f"{PASS_INDEX}#{ref}" for ref in read_index(root)}


def reachable_evidence(root: Path) -> set[str]:
    """Stored evidence a record still reaches: its pass or manifest and every object they
    name. Every ledger that cites manifests counts."""
    from claim_passes import PASS_INDEX
    from calibration_judgments import references as calibration_judgment_references
    checker = Checker(root)
    reached: set[str] = set()
    references = {ref for _, ref in inventory(root) if isinstance(ref, str)} | calibration_judgment_references(root)
    for ref in references:
        if not isinstance(ref, str):
            continue
        try:
            if ref.startswith("pass:") and ref in checker.pass_index:
                reached.add(f"{PASS_INDEX}#{ref}")
                address = checker.pass_index[ref]
            elif HEX.fullmatch(ref) and checker.objects.has(ref):
                address = ref
            else:
                continue
            reached.update(f"{OBJECTS}/{a}" for a in reachable(checker.objects, address))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, InvalidManifest):
            # An unreadable manifest keeps what it names; its record reports the defect.
            continue
    return reached


def unreachable_evidence(root: Path) -> list[str]:
    """Stored evidence no record reaches: superseded by a later pass or left by a stopped run.
    An object is named `parity/evidence_objects/<address>`, a pass `parity/passes.json#<id>`."""
    return sorted(_stored(root) - reachable_evidence(root))


def prune_evidence(root: Path) -> list[str]:
    """Delete stored evidence no record reaches: passes from the index, objects from the
    store. History keeps every superseded object."""
    from claim_passes import PASS_INDEX, write_index
    removed = unreachable_evidence(root)
    passes = {rel.split("#", 1)[1] for rel in removed if rel.startswith(f"{PASS_INDEX}#")}
    if passes:
        write_index(root, lambda index: [index.pop(ref, None) for ref in passes])
    prefix = f"{OBJECTS}/"
    dropped = {rel[len(prefix):] for rel in removed if rel.startswith(prefix)}
    if dropped:
        store = Store(root)
        store.compact(store.addresses() - dropped)
    return removed


def report(root: Path = ROOT, kind: str = "all") -> dict:
    return Checker(root).report(inventory(root, kind))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("all", "scientific", "calibration", "probe"), default="all")
    parser.add_argument("--show", metavar="REF", help="print the manifest a pass or address names")
    args = parser.parse_args()
    if args.show:
        print(json.dumps(Checker(ROOT).load(args.show), indent=2, sort_keys=True))
        return 0
    print(json.dumps(report(kind=args.kind), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
