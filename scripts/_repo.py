"""Repository root and UTF-8 text IO for scripts.

``Path.read_text()`` and ``Path.write_text()`` follow the process locale, and
``write_text`` translates ``\\n`` to ``os.linesep``. Windows cp1252 cannot
round-trip the support-matrix times sign. Go through these helpers (or pass
``encoding="utf-8"``) when a script reads or writes repository text.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# The workspace feature set the Rust tests build with (`workspace=` in scripts/test.sh);
# the scientific run uses the same set so a recorded pass means the same thing.
WORKSPACE_TEST_FEATURES = (
    "antecedent/calibration-internal",
    "antecedent-identify/test-util",
    "antecedent-io/calibration-internal",
    "antecedent-validate/calibration-internal",
)


def ensure_scripts_on_path() -> None:
    """Make sibling scripts importable.

    ``python3 scripts/foo.py`` already puts ``scripts/`` on ``sys.path``. An
    import of that module from another directory does not.
    """
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def read_text(path: Path | str, *, errors: str = "strict") -> str:
    return Path(path).read_text(encoding="utf-8", errors=errors)


def write_text(path: Path | str, data: str) -> int:
    return Path(path).write_text(data, encoding="utf-8", newline="\n")


def load_toml(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(read_text(path))


def load_toml_binary(path: Path | str) -> dict[str, Any]:
    """Parse a binary TOML file without universal-newline translation."""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def strict_git_head(root: Path | str) -> str:
    """Read HEAD, propagating Git failures to the caller."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def strict_git_dirty(root: Path | str) -> bool:
    """Whether tracked or normally reported untracked files are present."""
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
        ).strip()
    )


def cargo_target_feature_args(
    manifest: Path | str, target_args: list[str] | tuple[str, ...]
) -> list[str]:
    """Return only the declared required features for one selected Cargo target.

    Cargo skips a target with ``required-features`` unless the matching feature
    set is enabled. Evidence listing and execution both use this helper so they
    discover and run the same target without enabling unrelated features.
    """
    args = list(target_args)
    target_kind: str | None = None
    target_name: str | None = None
    for switch, kind in (
        ("--test", "test"),
        ("--bench", "bench"),
        ("--bin", "bin"),
        ("--example", "example"),
    ):
        if switch in args:
            index = args.index(switch)
            if index + 1 < len(args):
                target_kind = kind
                target_name = args[index + 1]
            break
    if target_kind is None and "--lib" in args:
        target_kind = "lib"

    if target_kind is None:
        return []

    cargo = load_toml(manifest)
    if target_kind == "lib":
        targets = [cargo.get("lib")] if isinstance(cargo.get("lib"), dict) else []
    else:
        targets = cargo.get(target_kind, [])
        if not isinstance(targets, list):
            targets = []
        targets = [
            target
            for target in targets
            if isinstance(target, dict) and target.get("name") == target_name
        ]
    if len(targets) > 1:
        raise ValueError(f"{manifest}: duplicate Cargo {target_kind} target {target_name!r}")
    if not targets:
        return []

    required = targets[0].get("required-features", [])
    if not isinstance(required, list) or not all(
        isinstance(feature, str) and feature for feature in required
    ):
        raise ValueError(
            f"{manifest}: invalid required-features for {target_kind} target {target_name!r}"
        )
    if not required:
        return []
    return ["--features", ",".join(required)]


def scientific_target_feature_args(manifest: Path, args: list[str]) -> list[str]:
    """Stable, manifest-declared scientific-build features for a Cargo target.

    Unlike Cargo `required-features`, these do not change ordinary test discovery.
    The evidence fingerprint and earning build must use the same selection.
    """
    cargo = load_toml(manifest)
    table = cargo.get("package", {}).get("metadata", {}).get("scientific-test-features", {})
    if not isinstance(table, dict):
        raise ValueError(f"{manifest}: scientific-test-features must be a table")
    target_name = "lib" if "--lib" in args else None
    if "--test" in args:
        index = args.index("--test")
        if index + 1 < len(args):
            target_name = args[index + 1]
    selected = table.get(target_name, [])
    if not isinstance(selected, list) or not all(isinstance(f, str) and f for f in selected):
        raise ValueError(f"{manifest}: invalid scientific features for target {target_name!r}")
    available = cargo.get("features", {})
    if any(f not in available for f in selected):
        raise ValueError(f"{manifest}: scientific target names an unknown feature")
    required = cargo_target_feature_args(manifest, args)
    features = set(selected) | (set(required[1].split(",")) if required else set())
    return ["--features", ",".join(sorted(features))] if features else []


def workspace_package_version(root: Path | None = None) -> str:
    """``[workspace.package] version`` from the repo ``Cargo.toml``."""
    cargo = load_toml((root or REPO_ROOT) / "Cargo.toml")
    version = cargo["workspace"]["package"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Cargo.toml workspace.package.version is missing")
    return version


def test_target(crate: str, stem: str, root: Path = REPO_ROOT) -> tuple[str, str]:
    """The test target that compiles `crates/<crate>/tests/<stem>.rs`, and the module prefix of its tests.

    A file with its own `[[test]]` target is that target with an empty prefix; a file
    folded into a `suite_NN.rs` aggregator by `scripts/consolidate_tests.py` runs inside
    the suite binary under the module named after the file (`<stem>::`).
    """
    crate_dir = root / "crates" / crate
    suites = sorted((crate_dir / "tests").glob("suite_*.rs"))
    # Cached per state of the manifest and the suite files: a status check asks this of
    # every row, and the manifests of large crates take most of that time to parse.
    state = tuple((str(path), _stat_state(path)) for path in (crate_dir / "Cargo.toml", *suites))
    key = (str(crate_dir), stem, state)
    if key not in _TEST_TARGETS:
        _TEST_TARGETS[key] = _test_target(crate_dir, stem, suites)
    return _TEST_TARGETS[key]


_TEST_TARGETS: dict[tuple, tuple[str, str]] = {}


def _stat_state(path: Path) -> tuple[int, int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size, info.st_ino


def _test_target(crate_dir: Path, stem: str, suites: list[Path]) -> tuple[str, str]:
    manifest = load_toml(crate_dir / "Cargo.toml")
    for entry in manifest.get("test", []):
        if entry.get("path", f"tests/{entry['name']}.rs") == f"tests/{stem}.rs":
            return entry["name"], ""
    include = re.compile(rf'#\[path\s*=\s*"{re.escape(stem)}\.rs"\]\s*mod\s+{re.escape(stem)}\s*;')
    for suite in suites:
        if include.search(suite.read_text(encoding="utf-8")):
            return suite.stem, f"{stem}::"
    return stem, ""
