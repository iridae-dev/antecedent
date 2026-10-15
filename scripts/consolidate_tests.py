#!/usr/bin/env python3
"""Fold integration-test files into a few binaries per crate.

Every file under `crates/*/tests/*.rs` is a separate test binary: its own link of the
crate under test and its own first-exec cost. This script keeps each calibration harness
as its own target (the gate scripts name them with `--test`) and groups the remaining
files into `suite_NN.rs` aggregators that include each file in place with `#[path]`, so
file paths cited by registries do not move.

    python3 scripts/consolidate_tests.py            # report only
    python3 scripts/consolidate_tests.py --write    # rewrite Cargo.toml and write suites
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PER_SUITE = 30
KEEP_PATTERNS = [
    r'#\[ignore\s*=\s*"calibration',
    r"common/calibration\.rs",
    r"mod calibration;",
    r"\bcrate::",
    r"static mut|OnceLock|lazy_static|thread_local",
    r"set_var\(",
]
SUITE_MARK = "// consolidated-suite"


def named_by_scripts() -> set[str]:
    """Test targets that a script or workflow selects with `--test <name>`."""
    names: set[str] = set()
    for pattern in ("scripts/*.sh", "scripts/*.py", ".github/workflows/*.yml"):
        for path in ROOT.glob(pattern):
            if path.name == "consolidate_tests.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            names.update(re.findall(r"--test\s+([A-Za-z0-9_]+)", text))
    return names


def declared_targets(manifest: str) -> dict[str, dict]:
    data = tomllib.loads(manifest)
    out: dict[str, dict] = {}
    for key in ("test", "bench"):
        for entry in data.get(key, []):
            out[entry.get("path", f"tests/{entry['name']}.rs")] = {"kind": key, **entry}
    return out


NAMED = named_by_scripts()


def plan_crate(crate: Path):
    tests = sorted((crate / "tests").glob("*.rs"))
    tests = [t for t in tests if not t.name.startswith("suite_")]
    if not tests:
        return None
    manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
    declared = declared_targets(manifest)
    keep: list[tuple[Path, dict]] = []
    fold: list[Path] = []
    for path in tests:
        rel = f"tests/{path.name}"
        text = path.read_text(encoding="utf-8", errors="ignore")
        included = [
            (path.parent / m.group(1)).resolve()
            for m in re.finditer(r'#\[path\s*=\s*"([^"]+)"\]', text)
        ]
        for target in included:
            if target.is_file():
                text += "\n" + target.read_text(encoding="utf-8", errors="ignore")
        if rel in declared or path.stem in NAMED or any(re.search(p, text, re.M) for p in KEEP_PATTERNS):
            keep.append((path, declared.get(rel, {})))
        else:
            fold.append(path)
    return crate, manifest, keep, fold


def suite_name(index: int) -> str:
    return f"suite_{index:02d}"


def render_suite(files: list[Path]) -> str:
    lines = [
        "//! Consolidated integration tests: one binary for this group of files, each",
        "//! included in place so its path, helpers and registry citations are unchanged.",
        "#![allow(clippy::all, reason = \"each included file carries its own lint scope\")]",
        SUITE_MARK,
        "",
    ]
    for path in files:
        lines.append(f'#[path = "{path.name}"]')
        lines.append(f"mod {path.stem};")
    return "\n".join(lines) + "\n"


def rewrite_manifest(manifest: str, keep, suites: list[str]) -> str:
    if "autotests" not in manifest:
        manifest = re.sub(r"(\[package\][^\[]*)", lambda m: m.group(1).rstrip("\n") + "\nautotests = false\n\n", manifest, count=1)
    existing = declared_targets(manifest)
    additions = []
    for path, meta in keep:
        rel = f"tests/{path.name}"
        if rel in existing:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        block = f'[[test]]\nname = "{path.stem}"\npath = "{rel}"\n'
        feat = re.search(r'^#!\[cfg\(feature\s*=\s*"([^"]+)"\)\]', text, re.M)
        if feat:
            block += f'required-features = ["{feat.group(1)}"]\n'
        additions.append(block)
    for name in suites:
        if f"tests/{name}.rs" in existing:
            continue
        additions.append(f'[[test]]\nname = "{name}"\npath = "tests/{name}.rs"\n')
    return manifest.rstrip("\n") + "\n\n" + "\n".join(additions)


def main() -> int:
    write = "--write" in sys.argv
    total_before = total_after = 0
    for crate in sorted((ROOT / "crates").iterdir()):
        plan = plan_crate(crate)
        if plan is None:
            continue
        crate, manifest, keep, fold = plan
        groups = [fold[i : i + MAX_PER_SUITE] for i in range(0, len(fold), MAX_PER_SUITE)]
        names = [suite_name(i + 1) for i in range(len(groups))]
        before = len(keep) + len(fold)
        after = len(keep) + len(groups)
        total_before += before
        total_after += after
        print(f"{crate.name:28} {before:4} files -> {after:3} binaries (keep {len(keep)}, fold {len(fold)})")
        if not write:
            continue
        for name, group in zip(names, groups):
            (crate / "tests" / f"{name}.rs").write_text(render_suite(group), encoding="utf-8")
        (crate / "Cargo.toml").write_text(rewrite_manifest(manifest, keep, names), encoding="utf-8")
        mapping = crate / "tests" / "suite_map.txt"
        mapping.write_text(
            "".join(f"{name}: {', '.join(p.stem for p in group)}\n" for name, group in zip(names, groups)),
            encoding="utf-8",
        )
    print(f"total {total_before} -> {total_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
