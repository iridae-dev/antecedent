#!/usr/bin/env python3
"""Classify every `#[ignore]`d Rust test and select the slow correctness lane.

An ignored test runs in no ordinary lane, so each one must say where it does
run. Every `#[ignore]`d test belongs to exactly one class:

  calibration   (a) repeated-sampling calibration, measurement or printing
                diagnostic; the calibration scripts own it (reason starts
                `calibration`, `coverage:`, `measurement`, `probe:` or
                `diagnostic:`, or names a stable-candidate run)
  writer        (b) golden or fixture writer, stays ignored (reason `writes ...`
                or an explicit oracle-update switch)
  timing        (c) timing or benchmark probe, stays ignored
  slow          (d) a correctness assertion that is too slow for the ordinary
                lanes; listed in parity/slow_correctness_tests.toml and run by
                `bash scripts/test.sh slow`
  other         (e) explicit exemption in the same registry with a reason

Classes (a)-(c) are read from the reason string of the attribute, so an ignored
test with no reason, or with a reason that fits none of them and no registry
row, fails. A registry row fails when its test no longer exists, is no longer
ignored, or is also matched by a rule. No command here runs a test.

    python3 scripts/check_ignored_tests.py                 # check (source only)
    python3 scripts/check_ignored_tests.py check --listing LISTING.json
    python3 scripts/check_ignored_tests.py report
    python3 scripts/check_ignored_tests.py slow-filter [--listing LISTING.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import tomllib
from _repo import REPO_ROOT
from test_evidence import libtest_name, rust_items, target_files

REGISTRY = "parity/slow_correctness_tests.toml"
RULES = (
    ("calibration", re.compile(r"^(calibration|coverage:|measurement|probe:|diagnostic:|Q08/E14|\d+ independent original-data)")),
    ("writer", re.compile(r"^(writes |run explicitly with UPDATE_\w+=1)")),
    ("timing", re.compile(r"^(timing harness|explicit performance measurement|prints bootstrap|ADR 0011 )")),
)
REASON = re.compile(r'^#\[ignore\s*=\s*"((?:[^"\\]|\\.)*)"\s*\]$', re.S)
TEST_ATTRS = ("#[test]", "#[tokio::test]")
SLOW_KEYS = {"path", "name", "reason", "expected_seconds"}
OTHER_KEYS = {"path", "name", "reason"}


def ignored_tests(root: Path) -> list[dict]:
    """Every `#[test]` with an `ignore` attribute in a crate source or test file."""
    found = []
    for base in (root / "crates", root / "python/src"):
        for path in sorted(base.rglob("*.rs")) if base.is_dir() else []:
            if "target" in path.relative_to(root).parts:
                continue
            if "ignore" not in path.read_text(encoding="utf-8", errors="ignore"):
                continue
            masked, items = rust_items(path)
            for item in items:
                if item.kind == "macro":
                    # Tests a macro generates have no fn of their own: the
                    # reason inside the macro body classifies all of them.
                    body = masked.code[item.body[0]:item.body[1]]
                    for m in re.finditer(r'#\[\s*ignore\s*(?:=\s*"((?:[^"\\]|\\.)*)")?\s*\]', body):
                        found.append({"path": path.relative_to(root).as_posix(), "name": None,
                                      "reason": m.group(1), "conditional": False, "macro": item.name})
                    continue
                if item.kind != "fn":
                    continue
                attrs = [re.sub(r"\s+", "", a) for a in item.attrs]
                if not any(a in TEST_ATTRS or a.startswith("#[tokio::test(") for a in attrs):
                    continue
                raw = next((a for a in item.attrs if re.match(r"#\[\s*ignore\b", a)), None)
                conditional = any("cfg_attr" in a and re.search(r"\bignore\b", a) for a in item.attrs)
                if raw is None and not conditional:
                    continue
                reason = None
                if raw is not None:
                    m = REASON.match(re.sub(r"\s+", " ", raw).replace("# [", "#[").strip())
                    reason = m.group(1) if m else None
                found.append({
                    "path": path.relative_to(root).as_posix(),
                    "name": item.name,
                    "reason": reason,
                    "conditional": conditional,
                })
    return found


def target_of(root: Path, rel: str) -> tuple[str, str, Path] | None:
    """(crate, cargo target, target root file) of a source file, or None.

    A file pulled in with `include!("name.rs")` belongs to its includer's target.
    """
    found = _target_of(root, rel)
    if found is None and rel.endswith(".rs"):
        path = root / rel
        needle = f'include!("{path.name}")'
        for sibling in sorted(path.parent.glob("*.rs")):
            if sibling != path and needle in sibling.read_text(encoding="utf-8", errors="ignore"):
                return target_of(root, sibling.relative_to(root).as_posix())
    return found


def _target_of(root: Path, rel: str) -> tuple[str, str, Path] | None:
    parts = Path(rel).parts
    if parts[0] == "python":
        crate_dir, rest = root / "python", parts[1:]
    elif parts[0] == "crates" and len(parts) > 3:
        crate_dir, rest = root / "crates" / parts[1], parts[2:]
    else:
        return None
    path = (root / rel).resolve()
    manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text(encoding="utf-8"))
    crate = manifest["package"]["name"]
    if rest[0] == "src":
        lib = crate_dir / manifest.get("lib", {}).get("path", "src/lib.rs")
        if lib.is_file() and path in target_files(lib.resolve(), crate_dir.resolve()):
            return crate, "lib", lib
        return None
    if rest[0] != "tests":
        return None
    entries = list(manifest.get("test", []))
    if manifest["package"].get("autotests", True):
        explicit = {e.get("path", f"tests/{e['name']}.rs") for e in entries}
        entries += [{"name": p.stem, "path": f"tests/{p.name}"}
                    for p in sorted((crate_dir / "tests").glob("*.rs"))
                    if f"tests/{p.name}" not in explicit]
    for entry in entries:
        root_file = crate_dir / entry.get("path", f"tests/{entry['name']}.rs")
        if root_file.is_file() and path in target_files(root_file.resolve(), crate_dir.resolve()):
            return crate, entry["name"], root_file
    return None


def load_registry(root: Path) -> tuple[list[dict], list[dict]]:
    data = tomllib.loads((root / REGISTRY).read_text(encoding="utf-8"))
    unknown = set(data) - {"slow", "other"}
    if unknown:
        raise ValueError(f"{REGISTRY}: unknown tables {sorted(unknown)}")
    return data.get("slow", []), data.get("other", [])


def registry_problems(root: Path, slow: list[dict], other: list[dict], ignored: list[dict]) -> list[str]:
    problems = []
    by_key: dict[tuple[str, str], list[dict]] = {}
    for t in ignored:
        if t["name"] is not None:
            by_key.setdefault((t["path"], t["name"]), []).append(t)
    seen = set()
    for table, rows, keys in (("slow", slow, SLOW_KEYS), ("other", other, OTHER_KEYS)):
        for row in rows:
            key = (row.get("path"), row.get("name"))
            label = f"[{table}] {key[0]}::{key[1]}"
            if set(row) != keys:
                problems.append(f"{label}: keys must be exactly {sorted(keys)}")
                continue
            if not all(isinstance(row[k], str) and row[k].strip() for k in ("path", "name", "reason")):
                problems.append(f"{label}: path, name and reason must be non-empty strings")
                continue
            if table == "slow" and not (isinstance(row["expected_seconds"], (int, float))
                                        and not isinstance(row["expected_seconds"], bool)
                                        and row["expected_seconds"] > 0):
                problems.append(f"{label}: expected_seconds must be a positive number")
            if key in seen:
                problems.append(f"{label}: listed twice")
                continue
            seen.add(key)
            if not (root / key[0]).is_file():
                problems.append(f"{label}: file does not exist")
                continue
            matches = by_key.get(key, [])
            if not matches:
                problems.append(f"{label}: no such test, or it is no longer #[ignore]d (remove the row or restore the ignore)")
                continue
            if len(matches) > 1:
                problems.append(f"{label}: name is ambiguous ({len(matches)} ignored tests in the file)")
                continue
            if target_of(root, key[0]) is None:
                problems.append(f"{label}: file is compiled into no cargo target")
            if matches[0]["reason"] is not None and (rule := rule_class(matches[0]["reason"])):
                problems.append(f"{label}: reason already classifies it as {rule}; drop the registry row")
    return problems


def rule_class(reason: str | None) -> str | None:
    if not reason:
        return None
    return next((name for name, pattern in RULES if pattern.search(reason)), None)


def classify(root: Path) -> tuple[list[dict], list[str]]:
    """Each ignored test with a `class`, plus the problems that prevent one."""
    slow, other = load_registry(root)
    ignored = ignored_tests(root)
    problems = registry_problems(root, slow, other, ignored)
    slow_keys = {(r["path"], r["name"]) for r in slow if "path" in r}
    other_keys = {(r["path"], r["name"]) for r in other if "path" in r}
    for t in ignored:
        key = (t["path"], t["name"])
        if t["name"] is None:
            t["class"] = rule_class(t["reason"])
            if t["class"] is None:
                problems.append(
                    f"{t['path']}: macro {t['macro']} generates ignored tests with reason {t['reason']!r}; "
                    "a macro-generated ignore must carry a calibration, writer or timing reason"
                )
        elif key in slow_keys:
            t["class"] = "slow"
        elif key in other_keys:
            t["class"] = "other"
        elif t["reason"] is None:
            t["class"] = None
            problems.append(f"{t['path']}::{t['name']}: #[ignore] needs a written reason (#[ignore = \"...\"])")
        elif (cls := rule_class(t["reason"])) is not None:
            t["class"] = cls
        else:
            t["class"] = None
            problems.append(
                f"{t['path']}::{t['name']}: ignored with reason {t['reason']!r} but in no class; "
                f"list it under [[slow]] or [[other]] in {REGISTRY}, or run it in an ordinary lane"
            )
    return ignored, problems


def slow_rows(root: Path) -> list[dict]:
    """Resolved identity of every slow-lane test."""
    slow, _ = load_registry(root)
    rows = []
    for row in slow:
        crate, target, root_file = target_of(root, row["path"])
        rows.append({
            "crate": crate, "cargo_target": target, "path": row["path"], "name": row["name"],
            "libtest_name": libtest_name(root / row["path"], row["name"], root_file),
            "expected_seconds": row["expected_seconds"],
        })
    return rows


def identity(row: dict) -> str:
    return row["crate"] + ("" if row["cargo_target"] == "lib" else f"::{row['cargo_target']}")


def listing_problems(root: Path, listing: dict, ignored: list[dict]) -> list[str]:
    """Compare a `nextest list --run-ignored only` JSON with source and registry."""
    problems = []
    suites = listing.get("rust-suites", {})
    for row in slow_rows(root):
        case = suites.get(identity(row), {}).get("testcases", {}).get(row["libtest_name"])
        if (not case or not case.get("ignored")
                or case.get("filter-match", {}).get("status", "matches") != "matches"):
            problems.append(f"slow test is not a selected ignored test in the listing: {identity(row)}::{row['libtest_name']}")
    # A #[path] module is named by its parent in the listing, so match the leaf
    # fn name within the binary; macro-generated tests are covered per binary.
    leaves, macros = set(), set()
    for t in ignored:
        target = target_of(root, t["path"])
        if target is None:
            continue
        ident = target[0] + ("" if target[1] == "lib" else f"::{target[1]}")
        (macros.add(ident) if t["name"] is None else leaves.add((ident, t["name"])))
    for ident, suite in suites.items():
        for name, case in suite.get("testcases", {}).items():
            if case.get("ignored") and (ident, name.rsplit("::", 1)[-1]) not in leaves and ident not in macros:
                problems.append(f"ignored test unseen by the source scan (cfg_attr or unparsed macro?): {ident}::{name}")
    return problems


def nextest_filter(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        target = "kind(lib)" if r["cargo_target"] == "lib" else f"binary(={r['cargo_target']})"
        parts.append(f"(package(={r['crate']}) and {target} and test(={r['libtest_name']}))")
    return " or ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="check", choices=("check", "report", "slow-filter"))
    parser.add_argument("--listing", type=Path, help="nextest list --run-ignored only --message-format json output")
    args = parser.parse_args(argv)
    root = REPO_ROOT
    ignored, problems = classify(root)
    if args.listing:
        problems += listing_problems(root, json.loads(args.listing.read_text(encoding="utf-8")), ignored)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.command == "slow-filter":
        rows = slow_rows(root)
        if not rows:
            print("slow lane selection is empty", file=sys.stderr)
            return 1
        print(nextest_filter(rows))
        return 0
    counts = Counter(t["class"] for t in ignored)
    summary = ", ".join(f"{k}={counts[k]}" for k in ("calibration", "writer", "timing", "slow", "other"))
    print(f"{len(ignored)} ignore sites classified (a macro counts once): {summary}")
    if args.command == "report":
        for t in sorted(ignored, key=lambda t: (t["class"], t["path"], t["name"] or "")):
            if t["class"] not in ("calibration",):
                print(f"  {t['class']:8} {t['path']}::{t['name'] or 'macro ' + t['macro']}  [{t['reason']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
