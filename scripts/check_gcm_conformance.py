#!/usr/bin/env python3
"""Static contract of the programmable-intervention, selection, falsification, replay and
intrinsic-influence conformance registry (parity/gcm_conformance.toml).

    python3 scripts/check_gcm_conformance.py
    python3 scripts/check_gcm_conformance.py --require-bound
    python3 scripts/check_gcm_conformance.py --self-test

The registry is executed by scripts/gate_gcm_conformance.sh; this check proves, without
running anything, that the execution has something to run:

- every clause the registry declares has at least one cited test, and every cited test
  exists (Rust `fn`, Python `def`) in the file it names;
- every benchmark is registered with Cargo, defines its entry function, asserts a regression
  bound against a baseline measured in the same process, is indexed in docs/hot_paths.md and is
  executed by the gate;
- every acceptance-manifest case in the areas this gate owns is bound to a test the gate runs
  (and, with `--require-bound`, none is left planned).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT as ROOT
from check_acceptance_manifest import evidence_problems

REGISTRY = "parity/gcm_conformance.toml"
MANIFEST = "parity/acceptance_manifest.toml"
GATE = "scripts/gate_gcm_conformance.sh"
# Areas whose bound cases must be tests this gate runs.
OWNED_AREAS = frozenset(
    {
        "shared_exogenous",
        "recalculation_bridge",
        "performance_conformance",
        "release_conformance",
    }
)


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def registry_problems(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    clauses = data.get("clauses")
    rows = data.get("conformance")
    benches = data.get("performance")
    if not isinstance(clauses, dict) or not clauses:
        return ["registry declares no [clauses]"]
    if not isinstance(rows, list) or not rows:
        return ["registry has no [[conformance]] rows"]
    seen: set[str] = set()
    covered: set[str] = set()
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"row without an id: {row}")
            continue
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
        clause = row.get("clause")
        if clause not in clauses:
            errors.append(f"{rid}: unknown clause {clause!r}")
        else:
            covered.add(clause)
        test, assertion = row.get("evidence_test"), row.get("evidence_assertion")
        if not isinstance(test, str) or not isinstance(assertion, str):
            errors.append(f"{rid}: needs evidence_test and evidence_assertion")
            continue
        errors.extend(
            evidence_problems(rid, {"status": "bound", "test": f"{test}::{assertion}"}, root)
        )
    for clause in sorted(set(clauses) - covered):
        errors.append(f"clause {clause} has no cited test")
    errors.extend(performance_problems(benches, covered, root))
    return errors


def performance_problems(benches: Any, covered: set[str], root: Path) -> list[str]:
    if not isinstance(benches, list) or not benches:
        return ["registry has no [[performance]] benchmark"]
    errors: list[str] = []
    gate = _read(root, GATE) if (root / GATE).is_file() else ""
    hot_paths = _read(root, "docs/hot_paths.md") if (root / "docs/hot_paths.md").is_file() else ""
    if '["performance"]' not in gate:
        errors.append(f"{GATE} does not run the registry's [[performance]] benchmarks")
    for bench in benches:
        bid = bench.get("id", "<no id>")
        path, name, package = (
            bench.get("file"),
            bench.get("bench"),
            bench.get("package"),
        )
        function = bench.get("function")
        if not all(isinstance(v, str) and v for v in (path, name, package, function)):
            errors.append(f"{bid}: needs package, bench, file and function")
            continue
        source = root / path
        if not source.is_file():
            errors.append(f"{bid}: bench file {path} does not exist")
            continue
        text = source.read_text(encoding="utf-8")
        if not re.search(rf"\bfn\s+{re.escape(function)}\b", text):
            errors.append(f"{bid}: {path} does not define {function}")
        if "MAX_REGRESSION" not in text or not re.search(
            r"assert!\(\s*ratio\s*<=\s*MAX_REGRESSION", text
        ):
            errors.append(f"{bid}: {path} must assert `ratio <= MAX_REGRESSION`")
        manifest = root / "crates" / package / "Cargo.toml"
        declared = (
            tomllib.loads(manifest.read_text(encoding="utf-8")).get("bench", [])
            if manifest.is_file()
            else []
        )
        if not any(b.get("name") == name and b.get("harness") is False for b in declared):
            errors.append(f"{bid}: {name} is not a harness = false [[bench]] of {package}")
        smoke = f"cargo bench -p {package} --bench {name}"
        if smoke not in hot_paths:
            errors.append(f"{bid}: docs/hot_paths.md has no smoke command `{smoke} -- --test`")
    return errors


def manifest_problems(
    manifest: dict[str, Any], data: dict[str, Any], require_bound: bool
) -> list[str]:
    cited = {
        f"{r['evidence_test']}::{r['evidence_assertion']}" for r in data.get("conformance", [])
    }
    cited |= {f"{b['file']}::{b['function']}" for b in data.get("performance", [])}
    errors: list[str] = []
    for case in manifest.get("case", []):
        if case.get("area") not in OWNED_AREAS:
            continue
        evidence = case.get("evidence", {})
        if evidence.get("status") != "bound":
            if require_bound:
                errors.append(f"{case['id']}: planned case in an area the conformance gate owns")
            continue
        if evidence.get("test") not in cited:
            errors.append(
                f"{case['id']}: bound to {evidence.get('test')}, which {GATE} does not run"
            )
    return errors


def problems(root: Path = ROOT, require_bound: bool = False) -> list[str]:
    data = tomllib.loads(_read(root, REGISTRY))
    errors = registry_problems(data, root)
    manifest = tomllib.loads(_read(root, MANIFEST))
    errors.extend(manifest_problems(manifest, data, require_bound))
    return errors


def self_test() -> int:
    good = tomllib.loads(_read(ROOT, REGISTRY))

    def corrupted(edit) -> dict[str, Any]:
        data = tomllib.loads(_read(ROOT, REGISTRY))
        edit(data)
        return data

    failures: list[str] = []

    def expect(label: str, data: dict[str, Any], needle: str | None) -> None:
        found = registry_problems(data)
        if needle is None and found:
            failures.append(f"{label}: unexpected {found[:2]}")
        elif needle is not None and not any(needle in e for e in found):
            failures.append(f"{label}: wanted {needle!r}, got {found[:2]}")

    expect("committed registry", good, None)
    expect(
        "clause without a test",
        corrupted(
            lambda d: d.update(
                conformance=[r for r in d["conformance"] if r["clause"] != "falsification_correct"]
            )
        ),
        "falsification_correct has no cited test",
    )
    expect(
        "missing test function",
        corrupted(lambda d: d["conformance"][0].update(evidence_assertion="no_such_test")),
        "does not define no_such_test",
    )
    expect(
        "missing test file",
        corrupted(lambda d: d["conformance"][0].update(evidence_test="crates/none/tests/none.rs")),
        "does not exist",
    )
    expect(
        "unknown clause",
        corrupted(lambda d: d["conformance"][0].update(clause="nowhere")),
        "unknown clause",
    )
    expect(
        "duplicate id",
        corrupted(lambda d: d["conformance"].append(dict(d["conformance"][0]))),
        "duplicate id",
    )
    expect(
        "bench without an entry function",
        corrupted(lambda d: d["performance"][0].update(function="absent_function")),
        "does not define absent_function",
    )
    expect(
        "bench not registered",
        corrupted(lambda d: d["performance"][0].update(bench="unregistered")),
        "not a harness = false [[bench]]",
    )
    manifest = {
        "case": [
            {
                "id": "c",
                "area": "shared_exogenous",
                "evidence": {"status": "bound", "test": "x.rs::y"},
            }
        ]
    }
    if not manifest_problems(manifest, good, False):
        failures.append("a bound case the gate does not run was accepted")
    planned = {
        "case": [
            {
                "id": "c",
                "area": "recalculation_bridge",
                "evidence": {"status": "planned"},
            }
        ]
    }
    if manifest_problems(planned, good, False) or not manifest_problems(planned, good, True):
        failures.append("planned cases are not handled by --require-bound")
    if failures:
        print("check_gcm_conformance self-test FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("check_gcm_conformance self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    unknown = [a for a in argv if a != "--require-bound"]
    if unknown:
        print(f"unknown arguments: {unknown}", file=sys.stderr)
        return 2
    errors = problems(require_bound="--require-bound" in argv)
    if errors:
        print("gcm conformance check FAILED:")
        for error in errors:
            print(f"  {error}")
        return 1
    data = tomllib.loads(_read(ROOT, REGISTRY))
    print(
        f"gcm conformance: {len(data['clauses'])} clauses, {len(data['conformance'])} tests, "
        f"{len(data['performance'])} benchmarks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
