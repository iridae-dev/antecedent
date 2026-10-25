#!/usr/bin/env python3
"""Check the route obligations (parity/route_obligations.toml).

Fails when

  missing        an executing route (a licensed coordinate or an executing closed rule) has no
                 obligation row
  stale          a row differs from what the recipes, the license table and
                 crates/antecedent/data/support_matrix.json now derive, names a route the
                 matrix does not license, or the matrix and the source tables disagree
  unreachable    a row cites a test that does not resolve as an executing checked-execution
                 test, a fixture that does not exist, or a calibration record that is not in
                 parity/coverage_records.toml
  recipe_only    a row is licensed without an explicit identification license, calibration
                 license and uncertainty, claims a basis other than the explicit license, or
                 an execution-only or closed route carries a license

    python3 scripts/check_route_obligations.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import gate_recipe_obligations as gate  # noqa: E402
import generate_route_obligations as gen  # noqa: E402


def citations(row: dict) -> list[tuple[str, str]]:
    """(test, assertion) pairs of the compiler execution a row cites."""
    pairs = []
    for entry in row.get("compiler_execution") or []:
        head, sep, tail = str(entry).partition(" = ")
        test, _, assertion = tail.partition("::")
        if sep and test and assertion:
            pairs.append((test, assertion))
    return pairs


def identification_citation(row: dict) -> tuple[str, str] | None:
    ident = str(row.get("identification_license") or "").split("|")
    return (ident[1], ident[2]) if len(ident) == 4 and ident[1] and ident[2] else None


def recipe_only_problems(row: dict) -> list[str]:
    rid = row.get("id")
    problems = []
    if row.get("kind") == "licensed":
        if row.get("license_basis") != "explicit_license":
            problems.append(f"recipe_only: {rid} is licensed on basis {row.get('license_basis')!r}, not the explicit license")
        ident = str(row.get("identification_license") or "").split("|")
        if len(ident) != 4 or not all(ident[:3]):
            problems.append(f"recipe_only: {rid} has no explicit identification license (evidence kind, test, assertion)")
        if not row.get("calibration_license"):
            problems.append(f"recipe_only: {rid} has no explicit calibration license (records or reason)")
        if not row.get("uncertainty"):
            problems.append(f"recipe_only: {rid} states no uncertainty")
        if not row.get("compiler_execution"):
            problems.append(f"recipe_only: {rid} cites no compiler execution")
    else:
        for name, none in (("license_basis", "none"), ("identification_license", "none"), ("calibration_license", "none")):
            if row.get(name) != none:
                problems.append(f"recipe_only: {rid} executes unlicensed yet carries {name}={row.get(name)!r}")
    return problems


def check(
    rows: list[dict],
    sources: dict,
    *,
    cite=None,
    fixture_exists=None,
    assertion_present=None,
    coverage_ids: set[str] | None = None,
) -> list[str]:
    problems: list[str] = []
    matrix = sources["matrix"]
    problems += [
        f"stale: {p}"
        for p in gate.cross_check(
            sources["licensed"], matrix.get("licensed") or [], sources["closed"], sources["na"],
            closed_snapshot=matrix.get("closed_rules") or [], na_snapshot=matrix.get("na_rules") or [],
            axes=sources["axes"],
        )
    ]
    expected = {r["id"]: r for r in gen.build(sources)}
    seen: dict[str, dict] = {}
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            problems.append(f"stale: {rid} appears twice")
        seen[rid] = row
    for rid in expected:
        if rid not in seen:
            problems.append(f"missing: no obligation row for executing route {rid}")
    licensed_keys = {gate.coordinate(c) for c in matrix.get("licensed") or []}
    for rid, row in seen.items():
        want = expected.get(rid)
        if want is None:
            why = "the matrix does not license it" if row.get("kind") == "licensed" and rid not in licensed_keys else "no executing route derives it"
            problems.append(f"stale: row {rid} has no executing route ({why})")
            problems += recipe_only_problems(row)
            continue
        differing = [n for n in gen.fields_of(want) if row.get(n) != want[n]]
        if differing:
            problems.append(f"stale: {rid} differs from the sources in {', '.join(differing)}")
        problems += recipe_only_problems(row)
        for test, assertion in citations(row):
            if cite is not None:
                problems += [f"unreachable: {rid}: {p}" for p in cite(test, assertion)]
        named = identification_citation(row)
        if named is not None and assertion_present is not None:
            problems += [f"unreachable: {rid}: {p}" for p in assertion_present(*named)]
        reach = row.get("evidence_reachability") or []
        ident = str(row.get("identification_license") or "").split("|")
        paths = list(reach) + ([ident[3]] if len(ident) == 4 and ident[3] else [])
        for path in paths:
            if fixture_exists is not None and not fixture_exists(path):
                problems.append(f"unreachable: {rid} cites {path!r}, which does not exist")
        cal = str(row.get("calibration_license") or "")
        if cal.startswith("records:") and coverage_ids is not None:
            for record in cal[len("records:"):].split(","):
                if record not in coverage_ids:
                    problems.append(f"unreachable: {rid} cites coverage record {record!r}, which is not recorded")
    return problems


def assertion_problems(root: Path, rel: str, name: str) -> list[str]:
    """The identification evidence test file exists and names the assertion."""
    path = root / rel
    if not path.is_file():
        return [f"identification test {rel!r} does not exist"]
    if name not in path.read_text(encoding="utf-8", errors="replace"):
        return [f"identification test {rel!r} does not name {name!r}"]
    return []


def run(root: Path = ROOT) -> list[str]:
    import functools

    import test_evidence  # noqa: E402

    cite = functools.lru_cache(maxsize=None)(
        lambda test, assertion: tuple(test_evidence.checked_execution_citation_problems(test, assertion))
    )

    path = root / gen.OUTPUT
    if not path.is_file():
        return [f"missing: {gen.OUTPUT} does not exist; run scripts/generate_route_obligations.py"]
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("obligation", [])
    records = tomllib.loads((root / "parity/coverage_records.toml").read_text(encoding="utf-8")).get("record", [])
    problems = check(
        rows,
        gen.load_sources(root),
        cite=cite,
        fixture_exists=lambda rel: (root / rel).exists(),
        assertion_present=lambda rel, name: assertion_problems(root, rel, name),
        coverage_ids={r["id"] for r in records},
    )
    if path.read_text(encoding="utf-8") != gen.render(gen.build(gen.load_sources(root))):
        problems.append(f"stale: {gen.OUTPUT} differs from its generator output")
    return problems


def main() -> int:
    problems = run()
    if problems:
        print(f"FAIL: route obligations: {len(problems)} problem(s)")
        for p in problems[:200]:
            print(f"FAIL: {p}")
        return 1
    print("route obligations match the recipes, the license table and the support matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
