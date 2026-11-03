#!/usr/bin/env python3
"""Check the route obligations (parity/route_obligations.toml).

Fails when

  missing        a supported route has no obligation row
  stale          a row differs from what the recipes, the support table and
                 crates/antecedent/data/support_matrix.json now derive, names a route the
                 matrix does not support, or the matrix and the source tables disagree
  unreachable    a row cites a test that does not resolve as an executing checked-execution
                 test, or a fixture that does not exist
  recipe_only    a row is supported without an explicit identification evidence and
                 uncertainty, claims a basis other than the explicit support, or is not a
                 supported row at all (a route that is not supported has no obligation)

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
    ident = str(row.get("identification_evidence") or "").split("|")
    return (ident[1], ident[2]) if len(ident) == 4 and ident[1] and ident[2] else None


def recipe_only_problems(row: dict) -> list[str]:
    rid = row.get("id")
    problems = []
    if row.get("kind") == "supported":
        if row.get("support_basis") != "explicit_support":
            problems.append(
                f"recipe_only: {rid} is supported on basis {row.get('support_basis')!r}, not the explicit support"
            )
        ident = str(row.get("identification_evidence") or "").split("|")
        if len(ident) != 4 or not all(ident[:3]):
            problems.append(
                f"recipe_only: {rid} has no explicit identification evidence (evidence kind, test, assertion)"
            )
        if not row.get("uncertainty"):
            problems.append(f"recipe_only: {rid} states no uncertainty")
        if not row.get("compiler_execution"):
            problems.append(f"recipe_only: {rid} cites no compiler execution")
    else:
        problems.append(f"recipe_only: {rid} is not a supported row (kind {row.get('kind')!r})")
    return problems


def check(
    rows: list[dict],
    sources: dict,
    *,
    cite=None,
    fixture_exists=None,
    assertion_present=None,
) -> list[str]:
    problems: list[str] = []
    matrix = sources["matrix"]
    problems += [
        f"stale: {p}"
        for p in gate.cross_check(
            sources["supported"],
            matrix.get("supported") or [],
            sources["closed"],
            sources["na"],
            closed_snapshot=matrix.get("closed_rules") or [],
            na_snapshot=matrix.get("na_rules") or [],
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
            problems.append(f"missing: no obligation row for supported route {rid}")
    supported_keys = {gate.coordinate(c) for c in matrix.get("supported") or []}
    for rid, row in seen.items():
        want = expected.get(rid)
        if want is None:
            why = (
                "the matrix does not support it"
                if row.get("kind") == "supported" and rid not in supported_keys
                else "no supported route derives it"
            )
            problems.append(f"stale: row {rid} has no supported route ({why})")
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
        ident = str(row.get("identification_evidence") or "").split("|")
        paths = list(reach) + ([ident[3]] if len(ident) == 4 and ident[3] else [])
        for path in paths:
            if fixture_exists is not None and not fixture_exists(path):
                problems.append(f"unreachable: {rid} cites {path!r}, which does not exist")
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
        lambda test, assertion: tuple(
            test_evidence.checked_execution_citation_problems(test, assertion)
        )
    )

    path = root / gen.OUTPUT
    if not path.is_file():
        return [f"missing: {gen.OUTPUT} does not exist; run scripts/generate_route_obligations.py"]
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("obligation", [])
    problems = check(
        rows,
        gen.load_sources(root),
        cite=cite,
        fixture_exists=lambda rel: (root / rel).exists(),
        assertion_present=lambda rel, name: assertion_problems(root, rel, name),
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
    print("route obligations match the recipes, the support table and the support matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
