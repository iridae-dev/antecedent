#!/usr/bin/env python3
"""Check the authoritative acceptance inventory (parity/acceptance_inventory.toml).

Fails when

  missing        a counterpart row, a manifest route, a method child, a scope entry for a route or
                 child, or a route obligation has no inventory row (or, for an obligation, no row
                 that carries it)
  conflict       the records disagree about a row: a status, a closure kind, a scope list, a child
                 gap or route, a closure against its residual limitation, or a route in no
                 conformance family. The inventory itself must record no conflict.
  unsupported    a licensed row has no explicit licence obligation, an obligation lacks its
                 identification or calibration licence, a coverage record is not recorded, or a
                 measured-record uncertainty has no record behind it
  stale          a row differs from what the records now derive, or the file differs from the
                 generator output

A route that executes without its claimed uncertainty or consumers is not a failure: the
inventory lists it `incomplete` with the reason.

    python3 scripts/check_acceptance_inventory.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import generate_acceptance_inventory as gen  # noqa: E402

SOURCE_BOUND = (
    ("status", "status"), ("closure_kind", "closure"), ("scope_status", "scope_status"),
    ("gap", "gap"), ("route", "route"), ("inference", "inference"),
)


def classify(message: str) -> str:
    return "unsupported" if "unsupported licensing evidence" in message or "explicit " in message or "coverage record" in message else "conflict"


def check(rows: list[dict], sources: dict) -> list[str]:
    problems: list[str] = []
    expected_rows = gen.build(sources)
    expected = {r["id"]: r for r in expected_rows}
    ix = gen.Index(sources)
    seen: dict[str, dict] = {}
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            problems.append(f"stale: {rid} appears twice")
        seen[rid] = row

    counts: dict[str, int] = {}
    for src in sources["counterparts"]:
        counts[src["id"]] = counts.get(src["id"], 0) + 1
    for cid, n in counts.items():
        if n > 1:
            problems.append(f"conflict: counterpart row {cid} appears {n} times")
    for rid in expected:
        if rid not in seen:
            problems.append(f"missing: no inventory row for {rid}")
    carried = {o for row in rows for o in row.get("obligations", [])}
    for ob in sources["obligations"]:
        if ob["id"] not in carried:
            problems.append(f"missing: route obligation {ob['id']} is carried by no inventory row")
    covered_routes = {row.get("route") for row in rows}
    for route in ix.manifest_routes:
        if route not in covered_routes:
            problems.append(f"missing: manifest route {route} has no inventory row")
    for child in ix.children:
        if child not in seen:
            problems.append(f"missing: method child {child} has no inventory row")
    for entry in sources["scope"]:
        if entry["source"] == "route_manifest" and entry["id"] not in covered_routes:
            problems.append(f"missing: release-scope route {entry['id']} has no inventory row")
        if entry["source"] == "method_children" and entry["id"] not in seen:
            problems.append(f"missing: release-scope child {entry['id']} has no inventory row")

    for rid, row in seen.items():
        want = expected.get(rid)
        if want is None:
            problems.append(f"stale: row {rid} names no counterpart row or manifest route")
            continue
        for message in row.get("conflicts", []):
            problems.append(f"{classify(message)}: {rid}: {message}")
        for message in want["conflicts"]:
            if message not in row.get("conflicts", []):
                problems.append(f"{classify(message)}: {rid}: {message}")
        differing = [n for n in gen.FIELDS if row.get(n) != want[n]]
        for name in differing:
            if name in dict(SOURCE_BOUND):
                problems.append(f"conflict: {rid}: {name} {row.get(name)!r} conflicts with the records ({want[name]!r})")
        stale = [n for n in differing if n not in dict(SOURCE_BOUND)]
        if stale:
            problems.append(f"stale: {rid} differs from the records in {', '.join(stale)}")
        if row.get("acceptance") == "complete" and (row.get("incomplete_reasons") or row.get("conflicts")):
            problems.append(f"conflict: {rid}: marked complete with open reasons")
        if row.get("acceptance") == "incomplete" and not row.get("incomplete_reasons") and not row.get("conflicts"):
            problems.append(f"conflict: {rid}: marked incomplete with no reason")
    return problems


def run(root: Path = ROOT) -> list[str]:
    path = root / gen.OUTPUT
    if not path.is_file():
        return [f"missing: {gen.OUTPUT} does not exist; run scripts/generate_acceptance_inventory.py"]
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("row", [])
    sources = gen.load_sources(root)
    problems = check(rows, sources)
    if path.read_text(encoding="utf-8") != gen.render(gen.build(sources)):
        problems.append(f"stale: {gen.OUTPUT} differs from its generator output")
    return problems


def main() -> int:
    problems = run()
    if problems:
        print(f"FAIL: acceptance inventory: {len(problems)} problem(s)")
        for p in problems[:200]:
            print(f"FAIL: {p}")
        return 1
    rows = tomllib.loads((ROOT / gen.OUTPUT).read_text(encoding="utf-8"))["row"]
    print(f"acceptance inventory reconciles the records ({gen.summary(rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
