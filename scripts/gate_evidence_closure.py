#!/usr/bin/env python3
"""Closure of the unit index (ADR 0031).

    python3 scripts/gate_evidence_closure.py [--self-test]

Fails when:
  * `parity/unit_index.json` or `parity/evidence_links.json` is stale or malformed;
  * a unit that derives as supported has no scientific pass;
  * a compiled route produced a result no unit decided, or a route whose unit is decided as it
    executes (a route cell outside parity/compiled_units.toml) ran under no traced test;
  * a coverage record is referenced by no unit and no retirement;
  * a unit's stated calibration reason contradicts its records: it says it measured
    nothing or reports no interval while a record measures it, or it names a boundary
    record and none measures it;
  * baseline evidence is neither linked by execution nor resolved (the residue), or a
    resolution is stale.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

UNMEASURED = {"estimator_grid_not_measured", "no_interval_reported"}
SUPPORTED = {"supported", "supported_with_limits"}


def problems() -> list[str]:
    import evidence_links as el
    import evidence_residue as er
    import unit_index as ui

    out: list[str] = []
    links = el.load()
    for table in ("passes", "records"):
        for ident, entry in links[table].items():
            for field in ("executed", "refused", "cited", "returned", "declared"):
                for unit in entry.get(field, []):
                    if len(unit) != 2:
                        out.append(f"evidence links: {table} {ident} has a malformed {field} unit")
    fresh = ui.render(ui.build())
    if not ui.OUT.is_file() or ui.OUT.read_text(encoding="utf-8") != fresh:
        out.append("parity/unit_index.json is stale: run python3 scripts/unit_index.py")
    data = json.loads(fresh)
    index = {(u["kind"], u["id"]): u for u in data["units"]}
    sources = er.claim_sources()

    for (kind, ident), unit in sorted(index.items()):
        if unit["state"]["support"] in SUPPORTED and not unit["scientific"]:
            out.append(f"{kind}:{ident}: supported with no scientific pass")
        row = sources.get((kind, ident)) or {}
        reason = row.get("calibration_reason")
        if reason in UNMEASURED and unit["calibration"]:
            out.append(f"{kind}:{ident}: states {reason} while {len(unit['calibration'])} "
                       "coverage record(s) measure it")
        if reason == "boundary_record" and not unit["calibration"]:
            out.append(f"{kind}:{ident}: states boundary_record and no coverage record measures it")

    referenced = set(data["records"])
    retired = {str(r["evidence"]).removeprefix("record:") for r in er.load_resolutions()
               if r.get("kind") == "retired" and str(r.get("evidence", "")).startswith("record:")}
    coverage = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
    orphans = sorted(str(r["id"]) for r in coverage.get("record", [])
                     if str(r["id"]) not in referenced and str(r["id"]) not in retired)
    for ident in orphans[:20]:
        out.append(f"coverage record {ident}: referenced by no unit")
    if len(orphans) > 20:
        out.append(f"... and {len(orphans) - 20} more coverage records referenced by no unit")

    out.extend(route_problems(links))

    rows, resolution_problems = er.residue()
    out.extend(f"evidence resolutions: {p}" for p in resolution_problems)
    if rows:
        out.append(f"evidence residue: {len(rows)} baseline evidence neither linked nor resolved "
                   "(python3 scripts/evidence_links.py residue)")
    return out


def compiled_routes() -> tuple[set[str], set[str]]:
    """(every name `LoweredProgram::route` can return, read from the route registry; the routes
    whose unit is decided as they execute). The others are bound to one unit in
    parity/compiled_units.toml, which scripts/gate_support.py closes against the registry."""
    import gate_support as gs

    registry = (ROOT / gs.PLAN_ROUTE_SOURCE).read_text(encoding="utf-8")
    families = (ROOT / gs.PROGRAM_FAMILIES_SOURCE).read_text(encoding="utf-8")
    posterior = set(re.findall(r'=> "(GraphPosterior[A-Za-z]+)",', registry))
    routes = gs.plan_route_variants(registry, families) | posterior
    bound = {row["lowering"] for row in tomllib.loads((ROOT / gs.COMPILED_UNITS).read_text(encoding="utf-8")).get("route", [])}
    return routes, routes - bound


def route_problems(links: dict, routes: tuple[set[str], set[str]] | None = None) -> list[str]:
    """Every compiled route that ran was decided by some unit, and every route whose unit is
    decided as it executes ran under a traced test. A single result may carry no unit: it
    reports no support standing, and a supported status without a unit reads as refused."""
    every, decided = compiled_routes() if routes is None else routes
    if not decided:
        return ["the route registry names no route whose unit is decided as it executes"]
    ran = links.get("routes") or {}
    out = [f"route {route}: ran, but no unit decided any of its results"
           for route in links.get("unowned_routes") or []]
    out.extend(f"route {route}: no traced test executes it" for route in sorted(decided - set(ran)))
    out.extend(f"route {route}: traced but not a compiled route" for route in sorted(set(ran) - every))
    return out


def self_test() -> None:
    routes = ({"Bound", "Decided", "Other"}, {"Decided", "Other"})
    ran = {"Decided": ["matrix\tx"], "Other": ["capability\ty"]}
    assert route_problems({"routes": ran, "unowned_routes": []}, routes) == []
    assert route_problems({"routes": ran, "unowned_routes": ["Other"]}, routes) == [
        "route Other: ran, but no unit decided any of its results"]
    assert route_problems({"routes": {"Decided": []}}, routes) == ["route Other: no traced test executes it"]
    assert route_problems({"routes": {**ran, "Gone": []}}, routes) == [
        "route Gone: traced but not a compiled route"]
    assert route_problems({"routes": ran}, (set(), set())) != []


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("evidence closure self-test: ok")
        return 0
    found = problems()
    for problem in found:
        print(f"evidence closure: {problem}", file=sys.stderr)
    if found:
        return 1
    print("evidence closure: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
