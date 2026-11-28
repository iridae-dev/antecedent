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
    resolution is stale;
  * against `parity/evidence_baseline.json`, a pass linked to a unit is missing from its index
    record, or a unit that had coverage records has none and no resolution settles them;
  * a unit states no_interval_reported while a coverage record measured on its exact
    coordinates and estimator shows its publisher returned an interval;
  * crate source that can read `antecedent_io_core::unit_calibration` states a calibration
    as a literal (`calibration: "not_measured"`, a `&str` constant of it, or a comparison
    with it) instead of reading its unit's state.
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
    import evidence_baseline as eb
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

    referenced = set(data["records"]) | set(data.get("unadmitted", {}))
    retired = {e.removeprefix("record:") for r in er.load_resolutions()
               if r.get("kind") in ("retired", "narrowed") for e in er.evidence_of(r) if e.startswith("record:")}
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

    out.extend(monotonicity_problems(
        {(i["kind"], i["id"]): i for i in eb.load()["items"]}, ui.resolve(data),
        er.settled_evidence(er.load_resolutions()),
        {(r["item"], r["evidence"]) for r in rows}, set(sources)))

    cells = [row for (kind, _), row in sources.items() if kind == "matrix"]
    published = published_intervals(cells, coverage.get("record", []))
    for unit, methods in (links.get("intervals") or {}).items():
        returned = set(methods) - {"none"}
        if returned:
            published.setdefault(tuple(unit.split("\t", 1)), set()).update(returned)
    out.extend(interval_problems(
        {key: row.get("calibration_reason") for key, row in sources.items()}, published))
    out.extend(calibration_literal_problems())
    return out


LOOKUP = "antecedent-io-core"
CALIBRATION_LITERAL = re.compile(
    r'calibration\w*"?\s*(?::|!=|==|,)\s*(?:&\w+\s*=\s*)?"not_measured"'
    r'|const\s+\w*CALIBRATION\w*\s*:\s*&(?:\'static\s+)?str\s*=\s*"not_measured"', re.I)
TEST_MODULE = re.compile(r"#\[cfg\(test\)\]\s*\n\s*(?:pub(?:\(\w+\))?\s+)?mod\s+\w+\s*\{")


def without_test_modules(text: str) -> str:
    """`text` with each `#[cfg(test)] mod` body blanked, keeping line numbers. Braces inside
    strings or comments are counted too, which can only blank less than the whole module."""
    out, at = [], 0
    for found in TEST_MODULE.finditer(text):
        if found.start() < at:
            continue
        depth, end = 1, found.end()
        while end < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        out.append(text[at:found.start()])
        out.append(re.sub(r"[^\n]", " ", text[found.start():end]))
        at = end
    out.append(text[at:])
    return "".join(out)


def lookup_crates() -> set[str]:
    """The unit calibration lookup's crate and the crates whose manifests depend on it."""
    out = {LOOKUP}
    for manifest in (ROOT / "crates").glob("*/Cargo.toml"):
        text = manifest.read_text(encoding="utf-8")
        if re.search(rf"^{LOOKUP}\s*=", text, re.M):
            out.add(manifest.parent.name)
    return out


def calibration_literal_problems(text_of=None) -> list[str]:
    out = []
    for crate in sorted(lookup_crates()):
        for path in sorted((ROOT / "crates" / crate / "src").rglob("*.rs")):
            if path.stem == "tests" or path.stem.endswith("_tests"):
                continue
            text = text_of(path) if text_of else path.read_text(encoding="utf-8")
            for n, line in enumerate(without_test_modules(text).splitlines(), 1):
                if not line.lstrip().startswith("//") and CALIBRATION_LITERAL.search(line):
                    out.append(f"{path.relative_to(ROOT)}:{n}: states calibration as a literal; "
                               "read antecedent_io_core::unit_calibration for its unit")
    return out


def monotonicity_problems(items: dict[tuple[str, str], dict], index: dict[tuple[str, str], dict],
                          resolved: dict[tuple[str, str], set[str]], residue: set[tuple[str, str]],
                          sources: set[tuple[str, str]]) -> list[str]:
    """Against the baseline: every scientific pass an item had is attached to its unit (or a
    construction on it) in the index, and a unit that had coverage records still has one.

    A pass the residue already reports, or a resolution settles, is not repeated here; the
    residue reads the execution links, this reads the join built from them, so what is left
    is evidence linked to the unit that the index still drops. A unit with no claim source
    is retired and the residue reports it. A construction on a cell is the successor unit
    the cell's evidence executes as."""
    successors: dict[tuple[str, str], list[dict]] = {}
    for (kind, ident), unit in index.items():
        if kind == "construction":
            successors.setdefault(("matrix", ident.partition("#")[0]), []).append(unit)

    def attached(key: tuple[str, str], half: str) -> set[str]:
        return {x for unit in [index.get(key, {}), *successors.get(key, [])] for x in unit.get(half, [])}

    out = []
    for key, item in sorted(items.items()):
        if key not in sources:
            continue
        label = f"{key[0]}:{key[1]}"
        done = resolved.get(key, set())
        if "*" in done:
            continue
        passes = {p for e in item["evidence"].values() for ps in e["scientific"].values() for p in ps}
        have = attached(key, "scientific")
        lost = sorted(p for p in passes - have
                      if f"pass:{p}" not in done and (label, f"pass:{p}") not in residue)
        out.extend(f"{label}: baseline pass {p} is linked to it but not in the unit index" for p in lost)
        records = {r for e in item["evidence"].values() for r in e["calibration"]}
        if records and not attached(key, "calibration") and any(f"record:{r}" not in done for r in records):
            state = index.get(key, {}).get("state", {}).get("calibration", "no unit")
            out.append(f"{label}: had {len(records)} coverage record(s) at the baseline and now has none "
                       f"(calibration {state})")
    return out


def published_intervals(cells: list[dict], records: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Matrix unit -> the interval methods of the coverage records measured on its exact
    coordinates and estimator: each record is an interval the publisher returned for it."""
    import collect_coverage_records as ccr
    import evidence_baseline as eb

    by_axes: dict[tuple, list[dict]] = {}
    for record in records:
        if record.get("interval_method") not in (None, "", "none"):
            axes = (record.get("query"), record.get("graph_class"), record.get("inference"), record.get("structure"))
            by_axes.setdefault(axes, []).append(record)
    out: dict[tuple[str, str], set[str]] = {}
    for cell in cells:
        axes = (cell.get("query"), cell.get("graph_class"), cell.get("inference"),
                ccr.supported_record_structure(cell, {}))
        methods = {str(r["interval_method"]) for r in by_axes.get(axes, [])
                   if ccr.supported_record_matches(cell, r)}
        if methods:
            out[("matrix", eb.cell_ident(cell))] = methods
    return out


def interval_problems(reasons: dict[tuple[str, str], str | None],
                      published: dict[tuple[str, str], set[str]]) -> list[str]:
    return [f"{kind}:{ident}: states no_interval_reported while its publisher returned "
            f"{', '.join(sorted(published[(kind, ident)]))} intervals"
            for (kind, ident), reason in sorted(reasons.items())
            if reason == "no_interval_reported" and published.get((kind, ident))]


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

    cell = ("matrix", "Q/Dag/explicit/Frequentist/none")
    item = {"evidence": {"v3.0.0": {"scientific": {"t.rs::a": ["p1", "p2"]}, "calibration": ["r1"]},
                         "pre": {"scientific": {"t.rs::b": ["p3"]}, "calibration": []}}}
    kept = {"scientific": ["p1", "p2", "p3"], "calibration": ["r1"], "state": {"calibration": "calibrated"}}
    bare = {"scientific": ["p1"], "calibration": [], "state": {"calibration": "no_interval_reported"}}
    construction = {"scientific": ["p2"], "calibration": ["r1"]}
    sources = {cell}
    assert monotonicity_problems({cell: item}, {cell: kept}, {}, set(), sources) == []
    assert monotonicity_problems({cell: item}, {cell: bare}, {}, set(), sources) == [
        f"{cell[0]}:{cell[1]}: baseline pass p2 is linked to it but not in the unit index",
        f"{cell[0]}:{cell[1]}: baseline pass p3 is linked to it but not in the unit index",
        f"{cell[0]}:{cell[1]}: had 1 coverage record(s) at the baseline and now has none "
        "(calibration no_interval_reported)"]
    on_construction = {cell: bare, ("construction", f"{cell[1]}#k"): construction}
    label = f"{cell[0]}:{cell[1]}"
    assert monotonicity_problems({cell: item}, on_construction, {}, {(label, "pass:p3")}, sources) == []
    assert monotonicity_problems({cell: item}, {cell: bare}, {cell: {"pass:p2", "pass:p3", "record:r1"}},
                                 set(), sources) == []
    assert monotonicity_problems({cell: item}, {cell: bare}, {cell: {"*"}}, set(), sources) == []
    assert monotonicity_problems({cell: item}, {}, {}, set(), set()) == []

    reasons = {cell: "no_interval_reported", ("matrix", "other"): "boundary_record",
               ("capability", "c"): "no_interval_reported"}
    published = {cell: {"bootstrap_se", "analytic_se"}, ("matrix", "other"): {"analytic_se"}}
    assert interval_problems(reasons, published) == [
        f"{label}: states no_interval_reported while its publisher returned analytic_se, bootstrap_se intervals"]
    assert interval_problems(reasons, {}) == []
    row = {"query": "Q", "graph_class": "Dag", "structure": "explicit", "inference": "Frequentist",
           "validation": "none", "estimators": ["e"]}
    record = {"query": "Q", "graph_class": "Dag", "structure": "fixed", "inference": "Frequentist",
              "validation": "none", "validation_schema": 2, "estimator": "e", "interval_method": "analytic_se",
              "id": "cov.q.dag.frequentist.analytic_se.l95.x.v2_validation_none"}
    assert published_intervals([row], [record, {**record, "estimator": "other", "interval_method": "x"},
                                        {**record, "interval_method": "none"}]) == {cell: {"analytic_se"}}
    assert published_intervals([{**row, "inference": "Bayesian"}], [record]) == {}

    for line in ('calibration: "not_measured",', 'pub const X_CALIBRATION: &str = "not_measured";',
                 '|| wire.calibration != "not_measured"', '("calibration", "not_measured".to_owned()),',
                 '"calibration":"not_measured"'):
        assert CALIBRATION_LITERAL.search(line), line
    for line in ('Self::NotMeasured => "not_measured",', '"calibrated" | "not_measured"',
                 'pub const NOT_MEASURED: &str = "not_measured";'):
        assert not CALIBRATION_LITERAL.search(line), line
    source = 'a\n#[cfg(test)]\nmod tests {\n    fn f() { x }\n}\nb\n'
    assert without_test_modules(source).splitlines()[5] == "b"
    assert "x" not in without_test_modules(source)


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
