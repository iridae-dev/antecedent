#!/usr/bin/env python3
"""One support rule for every supported unit, and the one support count.

A supported unit is one scientific combination, from exactly one of three sources:

  matrix cell     parity/support_cells.toml        one [[cell]] per (query, graph class,
                                                    structure, inference, validation)
  graphless row   parity/support_graphless.toml    one [[support]] per design-family method
  capability      parity/capability_records.toml   one supported [[record]] for a capability
                  outside the matrix               with no matrix axes (transport stages,
                                                    adaptive procedures, decisions, bridges)

A record whose scientific combination is already a matrix cell or graphless row names it with
`matrix_units` (selectors over the five cell axes) or `graphless_units` (selectors over family,
design, method). Such a record is an alias of those units: it is not counted again, and its
routes are entry points of them.

One rule decides support for every unit:

  provenance   a provenance record (provenance/*.toml) exists for the algorithm
  independent  an executed check against something other than this library's own output:
  check        known truth, an analytic or enumerated value, or an external/reference
               implementation. Internal consistency, a frozen own output (regression pin)
               and "implementation exists" never count.

API and stage routes listed under a record are entry points. They have no support status of
their own: a route is open only as an entry point of a supported unit, and refuses with
`not_supported` otherwise. Route counts are never a support count.

    python3 scripts/gate_support.py            # fail on any unit that violates the rule
    python3 scripts/gate_support.py --report   # print every violation, exit 0

Static source reading only: no cargo, no test execution. Cited tests are resolved as declared,
compiled, non-ignored test functions; executing them stays with the evidence runners.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()

import provenance_links  # noqa: E402
import test_evidence  # noqa: E402
from evidence_kinds import KNOWN_TRUTH_KINDS, fixture_kind_problems  # noqa: E402
from route_methods import IDS, parse_estimators  # noqa: E402

CELLS = "parity/support_cells.toml"
GRAPHLESS = "parity/support_graphless.toml"
RECORDS = "parity/capability_records.toml"
MANIFEST = "parity/capabilities/route_manifest.toml"

CELL_AXES = ("query", "graph_class", "structure", "inference", "validation")
GRAPHLESS_AXES = ("family", "design", "method")

# Fixture roles that are an independent scientific check. `truth` is exact agreement with a
# closed form, an enumerated model or a simulation truth; `parity` is agreement with a
# reference implementation. Positive, negative, artifact, invariance, budget, lifecycle and
# regression fixtures are software checks; a calibration fixture is a measured coverage
# attestation and is a separate claim (see check_capability_records / coverage records).
INDEPENDENT_FIXTURE_ROLES = ("truth", "parity")
# A record fixture may state its own evidence kind; only a known-truth kind counts.
CALIBRATION_ROLE = "calibration"


@dataclass
class Unit:
    kind: str  # matrix | graphless | capability
    ident: str
    problems: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.problems


def _load(rel: str) -> dict:
    return tomllib.loads((ROOT / rel).read_text(encoding="utf-8"))


_TEST_CACHE: dict[tuple[str, str], list[str]] = {}


def test_problems(path: str, name: str) -> list[str]:
    """Why `path::name` is not a declared, compiled, non-ignored test (empty = it is)."""
    key = (path, name)
    if key not in _TEST_CACHE:
        file = ROOT / path
        if not file.is_file():
            found = [f"{path}: missing"]
        elif path.endswith(".rs"):
            found = list(test_evidence.static_rust_test(file, name).problems)
        elif path.endswith(".py"):
            found = list(test_evidence.static_python_test(file, name))
        else:
            found = [f"{path}: not a Rust or Python test"]
        _TEST_CACHE[key] = found
    return _TEST_CACHE[key]


_SIMULATIONS: set[tuple[str, str]] | None = None


def calibration_simulations() -> set[tuple[str, str]]:
    """Registered repeated-sampling calibration checks (parity/v3_test_registry.toml)."""
    global _SIMULATIONS
    if _SIMULATIONS is None:
        _SIMULATIONS = {
            (r["evidence_test"], r["evidence_assertion"])
            for r in test_evidence.registered_calibration_simulations()
        }
    return _SIMULATIONS


class Provenance:
    """Which provenance record an implementation id resolves to."""

    def __init__(self) -> None:
        self.stems = provenance_links.stems(ROOT)
        self.estimators = parse_estimators((ROOT / IDS).read_text(encoding="utf-8"))

    def estimator(self, wire_id: str) -> str | None:
        """None when the estimator's family resolves to a record; else why not."""
        info = self.estimators.get(wire_id)
        if info is None:
            return f"estimator {wire_id!r} is not a closed EstimatorId"
        family = info["provenance"][0]
        members = provenance_links.FAMILY.get(family, (family,))
        missing = [m for m in members if provenance_links.resolve(m, self.stems) is None]
        if missing:
            return f"estimator {wire_id!r}: no provenance record for {', '.join(missing)}"
        return None

    def module(self, module: str) -> list[str]:
        return [s for s in provenance_links.records_for(module, ROOT) if s in self.stems]


# --------------------------------------------------------------------------- matrix cells
def cell_ident(cell: dict) -> str:
    return "/".join(str(cell.get(axis)) for axis in CELL_AXES)


def check_cell(cell: dict, prov: Provenance) -> Unit:
    unit = Unit("matrix", cell_ident(cell))
    estimators = cell.get("estimators") or []
    if not estimators:
        unit.problems.append("no provenance: the cell names no estimator")
    for est in estimators:
        why = prov.estimator(est)
        if why:
            unit.problems.append(f"no provenance: {why}")
    kind = cell.get("evidence_kind")
    if kind not in KNOWN_TRUTH_KINDS:
        unit.problems.append(
            f"no independent check: evidence_kind {kind} is not known truth or an external oracle"
        )
        return unit
    fixture = cell.get("known_truth_fixture")
    if not isinstance(fixture, str) or not (ROOT / fixture).exists():
        unit.problems.append(f"no independent check: known_truth_fixture {fixture!r} is missing")
    else:
        for why in fixture_kind_problems(ROOT, cell):
            unit.problems.append(f"no independent check: {why}")
    test, name = cell.get("evidence_test"), cell.get("evidence_assertion")
    if not (test and name):
        unit.problems.append("no independent check: no evidence_test/evidence_assertion")
    else:
        if (test, name) in calibration_simulations():
            unit.problems.append(
                f"no independent check: {test}::{name} is a calibration simulation, not an independent check"
            )
        for why in test_problems(test, name):
            unit.problems.append(f"no independent check: {test}::{name}: {why}")
    return unit


# --------------------------------------------------------------------------- graphless rows
def graphless_ident(row: dict) -> str:
    return f"{row['family']}:{row['design']}:{row['method']}:{row['inference_claim']}"


def graphless_routes(row: dict, manifest: dict) -> list[dict]:
    """The manifest routes that run this row: by route id, else by the evidence it cites."""
    prefix = f"graphless.{row['family']}.{row['design']}.{row['method']}"
    found = [r for r in manifest["route"] if r["id"] == prefix or r["id"].startswith(prefix + ".")]
    if found:
        return found
    cited = {row.get("known_truth_test"), row.get("retained_route_test")} - {None}
    by_route = {m["route"]: m for m in manifest.get("method") or []}
    return [
        r
        for r in manifest["route"]
        if r.get("retained_route_test") in cited
        or cited & set((by_route.get(r["id"]) or {}).get("evidence") or [])
    ]


def check_graphless(row: dict, prov: Provenance, manifest: dict) -> Unit:
    unit = Unit("graphless", graphless_ident(row))
    by_route = {m["route"]: m for m in manifest.get("method") or []}
    routes = graphless_routes(row, manifest)
    if not routes:
        unit.problems.append("no provenance: the route manifest has no route for this row")
    else:
        modules: list[str] = []
        pairs: list[str] = []
        for route in routes:
            method = by_route.get(route["id"]) or {}
            modules += [str(route["owning_module"]).split(" ")[0]] + list(method.get("implementation") or [])
            pairs += list(method.get("provenance") or [])
        named = [s for m in modules for s in prov.module(m)] + [
            p for p in pairs if provenance_links.resolve(p, prov.stems)
        ]
        if not named:
            unit.problems.append(
                f"no provenance: no provenance record names {', '.join(dict.fromkeys(modules))}"
            )
    # The independent check is `independent_check`; a row that names none falls back to its
    # `known_truth_test`. A registered calibration simulation never satisfies the rule.
    ref = row.get("independent_check") or row.get("known_truth_test")
    field_name = "independent_check" if row.get("independent_check") else "known_truth_test"
    if not isinstance(ref, str) or "::" not in ref:
        unit.problems.append("no independent check: no independent_check or known_truth_test")
    else:
        path, name = ref.split("::", 1)
        if (path, name) in calibration_simulations():
            unit.problems.append(
                f"no independent check: {field_name} {ref} is a calibration simulation, not an independent check"
            )
        for why in test_problems(path, name):
            unit.problems.append(f"no independent check: {field_name} {ref}: {why}")
    return unit


# --------------------------------------------------------------------------- capabilities
def selector_matches(selector: dict, row: dict, axes: tuple[str, ...]) -> bool:
    return all(row.get(k) == v for k, v in selector.items() if k in axes)


def record_modules(record: dict) -> list[str]:
    paths: list[str] = []
    for key in ("search_impl", "surface_rust", "surface", "surface_pyo3"):
        paths += [p for p in record.get(key) or [] if isinstance(p, str)]
    return paths


def record_provenance(record: dict, prov: Provenance) -> list[str]:
    """Provenance records for a capability: named by its id, or naming one of its modules."""
    ident, capability = record["id"], record.get("capability", record["id"])
    found = [s for s in (ident, capability) if s in prov.stems]
    for stem in sorted(prov.stems):
        if stem in found:
            continue
        text = _PROVENANCE_TEXT.setdefault(
            stem, (ROOT / "provenance" / f"{stem}.toml").read_text(encoding="utf-8")
        )
        if f'"{ident}"' in text or f"`{ident}`" in text or f"feature {ident}" in text:
            found.append(stem)
    for module in record_modules(record):
        found += [s for s in prov.module(module) if s not in found]
    return found


_PROVENANCE_TEXT: dict[str, str] = {}


def independent_fixtures(record: dict) -> tuple[list[dict], list[str]]:
    """Fixtures that are an independent check, and why others in the role set do not count."""
    good: list[dict] = []
    notes: list[str] = []
    for fx in record.get("fixtures") or []:
        if fx.get("role") not in INDEPENDENT_FIXTURE_ROLES:
            continue
        kind = fx.get("evidence_kind")
        if kind is not None and kind not in KNOWN_TRUTH_KINDS:
            notes.append(f"{fx['id']}: evidence_kind {kind}")
            continue
        test, name = fx.get("evidence_test"), fx.get("evidence_assertion")
        if not (test and name):
            notes.append(f"{fx['id']}: no cited test")
            continue
        if (test, name) in calibration_simulations():
            notes.append(f"{fx['id']}: {test}::{name} is a calibration simulation")
            continue
        why = test_problems(test, name)
        if why:
            notes.append(f"{fx['id']}: {test}::{name}: {why[0]}")
            continue
        good.append(fx)
    return good, notes


def check_capability(record: dict, prov: Provenance) -> Unit:
    unit = Unit("capability", record["id"])
    if not record_provenance(record, prov):
        unit.problems.append("no provenance: no provenance record names this capability or its modules")
    good, notes = independent_fixtures(record)
    if not good:
        have = Counter(f.get("role") for f in record.get("fixtures") or [])
        extra = "; ".join(notes[:2]) if notes else "no truth or parity fixture"
        calibration = have.get(CALIBRATION_ROLE, 0)
        unit.problems.append(
            "no independent check: "
            + extra
            + (f" ({calibration} calibration fixture(s) are coverage attestations, not a truth check)" if calibration else "")
        )
    return unit


# --------------------------------------------------------------------------- run
def evaluate() -> dict:
    prov = Provenance()
    cells = _load(CELLS).get("cell") or []
    rows = _load(GRAPHLESS).get("support") or []
    records = _load(RECORDS).get("record") or []
    manifest = _load(MANIFEST)

    matrix = [check_cell(c, prov) for c in cells]
    graphless = [check_graphless(r, prov, manifest) for r in rows]

    structural: list[str] = []
    aliased: list[tuple[str, int, int]] = []
    capabilities: list[Unit] = []
    closed = 0
    engineering: list[str] = []
    for rec in records:
        if rec.get("support_unit") is False:
            # Infrastructure that computes no scientific quantity: covered by
            # software tests and never counted as a support unit.
            engineering.append(rec["id"])
            continue
        if rec.get("status") != "supported":
            closed += 1
            continue
        m_sel = rec.get("matrix_units") or []
        g_sel = rec.get("graphless_units") or []
        if m_sel or g_sel:
            hit_m = [c for c in cells if any(selector_matches(s, c, CELL_AXES) for s in m_sel)]
            hit_g = [r for r in rows if any(selector_matches(s, r, GRAPHLESS_AXES) for s in g_sel)]
            for sel_list, hits, label in ((m_sel, hit_m, "matrix_units"), (g_sel, hit_g, "graphless_units")):
                for sel in sel_list:
                    axes = CELL_AXES if label == "matrix_units" else GRAPHLESS_AXES
                    bad = [k for k in sel if k not in axes]
                    if bad:
                        structural.append(f"{rec['id']}: {label} selector has unknown key(s) {bad}")
                    elif not any(selector_matches(sel, h, axes) for h in hits):
                        structural.append(f"{rec['id']}: {label} selector {sel} matches no unit")
            aliased.append((rec["id"], len(hit_m), len(hit_g)))
            continue
        capabilities.append(check_capability(rec, prov))
    return {
        "matrix": matrix,
        "graphless": graphless,
        "capabilities": capabilities,
        "closed_records": closed,
        "aliased": aliased,
        "engineering": engineering,
        "structural": structural,
    }


STALE = "independent check: a reviewed fixture or artifact changed since its review digest was bound"


def cause(problem: str) -> str:
    """The reason class of a problem, without the unit-specific names."""
    if "stale" in problem and "reference_review" in problem:
        return STALE
    if problem.startswith("no provenance"):
        if "no estimator" in problem:
            return "provenance: the cell names no estimator"
        if "estimator" in problem:
            return "provenance: an estimator has no provenance record"
        if "route manifest" in problem:
            return "provenance: no route for the row in the route manifest"
        if "names this capability" in problem:
            return "provenance: no provenance record names the capability or its modules"
        return "provenance: no provenance record names the implementing module"
    m = re.match(r"no independent check: evidence_kind (\w+) is not", problem)
    if m:
        return f"independent check: evidence_kind {m.group(1)} is not known truth or an external oracle"
    if "no truth or parity fixture" in problem:
        return "independent check: no truth or parity fixture"
    return "independent check: " + problem.removeprefix("no independent check: ")[:70]


def summarize(result: dict) -> dict:
    """The counts every human-facing page quotes: one supported-unit count."""
    groups = {k: result[k] for k in ("matrix", "graphless", "capabilities")}
    ok = {k: sum(1 for u in v if u.supported) for k, v in groups.items()}
    failing = [u for v in groups.values() for u in v if not u.supported]
    return {
        "matrix": ok["matrix"],
        "graphless": ok["graphless"],
        "capabilities": ok["capabilities"],
        "total": sum(ok.values()),
        "not_supported": len(failing),
        "listed_matrix": len(groups["matrix"]),
        "matrix_ok": {u.ident for u in groups["matrix"] if u.supported},
        "failing": failing,
    }


def headline(summary: dict) -> str:
    """The one support headline. Pages quote this line and no other support count."""
    return (
        f"Supported units: {summary['total']} (matrix cells {summary['matrix']}, "
        f"graphless rows {summary['graphless']}, capabilities outside the matrix "
        f"{summary['capabilities']}); not supported: {summary['not_supported']}"
    )


def main(argv: list[str]) -> int:
    report = "--report" in argv
    result = evaluate()
    summary = summarize(result)
    failing = summary["failing"]
    print(headline(summary))
    if result["aliased"]:
        names = ", ".join(f"{i} ({m} cells, {g} rows)" for i, m, g in result["aliased"])
        print(f"Aliases of matrix or graphless units, not counted again: {names}")
    if result["engineering"]:
        print(
            "Engineering capabilities (no scientific claim, not support units): "
            + ", ".join(sorted(result["engineering"]))
        )
    reasons: Counter[tuple[str, str]] = Counter()
    only_stale: Counter[str] = Counter()
    for unit in failing:
        causes = {cause(p) for p in unit.problems}
        for c in causes:
            reasons[(unit.kind, c)] += 1
        if causes == {STALE}:
            only_stale[unit.kind] += 1
    for (kind, reason), count in sorted(reasons.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {kind}: {reason}: {count} unit(s)")
    if only_stale:
        print(
            "  failing only on a stale review digest (rebind with scripts/bind_review_digests.py): "
            + ", ".join(f"{k} {v}" for k, v in sorted(only_stale.items()))
        )
    for item in result["structural"]:
        print(f"FAIL (link): {item}")
    for unit in failing:
        print(f"FAIL [{unit.kind}] {unit.ident}")
        for problem in unit.problems:
            print(f"    - {problem}")
    if (failing or result["structural"]) and not report:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
