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

  current      every scientific check the unit cites has a CURRENT PASSING record in the
  record       scientific-records ledger (parity/scientific_records.toml, maintained by
               scripts/scientific_records.py). A record is current while the fingerprint of
               everything the check depends on is unchanged. An owed or missing record
               (including a failing check, which is never recorded) makes the unit
               not_supported with missing evidence `owed: <selector>`, until the scientific
               lane re-earns it. The release build refuses the unit like any other
               not_supported route; the development build runs it with `would_refuse`.

API and stage routes listed under a record are entry points. They have no support status of
their own: a route is open only as an entry point of a supported unit, and refuses with
`not_supported` otherwise. Route counts are never a support count.

The same run checks the sources the units come from: the support axes against the live public
surface, the n/a and refusal-reason rules, the generated graphless supports, and the transport
stage registry.

    python3 scripts/gate_support.py                    # fail on any unit that violates the rule
    python3 scripts/gate_support.py --evidence-only    # the rule without ledger currency (the in-tree
                                                       # embedded status; the default generator mode)
    python3 scripts/gate_support.py --report           # print every violation and the not-supported
                                                       # breakdown (owed records vs other), exit 0
    python3 scripts/gate_support.py --cited-tests      # also resolve every cited test through
                                                       # `cargo test -- --list` (builds, never runs)
    python3 scripts/gate_support.py --cited-tests --matrix-shard 3   # one eighth of the matrix cells
    python3 scripts/gate_support.py --transport PATH   # check one transport registry (self-tests)
    python3 scripts/gate_support.py --self-test        # stubbed listings, no compile

The default run is static source reading: no cargo, no test execution. Cited tests are resolved
as declared, compiled, non-ignored test functions; executing them stays with the evidence runners.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()

import _repo  # noqa: E402
import provenance_links  # noqa: E402
import route_behaviour  # noqa: E402
import test_evidence  # noqa: E402
from evidence_kinds import EVIDENCE_KINDS as ALL_EVIDENCE_KINDS  # noqa: E402
from evidence_kinds import KNOWN_TRUTH_KINDS, fixture_kind_problems  # noqa: E402
from route_methods import IDS, parse_estimators  # noqa: E402
from test_evidence import (  # noqa: E402
    checked_execution_body_problems,
    closure,
    resolve_python_test,
    resolve_rust_test,
)
from transport_replay_contract import (  # noqa: E402
    posterior_consumer_problems,
    shared_evidence_problems,
    source_replay_problems,
)

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
    # Ledger selectors of the cited scientific checks whose record is owed.
    owed: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.problems

    def require_current(self, ledger: Ledger | None, test: str, name: str) -> None:
        """Not supported while the cited check has no current passing record."""
        selector = ledger.owed(test, name) if ledger is not None else None
        if selector is not None and selector not in self.owed:
            self.owed.append(selector)
            self.problems.append(f"{OWED_PREFIX}{selector}")

    @property
    def owed_only(self) -> bool:
        """Not supported for no reason other than owed scientific records."""
        return bool(self.problems) and all(p.startswith(OWED_PREFIX) for p in self.problems)


def _load(rel: str) -> dict:
    return _repo.load_toml(ROOT / rel)


OWED_PREFIX = "owed: "


class Ledger:
    """Which cited scientific checks have a current passing record in the ledger.

    Classification is `scientific_records.classify`: the fingerprint comes from
    `evidence_fingerprint`, the records from `parity/scientific_records.toml`. Nothing here
    fingerprints or reads records itself. A cited check that is not a scientific row can
    never be recorded, so it is owed."""

    def __init__(self) -> None:
        self._owed: dict[tuple[str, str], str] | None = None
        self._current: set[tuple[str, str]] = set()

    def _load(self) -> None:
        import scientific_records

        rows = scientific_records.all_scientific_rows(ROOT)
        owed, current = scientific_records.classify(ROOT, rows)
        key = lambda item: (item["evidence_test"], item["evidence_assertion"])  # noqa: E731
        self._owed = {key(i): i["selector"] for i in owed}
        self._current = {key(i) for i in current}

    def owed(self, test: str, name: str) -> str | None:
        """The ledger selector when the check's record is owed or missing; None when current."""
        if self._owed is None:
            self._load()
        assert self._owed is not None
        if (test, name) in self._current:
            return None
        if (test, name) in self._owed:
            return self._owed[(test, name)]
        from evidence_fingerprint import locate

        try:
            return locate(ROOT, {"evidence_test": test, "evidence_assertion": name})["selector"]
        except ValueError:
            return f"{test}::{name}"

    def counts(self) -> tuple[int, int]:
        if self._owed is None:
            self._load()
        assert self._owed is not None
        return len(self._current), len(self._owed)


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
    """Registered repeated-sampling calibration checks (parity/test_registry.toml)."""
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


def check_cell(cell: dict, prov: Provenance, ledger: Ledger | None = None) -> Unit:
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
        why_not = test_problems(test, name)
        for why in why_not:
            unit.problems.append(f"no independent check: {test}::{name}: {why}")
        if not why_not:
            unit.require_current(ledger, test, name)
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


def check_graphless(row: dict, prov: Provenance, manifest: dict, ledger: Ledger | None = None) -> Unit:
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
        why_not = test_problems(path, name)
        for why in why_not:
            unit.problems.append(f"no independent check: {field_name} {ref}: {why}")
        if not why_not:
            unit.require_current(ledger, path, name)
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


def check_capability(record: dict, prov: Provenance, ledger: Ledger | None = None) -> Unit:
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
    for fx in good:
        unit.require_current(ledger, fx["evidence_test"], fx["evidence_assertion"])
    return unit


# --------------------------------------------------------------------------- run
def evaluate(ledger: Ledger | None = None, evidence_only: bool = False) -> dict:
    """Every unit and its problems. `ledger` decides which cited checks are owed (default: the
    scientific-records ledger). `evidence_only` applies the rule without ledger currency
    (provenance plus a resolvable independent check): the in-tree embedded status, so the
    scientific lane can run the routes it re-earns."""
    prov = Provenance()
    ledger = None if evidence_only else (ledger if ledger is not None else Ledger())
    cells = _load(CELLS).get("cell") or []
    rows = _load(GRAPHLESS).get("support") or []
    records = _load(RECORDS).get("record") or []
    manifest = _load(MANIFEST)

    matrix = [check_cell(c, prov, ledger) for c in cells]
    graphless = [check_graphless(r, prov, manifest, ledger) for r in rows]

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
        capabilities.append(check_capability(rec, prov, ledger))
    return {
        "matrix": matrix,
        "graphless": graphless,
        "capabilities": capabilities,
        "closed_records": closed,
        "aliased": aliased,
        "engineering": engineering,
        "structural": structural,
        "records_current_owed": ledger.counts() if ledger is not None else (0, 0),
    }


OWED = "owed: a cited scientific check has no current passing record"
STALE = "independent check: a reviewed item or fixture changed since its review digest was bound"


def cause(problem: str) -> str:
    """The reason class of a problem, without the unit-specific names."""
    if problem.startswith(OWED_PREFIX):
        return OWED
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
        "not_supported_owed": sum(1 for u in failing if u.owed_only),
        "not_supported_other": sum(1 for u in failing if not u.owed_only),
        "listed_matrix": len(groups["matrix"]),
        "matrix_ok": {u.ident for u in groups["matrix"] if u.supported},
        "failing": failing,
    }


# The derived status fields of every embedded support file. They are a pure function of the
# evidence registries and the ledger's currency, never evidence themselves, so a fingerprint
# of an embedded file must drop exactly these keys.
DERIVED_STATUS_FIELDS = ("status", "missing_evidence")


def derived_status(unit: Unit) -> dict:
    """The embedded status of one unit: `supported`, or `not_supported` with its missing evidence."""
    if unit.supported:
        return {"status": "supported", "missing_evidence": []}
    return {"status": "not_supported", "missing_evidence": sorted(dict.fromkeys(unit.problems))}


def headline(summary: dict) -> str:
    """The one support headline. Pages quote this line and no other support count."""
    return (
        f"Supported units: {summary['total']} (matrix cells {summary['matrix']}, "
        f"graphless rows {summary['graphless']}, capabilities outside the matrix "
        f"{summary['capabilities']}); not supported: {summary['not_supported']} "
        f"(owed records only {summary.get('not_supported_owed', 0)}, "
        f"other {summary.get('not_supported_other', 0)})"
    )



# --------------------------------------------------------------------------- graphless citations
def graphless_cited() -> list[tuple[Path, str]]:
    import generate_graphless_support as graphless

    cited = sorted(
        {
            row[field]
            for row in graphless.load_rows()
            for field in ("known_truth_test", "retained_route_test")
        }
    )
    if not cited:
        sys.exit("graphless support evidence list is empty")
    out = []
    for citation in cited:
        path, function = citation.split("::")
        out.append((ROOT / path, function))
    return out


def graphless_cited_problems(
    cited: list[tuple[Path, str]], excluded: set[tuple[str, str]] | None = None
) -> list[str]:
    """Each cited graphless test must be listed by cargo, non-ignored, and in the software lane."""
    found = []
    for path, function in cited:
        relative = path.relative_to(ROOT).as_posix()
        if excluded and (relative, function) in excluded:
            found.append(f"{relative}::{function}: calibration simulation is excluded from the software lane")
            continue
        _, reasons = resolve_rust_test(path, function)
        found.extend(f"{relative}::{function}: {reason}" for reason in reasons)
    return found


def graphless_self_test() -> None:
    """Stub the cargo listing so the fail-closed behaviour is checked without compiling: a cited
    name listed only under another module, or listed as ignored, must fail; the exact listing
    must pass."""
    cited = graphless_cited()
    targets: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for path, function in cited:
        root_file, crate, target = test_evidence.target_root(path)
        targets.setdefault((crate, tuple(target)), set()).add(
            test_evidence.libtest_name(path, function, root_file)
        )
    path0, function0 = cited[0]
    root0, crate0, target0 = test_evidence.target_root(path0)
    key0 = (crate0, tuple(target0))
    full0 = test_evidence.libtest_name(path0, function0, root0)

    def run(name: str, listing: dict, ignored: set[str], want_fail: bool) -> None:
        test_evidence._CARGO_LIST_CACHE.clear()
        for key, names in listing.items():
            test_evidence._CARGO_LIST_CACHE[key] = (set(names), ignored if key == key0 else set(), None)
        got = graphless_cited_problems(cited)
        if bool(got) != want_fail:
            sys.exit(
                f"graphless self-test {name}: expected {'failure' if want_fail else 'pass'}, got {got}"
            )
        print(f"ok  {name}")

    run("exact listing passes", targets, set(), False)
    moved = {k: set(v) for k, v in targets.items()}
    moved[key0] = (moved[key0] - {full0}) | {f"elsewhere::tests::{function0}"}
    run("same function name in another module fails", moved, set(), True)
    run("cited test listed as ignored fails", targets, {full0}, True)
    excluded = {(path0.relative_to(ROOT).as_posix(), function0)}
    if not graphless_cited_problems([(path0, function0)], excluded):
        sys.exit("graphless self-test: excluded simulation must be incomplete")
    print("ok  excluded simulation is incomplete")
    print("graphless support self-test: ok")


def graphless_generated_problems() -> list[str]:
    """The generated graphless supports are current (generate_graphless_support.py --check)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_graphless_support.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return []
    return [line for line in (result.stdout + result.stderr).splitlines() if line.strip()] or [
        "generate_graphless_support.py --check failed"
    ]


# --------------------------------------------------------------------------- support matrix
MATRIX_EVIDENCE_KINDS = ALL_EVIDENCE_KINDS - {"process_attested"}


def regression_pin_problem(fixture: str, inference: str) -> str | None:
    """Why `fixture` cannot back known truth for a cell of this inference mode.

    A fixture whose `oracle.kind` is `regression_pin` holds frozen output of this library. Only
    the inference modes its oracle lists in `independent_inferences` (a part checked against an
    independent reference beside the pin) may cite it as truth."""
    path = ROOT / fixture / "expected.json"
    if not path.is_file():
        return None
    try:
        oracle = json.loads(path.read_text(encoding="utf-8")).get("oracle")
    except (OSError, ValueError):
        return None
    if (
        isinstance(oracle, dict)
        and oracle.get("kind") == "regression_pin"
        and inference not in (oracle.get("independent_inferences") or [])
    ):
        return (
            f"known_truth_fixture {fixture} is a regression_pin fixture (frozen output of "
            f"this library) and lists no independent {inference} reference; the cell is "
            "evidence_kind = regression_pin, not known truth"
        )
    return None


def matrix_problems(shard: str = "all", *, resolve: bool = False) -> tuple[list[str], str]:
    """The support-matrix contract: axes match the live surface, n/a and refusal rules are
    well-formed and disjoint, and every supported cell is well-formed. With `resolve`, every
    cited test of the shard (`all` or 0..7) is also resolved to an executing test through
    `cargo test -- --list`, and checked for its retained checked execution."""
    root = ROOT
    fail: list[str] = []
    matrix_shard = shard

    def in_shard(index: int) -> bool:
        return matrix_shard == "all" or (index - 1) % 8 == int(matrix_shard)

    if resolve and matrix_shard != "all":
        print(f"support matrix: resolving the cited tests of shard {matrix_shard}/8", flush=True)

    def load(rel: str) -> dict:
        path = root / rel
        if not path.is_file():
            fail.append(f"{rel}: missing")
            return {}
        try:
            return _repo.load_toml(path)
        except tomllib.TOMLDecodeError as exc:
            fail.append(f"{rel}: not valid TOML: {exc}")
            return {}


    axes = load("parity/support_axes.toml")
    na_doc = load("parity/support_n_a.toml")
    closed_doc = load("parity/support_refusal_reasons.toml")
    lic_doc = load("parity/support_cells.toml")

    queries = list(axes.get("queries") or [])
    stage_queries = list(axes.get("stage_queries") or [])
    off_axis_public_queries = list(axes.get("off_axis_public_queries") or [])
    public_query_companions = list(axes.get("public_query_companions") or [])
    graph_classes = list(axes.get("graph_classes") or [])
    off_axis_graph_classes = list(axes.get("off_axis_graph_classes") or [])
    structures = list(axes.get("structures") or [])
    inferences = list(axes.get("inferences") or [])
    validations = list(axes.get("validations") or [])
    all_queries = queries + stage_queries

    # --- live Python root query names -------------------------------------------
    init_text = (root / "python/antecedent/__init__.py").read_text(encoding="utf-8")
    mod = ast.parse(init_text)
    public: list[str] | None = None
    for node in mod.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                public = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        public.append(elt.value)
                break
    if public is None:
        fail.append("python/antecedent/__init__.py: could not parse __all__")
        public = []

    # Queries live between the "Queries" and "Graphs" comment blocks in __all__.
    q_start = init_text.find("# Queries")
    q_end = init_text.find("# Graphs")
    if q_start < 0 or q_end < 0 or q_end <= q_start:
        fail.append("python/antecedent/__init__.py: missing # Queries / # Graphs markers in __all__")
        live_queries: list[str] = []
    else:
        block = init_text[q_start:q_end]
        live_queries = re.findall(r'"([A-Za-z][A-Za-z0-9]+)"', block)

    # TransportQuery stays on the support-matrix query axis (supported trial-IPW
    # cell) but lives on `antecedent.transport.advanced`, not root `__all__`.
    root_queries = [q for q in queries if q != "TransportQuery"]
    # Public query class names can differ from the stable support query axis.
    # Keep those mappings explicit so adding a user-facing type does not silently
    # rename support cells or their evidence records.
    query_axis_aliases = {"NestedCounterfactual": "NestedCounterfactualEffect"}
    mapped_live_queries = [query_axis_aliases.get(q, q) for q in live_queries]
    # The root namespace is frozen (docs/api_naming.md). Geometric queries and the
    # companions that are actually re-exported there belong in the Queries block.
    # The design families are public on their stage modules, not at the root.
    public_names = set(public)
    root_companions = [name for name in public_query_companions if name in public_names]
    expected_root_queries = root_queries + root_companions
    if sorted(expected_root_queries) != sorted(mapped_live_queries):
        fail.append(
            "parity/support_axes.toml root query exports != python __all__ query names: "
            f"axes={sorted(expected_root_queries)} live={sorted(mapped_live_queries)}"
        )
    # Python spellings that are not the support-axis id.
    class_to_axis = {"LongitudinalRegime": "LongitudinalRegimeQuery"}
    stage_files = [
        root / "python/antecedent/experiment.py",
        root / "python/antecedent/policy.py",
        root / "python/antecedent/survival.py",
        root / "python/antecedent/regimes.py",
        root / "python/antecedent/query.py",
    ]
    found_public_types: set[str] = set()
    for path in stage_files:
        for name in re.findall(
            r"^class ([A-Za-z][A-Za-z0-9]*)\b", path.read_text(encoding="utf-8"), re.M
        ):
            found_public_types.add(class_to_axis.get(name, name))
    needed_public_types = set(off_axis_public_queries) | set(public_query_companions)
    missing_public_types = sorted(needed_public_types - found_public_types)
    if missing_public_types:
        fail.append(
            "support-axis public types missing from the stage or query modules: "
            + ", ".join(missing_public_types)
        )
    if set(off_axis_public_queries) & set(queries):
        fail.append("off_axis_public_queries overlaps geometric queries")
    if set(public_query_companions) & (set(queries) | set(off_axis_public_queries)):
        fail.append("public_query_companions overlaps query kinds")

    for name in ["Frequentist", "Bayesian"]:
        if name not in public:
            fail.append(f"{name} is an inference axis value but is not in python __all__")

    for name in stage_queries:
        if name == "TransportQuery":
            src_path = root / "python/antecedent/transport/_impl"
        else:
            src_path = root / "python/antecedent/interference.py"
        if src_path.is_dir():
            src = "\n".join(path.read_text(encoding="utf-8") for path in sorted(src_path.rglob("*.py")))
        elif src_path.is_file():
            src = src_path.read_text(encoding="utf-8")
        else:
            src = ""
        if f"class {name}" not in src:
            fail.append(f"{name}: no class {name} in {src_path.relative_to(root)}")

    # --- live GraphClass variants ------------------------------------------------
    accepted = (root / "crates/antecedent/src/accepted.rs").read_text(encoding="utf-8")
    m = re.search(r"pub enum GraphClass \{([^}]+)\}", accepted, re.S)
    if not m:
        fail.append("crates/antecedent/src/accepted.rs: could not parse GraphClass")
        live_graphs: list[str] = []
    else:
        live_graphs = re.findall(r"^\s+([A-Z][A-Za-z0-9]+),", m.group(1), re.M)
    # GraphClass must be on the axis. Classification-only extras (tier-rule
    # backgrounds) may appear in addition; they are not GraphClass variants.
    TIER_EXTRAS = {"CoDetermined", "Unknown"}
    if not set(live_graphs) <= set(graph_classes) | set(off_axis_graph_classes):
        fail.append(
            "parity/support_axes.toml graph_classes must contain GraphClass: "
            f"axes={sorted(graph_classes)} live={sorted(live_graphs)}"
        )
    unknown_extras = set(graph_classes) - set(live_graphs) - TIER_EXTRAS
    if unknown_extras:
        fail.append(
            "parity/support_axes.toml graph_classes has extras that are not "
            f"GraphClass or tier-rule axes: {sorted(unknown_extras)}"
        )
    if not set(graph_classes) >= TIER_EXTRAS:
        fail.append(
            "parity/support_axes.toml graph_classes must include "
            f"{sorted(TIER_EXTRAS)} (classification-only tier-rule axes)"
        )
    if set(off_axis_graph_classes) - set(live_graphs):
        fail.append("off_axis_graph_classes includes names absent from GraphClass")
    if set(off_axis_graph_classes) & set(graph_classes):
        fail.append("off_axis_graph_classes overlaps geometric graph_classes")

    expected_structures = {"explicit", "accepted", "graph_posterior"}
    if set(structures) != expected_structures:
        fail.append(f"structures must be {sorted(expected_structures)}; got {structures}")
    expected_inferences = {"Frequentist", "Bayesian"}
    if set(inferences) != expected_inferences:
        fail.append(f"inferences must be {sorted(expected_inferences)}; got {inferences}")
    expected_validations = {"none", "cheap", "full"}
    if set(validations) != expected_validations:
        fail.append(f"validations must be {sorted(expected_validations)}; got {validations}")

    for seq, label in (
        (queries, "queries"),
        (stage_queries, "stage_queries"),
        (off_axis_public_queries, "off_axis_public_queries"),
        (public_query_companions, "public_query_companions"),
        (graph_classes, "graph_classes"),
        (off_axis_graph_classes, "off_axis_graph_classes"),
        (structures, "structures"),
        (inferences, "inferences"),
        (validations, "validations"),
    ):
        if len(seq) != len(set(seq)):
            fail.append(f"parity/support_axes.toml {label} has duplicates")
        if any(not isinstance(x, str) or not x.strip() for x in seq):
            fail.append(f"parity/support_axes.toml {label} has an empty value")

    # --- n/a predicates ----------------------------------------------------------
    na_rules = na_doc.get("n_a") or []
    AXIS_KEYS = {
        "queries": set(all_queries),
        "graph_classes": set(graph_classes),
        "structures": set(structures),
        "inferences": set(inferences),
        "validations": set(validations),
    }
    for i, rule in enumerate(na_rules, 1):
        if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
            fail.append(f"parity/support_n_a.toml rule #{i}: missing reason")
        for key, legal in AXIS_KEYS.items():
            vals = rule.get(key)
            if vals is None:
                continue
            if not isinstance(vals, list) or not vals:
                fail.append(f"parity/support_n_a.toml rule #{i}: {key} must be a non-empty list")
                continue
            for v in vals:
                if v not in legal:
                    fail.append(
                        f"parity/support_n_a.toml rule #{i}: {key} value {v!r} is not an axis value"
                    )


    def rule_matches(rule: dict, cell: dict) -> bool:
        mapping = {
            "queries": "query",
            "graph_classes": "graph_class",
            "structures": "structure",
            "inferences": "inference",
            "validations": "validation",
        }
        for rule_key, cell_key in mapping.items():
            allowed = rule.get(rule_key)
            if allowed is not None and cell[cell_key] not in allowed:
                return False
        return True


    def is_n_a(cell: dict) -> bool:
        return any(rule_matches(rule, cell) for rule in na_rules)


    # Cited evidence is read through scripts/test_evidence.py, the one reader of test
    # source for the gates: the cited function must be an executing test and, with the
    # helpers it calls, must consume its fixture and build the row's axis values.

    closed_rules = closed_doc.get("closed") or []
    for i, rule in enumerate(closed_rules, 1):
        if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
            fail.append(f"parity/support_refusal_reasons.toml rule #{i}: missing reason")
        for key, legal in AXIS_KEYS.items():
            vals = rule.get(key)
            if vals is None:
                continue
            if not isinstance(vals, list) or not vals:
                fail.append(f"parity/support_refusal_reasons.toml rule #{i}: {key} must be a non-empty list")
                continue
            for v in vals:
                if v not in legal:
                    fail.append(
                        f"parity/support_refusal_reasons.toml rule #{i}: {key} value {v!r} is not an axis value"
                    )


    def is_closed(cell: dict) -> bool:
        return (not is_n_a(cell)) and any(rule_matches(rule, cell) for rule in closed_rules)


    # n/a and reason-backed refusals are disjoint. A too-broad n/a must not mask
    # a reason row in legacy-named support_refusal_reasons.toml.
    if all_queries and graph_classes and structures and inferences and validations:
        supported_keys_for_overlap = {
            (
                row.get("query"),
                row.get("graph_class"),
                row.get("structure"),
                row.get("inference"),
                row.get("validation"),
            )
            for row in (lic_doc.get("cell") or [])
        }
        overlap_n = 0
        for q, g, s, inf, v in product(all_queries, graph_classes, structures, inferences, validations):
            cell = {
                "query": q,
                "graph_class": g,
                "structure": s,
                "inference": inf,
                "validation": v,
            }
            closed_hit = next(
                (i for i, rule in enumerate(closed_rules, 1) if rule_matches(rule, cell)),
                None,
            )
            if closed_hit is None:
                continue
            if is_n_a(cell):
                overlap_n += 1
                if overlap_n <= 25:
                    fail.append(
                        f"parity/support_refusal_reasons.toml rule #{closed_hit} overlaps n/a cell "
                        f"{(q, g, s, inf, v)}"
                    )
            if (q, g, s, inf, v) in supported_keys_for_overlap:
                fail.append(
                    f"parity/support_refusal_reasons.toml rule #{closed_hit} overlaps supported cell "
                    f"{(q, g, s, inf, v)}"
                )
        if overlap_n > 25:
            fail.append(f"... {overlap_n - 25} more n/a/refusal-reason overlaps")

    # --- supported cells ----------------------------------------------------------
    cells = lic_doc.get("cell") or []
    seen: set[tuple[str, ...]] = set()
    required = (
        "query",
        "graph_class",
        "structure",
        "inference",
        "validation",
        "staged",
        "evidence_kind",
    )
    legal_q = set(all_queries)

    # Every staged supported row names the executing test and test function that
    # carries its evidence. Rows supported before this rule are frozen in
    # parity/_evidence_test_backlog.txt; the backlog only shrinks.
    missing_evidence: set[str] = set()


    # Estimator wire-ids recorded on supported rows (secondary axis; not cartesian).
    # When present, every entry must be a non-empty string. Empty lists are allowed
    # only when the row honestly has no estimator evidence (classify_estimator then
    # refuses every concrete EstimatorId, including the five not_measured families).
    def check_estimators(label: str, row: dict) -> None:
        named = row.get("estimators")
        if named is None:
            return
        if not isinstance(named, list):
            fail.append(f"{label}: estimators must be a list of wire-ids")
            return
        for est in named:
            if not isinstance(est, str) or not est.strip():
                fail.append(f"{label}: estimators entries must be non-empty strings")


    def check_evidence_test(label: str, row: dict) -> None:
        for problem in test_evidence.row_evidence_problems(row):
            fail.append(f"{label}: {problem}")


    # Every supported estimator on a row executes through a retained checked
    # operation: one checked_execution entry per estimator, each citing an
    # executing test that discards its builder, executes the plan, and inspects it.
    def check_checked_execution(label: str, row: dict) -> None:
        for problem in test_evidence.checked_execution_problems(row):
            fail.append(f"{label}: {problem}")


    for i, row in enumerate(cells, 1):
        if i % 50 == 0:
            print(f"support matrix: checked {i}/{len(cells)} supported cells", flush=True)
        label = f"parity/support_cells.toml cell #{i}"
        for key in required:
            if key not in row:
                fail.append(f"{label}: missing {key}")
        check_estimators(label, row)
        q = row.get("query")
        g = row.get("graph_class")
        s = row.get("structure")
        inf = row.get("inference")
        v = row.get("validation")
        if q not in legal_q:
            fail.append(f"{label}: query {q!r} is not an axis value")
        if g not in set(graph_classes):
            fail.append(f"{label}: graph_class {g!r} is not an axis value")
        if s not in set(structures):
            fail.append(f"{label}: structure {s!r} is not an axis value")
        if inf not in set(inferences):
            fail.append(f"{label}: inference {inf!r} is not an axis value")
        if v not in set(validations):
            fail.append(f"{label}: validation {v!r} is not an axis value")
        if row.get("staged") is not True:
            fail.append(f"{label}: staged must be true (analyze-only is not support)")
        kind = row.get("evidence_kind")
        if not isinstance(kind, str) or kind not in MATRIX_EVIDENCE_KINDS:
            fail.append(f"{label}: evidence_kind is required and must be a known kind")
        if row.get("status") is not None:
            fail.append(f"{label}: status is illegal on a matrix cell")
        fixture = row.get("known_truth_fixture")
        fail.extend(f"{label}: {problem}" for problem in fixture_kind_problems(root, row))
        if kind in KNOWN_TRUTH_KINDS:
            if not isinstance(fixture, str) or not fixture.strip():
                fail.append(f"{label}: {kind} requires known_truth_fixture")
            elif not (root / fixture).exists():
                fail.append(f"{label}: known_truth_fixture {fixture!r} does not exist")
            elif (why := regression_pin_problem(fixture, inf)) is not None:
                fail.append(f"{label}: {why}")
            elif not (row.get("evidence_test") and row.get("evidence_assertion")):
                fail.append(
                    f"{label}: {kind} requires evidence_test/evidence_assertion whose own "
                    "function consumes the fixture"
                )
        elif kind == "regression_pin":
            if fixture is not None:
                fail.append(
                    f"{label}: regression_pin must not name known_truth_fixture; a frozen "
                    "output of this library is not truth (name the pin fixture in limitations)"
                )
            if not str(row.get("limitations", "")).strip():
                fail.append(f"{label}: regression_pin requires limitations saying what is pinned")
            if not (row.get("evidence_test") and row.get("evidence_assertion")):
                fail.append(f"{label}: regression_pin requires evidence_test/evidence_assertion")
        elif kind == "internal_cross_check":
            if fixture is not None:
                fail.append(
                    f"{label}: internal_cross_check must not name known_truth_fixture; "
                    "that would present contextual data as independent truth evidence"
                )
            test_rel = row.get("evidence_test")
            assertion = row.get("evidence_assertion")
            if not isinstance(test_rel, str) or not test_rel.strip():
                fail.append(f"{label}: internal_cross_check requires evidence_test")
            elif not isinstance(assertion, str) or not assertion.strip():
                fail.append(f"{label}: internal_cross_check requires evidence_assertion")
        # Every named test must be an executing test function, whatever the kind.
        test_rel = row.get("evidence_test")
        assertion = row.get("evidence_assertion")
        has_test = isinstance(test_rel, str) and bool(test_rel.strip())
        has_assertion = isinstance(assertion, str) and bool(assertion.strip())
        if has_test != has_assertion:
            fail.append(f"{label}: evidence_test and evidence_assertion must be set together")
        elif has_test and resolve and in_shard(i):
            check_evidence_test(label, row)
        if resolve and in_shard(i):
            check_checked_execution(label, row)
        if row.get("staged") is True and not (has_test and has_assertion):
            missing_evidence.add("|".join(str(x) for x in (q, g, s, inf, v)))
        key = (q, g, s, inf, v)
        if key in seen:
            fail.append(f"{label}: duplicate cell {key}")
        seen.add(key)
        cell = {
            "query": q,
            "graph_class": g,
            "structure": s,
            "inference": inf,
            "validation": v,
        }
        if all(cell.values()) and is_n_a(cell):
            fail.append(f"{label}: cell is n/a under support_n_a.toml; cannot support it")
        if all(cell.values()) and is_closed(cell):
            fail.append(
                f"{label}: cell has a refusal reason under support_refusal_reasons.toml; cannot support it"
            )

    # --- cross-check-only cells need a known-truth sibling -------------------------
    # A cell whose only evidence is an internal cross-check (the estimator against an
    # in-test recomputation that may share its convention) and that carries no measured
    # coverage has nothing tying it to the truth. It is supported only while another cell of
    # the same query and inference mode has known-truth or external-oracle evidence, so the
    # estimand itself is checked against a simulation truth somewhere.
    _truth_kinds = KNOWN_TRUTH_KINDS
    _truthful = {
        (c.get("query"), c.get("inference")) for c in cells if c.get("evidence_kind") in _truth_kinds
    }
    for row in cells:
        if (
            row.get("evidence_kind") in {"internal_cross_check", "regression_pin"}
            and not row.get("calibration")
            and (row.get("query"), row.get("inference")) not in _truthful
        ):
            fail.append(
                f"{row.get('query')}/{row.get('graph_class')}/{row.get('structure')}/"
                f"{row.get('inference')}/{row.get('validation')}: {row.get('evidence_kind')} with no "
                "calibration and no known-truth sibling cell of the same query and inference "
                "mode; add known-truth evidence or a measured calibration"
            )

    # --- evidence_test ratchet ---------------------------------------------------
    backlog_rel = "parity/_evidence_test_backlog.txt"
    backlog_path = root / backlog_rel
    evidence_backlog: set[str] = set()
    if backlog_path.is_file():
        for line in backlog_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                evidence_backlog.add(line)
    for key in sorted(missing_evidence - evidence_backlog):
        fail.append(
            f"supported cell {key} is staged but has no evidence_test/evidence_assertion; "
            "name the executing test file and test function (do not add rows to "
            f"{backlog_rel} -- it is frozen)"
        )
    for key in sorted(evidence_backlog - missing_evidence):
        fail.append(
            f"{backlog_rel} lists {key} but that row now names its evidence (or is no "
            "longer supported) -- delete the line in the same change (the list only shrinks)"
        )

    # --- counts ------------------------------------------------------------------
    if all_queries and graph_classes and structures and inferences and validations:
        cartesian = 0
        n_a_count = 0
        reason_backed_refused_count = 0
        for q, g, s, inf, v in product(all_queries, graph_classes, structures, inferences, validations):
            cartesian += 1
            cell = {
                "query": q,
                "graph_class": g,
                "structure": s,
                "inference": inf,
                "validation": v,
            }
            if is_n_a(cell):
                n_a_count += 1
            elif is_closed(cell):
                reason_backed_refused_count += 1
        refused = cartesian - n_a_count - len(cells)
        unreasoned = refused - reason_backed_refused_count
        if refused < 0:
            fail.append("supported + n/a exceeds the cartesian product")
        if reason_backed_refused_count > refused:
            fail.append("reason-backed refusals exceed refused cells")
        if unreasoned:
            fail.append(
                f"{unreasoned} meaningful refused cell(s) have no reason in "
                "parity/support_refusal_reasons.toml; every refused cell must name why"
            )
    else:
        cartesian = n_a_count = reason_backed_refused_count = refused = 0
        fail.append("axes are incomplete; cannot form a cartesian product")

    summary = (
        f"Support matrix OK ({cartesian} cells; {len(cells)} supported; "
        f"{n_a_count} n/a; {reason_backed_refused_count} refused with reasons; "
        f"{refused - reason_backed_refused_count} refused without reasons)"
    )
    return fail, summary


def transport_problems(registry_path: Path | None = None, *, resolve: bool = False) -> list[str]:
    """The transport stage registry contract, independent of the analyze axes. With `resolve`,
    cited Rust tests are resolved through `cargo test -- --list`; otherwise by source reading.
    An explicit registry path exists for gate self-tests; evidence still resolves in this tree."""
    root = ROOT
    registry_path = registry_path or root / "parity/transport_stages.toml"

    def resolve_test(path: Path, assertion: str) -> list[str]:
        if not resolve:
            return test_problems(path.relative_to(ROOT).as_posix(), assertion)
        if path.suffix == ".py":
            return list(resolve_python_test(path, assertion))
        return list(resolve_rust_test(path, assertion)[1])

    registry = _repo.load_toml(registry_path)
    # Stage, status, reason code and claim of a record-owned route are generated from its
    # record (scripts/route_behaviour.py); a row states only the contract a stage adds.
    generated = route_behaviour.load(root / route_behaviour.GENERATED)
    merged, restated = route_behaviour.stage_routes(root, registry, generated)
    registry = {**registry, "routes": merged}
    owned = {r["route"] for entry in (generated or {}).values() for r in entry["routes"]}
    errors: list[str] = list(restated)
    if registry.get("version") != 1 or registry.get("default") != "closed":
        errors.append("transport stages require version 1 and default=closed")
    seen = set()
    for route in registry.get("routes", []):
        name = route.get("route")
        if not name or name in seen:
            errors.append(f"duplicate or missing transport route: {name}")
        seen.add(name)
        if route.get("stage") not in {
            "identify",
            "representation",
            "evaluate",
            "uncertainty",
            "prepare",
            "consume",
        }:
            errors.append(f"invalid stage: {name}")
        contract_keys = (
            "graph",
            "evidence",
            "functional",
            "guarantee",
            "evidence_test",
            "evidence_assertion",
        )
        if route.get("status") == "supported":
            # A record-owned route that states no contract of its own is evidenced by its
            # record's executed fixtures (scripts/check_capability_records.py).
            for key in () if name in owned and not any(route.get(k) for k in contract_keys) else contract_keys:
                if not route.get(key):
                    errors.append(f"{name}: missing {key}")
            path, assertion = route.get("evidence_test"), route.get("evidence_assertion")
            if path and assertion:
                evidence_path = root / path
                problems: list[str] = []
                if evidence_path.suffix in (".rs", ".py"):
                    problems = resolve_test(evidence_path, assertion)
                else:
                    problems = [f"{name}: evidence must be a collected Rust or Python test"]
                errors.extend(problems)
                # A public stage route executes through its retained checked plan:
                # the cited test drops its builder, executes, and inspects the plan.
                contract = route.get("execution_contract", "retained_checked_plan")
                if contract not in {
                    "retained_checked_plan", "source_bound_replay", "posterior_consumer_replay"
                }:
                    errors.append(f"{name}: unknown execution contract {contract}")
                elif not problems and contract == "posterior_consumer_replay":
                    errors.extend(
                        f"{name}: {problem}"
                        for problem in posterior_consumer_problems(
                            route, evidence_path.read_text(encoding="utf-8")
                        )
                    )
                elif not problems and contract == "source_bound_replay":
                    errors.extend(
                        f"{name}: {problem}"
                        for problem in source_replay_problems(
                            root, route, evidence_path.read_text(encoding="utf-8")
                        )
                    )
                elif (
                    not problems
                    and name.startswith("antecedent.transport.")
                    # The sensitivity module seals a decision artifact a consumer recomputes;
                    # it has no builder or retained plan to discard.
                    and not name.startswith("antecedent.transport.sensitivity.")
                ):
                    errors.extend(
                        f"{name}: {problem}"
                        for problem in checked_execution_body_problems(
                            closure(evidence_path, assertion), assertion
                        )
                    )
            if (name.startswith("antecedent.transport.") and route.get("stage") == "identify"
                    and route.get("guarantee") != "sound_incomplete"):
                pinned = {
                    "antecedent.transport.advanced.identify_classical": (
                        "complete_in_classical_evidence_scope",
                        "classical_complete_source_experimental_family",
                        "https://arxiv.org/abs/1312.7485v1",
                    ),
                    "antecedent.transport.advanced.identify_meta": (
                        "complete_in_classical_meta_evidence_scope",
                        "classical_complete_multi_source_experimental_families",
                        "https://proceedings.mlr.press/v31/bareinboim13a.pdf",
                    ),
                }
                if pinned.get(name) != (
                    route.get("guarantee"),
                    route.get("evidence"),
                    route.get("reference"),
                ):
                    errors.append(f"{name}: completeness requires its pinned classical evidence scope")
                path, assertion = (
                    route.get("conformance_test"),
                    route.get("conformance_assertion"),
                )
                if not path or not assertion:
                    errors.append(f"{name}: completeness requires consuming Rust branch conformance")
                else:
                    errors.extend(resolve_test(root / path, assertion))
        elif route.get("status") != "closed" or not route.get("reason_code"):
            errors.append(f"{name}: expected supported or a reason-backed closed contract")
    FAMILIES = {
        f"transport_{name}"
        for name in (
            "direct_regression",
            "standardize_regression",
            "target_only",
            "recursive_district",
            "negative_witness",
            "catalog_binding",
            "complementary_sources",
            "support_local",
            "grid_joint_inference",
            "multisample_inference",
            "prepared_lifecycle",
            "artifact_acceptance",
            "budget_refusal",
        )
    }
    EVIDENCE_CLASSES = {
        "external_parity",
        "exact_scm_truth",
        "internal_cross_check",
        "theoretical_witness",
        "statistical_calibration",
        "inferential_plumbing",
        "resampling_property",
    }
    roles: dict[str, dict[str, tuple[str, str]]] = {}
    for row in registry.get("fixture_evidence", []):
        rid, family, role = row.get("id"), row.get("family"), row.get("role")
        if family not in FAMILIES or role not in {"positive", "counterexample"}:
            errors.append(f"{rid}: unknown fixture family or role")
            continue
        if rid != f"{family}.{role}" or role in roles.setdefault(family, {}):
            errors.append(f"{rid}: id must be the unique <family>.<role>")
        if row.get("evidence_class") not in EVIDENCE_CLASSES or not row.get("limits"):
            errors.append(f"{rid}: requires a known evidence_class and stated limits")
        path, assertion = row.get("evidence_test", ""), row.get("evidence_assertion", "")
        roles[family][role] = (path, assertion)
        if path.endswith((".rs", ".py")):
            errors.extend(resolve_test(root / path, assertion))
        else:
            errors.append(f"{rid}: evidence must be a collected Rust or Python test")
    for family in sorted(FAMILIES):
        pair = roles.get(family, {})
        if set(pair) != {"positive", "counterexample"}:
            errors.append(f"{family}: requires a positive and a counterexample row")
        elif pair["positive"] == pair["counterexample"]:
            errors.append(f"{family}: one assertion cannot be both positive and counterexample")
    # Every supported route owns its evidence: one assertion cited by several routes
    # proves at most one of them, so a shared citation is an error, not a warning.
    errors.extend(shared_evidence_problems(registry.get("routes", [])))
    required = {
        "antecedent.transport.identify",
        "antecedent.transport.reload_lowered_expression",
    }
    if not required.issubset(seen):
        errors.append("missing current public transport stage")
    coverage_ids = {
        row.get("id")
        for row in _repo.load_toml(root / "parity/coverage_records.toml").get("record", [])
    }
    for route in registry.get("routes", []):
        for cid in route.get("coverage") or []:
            if cid not in coverage_ids:
                errors.append(f"{route.get('route')}: unknown coverage record {cid}")
        if route.get("coverage") and route.get("calibration_reason"):
            errors.append(
                f"{route.get('route')}: a route cannot cite coverage records and also state "
                "calibration_reason; the records must measure the route's own estimator"
            )
        for key in ("scm", "tolerance", "calibration_reason"):
            value = route.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"{route.get('route')}: {key} must be a non-empty string")
    for row in registry.get("fixture_evidence", []):
        for key in ("scm", "tolerance"):
            value = row.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"{row.get('id')}: {key} must be a non-empty string")
    return errors


# --------------------------------------------------------------------------- run
def count_main(report: bool, evidence_only: bool = False) -> int:
    result = evaluate(evidence_only=evidence_only)
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
    current, owed = result.get("records_current_owed", (0, 0))
    print(
        f"Scientific records: {current} current, {owed} owed; not supported: "
        f"{summary['not_supported_owed']} unit(s) on owed records only, "
        f"{summary['not_supported_other']} on other reasons"
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
            "  failing only on a stale review digest (re-read the changed items, then rebind with scripts/bind_review_digests.py): "
            + ", ".join(f"{k} {v}" for k, v in sorted(only_stale.items()))
        )
    for item in result["structural"]:
        print(f"FAIL (link): {item}")
    for unit in failing:
        if unit.owed_only:
            print(f"FAIL [{unit.kind}] {unit.ident}: " + "; ".join(unit.problems))
            continue
        print(f"FAIL [{unit.kind}] {unit.ident}")
        for problem in unit.problems:
            print(f"    - {problem}")
    if (failing or result["structural"]) and not report:
        return 1
    return 0


def structure_problems() -> list[str]:
    """The sources the units come from, read statically."""
    found, summary = matrix_problems()
    print(summary if not found else "support matrix: FAILED")
    found = [f"support matrix: {p}" for p in found]
    transport = transport_problems()
    if not transport:
        print("transport stage registry: ok")
    found += [f"transport stages: {p}" for p in transport]
    graphless_current = graphless_generated_problems()
    if not graphless_current:
        print("generated graphless supports: current")
    found += [f"graphless supports: {p}" for p in graphless_current]
    return found


def cited_test_problems(shard: str) -> list[str]:
    """Every cited test resolves to a compiled, non-ignored test (builds test binaries to list
    them; runs nothing)."""
    found = []
    whole = shard in ("all", "0")  # the graphless and transport citations ride with the first shard
    if whole:
        cited = graphless_cited()
        print(f"== resolving {len(cited)} graphless evidence citation(s) via cargo test -- --list", flush=True)
        excluded = (
            set()
            if os.environ.get("ANTECEDENT_RUN_CALIBRATION_SIMULATIONS") == "1"
            else {(r["evidence_test"], r["evidence_assertion"]) for r in test_evidence.registered_calibration_simulations()}
        )
        found += graphless_cited_problems(cited, excluded)
    matrix, summary = matrix_problems(shard, resolve=True)
    found += [f"support matrix: {p}" for p in matrix]
    if whole:
        found += [f"transport stages: {p}" for p in transport_problems(resolve=True)]
    if not found:
        print(f"cited support evidence: ok ({summary})")
    return found


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        graphless_self_test()
        return 0
    if "--transport" in argv:
        path = Path(argv[argv.index("--transport") + 1])
        found = transport_problems(path)
        if found:
            print("Transport stage gate FAILED:\n" + "\n".join(f" - {e}" for e in found))
            return 1
        print("Transport stage contracts OK")
        return 0
    report = "--report" in argv
    evidence_only = "--evidence-only" in argv
    if "--cited-tests" in argv:
        shard = argv[argv.index("--matrix-shard") + 1] if "--matrix-shard" in argv else "all"
        found = cited_test_problems(shard)
        for problem in found:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1 if found and not report else 0
    status = count_main(report, evidence_only)
    found = structure_problems()
    for problem in found:
        print(f"FAIL: {problem}")
    return 1 if (status or (found and not report)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
