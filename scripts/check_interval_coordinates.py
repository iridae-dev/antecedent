#!/usr/bin/env python3
"""2.2 closure: zero newly introduced unmeasured interval coordinates, for every 2.2 record.

This is scripts/check_a_intervals.py generalised from the 2.2A records to ALL records of
parity/promotion_2_2.toml (milestone A and B), importing its helpers read-only. For each record:

  (a) no licensed route has an owning-registry row (parity/transport_stages.toml [[routes]],
      parity/support_licensed.toml cells) carrying `estimator_grid_not_measured`;
  (b) every route whose stage is `uncertainty` is `closed` (with an executed refusal test that
      names its reason code) or licensed with EVERY coverage record its record allocates present
      in parity/coverage_records.toml; a `nominal` claim is refused; a `calibrated` claim must
      allocate coverage record ids, and each allocated id must be emitted by a test under
      crates/*/tests (or python/tests) and registered in scripts/gate_calibration.sh, so a
      measurement can actually produce it at the cut;
      Emission means the id (or its last component, or a `...<component>` composed form) is a STRING
      LITERAL in the body of an `#[ignore]` test function (Rust; comments do not count) or of a
      function (Python), read through scripts/promotion_source.py; wiring means the component is a registered group name on a `run_*` call line
      of scripts/gate_calibration.sh (a comment does not count; a `carried_forward` record is not
      measured at the cut, so only its emission is required). The same holds for every record that
      allocates ids, whatever its claim or status (`carried_forward` included);
  (b2) hidden reservations: a `cov.<...>` id mentioned in any text field of a record but absent from
      that record's `coverage_records` is an error (an id reserved in prose is invisible to (b));
  (b3) orphan registrations: a `run_*` call under a `== 2.2...` header of scripts/gate_calibration.sh
      whose test emits a coverage record (its own name is a literal in its body) must be owned by
      some 2.2 record's `coverage_records` (else it is measured at the cut with no owner);
  (c) every public interval-bearing name on the record's surface files (the workstream's table in
      check_a_intervals.INTERVAL_SURFACES for the 2.2A workstreams, otherwise the record's own
      `surface` and `surface_rust` files minus its `search_impl`) sits behind a closed or covered
      uncertainty route.

Calibration state is reported PER RECORD, for EVERY record that allocates coverage ids (not only
`calibrated` claims, so a `carried_forward` record's pending measurement stays visible) (2.2B repeats workstream ids of 2.2A, so the workstream
key of check_a_intervals would collide): PENDING_CALIBRATION while an allocated record is absent,
PASS once all are present and attested (scripts/calibration_facets.py), FAIL otherwise.

    python3 scripts/check_interval_coordinates.py                 # static check
    python3 scripts/check_interval_coordinates.py --json          # machine-readable result
    python3 scripts/check_interval_coordinates.py --run-refusals  # also execute cited refusal tests
    python3 scripts/check_interval_coordinates.py --self-test

Exit status 1 on FAIL; PENDING_CALIBRATION is not a failure here (the exit gates decide).
Environment: INTERVALS_22_ROOT (tree to read, default the repo), INTERVALS_22_SKIP_ATTEST=1
(self-test only).
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("INTERVALS_22_ROOT", REPO))
RELEASE = "2.2"
# check_a_intervals reads its tree from this variable at import; keep both on one tree.
os.environ["A_INTERVALS_ROOT"] = str(ROOT)
if os.environ.get("INTERVALS_22_SKIP_ATTEST") == "1":
    os.environ["A_INTERVALS_SKIP_ATTEST"] = "1"
sys.path.insert(0, str(REPO / "scripts"))
import check_a_intervals as cai  # noqa: E402  (read-only reuse)
from promotion_source import ignored_test_literals, python_fn_literals, rust_items  # noqa: E402

NOMINAL = cai.NOMINAL
WITHHOLD_REASONS = cai.WITHHOLD_REASONS
REGISTRY_FILES = [*cai.REGISTRY_FILES, "scripts/gate_calibration.sh"]


def load(rel: str) -> dict:
    p = ROOT / rel
    return tomllib.loads(p.read_text()) if p.is_file() else {}


def surface_interval_findings(rel: str) -> list[tuple[int, str]]:
    if RELEASE != "2.3" or not rel.endswith(".py"):
        return cai.interval_findings(rel)
    # Python keyword arguments are uses, not new public interval declarations.
    found = []
    def declarations(nodes):
        for node in nodes:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                declarations(node.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and "interval" in node.name.lower():
                    found.append((node.lineno, node.name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if not name.startswith("_") and "interval" in name.lower():
                    found.append((node.lineno, name))
    declarations(ast.parse((ROOT / rel).read_text()).body)
    return found


def surfaces_of(record: dict) -> list[str]:
    ws = cai.workstream(record)
    if RELEASE == "2.2" and record.get("milestone") == "A" and ws in cai.INTERVAL_SURFACES:
        return list(cai.INTERVAL_SURFACES[ws])
    skip = set(record.get("search_impl") or [])
    files = [*(record.get("surface") or []), *(record.get("surface_rust") or [])]
    if RELEASE == "2.3":
        files += [*(record.get("surface_exports") or []), *(record.get("surface_pyo3") or [])]
    return list(dict.fromkeys(f for f in files if f not in skip))


COV_ID = re.compile(r"cov\.[a-z_.0-9]+")


def emitted_by_a_test(coverage_id: str) -> bool:
    """The id, its last component (the claim/test name) or a `...<component>` composed form is a
    string literal in an `#[ignore]` test body (Rust) or a function body (Python); a comment or a
    mere substring of another token does not count."""
    comp = coverage_id.rsplit(".", 1)[-1]

    def hit(lit: str) -> bool:
        return lit in (comp, coverage_id) or lit.endswith("." + comp)

    # Some crates keep their ignored calibration tests in a #[cfg(test)]
    # module under src (antecedent-estimate does); both layouts emit records.
    for p in [*ROOT.glob("crates/*/tests/**/*.rs"), *ROOT.glob("crates/*/src/**/*.rs")]:
        if comp not in p.read_text():
            continue
        if any(hit(lit) for lits in ignored_test_literals(p).values() for lit in lits):
            return True
    for p in ROOT.glob("python/tests/**/*.py"):
        if comp not in p.read_text():
            continue
        if any(hit(lit) for lits in python_fn_literals(p).values() for lit in lits):
            return True
    return False


def registered_groups(gate_text: str) -> set[str]:
    """Names on `run_*` call lines of gate_calibration.sh (comments and echoes excluded)."""
    names: set[str] = set()
    for line in gate_text.splitlines():
        code = line.split("#", 1)[0].strip()
        if re.match(r"run_\w+\b", code):
            names.update(re.findall(r"[A-Za-z_][\w]*", code)[1:])
    return names


def registrations(gate_text: str) -> list[dict]:
    """`run_<fn> <name>` calls under each `echo "== <header>"`, with the wrapper's `--test` target."""
    targets: dict[str, str] = {}
    current_fn: str | None = None
    for line in gate_text.splitlines():
        m = re.match(r"(run_\w+)\(\) \{", line)
        if m:
            current_fn = m.group(1)
        elif current_fn and (t := re.search(r"--test (\w+)", line)):
            targets[current_fn] = t.group(1)
        elif line.startswith("}"):
            current_fn = None
    out, header = [], ""
    for line in gate_text.splitlines():
        h = re.match(r'echo "== (.*?) ==', line)
        if h:
            header = h.group(1)
        m = re.match(r"(run_\w+) +(?:\S+ +)?(\w+)\s*$", line)
        if m and m.group(1) in targets and not line.startswith("#"):
            out.append(
                {
                    "header": header,
                    "name": m.group(2),
                    "target": targets[m.group(1)],
                    "fn": m.group(1),
                }
            )
    return out


def text_fields(
    value: object, skip: tuple[str, ...] = ("coverage_records",)
) -> list[str]:
    """Every string in a record (recursively) outside the skipped keys."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            t for k, v in value.items() if k not in skip for t in text_fields(v, skip)
        ]
    if isinstance(value, list):
        return [t for v in value for t in text_fields(v, skip)]
    return []


def registered_output_ids(gate_text: str) -> dict[tuple[str, str], set[str]]:
    """Read each registered target once; keep literals bound to its actual test."""
    targets: dict[str, dict[str, set[str]]] = {}
    result = {}
    for registration in registrations(gate_text):
        target, name = registration["target"], registration["name"]
        if target not in targets:
            tests: dict[str, set[str]] = {}
            for source in ROOT.glob(f"crates/*/tests/**/{target}.rs"):
                for test, literals in ignored_test_literals(source).items():
                    tests.setdefault(test, set()).update(literals)
            targets[target] = tests
        result[(target, name)] = targets[target].get(name, set())
    return result


def check() -> dict:
    errors: list[str] = []
    notes: list[str] = []
    refusals: list[dict] = []
    calibration: dict[str, dict] = {}
    promotion_path = f"parity/promotion_{RELEASE.replace('.', '_')}.toml"
    promotion = load(promotion_path)
    stages = load("parity/transport_stages.toml")
    support = load("parity/support_licensed.toml")
    present = {
        r.get("id") for r in load("parity/coverage_records.toml").get("record", [])
    }
    gate_text = (
        (ROOT / "scripts/gate_calibration.sh").read_text()
        if (ROOT / "scripts/gate_calibration.sh").is_file()
        else ""
    )
    groups = registered_groups(gate_text)
    registered_literals = registered_output_ids(gate_text)
    registered_ids = set().union(*registered_literals.values()) if registered_literals else set()
    records = promotion.get("record", [])
    if not records:
        errors.append(f"no record found in {promotion_path}")
    stage_rows = {row.get("route"): row for row in stages.get("routes", [])}
    scoped_routes: set[str] = set()
    diagnostic_groups: set[tuple[str, str]] = set()
    allocations_by_record = {r["id"]: set(r.get("coverage_records") or []) | set(r.get("candidate_coverage_records") or []) for r in records}
    for record in records:
        rid = record["id"]
        for declaration in record.get("diagnostic_measurement_tests") or []:
            parts = declaration.split("::") if isinstance(declaration, str) else []
            if len(parts) != 2:
                errors.append(f"{rid}: diagnostic measurement needs an exact source::test")
                continue
            source, test = parts
            path = ROOT / source
            suites = {
                output.get("measurement_suite", "").split("::", 1)[0]
                for output in record.get("inference_outputs") or []
            }
            if (
                source not in suites or not path.is_file()
                or test not in ignored_test_literals(path)
                or "CoverageTally" in path.read_text()
                or any(COV_ID.search(literal) for literal in ignored_test_literals(path).get(test, []))
            ):
                errors.append(f"{rid}: diagnostic measurement {declaration!r} lacks its actual noncoverage ignored suite allocation")
                continue
            key = (path.stem, test)
            if key not in registered_literals:
                errors.append(f"{rid}: diagnostic measurement {declaration!r} is not registered at its actual target")
            else:
                diagnostic_groups.add(key)
        active = list(record.get("coverage_records") or [])
        candidates = list(record.get("candidate_coverage_records") or [])
        allocated = active + candidates
        if len(allocated) != len(set(allocated)):
            errors.append(f"{rid}: active and candidate coverage allocations must be distinct")
        missing_active = [cid for cid in active if cid not in present]
        missing = [c for c in allocated if c not in present]
        claim = record.get("inference_claim")
        if claim == "nominal":
            errors.append(
                f"{rid}: inference_claim 'nominal' ships an interval no record measures"
            )
        if claim == "calibrated" and not active:
            errors.append(
                f"{rid}: a calibrated claim must allocate its coverage record ids"
            )
        if (
            active
            and claim != "calibrated"
            and record.get("status") != "carried_forward"
        ):
            errors.append(
                f"{rid}: allocates coverage records but claims {claim!r}, not calibrated"
            )
        for cid in allocated:
            if not emitted_by_a_test(cid):
                errors.append(
                    f"{rid}: coverage id {cid} is emitted by no `#[ignore]` test body (string literal) "
                    "under crates/*/tests, crates/*/src, or python/tests"
                )
            # a carried_forward record is deliberately not measured at the cut, so it need not be
            # registered in gate_calibration.sh (its emitting test must still exist)
            if (
                record.get("status") != "carried_forward"
                and cid.rsplit(".", 1)[-1] not in groups
                and cid not in registered_ids
            ):
                errors.append(
                    f"{rid}: coverage id {cid} is not a registered group name on a run_* line of "
                    "scripts/gate_calibration.sh"
                )
        inherited = set()
        for output in record.get("inference_outputs") or []:
            inherited_ids = set(output.get("inherited_coverage_records") or [])
            parents = output.get("inherited_from_records") or []
            parent_ids = set()
            for parent in parents:
                if parent == rid or parent not in allocations_by_record:
                    errors.append(f"{rid}: inherited coverage names an invalid owning record {parent!r}")
                else:
                    parent_ids.update(allocations_by_record[parent])
            # Existing measured records are original methods with independently registered
            # emission. Unmeasured candidate inheritance must identify its actual owner.
            invalid = inherited_ids - (present | parent_ids)
            for cid in sorted(invalid):
                errors.append(f"{rid}: inherited coverage id {cid} has no measured record or explicit candidate owner")
            inherited.update(inherited_ids - invalid)
        hidden = sorted(
            {
                i.rstrip(".")
                for t in text_fields(record)
                for i in COV_ID.findall(t)
                if i.rstrip(".") not in allocated and i.rstrip(".") not in inherited
            }
        )
        for cid in hidden:
            errors.append(
                f"{rid}: coverage id {cid} is mentioned in the record's text but absent from its "
                "coverage_records (a hidden reserved id the calibration checks cannot see)"
            )
        open_uncertainty = closed_uncertainty = False
        for route in record.get("routes") or []:
            name, status, stage = (
                route.get("name"),
                route.get("status"),
                route.get("stage"),
            )
            scoped_routes.add(name)
            row = stage_rows.get(name)
            if status == "licensed" and row is not None:
                if row.get("calibration_reason") == NOMINAL or NOMINAL in json.dumps(
                    row
                ):
                    errors.append(
                        f"{rid}: licensed route {name} has a transport_stages row carrying {NOMINAL}"
                    )
            if status == "licensed" and route.get("registry") == "support_licensed":
                for cell in support.get("cell", []):
                    if (
                        cell.get("query") == route.get("query")
                        and cell.get("contrast") == route.get("contrast")
                        and (
                            cell.get("calibration_reason") == NOMINAL
                            or NOMINAL in json.dumps(cell)
                        )
                    ):
                        errors.append(
                            f"{rid}: licensed route {name} has a support cell carrying {NOMINAL}"
                        )
            if stage == "uncertainty":
                if status == "licensed":
                    open_uncertainty = True
                    if missing_active:
                        errors.append(
                            f"{rid}: uncertainty route {name} is licensed but coverage records are "
                            f"absent: {', '.join(missing_active)}"
                        )
                elif status == "closed":
                    closed_uncertainty = True
                else:
                    errors.append(
                        f"{rid}: uncertainty route {name} has status {status!r}"
                    )
            if status == "closed" and (
                stage == "uncertainty" or route.get("reason_code") in WITHHOLD_REASONS
            ):
                test, assertion = (
                    route.get("refusal_test"),
                    route.get("refusal_assertion"),
                )
                if not test or not assertion:
                    # A frozen record has no executed test yet; in_progress/promoted must.
                    if record.get("status") in ("in_progress", "promoted"):
                        errors.append(
                            f"{rid}: closed route {name} cites no executed refusal test"
                        )
                    continue
                path = ROOT / test
                if not path.is_file():
                    errors.append(f"{rid}: {name}: refusal test file {test} is missing")
                    continue
                body = cai.refusal_closure(path, assertion)
                if body is None:
                    errors.append(
                        f"{rid}: {name}: refusal test {test}::{assertion} does not exist"
                    )
                elif not cai.names_reason(body, str(route.get("reason_code"))):
                    errors.append(
                        f"{rid}: {name}: refusal test {test}::{assertion} never names {route.get('reason_code')!r}"
                    )
                refusals.append(
                    {"record": rid, "route": name, "test": test, "assertion": assertion}
                )
        findings = []
        for rel in surfaces_of(record):
            if not (ROOT / rel).is_file():
                errors.append(f"{rid}: interval surface file {rel} is missing")
                continue
            findings += [(rel, line, n) for line, n in surface_interval_findings(rel)]
        # A wrapper may echo an interval licensed by its input route without
        # constructing its own uncertainty. The record must name those public
        # fields explicitly; any new interval-bearing name still needs a route.
        descriptors = record.get("noninferential_interval_fields") or []
        for descriptor in descriptors:
            name = descriptor.get("name")
            why = descriptor.get("why")
            if not isinstance(name, str) or not isinstance(why, str) or not why.strip():
                errors.append(f"{rid}: interval status descriptors need a name and explanation")
                continue
            matches = [
                row for row in findings
                if row[2] == name and row[0].endswith(".py")
                and (
                    re.match(
                        rf"\s*{re.escape(name)}\s*:\s*str\s*(?:#.*)?$",
                        (ROOT / row[0]).read_text().splitlines()[row[1] - 1],
                    )
                    or (
                        re.match(
                            rf"\s*def {re.escape(name)}\(self\) -> str:",
                            (ROOT / row[0]).read_text().splitlines()[row[1] - 1],
                        )
                        and (ROOT / row[0]).read_text().splitlines()[row[1] - 2].strip() == "@property"
                    )
                )
            ]
            matches += [
                row for row in findings if row[2] == name and row[0].endswith(".rs")
                and re.match(
                    rf"\s*pub {re.escape(name)}\s*:\s*&'static str\s*,",
                    (ROOT / row[0]).read_text().splitlines()[row[1] - 1],
                )
            ]
            if not matches:
                errors.append(f"{rid}: noninferential descriptor {name!r} is not a string field")
            findings = [row for row in findings if row not in matches]
        for name in record.get("internal_interval_fields") or []:
            matches = [
                row for row in findings if row[2] == name and row[0].endswith(".rs")
                and re.search(
                    rf'#\[cfg\(feature = "calibration-internal"\)\]\s*#\[doc\(hidden\)\]\s*pub fn {re.escape(name)}\b',
                    (ROOT / row[0]).read_text(),
                )
            ]
            if not matches:
                errors.append(f"{rid}: internal interval {name!r} lacks its explicit hidden calibration-only gate")
            findings = [row for row in findings if row not in matches]
        inherited = set(record.get("inherited_interval_fields") or [])
        if inherited:
            seen = {name for _, _, name in findings}
            for name in sorted(inherited - seen):
                errors.append(f"{rid}: inherited interval field {name!r} is absent from its surface")
            findings = [row for row in findings if row[2] not in inherited]
        if findings and not (closed_uncertainty or (open_uncertainty and not missing_active)):
            first = findings[0]
            errors.append(
                f"{rid}: public interval-bearing output {first[2]} ({first[0]}:{first[1]}) has no "
                "closed uncertainty route and no coverage records"
            )
        if findings:
            notes.append(
                f"{rid}: {len(findings)} interval-bearing public names, all behind "
                f"{'a closed route' if closed_uncertainty else 'covered routes'}"
            )
        if allocated:
            if missing:
                calibration[rid] = {
                    "status": "PENDING_CALIBRATION",
                    "detail": f"{len(missing)} of {len(allocated)} coverage records absent",
                    "missing": missing,
                    "workstream": cai.workstream(record),
                    "milestone": record.get("milestone"),
                }
            else:
                ok, message = cai.attested()
                calibration[rid] = {
                    "status": "PASS" if ok else "FAIL",
                    "detail": "records present and attested"
                    if ok
                    else f"records present, not attested: {message}",
                    "workstream": cai.workstream(record),
                    "milestone": record.get("milestone"),
                }
                if not ok:
                    errors.append(
                        f"{rid}: coverage records present but not attested: {message}"
                    )
    owned = {
        cid.rsplit(".", 1)[-1]
        for r in records
        for cid in [*(r.get("coverage_records") or []), *(r.get("candidate_coverage_records") or [])]
    }
    for reg in registrations(gate_text):
        if (
            not reg["header"].startswith(RELEASE)
            or reg["name"] in owned
            or (reg["target"], reg["name"]) in diagnostic_groups
            or any(
                cid in registered_literals[(reg["target"], reg["name"])]
                for record in records
                for cid in [*(record.get("coverage_records") or []), *(record.get("candidate_coverage_records") or [])]
            )
        ):
            continue
        for p in ROOT.glob(f"crates/*/tests/**/{reg['target']}.rs"):
            if reg["name"] in ignored_test_literals(p).get(reg["name"], []):
                errors.append(
                    f"scripts/gate_calibration.sh registers {reg['name']} (under '== {reg['header']}') "
                    f"and its test emits that coverage record, but no {RELEASE} record lists it in "
                    "coverage_records (measured at the cut without an owner)"
                )
    legacy = [
        n
        for n, row in stage_rows.items()
        if n not in scoped_routes
        and NOMINAL in json.dumps(row)
        and row.get("status") == "licensed"
    ]
    if legacy:
        notes.append(
            f"{len(legacy)} pre-2.2 licensed rows keep {NOMINAL} (backlog, not 2.2 routes)"
        )
    return {
        "intervals": "FAIL" if errors else "PASS",
        "calibration": calibration,
        "errors": errors,
        "notes": notes,
        "refusal_tests": [
            dict(t) for t in {tuple(sorted(r.items())) for r in refusals}
        ],
        "records": [r["id"] for r in records],
    }


def report(result: dict) -> None:
    print(f"{RELEASE} records checked: {len(result['records'])}")
    for note in result["notes"]:
        print(f"  note: {note}")
    for rid, state in sorted(result["calibration"].items()):
        print(f"  calibration {rid}: {state['status']} ({state['detail']})")
    for error in result["errors"]:
        print(f"  FAIL: {error}")
    print(f"interval coordinates (all {RELEASE} records): {result['intervals']}")


# ----------------------------------------------------------------------------- self-test
SYNTH_SURFACE = "crates/synth/src/lib.rs"
SYNTH_TEST = "crates/synth/tests/refusal.rs"


def self_test() -> int:
    global ROOT
    failures: list[str] = []
    real = REPO
    base = Path(tempfile.mkdtemp(prefix="intervals22_"))
    try:
        promotion = tomllib.loads((real / "parity/promotion_2_2.toml").read_text())
        needed = set(REGISTRY_FILES) - {"parity/coverage_records.toml"}
        for rec in promotion["record"]:
            needed |= set(surfaces_of(rec))
            for route in rec.get("routes") or []:
                if route.get("refusal_test"):
                    needed.add(route["refusal_test"])
            for cid in rec.get("coverage_records") or []:
                for p in [*real.glob("crates/*/tests/**/*.rs"), *real.glob("crates/*/src/**/*.rs")]:
                    if cid.rsplit(".", 1)[-1] in p.read_text():
                        needed.add(str(p.relative_to(real)))
        for rel in sorted(needed):
            src = real / rel
            if src.is_file():
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, base / rel)
        (base / "parity/coverage_records.toml").write_text("")
        synth = base / SYNTH_SURFACE
        synth.parent.mkdir(parents=True, exist_ok=True)
        synth.write_text("pub struct R {\n    pub interval: f64,\n}\n")
        t = base / SYNTH_TEST
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(
            "#[test]\nfn refuses() {\n"
            '    let _ = "cell_not_licensed";\n}\n\n'
            '#[test]\n#[ignore = "coverage"]\nfn synth_alpha() {\n    record("synth_alpha");\n}\n\n'
            '#[test]\n#[ignore = "coverage"]\nfn synth_beta() {\n'
            '    let _key = "cov.synth.b.l95.synth_beta"; // a composed id also counts\n}\n'
        )
        b_record = """
[[record]]
id = "2.2B.X3.synthetic"
workstream = "X3"
milestone = "B"
work_package = "B3"
status = "in_progress"
inference_claim = "calibrated"
coverage_records = ["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"]
surface = ["crates/synth/src/lib.rs"]
routes = [
  { name = "synth.uncertainty", stage = "uncertainty", status = "closed", reason_code = "cell_not_licensed", refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses" },
]
"""
        promo = base / "parity/promotion_2_2.toml"
        # Keep the live A records, drop the live B ones: the synthetic B record below stands in for
        # them, so a B package's in-flight edits cannot break this self-test.
        blocks = re.split(r"(?m)^(?=\[\[record\]\])", promo.read_text())
        kept = [b for b in blocks if 'milestone = "B"' not in b]
        promo.write_text("".join(kept) + b_record)
        # Preserve the evidence for live licensed routes in the copied A/E
        # records. Only the synthetic B record's coverage varies by case.
        baseline_coverage = {
            coverage_id
            for record in tomllib.loads(promo.read_text())["record"]
            if any(route.get("stage") == "uncertainty" and route.get("status") == "licensed"
                   for route in record.get("routes") or [])
            and record.get("id") != "2.2B.X3.synthetic"
            for coverage_id in record.get("coverage_records") or []
        }
        (base / "parity/coverage_records.toml").write_text(
            "".join(f'[[record]]\nid = "{coverage_id}"\n' for coverage_id in sorted(baseline_coverage))
        )
        gate = base / "scripts/gate_calibration.sh"
        # The live 2.2B calibration groups go with the live B records (the synthetic ones replace them).
        live_gate = re.sub(
            r'(?ms)^echo "== 2\.2B .*?(?=^echo "== (?!2\.2B )|\Z)', "", gate.read_text()
        )
        gate.write_text(
            live_gate + '\necho "== 2.2B synthetic coverage (antecedent) =="\n'
            "run_synth() {\n"
            '  local filter="$1"\n'
            '  check "synth: ${filter}" cargo test --release -p synth --test refusal "$filter" -- --ignored\n'
            "}\n"
            "run_synth synth_alpha\n"
            "run_synth synth_beta\n"
        )
        os.environ["INTERVALS_22_SKIP_ATTEST"] = "1"
        os.environ["A_INTERVALS_SKIP_ATTEST"] = "1"

        def run(edit=None, coverage: list[str] | None = None, edit_extra=None) -> dict:
            global ROOT
            case = Path(tempfile.mkdtemp(prefix="intervals22_case_"))
            try:
                shutil.copytree(base, case, dirs_exist_ok=True)
                if coverage is not None:
                    (case / "parity/coverage_records.toml").write_text(
                        "".join(f'[[record]]\nid = "{c}"\n' for c in sorted(baseline_coverage | set(coverage)))
                    )
                if edit is not None:
                    edit(case)
                if edit_extra is not None:
                    edit_extra(case)
                ROOT = case
                cai.ROOT = case
                return check()
            finally:
                ROOT = real
                cai.ROOT = real
                shutil.rmtree(case, ignore_errors=True)

        def edit(rel: str, old: str, new: str):
            def apply(case: Path) -> None:
                p = case / rel
                text = p.read_text()
                if old not in text:
                    raise LookupError(f"{old[:60]!r} not found in {rel}")
                p.write_text(text.replace(old, new, 1))

            return apply

        def expect(label: str, result: dict, needle: str | None) -> None:
            if needle is None:
                if result["intervals"] != "PASS":
                    failures.append(f"'{label}' should pass: {result['errors']}")
            elif result["intervals"] != "FAIL" or not any(
                needle in e for e in result["errors"]
            ):
                failures.append(
                    f"'{label}' did not report {needle!r}: {result['errors']}"
                )

        promo_rel = "parity/promotion_2_2.toml"
        expect("baseline (A records and a synthetic B record)", run(), None)
        state = run()["calibration"].get("2.2B.X3.synthetic", {}).get("status")
        if state != "PENDING_CALIBRATION":
            failures.append(
                f"absent B records must be PENDING_CALIBRATION, got {state}"
            )
        closed_head = '{ name = "synth.uncertainty", stage = "uncertainty", status = "closed", reason_code = "cell_not_licensed", refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses" }'
        opened = '{ name = "synth.uncertainty", stage = "uncertainty", status = "licensed", claim = "calibrated" }'
        expect(
            "B uncertainty licensed without records",
            run(edit(promo_rel, closed_head, opened)),
            "licensed but coverage records are absent",
        )
        covered = run(
            edit(promo_rel, closed_head, opened),
            coverage=["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"],
        )
        expect("B uncertainty licensed with records", covered, None)
        if covered["calibration"].get("2.2B.X3.synthetic", {}).get("status") != "PASS":
            failures.append(
                f"present B records must report PASS: {covered['calibration']}"
            )
        expect(
            "B record with a nominal claim",
            run(
                edit(
                    promo_rel,
                    'inference_claim = "calibrated"\ncoverage_records = ["cov.synth',
                    'inference_claim = "nominal"\ncoverage_records = ["cov.synth',
                )
            ),
            "ships an interval no record measures",
        )
        expect(
            "B calibrated claim without ids",
            run(
                edit(
                    promo_rel,
                    'coverage_records = ["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"]',
                    "coverage_records = []",
                )
            ),
            "must allocate its coverage record ids",
        )
        expect(
            "B coverage id emitted by no test",
            run(edit(SYNTH_TEST, 'record("synth_alpha")', 'record("other")')),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "B coverage id only in a comment of its test is not emitted",
            run(
                edit(SYNTH_TEST, 'record("synth_alpha");', '// record("synth_alpha");')
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "B coverage id literal in a test that is not #[ignore] is not emitted",
            run(
                edit(
                    SYNTH_TEST,
                    '#[ignore = "coverage"]\nfn synth_alpha',
                    "fn synth_alpha",
                )
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "B coverage id as a substring of another identifier is not emitted",
            run(edit(SYNTH_TEST, 'record("synth_alpha")', "let synth_alpha_v2 = 1")),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "B coverage id not registered in gate_calibration.sh",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "run_synth other",
                )
            ),
            "not a registered group name",
        )
        expect(
            "B coverage id only in a comment of gate_calibration.sh is not registered",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "# run_synth synth_alpha",
                )
            ),
            "not a registered group name",
        )
        expect(
            "B coverage id reserved only in the record's notes (hidden)",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"\ncoverage_records = ["cov.synth.b.l95.synth_alpha"',
                    'status = "in_progress"\ninference_notes = "also reserves cov.synth.b.l95.synth_gamma."\ninference_claim = "calibrated"\ncoverage_records = ["cov.synth.b.l95.synth_alpha"',
                )
            ),
            "hidden reserved id",
        )

        def register_orphan(case: Path) -> None:
            with (case / "scripts/gate_calibration.sh").open("a") as fh:
                fh.write("run_synth synth_gamma\n")
            with (case / SYNTH_TEST).open("a") as fh:
                fh.write(
                    '\n#[test]\n#[ignore = "coverage"]\nfn synth_gamma() {\n    record("synth_gamma");\n}\n'
                )

        expect(
            "a 2.2 calibration registration that emits a record no 2.2 record owns",
            run(register_orphan),
            "without an owner",
        )
        expect(
            "a carried_forward record's allocated ids are still checked and reported",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"',
                    'status = "carried_forward"\ninference_claim = "calibrated"',
                ),
                edit_extra=edit(SYNTH_TEST, 'record("synth_alpha")', 'record("other")'),
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "a carried_forward record need not be registered in gate_calibration.sh",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"',
                    'status = "carried_forward"\ninference_claim = "calibrated"',
                ),
                edit_extra=edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "# run_synth synth_alpha",
                ),
            ),
            None,
        )
        cf = run(
            edit(
                promo_rel,
                'status = "in_progress"\ninference_claim = "calibrated"',
                'status = "carried_forward"\ninference_claim = "calibrated"',
            )
        )
        if (
            cf["calibration"].get("2.2B.X3.synthetic", {}).get("status")
            != "PENDING_CALIBRATION"
        ):
            failures.append(
                f"carried_forward record must report its calibration: {cf['calibration']}"
            )
        nc = run(
            edit(
                promo_rel,
                'inference_claim = "calibrated"\ncoverage_records',
                'inference_claim = "assumption_range"\ncoverage_records',
            )
        )
        if nc["calibration"].get("2.2B.X3.synthetic", {}).get(
            "status"
        ) != "PENDING_CALIBRATION" or not any(
            "not calibrated" in e for e in nc["errors"]
        ):
            failures.append(
                f"non-calibrated claim with ids must error and still report calibration: {nc}"
            )
        expect(
            "B interval surface without a closed route",
            run(
                edit(
                    promo_rel,
                    closed_head,
                    '{ name = "synth.point", stage = "evaluate", status = "licensed", claim = "point_only" }',
                )
            ),
            "has no closed uncertainty route",
        )
        expect(
            "an inverse-owned interval is not hidden by inherited forward metadata",
            run(
                edit(
                    "python/antecedent/_inverse.py",
                    "    interval: tuple[float | None, float | None] | None",
                    "    inverse_interval: tuple[float | None, float | None] | None\n"
                    "    interval: tuple[float | None, float | None] | None",
                )
            ),
            "has no closed uncertainty route",
        )
        expect(
            "B refusal test never names the reason",
            run(edit(SYNTH_TEST, '"cell_not_licensed"', '"other"')),
            "never names",
        )

        def freeze_and_drop_test(case: Path) -> None:
            text = (case / promo_rel).read_text()
            text = text.replace(
                ', refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses"',
                "",
            )
            text = text.replace(
                'status = "in_progress"\ninference_claim = "calibrated"',
                'status = "frozen"\ninference_claim = "calibrated"',
            )
            (case / promo_rel).write_text(text)

        expect(
            "B frozen record may omit its refusal test", run(freeze_and_drop_test), None
        )
        expect(
            "B in_progress record needs its refusal test",
            run(
                edit(
                    promo_rel,
                    ', refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses"',
                    "",
                )
            ),
            "cites no executed refusal test",
        )
        # A-record coverage still checked through the generalised path.
        x1 = next(
            r
            for r in promotion["record"]
            if r["id"].endswith("X1.multi_source_limited_experiment")
        )
        expect(
            "A record: licensed row carrying estimator_grid_not_measured",
            run(
                edit(
                    "parity/transport_stages.toml",
                    f'route = "{next(r["name"] for r in x1["routes"] if r["status"] == "licensed" and r["stage"] == "prepare")}"\n',
                    f'route = "{next(r["name"] for r in x1["routes"] if r["status"] == "licensed" and r["stage"] == "prepare")}"\ncalibration_reason = "{NOMINAL}"\n',
                )
            ),
            "carrying estimator_grid_not_measured",
        )
    finally:
        ROOT = real
        shutil.rmtree(base, ignore_errors=True)
    if failures:
        print("check_interval_coordinates self-test FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("check_interval_coordinates self-test: ok")
    return 0


def self_test_23() -> int:
    """Adversarial ownership checks on an isolated 2.3 tree, without measurements."""
    global ROOT, RELEASE
    original_root, original_release, original_cai_root = ROOT, RELEASE, cai.ROOT
    failures = []
    with tempfile.TemporaryDirectory(prefix="intervals23_") as directory:
        ROOT = Path(directory)
        RELEASE = "2.3"
        cai.ROOT = ROOT
        def write(path: str, contents: str) -> None:
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        coverage_id = "cov.synth.dag.frequentist.l95.mean0"
        surface = "python/antecedent/synth.py"
        write(surface, "class Result:\n    interval_status: str\n    def summarize(self):\n        return result(interval=(0, 1))\n")
        write("crates/synth/tests/calibration.rs",
              '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
              f'    emit("{coverage_id}");\n}}\n')
        write("scripts/gate_calibration.sh",
              'run_synth() {\n  cargo test --test calibration "$1" -- --ignored\n}\n'
              'echo "== 2.3 synthetic =="\nrun_synth whole_method\n')
        record = (
            '[[record]]\nid = "2.3A.F1.synthetic"\nmilestone = "A"\n'
            'status = "promoted"\ninference_claim = "point_only"\n'
            f'candidate_coverage_records = ["{coverage_id}"]\n'
            f'surface = ["{surface}"]\n'
            'noninferential_interval_fields = [{name="interval_status", why="status only"}]\n'
        )
        write("parity/promotion_2_3.toml", record)
        try:
            def expect(label: str, needle: str | None) -> dict:
                # Synthetic mutations deliberately reuse a path between checks.
                rust_items.cache_clear()
                result = check()
                if needle is None and result["errors"]:
                    failures.append(f"{label}: {result['errors']}")
                elif needle is not None and not any(needle in error for error in result["errors"]):
                    failures.append(f"{label}: missing {needle!r}: {result['errors']}")
                return result
            baseline = expect("scoped candidate allocation and typed status", None)
            if baseline["calibration"].get("2.3A.F1.synthetic", {}).get("status") != "PENDING_CALIBRATION":
                failures.append("unmeasured diagnostic candidate lost its pending state")
            write(surface, "class Result:\n    interval_status: float\n")
            expect("numeric output cannot masquerade as a descriptor", "not a string field")
            write(surface, "class Result:\n    interval_status: str\n    interval: tuple[float, float]\n")
            expect("real interval needs its own route", "has no closed uncertainty route")
            write(surface, "class Result:\n    interval_status: str\n")
            write("parity/promotion_2_3.toml", record.replace('inference_claim = "point_only"', 'inference_claim = "calibrated"'))
            expect("candidate allocation does not license calibration", "must allocate its coverage record ids")
            write("parity/promotion_2_3.toml", record)
            write("scripts/gate_calibration.sh",
                  'run_synth() {\n cargo test --test wrong_target "$1" -- --ignored\n}\n'
                  'echo "== 2.3 synthetic =="\nrun_synth whole_method\n')
            expect("same test name in another target is not registration", "not a registered group name")
            write("scripts/gate_calibration.sh",
                  'run_synth() {\n cargo test --test calibration "$1" -- --ignored\n}\n'
                  'echo "== 2.3 synthetic =="\nrun_synth whole_method\n')
            write("crates/synth/tests/calibration.rs",
                  '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                  f'    // emit("{coverage_id}");\n}}\n')
            expect("comment is not actual candidate emission", "emitted by no")
            inherited_record = (
                '\n[[record]]\nid="2.3A.F2.inherited"\nmilestone="A"\n'
                'status="promoted"\ninference_claim="point_only"\n'
                'inference_outputs=[{inherited_from_records=["2.3A.F1.synthetic"],'
                f'inherited_coverage_records=["{coverage_id}"]}}]\n'
            )
            write("parity/promotion_2_3.toml", record + inherited_record)
            write("crates/synth/tests/calibration.rs",
                  '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                  f'    emit("{coverage_id}");\n}}\n')
            expect("candidate inheritance names actual separate owner", None)
            write("parity/promotion_2_3.toml", record + inherited_record.replace('"2.3A.F1.synthetic"', '"invented.owner"'))
            expect("inherited reservation without owner refuses", "invalid owning record")
            write("parity/promotion_2_3.toml", record + inherited_record.replace('inference_claim="point_only"', 'inference_claim="calibrated"'))
            expect("candidate inheritance cannot license child calibration", "must allocate its coverage record ids")
            diagnostic_record = record.replace(
                f'candidate_coverage_records = ["{coverage_id}"]\n',
                'diagnostic_measurement_tests = ["crates/synth/tests/calibration.rs::whole_method"]\n'
                'inference_outputs = [{measurement_suite="crates/synth/tests/calibration.rs"}]\n',
            )
            write("parity/promotion_2_3.toml", diagnostic_record)
            write("crates/synth/tests/calibration.rs",
                  '#[test]\n#[ignore = "precision"]\nfn whole_method() {\n'
                  '    metric("whole_method");\n}\n')
            expect("owned precision group does not fabricate a coverage record", None)
            write("crates/synth/tests/calibration.rs",
                  '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                  f'    emit("{coverage_id}");\n}}\n')
            expect("coverage cannot be concealed as a diagnostic", "noncoverage ignored suite")
        finally:
            ROOT, RELEASE, cai.ROOT = original_root, original_release, original_cai_root
    if failures:
        print("check_interval_coordinates 2.3 self-test FAILED:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("check_interval_coordinates 2.3 self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return max(self_test(), self_test_23())
    global RELEASE
    if "--release" in argv:
        index = argv.index("--release")
        if index + 1 >= len(argv) or argv[index + 1] not in ("2.2", "2.3"):
            print("--release requires 2.2 or 2.3")
            return 2
        RELEASE = argv[index + 1]
    result = check()
    if "--run-refusals" in argv:
        failures = cai.run_refusals([dict(t) for t in result["refusal_tests"]])
        result["errors"].extend(f"refusal test failed: {f}" for f in failures)
        result["intervals"] = "FAIL" if result["errors"] else "PASS"
    if "--json" in argv:
        print(json.dumps(result))
    else:
        report(result)
    return 1 if result["intervals"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
