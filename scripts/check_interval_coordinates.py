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
# check_a_intervals reads its tree from this variable at import; keep both on one tree.
os.environ["A_INTERVALS_ROOT"] = str(ROOT)
if os.environ.get("INTERVALS_22_SKIP_ATTEST") == "1":
    os.environ["A_INTERVALS_SKIP_ATTEST"] = "1"
sys.path.insert(0, str(REPO / "scripts"))
import check_a_intervals as cai  # noqa: E402  (read-only reuse)
from promotion_source import ignored_test_literals, python_fn_literals  # noqa: E402

NOMINAL = cai.NOMINAL
WITHHOLD_REASONS = cai.WITHHOLD_REASONS
REGISTRY_FILES = [*cai.REGISTRY_FILES, "scripts/gate_calibration.sh"]


def load(rel: str) -> dict:
    p = ROOT / rel
    return tomllib.loads(p.read_text()) if p.is_file() else {}


def surfaces_of(record: dict) -> list[str]:
    ws = cai.workstream(record)
    if record.get("milestone") == "A" and ws in cai.INTERVAL_SURFACES:
        return list(cai.INTERVAL_SURFACES[ws])
    skip = set(record.get("search_impl") or [])
    files = [*(record.get("surface") or []), *(record.get("surface_rust") or [])]
    return [f for f in files if f not in skip]


COV_ID = re.compile(r"cov\.[a-z_.0-9]+")


def emitted_by_a_test(coverage_id: str) -> bool:
    """The id, its last component (the claim/test name) or a `...<component>` composed form is a
    string literal in an `#[ignore]` test body (Rust) or a function body (Python); a comment or a
    mere substring of another token does not count."""
    comp = coverage_id.rsplit(".", 1)[-1]

    def hit(lit: str) -> bool:
        return lit in (comp, coverage_id) or lit.endswith("." + comp)

    for p in ROOT.glob("crates/*/tests/**/*.rs"):
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


def check() -> dict:
    errors: list[str] = []
    notes: list[str] = []
    refusals: list[dict] = []
    calibration: dict[str, dict] = {}
    promotion = load("parity/promotion_2_2.toml")
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
    records = promotion.get("record", [])
    if not records:
        errors.append("no record found in parity/promotion_2_2.toml")
    stage_rows = {row.get("route"): row for row in stages.get("routes", [])}
    scoped_routes: set[str] = set()
    for record in records:
        rid = record["id"]
        allocated = list(record.get("coverage_records") or [])
        missing = [c for c in allocated if c not in present]
        claim = record.get("inference_claim")
        if claim == "nominal":
            errors.append(
                f"{rid}: inference_claim 'nominal' ships an interval no record measures"
            )
        if claim == "calibrated" and not allocated:
            errors.append(
                f"{rid}: a calibrated claim must allocate its coverage record ids"
            )
        if (
            allocated
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
                    "under crates/*/tests or python/tests"
                )
            # a carried_forward record is deliberately not measured at the cut, so it need not be
            # registered in gate_calibration.sh (its emitting test must still exist)
            if (
                record.get("status") != "carried_forward"
                and cid.rsplit(".", 1)[-1] not in groups
            ):
                errors.append(
                    f"{rid}: coverage id {cid} is not a registered group name on a run_* line of "
                    "scripts/gate_calibration.sh"
                )
        hidden = sorted(
            {
                i.rstrip(".")
                for t in text_fields(record)
                for i in COV_ID.findall(t)
                if i.rstrip(".") not in allocated
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
                    if missing:
                        errors.append(
                            f"{rid}: uncertainty route {name} is licensed but coverage records are "
                            f"absent: {', '.join(missing)}"
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
            findings += [(rel, line, n) for line, n in cai.interval_findings(rel)]
        if findings and not (closed_uncertainty or (open_uncertainty and not missing)):
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
        for cid in r.get("coverage_records") or []
    }
    for reg in registrations(gate_text):
        if not reg["header"].startswith("2.2") or reg["name"] in owned:
            continue
        for p in ROOT.glob(f"crates/*/tests/**/{reg['target']}.rs"):
            if reg["name"] in ignored_test_literals(p).get(reg["name"], []):
                errors.append(
                    f"scripts/gate_calibration.sh registers {reg['name']} (under '== {reg['header']}') "
                    "and its test emits that coverage record, but no 2.2 record lists it in "
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
    print(f"2.2 records checked: {len(result['records'])}")
    for note in result["notes"]:
        print(f"  note: {note}")
    for rid, state in sorted(result["calibration"].items()):
        print(f"  calibration {rid}: {state['status']} ({state['detail']})")
    for error in result["errors"]:
        print(f"  FAIL: {error}")
    print(f"interval coordinates (all 2.2 records): {result['intervals']}")


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
                for p in list(real.glob("crates/*/tests/**/*.rs")):
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
                        "".join(f'[[record]]\nid = "{c}"\n' for c in coverage)
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


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
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
