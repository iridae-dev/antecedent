#!/usr/bin/env python3
"""2.2 A exit gate, interval-coordinate check: zero newly introduced unmeasured intervals.

For every 2.2A record of parity/promotion_2_2.toml this proves that no licensed route, row or
artifact ships an interval without a matching coverage record:

  (a) no licensed route of a 2.2A record has an owning-registry row (parity/transport_stages.toml
      [[routes]], parity/support_licensed.toml cells) that carries `estimator_grid_not_measured`
      (as `calibration_reason` or anywhere in its text);
  (b) every route whose stage is `uncertainty` is `closed` (and cites an executed refusal test) or
      is licensed with EVERY coverage record its record allocates present in
      parity/coverage_records.toml;
  (c) every public interval-bearing output of a 2.2A surface (Rust result types, wire structs and
      Python result types found by grep) belongs to a workstream whose interval route is closed
      or covered, and the closed-route refusal tests that show the runtime status is withheld
      (`cell_not_licensed` and the other closed reasons) exist, name their reason, and, with
      `--run-refusals`, are executed here.

It also reports, per calibrated 2.2A workstream (X1, X4), whether its allocated coverage records
exist and are attested:

  PENDING_CALIBRATION   at least one allocated record is absent from coverage_records.toml
  PASS                  every allocated record exists and `calibration_facets.py status --require`
                        attests them (the tree matches the code they were measured at)
  FAIL                  records exist but are not attested, or a route is open without them

    python3 scripts/check_a_intervals.py                 # static check
    python3 scripts/check_a_intervals.py --json          # machine-readable result on stdout
    python3 scripts/check_a_intervals.py --run-refusals  # also execute the cited refusal tests
    python3 scripts/check_a_intervals.py --self-test     # synthetic violations must fail

Exit status 1 on any FAIL. PENDING_CALIBRATION is not a failure here; the gate decides.
Environment: A_INTERVALS_ROOT (tree to read, default the repo), A_INTERVALS_SKIP_ATTEST=1
(self-test only: treat present records as attested).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("A_INTERVALS_ROOT", REPO))
sys.path.insert(0, str(REPO / "scripts"))
from promotion_source import closure_code  # noqa: E402  (the one reader of test bodies for the gates)

NOMINAL = "estimator_grid_not_measured"
# Closed reasons an interval or an aggregate can be refused with; their refusal tests are the
# runtime proof that the unmeasured coordinate is withheld.
WITHHOLD_REASONS = {
    "cell_not_licensed",
    "estimator_inference_mismatch",
    "scenario_aggregate_not_licensed",
}

# Public surfaces where an interval-bearing output could live, per workstream: the Rust result
# and wire types and the Python result types of each 2.2A cell.
INTERVAL_SURFACES = {
    "X1": [
        "crates/antecedent-io/src/mz_transport_artifact.rs",
        "crates/antecedent-estimate/src/statistical_transport.rs",
        "crates/antecedent/src/analysis/mz_transport.rs",
        "python/antecedent/transport/_multi_source.py",
    ],
    "X2": [
        "crates/antecedent-estimate/src/transport_scenarios.rs",
        "crates/antecedent-io/src/transport_scenario_artifact.rs",
        "crates/antecedent/src/analysis/transport_scenarios.rs",
        "python/antecedent/transport/_scenarios.py",
    ],
    "X4": [
        "crates/antecedent-estimate/src/learned_continuous.rs",
        "crates/antecedent-io/src/learned_continuous_artifact.rs",
        "crates/antecedent/src/analysis/learned_continuous.rs",
        "python/antecedent/transport/_learned_continuous.py",
    ],
    "X5": [
        "crates/antecedent-estimate/src/temporal_transport.rs",
        "crates/antecedent-io/src/temporal_transport_artifact.rs",
        "crates/antecedent/src/analysis/temporal_transport.rs",
        "python/antecedent/transport/_temporal.py",
    ],
    "X8": [
        "crates/antecedent/src/cross_world.rs",
        "crates/antecedent-io/src/cross_world_artifact.rs",
        "python/antecedent/cross_world.py",
    ],
    "X9": [
        "crates/antecedent-estimate/src/mixed_source.rs",
        "crates/antecedent-io/src/mixed_source_artifact.rs",
        "crates/antecedent/src/analysis/mixed_source.rs",
        "python/antecedent/transport/_mixed_source.py",
    ],
}
REGISTRY_FILES = [
    "parity/promotion_2_2.toml",
    "parity/transport_stages.toml",
    "parity/support_licensed.toml",
    "parity/support_closed.toml",
    "parity/coverage_records.toml",
]
# `pub interval: ...`, `pub mean_intervals: ...`, `pub fn interval(`, python `interval:` fields,
# `def interval(` and dict keys such as "mean_intervals".
RUST_INTERVAL = re.compile(r"^\s*pub(?:\([a-z]+\))?\s+(?:fn\s+)?(\w*interval\w*)\b", re.I)
PY_INTERVAL = re.compile(r"^\s*(?:def\s+)?(\w*interval\w*)\s*[:(=]", re.I)


def load(rel: str) -> dict:
    path = ROOT / rel
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def workstream(record: dict) -> str:
    return str(record.get("workstream") or record["id"].split(".")[1])


def coverage_ids() -> set[str]:
    return {r.get("id") for r in load("parity/coverage_records.toml").get("record", [])}


def refusal_closure(path: Path, name: str) -> str | None:
    """Comment-free code of test `name` in `path` plus the helpers it reaches
    (scripts/promotion_source.py); None when no such function exists. A reason
    code in a comment, or in an unrelated function that merely follows the test
    in the file, is not the test naming it."""
    try:
        return closure_code(path, name) or None
    except (ValueError, SyntaxError, OSError):
        return None


def names_reason(body: str, reason: str) -> bool:
    """Whether `body` names `reason` as a whole token (not as part of a longer one)."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(reason)}(?![A-Za-z0-9_])", body) is not None


def attested() -> tuple[bool, str]:
    """Every coverage record in this tree matches the code it was measured at."""
    if os.environ.get("A_INTERVALS_SKIP_ATTEST") == "1":
        return True, "attestation skipped (self-test)"
    proc = subprocess.run(
        [sys.executable, "scripts/calibration_facets.py", "status", "--require"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    return proc.returncode == 0, tail[0]


def interval_findings(rel: str) -> list[tuple[int, str]]:
    path = ROOT / rel
    if not path.is_file():
        return []
    pattern = PY_INTERVAL if rel.endswith(".py") else RUST_INTERVAL
    found = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        hit = pattern.match(line)
        if hit:
            found.append((number, hit.group(1)))
    return found


def check() -> dict:
    errors: list[str] = []
    notes: list[str] = []
    refusals: list[dict] = []
    calibration: dict[str, dict] = {}
    promotion = load("parity/promotion_2_2.toml")
    stages = load("parity/transport_stages.toml")
    support = load("parity/support_licensed.toml")
    present = coverage_ids()
    records = [r for r in promotion.get("record", []) if r.get("milestone") == "A"]
    if not records:
        errors.append("no 2.2A record found in parity/promotion_2_2.toml")

    stage_rows = {row.get("route"): row for row in stages.get("routes", [])}
    scoped_routes: set[str] = set()
    for record in records:
        rid, ws = record["id"], workstream(record)
        allocated = list(record.get("coverage_records") or [])
        missing = [c for c in allocated if c not in present]
        claim = record.get("inference_claim")
        if claim == "nominal":
            errors.append(f"{rid}: inference_claim 'nominal' ships an interval no record measures")
        if claim == "calibrated" and not allocated:
            errors.append(f"{rid}: a calibrated claim must allocate its coverage record ids")
        open_uncertainty = False
        closed_uncertainty = False
        for route in record.get("routes") or []:
            name, status, stage = route.get("name"), route.get("status"), route.get("stage")
            scoped_routes.add(name)
            row = stage_rows.get(name)
            # (a) no licensed row carries the unmeasured-interval reason.
            if status == "licensed" and row is not None:
                text = json.dumps(row)
                if row.get("calibration_reason") == NOMINAL or NOMINAL in text:
                    errors.append(
                        f"{rid}: licensed route {name} has a transport_stages row carrying {NOMINAL}"
                    )
            if status == "licensed" and route.get("registry") == "support_licensed":
                for cell in support.get("cell", []):
                    if (
                        cell.get("query") == route.get("query")
                        and cell.get("contrast") == route.get("contrast")
                        and (cell.get("calibration_reason") == NOMINAL or NOMINAL in json.dumps(cell))
                    ):
                        errors.append(f"{rid}: licensed route {name} has a support cell carrying {NOMINAL}")
            # (b) an uncertainty route is closed, or licensed with every record present.
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
                    errors.append(f"{rid}: uncertainty route {name} has status {status!r}")
            if status == "closed" and (stage == "uncertainty" or route.get("reason_code") in WITHHOLD_REASONS):
                test, assertion = route.get("refusal_test"), route.get("refusal_assertion")
                if not test or not assertion:
                    errors.append(f"{rid}: closed route {name} cites no executed refusal test")
                    continue
                path = ROOT / test
                if not path.is_file():
                    errors.append(f"{rid}: {name}: refusal test file {test} is missing")
                    continue
                body = refusal_closure(path, assertion)
                if body is None:
                    errors.append(f"{rid}: {name}: refusal test {test}::{assertion} does not exist")
                elif not names_reason(body, str(route.get("reason_code"))):
                    errors.append(
                        f"{rid}: {name}: refusal test {test}::{assertion} never names "
                        f"{route.get('reason_code')!r}"
                    )
                refusals.append({"record": rid, "route": name, "test": test, "assertion": assertion})
        # (c) public interval-bearing outputs must sit behind a closed or covered route.
        surfaces = INTERVAL_SURFACES.get(ws)
        if surfaces is None:
            errors.append(f"{rid}: workstream {ws} has no interval-surface entry in check_a_intervals.py")
            surfaces = []
        findings = []
        for rel in surfaces:
            if not (ROOT / rel).is_file():
                errors.append(f"{rid}: interval surface file {rel} is missing")
                continue
            findings += [(rel, line, name) for line, name in interval_findings(rel)]
        if findings and not (closed_uncertainty or (open_uncertainty and not missing)):
            first = findings[0]
            errors.append(
                f"{rid}: public interval-bearing output {first[2]} ({first[0]}:{first[1]}) has no "
                "closed uncertainty route and no coverage records"
            )
        if findings:
            notes.append(f"{rid}: {len(findings)} interval-bearing public names, all behind "
                         f"{'a closed route' if closed_uncertainty else 'covered routes'}")
        # Calibration state of a calibrated workstream.
        if claim == "calibrated":
            if missing:
                calibration[ws] = {
                    "status": "PENDING_CALIBRATION",
                    "detail": f"{len(missing)} of {len(allocated)} coverage records absent",
                    "missing": missing,
                }
            else:
                ok, message = attested()
                if ok:
                    calibration[ws] = {"status": "PASS", "detail": "records present and attested"}
                else:
                    calibration[ws] = {"status": "FAIL", "detail": f"records present, not attested: {message}"}
                    errors.append(f"{rid}: coverage records present but not attested: {message}")
    # Legacy rows are reported, not judged: they predate 2.2A.
    legacy = [
        name
        for name, row in stage_rows.items()
        if name not in scoped_routes and NOMINAL in json.dumps(row) and row.get("status") == "licensed"
    ]
    if legacy:
        notes.append(f"{len(legacy)} pre-2.2A licensed rows keep {NOMINAL} (not 2.2A routes)")
    return {
        "intervals": "FAIL" if errors else "PASS",
        "calibration": calibration,
        "errors": errors,
        "notes": notes,
        "refusal_tests": [dict(t) for t in {tuple(sorted(r.items())) for r in refusals}],
        "records": [r["id"] for r in records],
    }


def run_refusals(tests: list[dict]) -> list[str]:
    """Execute every cited closed-route refusal test; return failures."""
    failures = []
    rust: dict[tuple[str, str], list[str]] = {}
    python: list[str] = []
    for t in tests:
        if t["test"].endswith(".py"):
            node = f"{t['test'].removeprefix('python/')}::{t['assertion']}"
            if node not in python:
                python.append(node)
            continue
        parts = Path(t["test"]).parts
        crate = parts[1] if parts[0] == "crates" else None
        if crate is None:
            failures.append(f"cannot place refusal test {t['test']} in a crate")
            continue
        names = rust.setdefault((crate, Path(t["test"]).stem), [])
        if t["assertion"] not in names:
            names.append(t["assertion"])
    for (crate, target), names in sorted(rust.items()):
        cmd = ["cargo", "test", "-p", crate, "--test", target, "--", "--exact", *names]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        passed = len(re.findall(r"^test .* \.\.\. ok$", proc.stdout, re.M))
        print(f"  cargo test -p {crate} --test {target} ({len(names)} refusal test(s)): "
              f"{'ok' if proc.returncode == 0 and passed >= len(names) else 'FAILED'}", file=sys.stderr)
        if proc.returncode != 0 or passed < len(names):
            failures.append(f"{crate}/{target}: {' '.join(names)}\n{proc.stdout[-1500:]}{proc.stderr[-800:]}")
    if python:
        cmd = ["uv", "run", "--project", str(ROOT / "python"), "pytest", "-q", "-p", "no:cacheprovider", *python]
        proc = subprocess.run(cmd, cwd=ROOT / "python", capture_output=True, text=True)
        print(f"  pytest ({len(python)} refusal test(s)): {'ok' if proc.returncode == 0 else 'FAILED'}", file=sys.stderr)
        if proc.returncode != 0:
            failures.append(f"pytest {' '.join(python)}\n{proc.stdout[-1500:]}{proc.stderr[-800:]}")
    return failures


def report(result: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result))
        return
    print(f"2.2A records checked: {len(result['records'])}")
    for note in result["notes"]:
        print(f"  note: {note}")
    for ws, state in sorted(result["calibration"].items()):
        print(f"  calibration {ws}: {state['status']} ({state['detail']})")
    for error in result["errors"]:
        print(f"  FAIL: {error}")
    print(f"interval coordinates: {result['intervals']}")


# ----------------------------------------------------------------------------- self-test


def self_test() -> int:
    """Each synthetic violation, in a temp copy of the registries, must fail; committed files
    are never written. The baseline copy must pass."""
    global ROOT
    real_root = ROOT
    failures: list[str] = []
    baseline = Path(tempfile.mkdtemp(prefix="a_intervals_"))
    try:
        needed = set(REGISTRY_FILES) - {"parity/coverage_records.toml"}
        promotion = tomllib.loads((real_root / "parity/promotion_2_2.toml").read_text())
        for record in (r for r in promotion["record"] if r.get("milestone") == "A"):
            if record.get("milestone") != "A":
                continue
            needed |= set(INTERVAL_SURFACES.get(workstream(record), []))
            for route in record.get("routes") or []:
                if route.get("refusal_test"):
                    needed.add(route["refusal_test"])
        for rel in sorted(needed):
            src = real_root / rel
            if src.is_file():
                (baseline / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, baseline / rel)
        (baseline / "parity/coverage_records.toml").write_text("")
        os.environ["A_INTERVALS_SKIP_ATTEST"] = "1"

        def run(edit=None, coverage: list[str] | None = None) -> dict:
            global ROOT
            case = Path(tempfile.mkdtemp(prefix="a_intervals_case_"))
            try:
                shutil.copytree(baseline, case, dirs_exist_ok=True)
                if coverage is not None:
                    (case / "parity/coverage_records.toml").write_text(
                        "".join(f'[[record]]\nid = "{c}"\n' for c in coverage)
                    )
                if edit is not None:
                    edit(case)
                ROOT = case
                return check()
            finally:
                ROOT = real_root
                shutil.rmtree(case, ignore_errors=True)

        def edit_file(rel: str, old: str, new: str):
            def apply(case: Path) -> None:
                path = case / rel
                text = path.read_text()
                if old not in text:
                    raise LookupError(f"{old[:60]!r} not found in {rel}")
                path.write_text(text.replace(old, new, 1))

            return apply

        def expect(label: str, result: dict, *, fail_with: str | None) -> None:
            if fail_with is None:
                if result["intervals"] != "PASS":
                    failures.append(f"'{label}' should pass: {result['errors']}")
                else:
                    print(f"self-test ok: '{label}' passes")
                return
            if result["intervals"] != "FAIL" or not any(fail_with in e for e in result["errors"]):
                failures.append(f"'{label}' did not report {fail_with!r}: {result['errors']}")
            else:
                print(f"self-test ok: '{label}' fails")

        x1 = next(r for r in promotion["record"] if workstream(r) == "X1" and r.get("milestone") == "A")
        x1_records = list(x1["coverage_records"])
        x1_route = next(r for r in x1["routes"] if r["stage"] == "uncertainty")
        promo = "parity/promotion_2_2.toml"
        licensed_route = (
            f'{{ name = "{x1_route["name"]}", stage = "uncertainty", status = "licensed", claim = "calibrated" }}'
        )
        closed_route_head = f'{{ name = "{x1_route["name"]}", stage = "uncertainty", status = "closed"'

        def open_x1(case: Path) -> None:
            path = case / promo
            text = path.read_text()
            start = text.index(closed_route_head)
            end = text.index("},", start) + 1
            path.write_text(text[:start] + licensed_route + text[end:])

        expect("baseline copy", run(), fail_with=None)
        result = run()
        state = result["calibration"].get("X1", {}).get("status")
        if state != "PENDING_CALIBRATION":
            failures.append(f"absent records must be PENDING_CALIBRATION, got {state}")
        else:
            print("self-test ok: absent coverage records report PENDING_CALIBRATION")
        expect("uncertainty route licensed without coverage records", run(open_x1),
               fail_with="licensed but coverage records are absent")
        covered = run(open_x1, coverage=x1_records)
        expect("uncertainty route licensed with every coverage record present", covered, fail_with=None)
        if covered["calibration"].get("X1", {}).get("status") != "PASS":
            failures.append(f"present records must report PASS: {covered['calibration']}")
        else:
            print("self-test ok: present and attested coverage records report PASS")
        half = run(open_x1, coverage=x1_records[:1])
        expect("only one of two coverage records present", half, fail_with="licensed but coverage records are absent")
        stale = run(coverage=x1_records[:1])
        if stale["calibration"]["X1"]["status"] != "PENDING_CALIBRATION":
            failures.append("one of two records present must stay PENDING_CALIBRATION")
        else:
            print("self-test ok: one of two records stays PENDING_CALIBRATION")
        licensed_name = next(
            r["name"] for r in x1["routes"] if r["status"] == "licensed" and r["stage"] == "prepare"
        )
        row_head = f'route = "{licensed_name}"\n'
        expect(
            "licensed row carries estimator_grid_not_measured",
            run(edit_file("parity/transport_stages.toml", row_head, row_head + f'calibration_reason = "{NOMINAL}"\n')),
            fail_with="carrying estimator_grid_not_measured",
        )
        expect(
            "nominal claim",
            run(edit_file(promo, 'inference_claim = "calibrated"', 'inference_claim = "nominal"')),
            fail_with="ships an interval no record measures",
        )
        expect(
            "closed route without refusal evidence",
            run(edit_file(promo, f', refusal_assertion = "{x1_route["refusal_assertion"]}"', "")),
            fail_with="cites no executed refusal test",
        )
        expect(
            "refusal test that does not exist",
            run(edit_file(promo, f'refusal_assertion = "{x1_route["refusal_assertion"]}"', 'refusal_assertion = "no_such_test"')),
            fail_with="does not exist",
        )
        x5 = next(r for r in promotion["record"] if workstream(r) == "X5" and r.get("milestone") == "A")
        x5_heads = [
            f'{{ name = "{r["name"]}", stage = "uncertainty", status = "closed"'
            for r in x5["routes"]
            if r["stage"] == "uncertainty"
        ]

        def drop_x5_interval_route(case: Path) -> None:
            # X5 declares one closed uncertainty route per surface (Rust facade, Python stage);
            # the violation is having none, so every one of them goes.
            path = case / promo
            text = path.read_text()
            for head in x5_heads:
                start = text.index(head)
                end = text.index("},", start) + 2
                text = text[:start] + text[end:]
            path.write_text(text)

        expect("interval-bearing surface with no closed route", run(drop_x5_interval_route),
               fail_with="has no closed uncertainty route")
        fresh_surface = INTERVAL_SURFACES["X8"][0]

        def add_public_interval(case: Path) -> None:
            path = case / fresh_surface
            path.write_text(path.read_text() + "\npub struct SelfTestLeak {\n    pub interval: Option<(f64, f64)>,\n}\n")

        def drop_x8_route(case: Path) -> None:
            path = case / promo
            text = path.read_text()
            head = '{ name = "antecedent.cross_world.uncertainty"'
            start = text.index(head)
            end = text.index("},", start) + 2
            path.write_text(text[:start] + text[end:])

        def leak_without_route(case: Path) -> None:
            add_public_interval(case)
            drop_x8_route(case)

        expect("new public interval field without a closed route", run(leak_without_route),
               fail_with="public interval-bearing output")
        expect("public interval field behind a closed route", run(add_public_interval), fail_with=None)
    except (LookupError, ValueError, StopIteration) as err:
        failures.append(f"self-test could not build a case: {err!r}")
    finally:
        ROOT = real_root
        shutil.rmtree(baseline, ignore_errors=True)
    if failures:
        for failure in failures:
            print(f"SELF-TEST FAIL: {failure}")
        return 1
    print("check_a_intervals self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    result = check()
    if "--run-refusals" in argv and result["intervals"] == "PASS":
        print("executing the cited closed-route refusal tests:", file=sys.stderr)
        failures = run_refusals(result["refusal_tests"])
        if failures:
            result["intervals"] = "FAIL"
            result["errors"] += [f"refusal test failed: {f}" for f in failures]
    report(result, as_json="--json" in argv)
    return 1 if result["intervals"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
