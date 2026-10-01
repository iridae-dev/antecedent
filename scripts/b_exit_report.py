#!/usr/bin/env python3
"""2.2 B exit gate (scaffold): join registered story results and record state into one table.

The per-package registry `B_PACKAGES` below is where each 2.2B package owner adds the
end-to-end story tests of their package (Rust integration-test targets and Python test files,
all of which must pass from clean preparation through independent artifact consumption). Nothing
is registered yet: no story test exists for any B package, so every package is
PENDING_IMPLEMENTATION until its record is past `frozen` AND its stories are registered and pass.

    python3 scripts/b_exit_report.py --list-tests                      # TSV of tests to run
    python3 scripts/b_exit_report.py --runs runs.tsv --intervals iv.json \\
        [--require-calibrated] [--require-implemented] [--release]
    python3 scripts/b_exit_report.py --self-test

`runs.tsv` has one line per executed test target: `kind<TAB>package<TAB>target<TAB>rc<TAB>log`
(kind is `rust` or `python`; `log` is a file holding the captured output). `iv.json` is the output
of `scripts/check_interval_coordinates.py --json`.

STORY RULES (strict, deterministic; the registry stays empty until package owners add stories):
  * a package reads PASS only if its `B_PACKAGES` entry registers story files (non-empty `rust` or
    `python`) AND lists `evidence`: the route-evidence assertions the story covers, by test name;
  * every evidence name must be a test function (`fn name` / `def name`) in a registered story file,
    or in a fixture file the package's record already cites (`fixtures[].evidence_test` with
    `evidence_assertion`, or a route's `refusal_test` with `refusal_assertion`);
  * at least one evidence name must live in a story file, and every one that does must appear as
    passed in that story's run log (`test <name> ... ok` for Rust; `PASSED ...::<name>` for pytest -rA),
    so a green run of unrelated tests proves nothing;
  * a `promoted` package one of whose records (`carried_forward` included) allocates coverage ids
    never passes without a calibration reading for it: a missing or pending reading is
    PENDING_CALIBRATION (live record) or a CARRIED_FORWARD calibration row (carried record).

Recommended story list (one line each; owners register these in `B_PACKAGES`):
  B1  prepare latent-confounded selection ADMG + conditional query -> point vs enumerated truth ->
      export -> independent consume; non-transportable diagram -> `transport_not_certified`;
      counted laws refused.
  B2  prepare trial + dose + grid -> smoothed psi_h -> export/consume; tampered artifact fails;
      `.interval()` refuses `cell_not_licensed`.
  B3  z baseline -> joint 2-factor deviation with tipping frontier -> export/consume; re-sealed
      mutated threshold/factor set fails replay; union never labelled a CI.
  B4  failed X1 (and X9) decision -> plan studies -> export/replay; bounds-exceeded catalog is
      inconclusive, never 'sufficient'; arrival flips the decision.
  B5  confounded ADMG ETT vs Y0 fixture -> evaluate -> export/consume; non-identified event
      refuses; uncertainty refuses.
  B6  binary m-graph with item-missing variables -> recover joint law -> feed downstream ID ->
      export/consume; out-of-class mechanism refuses; `prepare_empirical` refuses
      `cell_not_licensed`.

Per-package status:
  PASS                   record `promoted`, stories registered with evidence, every registered story
                         target ran and passed, evidence passed, and any allocated coverage is
                         measured and attested
  FAIL                   a registered story failed, did not run or ran zero tests; the record is
                         unregistered here; or the record's interval state is FAIL
  PENDING_IMPLEMENTATION no record yet, record `frozen`, no story tests registered, or the record
                         is `in_progress` with nothing left but its closed routes
  PENDING_CALIBRATION    stories pass but an allocated coverage record is absent (measured once at
                         the 2.2 cut)
  CARRIED_FORWARD        the record is `carried_forward` (its refusal stays visible; 2.2 ships); also
                         a separate calibration row for a carried record's unmeasured coverage ids

Exit status: 1 on any FAIL. PENDING_CALIBRATION is nonzero only with `--require-calibrated`,
PENDING_IMPLEMENTATION only with `--require-implemented`; `--release` is both (the cut).
Environment: B_EXIT_ROOT (tree to read, default the repo).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("B_EXIT_ROOT", REPO))

# ---------------------------------------------------------------------------------------------
# STORY REGISTRY. Each B package adds its end-to-end story tests here (additive, one entry per
# package; do not invent tests that do not exist):
#   "rust":   [("<crate>", "<integration test target>")]   run as `cargo test -p <crate> --test <target>`
#   "python": ["tests/<file>.py"]                         run from python/ with pytest
#   "evidence": ["<test fn name>", ...]  the route-evidence assertions the story covers (required
#                                        as soon as a story is registered; see STORY RULES above)
# A story test must take its package's cell from clean preparation through independent artifact
# consumption and its typed refusal (mirror crates/antecedent/tests/a_exit_gate.rs and
# python/tests/test_a_exit_gate.py).
# ---------------------------------------------------------------------------------------------
B_PACKAGES: dict[str, dict] = {
    "B1": {
        "workstream": "X2",
        "title": "latent-confounded ADMG transport row",
        "rust": [],
        "python": [],
        "evidence": [],
    },
    "B2": {
        "workstream": "X4",
        "title": "smoothed dose-response transport grid",
        "rust": [],
        "python": [],
        "evidence": [],
    },
    "B3": {
        "workstream": "X3",
        "title": "joint mechanism deviations + sampling uncertainty",
        "rust": [],
        "python": [],
        "evidence": [],
    },
    "B4": {
        "workstream": "X6",
        "title": "planning over restricted catalogs / competing designs",
        "rust": [],
        "python": [],
        "evidence": [],
    },
    "B5": {
        "workstream": "X8",
        "title": "bounded ADMG counterfactual-ID cell",
        "rust": [],
        "python": [],
        "evidence": [],
    },
    "B6": {
        "workstream": "X10",
        "title": "exact binary causal observation recovery",
        "rust": [],
        "python": [],
        "evidence": [],
    },
}

PASSED_RUST = re.compile(r"^test result: ok\. (\d+) passed; 0 failed", re.M)
PASSED_PY = re.compile(r"\b(\d+) passed\b")
FAILED_PY = re.compile(r"\b\d+ (failed|error)")


def b_records(root: Path) -> dict[str, list[dict]]:
    """Every milestone-B record by work package (a package may own several, e.g. a carried one)."""
    p = root / "parity/promotion_2_2.toml"
    records = tomllib.loads(p.read_text()).get("record", []) if p.is_file() else []
    out: dict[str, list[dict]] = {}
    for r in records:
        if r.get("milestone") == "B":
            out.setdefault(r["work_package"], []).append(r)
    return out


def list_tests() -> list[str]:
    out = []
    for pkg, entry in B_PACKAGES.items():
        out += [f"rust\t{pkg}\t{crate}\t{target}" for crate, target in entry["rust"]]
        out += [f"python\t{pkg}\t{path}" for path in entry["python"]]
    return out


def parse_runs(text: str) -> dict[tuple[str, str, str], tuple[int, str]]:
    runs = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 5:
            kind, pkg, target, rc, log = parts
            body = Path(log).read_text() if Path(log).is_file() else ""
            runs[(kind, pkg, target)] = (int(rc), body)
    return runs


def _rust_files(root: Path, crate: str, target: str) -> list[Path]:
    base = root / "crates" / crate / "tests"
    return [
        p for p in (base / f"{target}.rs", base / target / "main.rs") if p.is_file()
    ]


def _defines(text: str, name: str, python: bool) -> bool:
    pat = (
        rf"^\s*(?:async )?def {re.escape(name)}\("
        if python
        else rf"\bfn {re.escape(name)}\s*[(<]"
    )
    return re.search(pat, text, re.M) is not None


def _fixture_evidence(recs: list[dict]) -> list[tuple[str, str]]:
    """(file, test name) pairs the package's records already cite as evidence."""
    out: list[tuple[str, str]] = []
    for r in recs:
        for f in r.get("fixtures") or []:
            if f.get("evidence_test") and f.get("evidence_assertion"):
                out.append((f["evidence_test"], f["evidence_assertion"]))
        for route in r.get("routes") or []:
            if route.get("refusal_test") and route.get("refusal_assertion"):
                out.append((route["refusal_test"], route["refusal_assertion"]))
    return out


def evidence_problems(
    pkg: str, entry: dict, runs: dict, recs: list[dict], root: Path
) -> list[str]:
    """Why the package's registered stories do not evidence its routes (empty: they do)."""
    evidence = list(entry.get("evidence") or [])
    if not evidence:
        return [
            f"{pkg}: stories are registered but B_PACKAGES lists no `evidence` assertions (the story "
            "file must name the route evidence tests it covers)"
        ]
    story_files: list[
        tuple[str, str, Path, str]
    ] = []  # (kind, target key, file, log body)
    for crate, target in entry["rust"]:
        got = runs.get(("rust", pkg, f"{crate}/{target}"))
        for f in _rust_files(root, crate, target):
            story_files.append(("rust", f"{crate}/{target}", f, got[1] if got else ""))
    for path in entry["python"]:
        got = runs.get(("python", pkg, path))
        f = root / "python" / path
        if f.is_file():
            story_files.append(("python", path, f, got[1] if got else ""))
    cited = _fixture_evidence(recs)
    problems: list[str] = []
    in_story = 0
    for name in evidence:
        homes = [
            (kind, key, body)
            for kind, key, f, body in story_files
            if _defines(f.read_text(), name, kind == "python")
        ]
        if homes:
            in_story += 1
            for kind, key, body in homes:
                pat = (
                    rf"^test (?:\S+::)?{re.escape(name)} \.\.\. ok$"
                    if kind == "rust"
                    else rf"^PASSED \S+::{re.escape(name)}\b"
                )
                if not re.search(pat, body, re.M):
                    problems.append(
                        f"{pkg}: evidence test {name} is in story {key} but did not pass in its run log"
                    )
            continue
        fixture_ok = any(
            n == name
            and (root / rel).is_file()
            and _defines((root / rel).read_text(), name, rel.endswith(".py"))
            for rel, n in cited
        )
        if not fixture_ok:
            problems.append(
                f"{pkg}: evidence test {name} is neither a test function of a registered story file "
                "nor an already-cited record fixture assertion"
            )
    if evidence and not in_story:
        problems.append(
            f"{pkg}: no evidence test lives in a registered story file (fixture citations alone "
            "are not a story)"
        )
    return problems


def story_state(
    pkg: str,
    entry: dict,
    runs: dict,
    recs: list[dict] | None = None,
    root: Path | None = None,
) -> tuple[str, str]:
    """(PASS|FAIL|NONE, detail) over the package's registered story targets."""
    targets = [("rust", f"{c}/{t}") for c, t in entry["rust"]] + [
        ("python", p) for p in entry["python"]
    ]
    if not targets:
        return "NONE", "no story test registered in B_PACKAGES"
    total = 0
    for kind, target in targets:
        got = runs.get((kind, pkg, target))
        if got is None:
            return "FAIL", f"story target {target} did not run"
        rc, body = got
        if rc != 0:
            return "FAIL", f"story target {target} failed (rc={rc})"
        if kind == "rust":
            n = sum(int(x) for x in PASSED_RUST.findall(body))
        else:
            n = sum(int(x) for x in PASSED_PY.findall(body))
            if FAILED_PY.search(body):
                return "FAIL", f"story target {target} reports failures"
        if n == 0:
            return "FAIL", f"story target {target} ran zero tests"
        total += n
    problems = evidence_problems(pkg, entry, runs, recs or [], root or ROOT)
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", f"{total} story test(s) passed"


def _allocated(rec: dict) -> list[str]:
    return list(rec.get("coverage_records") or [])


def evaluate(
    records: dict[str, dict | list[dict]],
    runs: dict,
    intervals: dict,
    *,
    require_calibrated: bool,
    require_implemented: bool,
    root: Path | None = None,
) -> tuple[list[str], int]:
    calibration = intervals.get("calibration", {})
    rows: list[tuple[str, str, str]] = []
    by_pkg: dict[str, list[dict]] = {
        k: (v if isinstance(v, list) else [v]) for k, v in records.items()
    }
    for pkg, entry in B_PACKAGES.items():
        recs = by_pkg.get(pkg, [])
        label = f"{pkg} {entry['workstream']}"
        if not recs:
            rows.append(
                (
                    label,
                    "PENDING_IMPLEMENTATION",
                    "no record in parity/promotion_2_2.toml",
                )
            )
            continue
        live = [r for r in recs if r.get("status") != "carried_forward"]
        carried = [r for r in recs if r.get("status") == "carried_forward"]
        extra: list[tuple[str, str, str]] = []
        for r in carried:
            ids = _allocated(r)
            cal = calibration.get(r["id"])
            if ids and (cal is None or cal.get("status") != "PASS"):
                why = (
                    cal["detail"]
                    if cal
                    else "no calibration reading in the interval check"
                )
                extra.append(
                    (
                        f"{pkg} calibration",
                        "CARRIED_FORWARD",
                        f"{r['id']}: {len(ids)} coverage id(s) carried with the record ({why})",
                    )
                )
        if not live:
            rows.append(
                (
                    label,
                    "CARRIED_FORWARD",
                    "every record of the package is carried_forward",
                )
            )
            rows.extend(extra)
            continue
        rec = live[0]
        status = rec.get("status")
        if status == "frozen":
            rows.append((label, "PENDING_IMPLEMENTATION", "record frozen"))
            rows.extend(extra)
            continue
        state, detail = story_state(pkg, entry, runs, recs, root)
        if state == "FAIL":
            rows.append((label, "FAIL", detail))
            continue
        if state == "NONE":
            rows.append((label, "PENDING_IMPLEMENTATION", f"record {status}; {detail}"))
            rows.extend(extra)
            continue
        cal_fail = cal_pending = None
        for r in live:
            cal = calibration.get(r["id"])
            if cal is not None and cal["status"] == "FAIL":
                cal_fail = cal["detail"]
            elif cal is not None and cal["status"] == "PENDING_CALIBRATION":
                cal_pending = cal["detail"]
            elif cal is None and _allocated(r):
                # allocated coverage ids with no reading at all never pass silently
                cal_pending = (
                    f"{r['id']}: {len(_allocated(r))} allocated coverage id(s) have no calibration "
                    "reading"
                )
        if cal_fail is not None:
            rows.append((label, "FAIL", f"interval calibration: {cal_fail}"))
        elif cal_pending is not None:
            rows.append((label, "PENDING_CALIBRATION", f"{detail}; {cal_pending}"))
        elif status != "promoted":
            rows.append(
                (label, "PENDING_IMPLEMENTATION", f"{detail}; record still {status}")
            )
        else:
            rows.append((label, "PASS", detail))
        rows.extend(extra)
    for pkg in sorted(set(by_pkg) - set(B_PACKAGES)):
        rows.append(
            (
                pkg,
                "FAIL",
                f"record {by_pkg[pkg][0]['id']} has a work package unregistered in B_PACKAGES",
            )
        )
    rows.append(
        (
            "intervals",
            intervals.get("intervals", "FAIL"),
            "zero newly introduced unmeasured interval coordinates (all 2.2 records)",
        )
    )
    rows.append(
        (
            "B7 X7",
            "CARRIED_FORWARD",
            "GPU lane carried forward to 2.3+; no code in 2.2 (CPU-only 2.1 baseline)",
        )
    )
    width = max(len(r[0]) for r in rows)
    lines = ["2.2 B exit gate", ""]
    lines += [
        f"{item:<{width}}  {status:<22}  {detail}" for item, status, detail in rows
    ]
    for error in intervals.get("errors", []):
        lines.append(f"  interval check: {error}")
    statuses = {r[1] for r in rows}
    lines.append("")
    if "FAIL" in statuses:
        lines.append("verdict: FAIL")
        return lines, 1
    bad = []
    if "PENDING_CALIBRATION" in statuses and require_calibrated:
        bad.append("PENDING_CALIBRATION")
    if "PENDING_IMPLEMENTATION" in statuses and require_implemented:
        bad.append("PENDING_IMPLEMENTATION")
    if bad:
        lines.append(f"verdict: {' + '.join(bad)} (required flags: not accepted)")
        return lines, 1
    if "PENDING_CALIBRATION" in statuses or "PENDING_IMPLEMENTATION" in statuses:
        pend = sorted(s for s in statuses if s.startswith("PENDING"))
        lines.append(f"verdict: {' + '.join(pend)} (scaffold: nothing failed)")
        return lines, 0
    lines.append("verdict: PASS")
    return lines, 0


def self_test() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tree"

        def put(rel: str, text: str) -> None:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)

        def log(name: str, body: str) -> Path:
            p = Path(t) / name
            p.write_text(body)
            return p

        ev = "b1_routes_refuse_uncertainty"
        ev2 = "b1_point_equals_truth"
        rust_ok = log(
            "ok_rust.log",
            f"test {ev} ... ok\ntest {ev2} ... ok\ntest result: ok. 2 passed; 0 failed; 0 ignored\n",
        )
        rust_no_ev = log(
            "no_ev.log",
            f"test {ev2} ... ok\ntest other ... ok\ntest result: ok. 2 passed; 0 failed; 0 ignored\n",
        )
        zero = log("zero.log", "test result: ok. 0 passed; 0 failed; 0 ignored\n")
        py_ok = log(
            "ok_py.log",
            f"PASSED tests/test_b1_exit_gate.py::{ev}\n===== 1 passed in 0.5s =====\n",
        )
        py_unrelated = log(
            "py_unrelated.log",
            "PASSED tests/test_b1_exit_gate.py::test_other\n===== 1 passed in 0.5s =====\n",
        )
        bad_py = log("bad_py.log", "===== 1 failed, 2 passed in 0.5s =====\n")
        rust_src = f"#[test]\nfn {ev}() {{}}\n#[test]\nfn {ev2}() {{}}\n"
        py_src = f"def {ev}():\n    pass\n"
        put("crates/antecedent/tests/b1_exit_gate.rs", rust_src)
        put("python/tests/test_b1_exit_gate.py", py_src)
        put(
            "crates/antecedent/tests/fixture_cited.rs",
            "#[test]\nfn cited_fixture_fn() {}\n",
        )
        saved = {
            k: {a: list(b) if isinstance(b, list) else b for a, b in v.items()}
            for k, v in B_PACKAGES.items()
        }
        try:
            B_PACKAGES["B1"]["rust"] = [("antecedent", "b1_exit_gate")]
            B_PACKAGES["B1"]["python"] = ["tests/test_b1_exit_gate.py"]
            B_PACKAGES["B1"]["evidence"] = [ev, ev2]

            def rec(
                pkg: str, status: str, claim: str = "point_only", ids=None, rid=None
            ) -> dict:
                return {
                    "id": rid or f"2.2B.{pkg}",
                    "work_package": pkg,
                    "milestone": "B",
                    "status": status,
                    "inference_claim": claim,
                    "coverage_records": list(ids or []),
                }

            def runs(rust=rust_ok, py=py_ok, rc_rust=0, rc_py=0, skip_py=False) -> dict:
                lines = [f"rust\tB1\tantecedent/b1_exit_gate\t{rc_rust}\t{rust}"]
                if not skip_py:
                    lines.append(
                        f"python\tB1\ttests/test_b1_exit_gate.py\t{rc_py}\t{py}"
                    )
                return parse_runs("\n".join(lines))

            iv_ok = {"intervals": "PASS", "calibration": {}, "errors": []}
            iv_pending = {
                "intervals": "PASS",
                "calibration": {
                    "2.2B.B1": {
                        "status": "PENDING_CALIBRATION",
                        "detail": "2 of 2 absent",
                    }
                },
                "errors": [],
            }
            iv_fail = {"intervals": "FAIL", "calibration": {}, "errors": ["x"]}

            def expect(label, records, rn, iv, code, verdict, cal=False, impl=False):
                lines, rc = evaluate(
                    records,
                    rn,
                    iv,
                    require_calibrated=cal,
                    require_implemented=impl,
                    root=root,
                )
                text = "\n".join(lines)
                if rc != code or verdict not in text:
                    failures.append(
                        f"'{label}': rc={rc} (want {code}), want {verdict!r}\n{text}"
                    )

            # ---- all six promoted and passing (every package its own stories)
            full = {p: rec(p, "promoted") for p in B_PACKAGES}
            lines_all = []
            for p in B_PACKAGES:
                if p == "B1":
                    continue
                n = p.lower()
                put(
                    f"crates/antecedent/tests/{n}_exit_gate.rs",
                    f"#[test]\nfn {n}_evidence() {{}}\n",
                )
                B_PACKAGES[p]["rust"] = [("antecedent", f"{n}_exit_gate")]
                B_PACKAGES[p]["evidence"] = [f"{n}_evidence"]
                ok = log(
                    f"{n}.log",
                    f"test {n}_evidence ... ok\ntest result: ok. 1 passed; 0 failed\n",
                )
                lines_all.append(f"rust\t{p}\tantecedent/{n}_exit_gate\t0\t{ok}")
            lines_all += [
                f"rust\tB1\tantecedent/b1_exit_gate\t0\t{rust_ok}",
                f"python\tB1\ttests/test_b1_exit_gate.py\t0\t{py_ok}",
            ]
            expect(
                "all promoted and passing",
                full,
                parse_runs("\n".join(lines_all)),
                iv_ok,
                0,
                "verdict: PASS",
                True,
                True,
            )
            for p in list(B_PACKAGES):
                if p != "B1":
                    B_PACKAGES[p]["rust"] = []
                    B_PACKAGES[p]["evidence"] = []
            expect(
                "scaffold: nothing registered is pending, exit 0",
                {},
                {},
                iv_ok,
                0,
                "scaffold: nothing failed",
            )
            expect(
                "pending implementation fails --require-implemented",
                {},
                {},
                iv_ok,
                1,
                "not accepted",
                False,
                True,
            )
            expect(
                "frozen record is pending implementation",
                {"B1": rec("B1", "frozen")},
                {},
                iv_ok,
                0,
                "record frozen",
            )
            expect(
                "registered stories + evidence, promoted: B1 passes",
                {"B1": rec("B1", "promoted")},
                runs(),
                iv_ok,
                0,
                "B1 X2      PASS",
            )
            expect(
                "a failed Rust story fails",
                {"B1": rec("B1", "promoted")},
                runs(rc_rust=101),
                iv_ok,
                1,
                "verdict: FAIL",
            )
            expect(
                "a failed Python story fails",
                {"B1": rec("B1", "promoted")},
                runs(py=bad_py),
                iv_ok,
                1,
                "verdict: FAIL",
            )
            expect(
                "a nonzero Python rc fails",
                {"B1": rec("B1", "promoted")},
                runs(rc_py=1),
                iv_ok,
                1,
                "verdict: FAIL",
            )
            expect(
                "zero tests run fails",
                {"B1": rec("B1", "promoted")},
                runs(rust=zero),
                iv_ok,
                1,
                "ran zero tests",
            )
            expect(
                "a story that never ran fails",
                {"B1": rec("B1", "promoted")},
                runs(skip_py=True),
                iv_ok,
                1,
                "did not run",
            )
            expect(
                "pending calibration exits 0 by default",
                {"B1": rec("B1", "in_progress", "calibrated")},
                runs(),
                iv_pending,
                0,
                "PENDING_CALIBRATION",
            )
            expect(
                "pending calibration fails --require-calibrated",
                {"B1": rec("B1", "in_progress", "calibrated")},
                runs(),
                iv_pending,
                1,
                "not accepted",
                True,
            )
            expect(
                "a failed story outranks pending calibration",
                {"B1": rec("B1", "in_progress", "calibrated")},
                runs(rc_rust=1),
                iv_pending,
                1,
                "verdict: FAIL",
            )
            expect(
                "in_progress point-only record is not a pass",
                {"B1": rec("B1", "in_progress")},
                runs(),
                iv_ok,
                0,
                "record still in_progress",
            )
            expect(
                "record with no registered story is pending",
                {"B2": rec("B2", "promoted")},
                runs(),
                iv_ok,
                0,
                "no story test registered",
            )
            expect("interval violation fails", {}, {}, iv_fail, 1, "verdict: FAIL")
            expect(
                "unregistered package fails",
                {"B9": rec("B9", "promoted")},
                {},
                iv_ok,
                1,
                "unregistered",
            )
            expect(
                "carried_forward is accepted",
                {"B3": rec("B3", "carried_forward")},
                {},
                iv_ok,
                0,
                "CARRIED_FORWARD",
                True,
                False,
            )

            # ---- D4: evidence rules. A green story run is not enough.
            promoted = {"B1": rec("B1", "promoted")}
            expect(
                "a story whose run log lacks the evidence tests fails",
                promoted,
                runs(rust=rust_no_ev),
                iv_ok,
                1,
                f"evidence test {ev} is in story",
            )
            expect(
                "a python evidence test that did not pass fails",
                promoted,
                runs(py=py_unrelated),
                iv_ok,
                1,
                "did not pass in its run log",
            )
            B_PACKAGES["B1"]["evidence"] = []
            expect(
                "stories registered with no evidence list fail",
                promoted,
                runs(),
                iv_ok,
                1,
                "lists no `evidence`",
            )
            B_PACKAGES["B1"]["evidence"] = [ev, "b1_ghost_test"]
            expect(
                "an evidence name defined nowhere fails",
                promoted,
                runs(),
                iv_ok,
                1,
                "b1_ghost_test is neither",
            )
            B_PACKAGES["B1"]["evidence"] = ["cited_fixture_fn"]
            fixture_rec = rec("B1", "promoted")
            fixture_rec["fixtures"] = [
                {
                    "evidence_test": "crates/antecedent/tests/fixture_cited.rs",
                    "evidence_assertion": "cited_fixture_fn",
                }
            ]
            expect(
                "evidence only in a cited fixture is not a story",
                {"B1": fixture_rec},
                runs(),
                iv_ok,
                1,
                "no evidence test lives in a registered story file",
            )
            B_PACKAGES["B1"]["evidence"] = [ev, "cited_fixture_fn"]
            expect(
                "evidence in a story file plus a cited fixture passes",
                {"B1": fixture_rec},
                runs(),
                iv_ok,
                0,
                "B1 X2      PASS",
            )
            B_PACKAGES["B1"]["evidence"] = [ev, ev2]

            # ---- D3: allocated coverage ids never pass without a calibration reading
            ids = ["cov.x.l95.a"]
            expect(
                "promoted with allocated ids and no calibration reading is pending",
                {"B1": rec("B1", "promoted", "assumption_range", ids)},
                runs(),
                iv_ok,
                0,
                "PENDING_CALIBRATION",
            )
            expect(
                "... and fails --require-calibrated",
                {"B1": rec("B1", "promoted", "assumption_range", ids)},
                runs(),
                iv_ok,
                1,
                "not accepted",
                True,
            )
            iv_pass = {
                "intervals": "PASS",
                "calibration": {"2.2B.B1": {"status": "PASS", "detail": "ok"}},
                "errors": [],
            }
            expect(
                "promoted with allocated ids and a passing reading passes",
                {"B1": rec("B1", "promoted", "calibrated", ids)},
                runs(),
                iv_pass,
                0,
                "B1 X2      PASS",
            )
            carried = rec(
                "B1", "carried_forward", "calibrated", ids, rid="2.2B.B1.carried"
            )
            expect(
                "a carried_forward sibling's unmeasured ids are a CARRIED_FORWARD calibration row",
                {"B1": [rec("B1", "promoted"), carried]},
                runs(),
                iv_ok,
                0,
                "B1 calibration",
                True,
                False,
            )
            expect(
                "...and the promoted record still passes",
                {"B1": [rec("B1", "promoted"), carried]},
                runs(),
                iv_ok,
                0,
                "B1 X2           PASS",
            )
            expect(
                "a carried record with a failing reading does not hide behind the carry",
                {"B1": [rec("B1", "promoted", "calibrated", ids), carried]},
                runs(),
                {
                    "intervals": "PASS",
                    "calibration": {
                        "2.2B.B1": {"status": "FAIL", "detail": "not attested"}
                    },
                    "errors": [],
                },
                1,
                "interval calibration",
            )
        finally:
            B_PACKAGES.clear()
            B_PACKAGES.update(saved)
    if failures:
        for f in failures:
            print(f"SELF-TEST FAIL: {f}")
        return 1
    print("b_exit_report self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    if "--list-tests" in argv:
        print("\n".join(list_tests()))
        return 0
    args = {}
    it = iter(
        a
        for a in argv
        if a not in ("--require-calibrated", "--require-implemented", "--release")
    )
    for k in it:
        args[k] = next(it, "")
    runs = parse_runs(Path(args["--runs"]).read_text()) if "--runs" in args else {}
    intervals = (
        json.loads(Path(args["--intervals"]).read_text())
        if "--intervals" in args
        else {}
    )
    release = "--release" in argv
    lines, code = evaluate(
        b_records(ROOT),
        runs,
        intervals,
        require_calibrated=release or "--require-calibrated" in argv,
        require_implemented=release or "--require-implemented" in argv,
    )
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
