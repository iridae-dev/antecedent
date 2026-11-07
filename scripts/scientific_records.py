#!/usr/bin/env python3
"""Recorded passes of the scientific assertions, current while their dependency manifests match.

    scientific_records.py status [--check]   owed versus current records; --check exits 1 when any
                                             is owed and executes nothing (CI and releases)
    scientific_records.py run [--all] [--only PATTERN]
                                             earn the owed assertions and record their passes

`parity/scientific_records.toml` holds one row per assertion that passed, with the
content-addressed dependency manifest it earned (`scripts/evidence_earning.py`). `status`
checks manifests with `scripts/evidence_manifests.py` only: no compiler, Cargo or test runs.
A row without a manifest, or whose manifest no longer matches, is owed.

`run` earns owed Rust assertions by building each test target once with coverage
instrumentation and running each assertion exactly; owed Python assertions run in one pytest
batch. A rerun withdraws the old pass first, so a failed or interrupted run never leaves a
superseded pass standing. `--all` or ANTECEDENT_SCIENTIFIC_ALL=1 reruns everything.

A recorded pass is evidence for support only. It is not a calibration measurement and no
coverage or support claim reads it as one.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from _repo import REPO_ROOT, scientific_target_feature_args, load_toml
from evidence_fingerprint import Fingerprinter, UnresolvedInput, locate

RECORDS = "parity/scientific_records.toml"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

HEADER = """# Recorded passes of the scientific assertions (see scripts/scientific_records.py).
#
# One [[record]] per scientific assertion that passed, written by
# `python3 scripts/scientific_records.py run`, the one place scientific assertions
# run. A record is current while its dependency manifest matches; an owed
# assertion reruns, a current one carries forward. A recorded pass is evidence for
# support only. It is never a calibration measurement or a coverage claim.
#
# test         exact selector: rust `crate::binary::module::name` (`lib` for the
#              library target) or python node id relative to python/
# roles        evidence roles of the citing claims
# passed       always true; failures are reported, never recorded
# commit       HEAD when the pass was recorded
# dirty        whether the working tree differed from that commit
# fingerprint  sha256 of the assertion's dependency closure when it passed (identity only)
# recorded_at  UTC time of the pass
# dependency_manifest  parity/evidence_manifests/<sha256>.json: what the pass depended on
"""


def load_records(root: Path) -> dict[str, dict]:
    path = root / RECORDS
    if not path.is_file():
        return {}
    data = load_toml(path)
    return {r["test"]: r for r in data.get("record", [])}


def write_records(root: Path, records: dict[str, dict]) -> None:
    out = [HEADER]
    for test in sorted(records):
        r = records[test]
        out.append(
            "\n[[record]]\n"
            f"test = {json.dumps(test)}\n"
            f"roles = {json.dumps(list(r['roles']))}\n"
            "passed = true\n"
            f"commit = {json.dumps(r['commit'])}\n"
            f"dirty = {'true' if r.get('dirty') else 'false'}\n"
            f"fingerprint = {json.dumps(r['fingerprint'])}\n"
            f"recorded_at = {json.dumps(r['recorded_at'])}\n"
            + (f"dependency_manifest = {json.dumps(r['dependency_manifest'])}\n" if r.get("dependency_manifest") else "")
        )
    (root / RECORDS).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORDS).write_text("".join(out), encoding="utf-8", newline="\n")


def git_state(root: Path) -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    return head.stdout.strip() or "unknown", bool(dirty.stdout.strip())


def classify(root: Path, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(owed, current) rows, each annotated with its selector. A row is current only when its
    recorded pass has a dependency manifest whose inputs and declarations still match."""
    from evidence_earning import binding, declarations
    from evidence_manifests import Checker

    records = load_records(root)
    checker = None
    owed, current = [], []
    for row in rows:
        info = locate(root, row)
        rec = records.get(info["selector"])
        item = {**row, "selector": info["selector"], "fingerprint": None, "manifest_reasons": ["no_manifest"]}
        if rec and rec.get("passed") is True and rec.get("dependency_manifest"):
            checker = checker or Checker(root)
            status = checker.check(binding("scientific", info["selector"], rec.get("fingerprint")), rec["dependency_manifest"])
            if status["current"] and checker.load(rec["dependency_manifest"]).get("declarations") != declarations(row):
                status["current"] = False
                status["reasons"].append("scientific_declarations_changed")
            item.update(fingerprint=rec.get("fingerprint"), manifest_reasons=status["reasons"])
            if status["current"]:
                current.append(item)
                continue
        owed.append(item)
    return owed, current

def record_passes(root: Path, items: list[dict]) -> int:
    """Merge passing rows, each with the dependency manifest it earned, into the ledger."""
    if not items:
        return 0
    commit, dirty = git_state(root)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = load_records(root)  # reread: another gate may have written meanwhile
    for item in items:
        records[item["selector"]] = {
            "roles": item.get("roles") or ["scientific"],
            "commit": commit,
            "dirty": dirty,
            "fingerprint": item["fingerprint"],
            "recorded_at": now,
            "dependency_manifest": item["dependency_manifest"],
        }
    write_records(root, records)
    return len(items)


def withdraw_records(root: Path, items: list[dict]) -> None:
    """A rerun supersedes old passes, including if it fails or is interrupted."""
    records = load_records(root)
    changed = False
    for item in items:
        if item["selector"] in records:
            del records[item["selector"]]
            changed = True
    if changed:
        write_records(root, records)


def all_scientific_rows(root: Path) -> list[dict]:
    import evidence_audit

    return evidence_audit.scientific_rows(
        evidence_audit.build_report(root)["claims"], evidence_audit.registered_scientific_tests(root)
    )


# --------------------------------------------------------------------------
# Execution capture
# --------------------------------------------------------------------------

PASSED_ONE = re.compile(r"test result: ok\. 1 passed")

def line_coverage(export: dict) -> dict[str, set[int]]:
    """Resolved source path -> lines where a covered region starts, from an llvm-cov export."""
    coverage: dict[str, set[int]] = {}
    for block in export.get("data", []):
        for item in block.get("files", []):
            coverage[str(Path(item["filename"]).resolve())] = {
                s[0] for s in item.get("segments", []) if s[2] > 0 and s[3]
            }
    return coverage


def llvm_tool(name: str) -> str:
    """Path of an `llvm-tools-preview` binary for the active toolchain."""
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], text=True, capture_output=True).stdout.strip()
    for candidate in sorted(Path(sysroot, "lib", "rustlib").glob(f"*/bin/{name}")):
        return str(candidate)
    raise SystemExit(f"{name} not found; run `rustup component add llvm-tools-preview`")


def capture_env(root: Path) -> dict[str, str]:
    """Build environment for coverage-instrumented test binaries in their own target directory."""
    rustflags = os.environ.get("RUSTFLAGS", "")
    return {
        **os.environ,
        "CARGO_TARGET_DIR": os.environ.get("ANTECEDENT_CAPTURE_TARGET_DIR", str(root / "target/cov")),
        "RUSTFLAGS": f"{rustflags} -C instrument-coverage".strip(),
        "CARGO_PROFILE_TEST_OPT_LEVEL": os.environ.get("CARGO_PROFILE_TEST_OPT_LEVEL", "2"),
        "CARGO_PROFILE_TEST_DEBUG": os.environ.get("CARGO_PROFILE_TEST_DEBUG", "line-tables-only"),
        "LLVM_PROFILE_FILE": os.devnull,
    }


def build_command(root: Path, crate: str, target: tuple[str, ...]) -> list[str]:
    """`cargo test --no-run` for one crate target, reporting its test executable as JSON."""
    features = scientific_target_feature_args(root / "crates" / crate / "Cargo.toml", list(target))
    return ["cargo", "test", "--no-run", "--message-format=json", "-p", crate, *target, *features]


def built_executable(stdout: str) -> str | None:
    """The test executable a `cargo test --no-run --message-format=json` build produced."""
    found = None
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg.get("profile", {}).get("test") and msg.get("executable"):
            found = msg["executable"]
    return found


def capture_one(root: Path, binary: str, info: dict, env: dict[str, str], run=subprocess.run) -> dict[str, set[int]] | None:
    """Run one Rust assertion's instrumented binary; its line coverage, or `None` on failure."""
    libtest = info["selector"].split("::", 2)[2]
    with tempfile.TemporaryDirectory(prefix="antecedent-capture-") as tmp:
        raw, data = Path(tmp) / "run.profraw", Path(tmp) / "run.profdata"
        done = run(
            [binary, "--exact", libtest, "--include-ignored", "--test-threads", "1"],
            cwd=root / "crates" / info["crate"],
            env={**env, "LLVM_PROFILE_FILE": str(raw)},
            text=True,
            capture_output=True,
        )
        if done.returncode != 0 or not PASSED_ONE.search((done.stdout or "") + (done.stderr or "")):
            print(f"FAIL: {info['selector']}: not captured\n{((done.stdout or '') + (done.stderr or ''))[-400:]}")
            return None
        merged = run([llvm_tool("llvm-profdata"), "merge", "-sparse", str(raw), "-o", str(data)], text=True, capture_output=True)
        if merged.returncode != 0:
            print(f"FAIL: {info['selector']}: profile merge failed\n{merged.stderr[-400:]}")
            return None
        export = run(
            [llvm_tool("llvm-cov"), "export", f"-instr-profile={data}", binary],
            text=True,
            capture_output=True,
        )
        if export.returncode != 0:
            print(f"FAIL: {info['selector']}: llvm-cov export failed\n{export.stderr[-400:]}")
            return None
        data = json.loads(export.stdout)
        return line_coverage(data)


def earn_rust(root: Path, items: list[dict], run=subprocess.run) -> int:
    """Earn the given Rust rows: build each test target once with coverage instrumentation
    (through the compiler inventory driver when `rustc-dev` is installed), run each assertion
    exactly, and publish its manifest. The ledger is written after every target, so an
    interrupted run keeps what it finished. Returns the number of failures."""
    from evidence_earning import require_compile_environment
    require_compile_environment(root)
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[dict, dict]]] = {}
    for row in items:
        info = locate(root, row)
        groups.setdefault((info["crate"], tuple(info["target"])), []).append((row, info))
    print(f"scientific records: earning {len(items)} Rust assertion(s) in {len(groups)} target(s)", flush=True)
    if not groups:
        return 0
    from evidence_earning import Snapshot, binding, build_analyzer, compile_time_code, project_snapshot

    fp = Fingerprinter(root, scientific_build=True)
    env = capture_env(root)
    # Only this run's compilations describe the code being earned: an inventory from
    # an earlier compilation cannot say a body is still eligible. A crate Cargo does
    # not recompile has no inventory and keeps its complete source.
    inventory_dir = root / "target/evidence-compiler-inventory"
    shutil.rmtree(inventory_dir, ignore_errors=True)
    inventory_dir.mkdir(parents=True)
    # Body projection needs a compilation with no compile-time code (see project_snapshot).
    compile_time = compile_time_code(root)
    projectable = compile_time is not None and any(not compile_time.get(crate, {crate}) for crate, _ in groups)
    driver = build_analyzer(root, root / "target/evidence-analyzer") if projectable else None
    if driver is not None:
        env.update(RUSTC_WORKSPACE_WRAPPER=str(driver), ANTECEDENT_ANALYZER_OUTPUT=str(inventory_dir))
    recorded, failed = 0, 0
    for (crate, target), todo in sorted(groups.items()):
        from test_evidence import rust_test_features
        args = scientific_target_feature_args(root / "crates" / crate / "Cargo.toml", list(target))
        configured = set(args[1].split(",")) if args else set()
        required = set().union(*(rust_test_features(root / row["evidence_test"], row["evidence_assertion"])
                                 for row, _ in todo))
        if not required <= configured:
            raise RuntimeError(f"scientific target {crate} {' '.join(target)} is missing declared features: "
                               + ", ".join(sorted(required - configured)))
        snapshots = {
            info["selector"]: Snapshot(root, row, binding("scientific", info["selector"], None), fp)
            for row, info in todo
        }
        withdraw_records(root, [info for _, info in todo])
        built = run(build_command(root, crate, target), cwd=root, env=env, text=True, capture_output=True)
        binary = built_executable(built.stdout or "") if built.returncode == 0 else None
        if binary is None:
            print(f"FAIL: {crate} {' '.join(target)}: instrumented build failed\n{(built.stderr or '')[-400:]}")
            failed += len(todo)
            continue
        earned = []
        for row, info in todo:
            coverage = capture_one(root, binary, info, env, run)
            if coverage is None:
                failed += 1
                continue
            snapshot = snapshots[info["selector"]]
            try:
                if driver is not None:
                    project_snapshot(snapshot, inventory_dir, coverage)
                ref = snapshot.publish()
            except RuntimeError as error:
                print(f"FAIL: {info['selector']}: {error}")
                failed += 1
                continue
            earned.append({**row, "selector": info["selector"], "dependency_manifest": ref,
                           "fingerprint": snapshot.manifest["record"]["fingerprint"]})
        recorded += record_passes(root, earned)
        print(f"  {crate} {' '.join(target)}: {len(earned)}/{len(todo)} earned", flush=True)
    print(f"scientific records: earned {recorded} Rust pass(es); {failed} failed")
    return failed


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def force_all(args: argparse.Namespace | None = None) -> bool:
    return bool(getattr(args, "all", False)) or os.environ.get("ANTECEDENT_SCIENTIFIC_ALL") == "1"


def cmd_status(root: Path, args: argparse.Namespace) -> int:
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows)
    print(f"scientific records: {len(current)} current, {len(owed)} owed of {len(rows)}")
    for item in owed:
        print(f"  owed: {item['selector']} ({', '.join(item['manifest_reasons'][:3])})")
    return int(args.check and bool(owed))


def python_environment_ok(root: Path) -> bool:
    for candidate in (root / "python/.venv/bin/python", root / "python/.venv/Scripts/python.exe"):
        if candidate.is_file():
            return subprocess.run([str(candidate), str(root / "scripts/native_build_identity.py"), "check"]).returncode == 0
    print("FAIL: the Python conformance environment is missing; run bash scripts/test.sh python")
    return False


def matches_only(item: dict, patterns: list[str]) -> bool:
    """Does a row match any `--only` pattern (substring or glob) on its selector, test path or id?"""
    if not patterns:
        return True
    fields = [str(item.get(k) or "") for k in ("selector", "evidence_test", "id")]
    return any(
        pattern in field or fnmatch.fnmatchcase(field, pattern) for pattern in patterns for field in fields
    )


def run_batch(root: Path, batch: list[dict]) -> tuple[int, set[str]]:
    """Execute one batch through run_evidence_rows.py; its exit status and the selectors proven passed."""
    with tempfile.TemporaryDirectory(prefix="antecedent-scientific-") as tmp:
        ledger = Path(tmp) / "selection.toml"
        passed_out = Path(tmp) / "passed.json"
        ledger.write_text(
            "".join(
                "[[capabilities]]\n"
                f"id = {json.dumps(i['selector'])}\n"
                f"evidence_test = {json.dumps(i['evidence_test'])}\n"
                f"evidence_assertion = {json.dumps(i['evidence_assertion'])}\n\n"
                for i in batch
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            # The optimized test profile of scripts/test.sh, unless the caller chose one.
            "CARGO_PROFILE_TEST_OPT_LEVEL": os.environ.get("CARGO_PROFILE_TEST_OPT_LEVEL", "2"),
            "CARGO_PROFILE_TEST_DEBUG": os.environ.get("CARGO_PROFILE_TEST_DEBUG", "line-tables-only"),
            "ANTECEDENT_EVIDENCE_PASSED_OUT": str(passed_out),
        }
        code = subprocess.run(
            [sys.executable, str(root / "scripts/run_evidence_rows.py"), str(root), str(ledger), str(root),
             "capabilities", "scientific"],
            cwd=root, env=env,
        ).returncode
        passed = set(json.loads(passed_out.read_text(encoding="utf-8"))) if passed_out.is_file() else set()
    return code, passed


def cmd_run(root: Path, args: argparse.Namespace) -> int:
    from evidence_earning import require_compile_environment
    require_compile_environment(root)
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows)
    only = list(getattr(args, "only", None) or [])
    if only and not any(matches_only(i, only) for i in owed + current):
        print(f"FAIL: --only {only!r} matches no scientific assertion")
        return 2
    candidates = (owed + current) if force_all(args) else owed
    run_items = [i for i in candidates if matches_only(i, only)]
    print(
        f"scientific records: running {len(run_items)} of {len(rows)} "
        f"({len(current) if not force_all(args) else 0} carried forward"
        + (f", filtered by --only {only}" if only else "")
        + ")",
        flush=True,
    )
    if not run_items:
        return 0
    if any(i["evidence_test"].endswith(".py") for i in run_items) and not python_environment_ok(root):
        return 1
    # Each batch is recorded as soon as it finishes, so an interrupted or partly failing
    # run keeps every pass it earned.
    status, recorded = 0, 0
    rust = [i for i in run_items if not i["evidence_test"].endswith(".py")]
    python = [i for i in run_items if i["evidence_test"].endswith(".py")]
    if rust:
        failed = earn_rust(root, rust)
        recorded += len(rust) - failed
        status = status or int(bool(failed))
    if python:
        print(f"scientific records: batch python ({len(python)})", flush=True)
        from evidence_earning import Snapshot, binding

        fp = Fingerprinter(root, scientific_build=True)
        snapshots = {i["selector"]: Snapshot(root, i, binding("scientific", i["selector"], None), fp) for i in python}
        withdraw_records(root, python)
        code, passed = run_batch(root, python)
        verified = []
        for item in python:
            if item["selector"] not in passed:
                continue
            snapshot = snapshots[item["selector"]]
            try:
                ref = snapshot.publish()
            except RuntimeError as error:
                print(f"FAIL: {item['selector']}: {error}")
                status = 1
                continue
            verified.append({**item, "dependency_manifest": ref, "fingerprint": snapshot.manifest["record"]["fingerprint"]})
        recorded += record_passes(root, verified)
        status = status or code
    print(f"scientific records: recorded {recorded} pass(es); {len(run_items) - recorded} not recorded")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument(
        "--check", action="store_true", help="exit 1 when any assertion is owed; executes nothing"
    )
    run = sub.add_parser("run")
    run.add_argument("--all", action="store_true", help="rerun and re-record every assertion (manual)")
    run.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PATTERN",
        help="limit to rows whose selector, test path or id contains PATTERN or matches it as a glob (repeatable)",
    )
    args = parser.parse_args()
    try:
        return {"status": cmd_status, "run": cmd_run}[args.command](REPO_ROOT, args)
    except UnresolvedInput as problem:
        # An input the fingerprint cannot account for owes its record; it is never silently current.
        print(f"FAIL: a scientific assertion has an undeclared input: {problem}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
