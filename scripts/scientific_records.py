#!/usr/bin/env python3
"""Recorded passes of the scientific assertions, carried forward while nothing they depend on changed.

    scientific_records.py status [--check]   owed versus current records
    scientific_records.py run [--all]        run the owed assertions, record passes
    scientific_records.py capture [--all]    run Rust assertions under coverage, record the
                                             source each executes

`parity/scientific_records.toml` holds one row per scientific assertion that
passed: its exact selector, roles, the commit it passed at, a fingerprint of
everything the assertion depends on, and when. A row is current while the
assertion's fingerprint is unchanged, so a no-change rerun executes nothing and
a leaf-crate edit owes only the assertions whose dependency closure contains
that crate. `--all` or ANTECEDENT_SCIENTIFIC_ALL=1 (a release candidate) runs
and re-records everything.

`run` executes the owed Rust assertions in one `cargo nextest run` (optimized
test profile, as scripts/test.sh) and the owed Python assertions in one pytest
batch, through scripts/run_evidence_rows.py, then records the passes. The
default engineering lanes exclude these assertions, so nothing else runs them.

A recorded pass is evidence for support only. It is not a calibration
measurement and no coverage or support claim reads it as one.

A Rust assertion that has been captured is fingerprinted over the source files it
executed (from `cargo llvm-cov`, which needs the `llvm-tools-preview` rustup component), not
over its crate-dependency closure: the row stores `executed` and `capture_test`, the digest of
its test-side files at capture time. An edit to a crate the test never reaches leaves the row
current. A row without a capture, or whose test files changed since it, uses the closure
fingerprint until `capture` is run again. Python rows always use the closure.

The fingerprint is `scripts/evidence_fingerprint.py`: one implementation shared with the
calibration flow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from _repo import REPO_ROOT, cargo_target_feature_args, read_text
from evidence_fingerprint import Fingerprinter, locate

RECORDS = "parity/scientific_records.toml"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

HEADER = """# Recorded passes of the scientific assertions (see scripts/scientific_records.py).
#
# One [[record]] per scientific assertion that passed, written by
# `python3 scripts/scientific_records.py run`, the one place scientific assertions
# run. A record is current while its fingerprint matches; an owed
# assertion reruns, a current one carries forward. A recorded pass is evidence for
# support only. It is never a calibration measurement or a coverage claim.
#
# test         exact selector: rust `crate::binary::module::name` (`lib` for the
#              library target) or python node id relative to python/
# roles        evidence roles of the citing claims
# passed       always true; failures are reported, never recorded
# commit       HEAD when the pass was recorded
# dirty        whether the working tree differed from that commit
# fingerprint  sha256 over the assertion's dependency closure, or over the source files
#              it executed when `executed` is present
# executed     source files the assertion ran under coverage (written by `capture`)
# capture_test digest of the test-side files when `executed` was captured; a different
#              digest voids the capture
# recorded_at  UTC time of the pass
"""


def load_records(root: Path) -> dict[str, dict]:
    path = root / RECORDS
    if not path.is_file():
        return {}
    data = tomllib.loads(read_text(path))
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
            + (
                f"capture_test = {json.dumps(r['capture_test'])}\n"
                f"executed = {json.dumps(list(r['executed']))}\n"
                if r.get("executed") is not None and r.get("capture_test")
                else ""
            )
        )
    (root / RECORDS).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORDS).write_text("".join(out), encoding="utf-8", newline="\n")


def git_state(root: Path) -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    return head.stdout.strip() or "unknown", bool(dirty.stdout.strip())


def is_current(records: dict[str, dict], selector: str, fingerprint: str) -> bool:
    rec = records.get(selector)
    return bool(rec and rec.get("passed") is True and rec.get("fingerprint") == fingerprint)


def valid_capture(rec: dict | None, fp: Fingerprinter, row: dict) -> list[str] | None:
    """The executed-source list of a record whose test files are unchanged since its capture."""
    if not rec or rec.get("executed") is None or not rec.get("capture_test"):
        return None
    if fp.test_digest(row) != rec["capture_test"]:
        return None
    return list(rec["executed"])


def classify(root: Path, rows: list[dict], fp: Fingerprinter) -> tuple[list[dict], list[dict]]:
    """(owed, current) rows, each annotated with selector and fingerprint."""
    records = load_records(root)
    owed, current = [], []
    for row in rows:
        info = locate(root, row)
        executed = valid_capture(records.get(info["selector"]), fp, row)
        item = {
            **row,
            "selector": info["selector"],
            "fingerprint": fp.fingerprint(row, executed),
            "executed": executed,
            "capture_test": fp.test_digest(row) if executed is not None else None,
        }
        (current if is_current(records, item["selector"], item["fingerprint"]) else owed).append(item)
    return owed, current


def record_passes(root: Path, items: list[dict]) -> int:
    """Merge passing rows (annotated by classify) into the ledger."""
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
            **(
                {"executed": item["executed"], "capture_test": item["capture_test"]}
                if item.get("executed") is not None and item.get("capture_test")
                else {}
            ),
        }
    write_records(root, records)
    return len(items)


def all_scientific_rows(root: Path) -> list[dict]:
    import evidence_audit

    return evidence_audit.scientific_rows(
        evidence_audit.build_report(root)["claims"], evidence_audit.registered_scientific_tests(root)
    )


# --------------------------------------------------------------------------
# Execution capture
# --------------------------------------------------------------------------

PASSED_ONE = re.compile(r"test result: ok\. 1 passed")
NOT_SOURCE = re.compile(r"(^|/)(tests|benches|examples|target)/|/\.cargo/|/rustc/")


def executed_files(export: dict, root: Path) -> list[str]:
    """Repo-relative source files with at least one covered line in an llvm-cov export."""
    root = root.resolve()
    found: set[str] = set()
    for block in export.get("data", []):
        for entry in block.get("files", []):
            if entry.get("summary", {}).get("lines", {}).get("covered", 0) <= 0:
                continue
            try:
                rel = Path(entry["filename"]).resolve().relative_to(root).as_posix()
            except (ValueError, KeyError, OSError):
                continue
            if rel.endswith(".rs") and not NOT_SOURCE.search(rel):
                found.add(rel)
    return sorted(found)


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
    features = cargo_target_feature_args(root / "crates" / crate / "Cargo.toml", list(target))
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


def capture_one(root: Path, binary: str, info: dict, env: dict[str, str], run=subprocess.run) -> list[str] | None:
    """Run one Rust assertion's instrumented binary; its executed source files, or `None` on failure."""
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
            [llvm_tool("llvm-cov"), "export", "-summary-only", f"-instr-profile={data}", binary],
            text=True,
            capture_output=True,
        )
        if export.returncode != 0:
            print(f"FAIL: {info['selector']}: llvm-cov export failed\n{export.stderr[-400:]}")
            return None
        return executed_files(json.loads(export.stdout), root)


def cmd_capture(root: Path, args: argparse.Namespace, run=subprocess.run) -> int:
    """Build each test target once with coverage instrumentation, then profile each assertion.

    Records are written after every target, so an interrupted capture keeps what it finished.
    """
    fp = Fingerprinter(root)
    records = load_records(root)
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[dict, dict]]] = {}
    for row in all_scientific_rows(root):
        info = locate(root, row)
        if info["lang"] != "rust":
            continue
        if force_all(args) or valid_capture(records.get(info["selector"]), fp, row) is None:
            groups.setdefault((info["crate"], tuple(info["target"])), []).append((row, info))
    total = sum(len(v) for v in groups.values())
    print(f"scientific records: capturing {total} Rust assertion(s) in {len(groups)} target(s)", flush=True)
    env = capture_env(root)
    recorded, failed = 0, 0
    for (crate, target), todo in sorted(groups.items()):
        built = run(build_command(root, crate, target), cwd=root, env=env, text=True, capture_output=True)
        binary = built_executable(built.stdout or "") if built.returncode == 0 else None
        if binary is None:
            print(f"FAIL: {crate} {' '.join(target)}: instrumented build failed\n{(built.stderr or '')[-400:]}")
            failed += len(todo)
            continue
        items = []
        for row, info in todo:
            executed = capture_one(root, binary, info, env, run)
            if executed is None:
                failed += 1
                continue
            items.append(
                {
                    **row,
                    "selector": info["selector"],
                    "executed": executed,
                    "capture_test": fp.test_digest(row),
                    "fingerprint": fp.fingerprint(row, executed),
                }
            )
        recorded += record_passes(root, items)
        print(f"  {crate} {' '.join(target)}: {len(items)}/{len(todo)} captured", flush=True)
    print(f"scientific records: captured {recorded}; {failed} failed")
    return int(bool(failed))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def force_all(args: argparse.Namespace | None = None) -> bool:
    return bool(getattr(args, "all", False)) or os.environ.get("ANTECEDENT_SCIENTIFIC_ALL") == "1"


def cmd_status(root: Path, args: argparse.Namespace) -> int:
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows, Fingerprinter(root))
    records = load_records(root)
    captured = sum(1 for i in rows if records.get(locate(root, i)["selector"], {}).get("executed") is not None)
    print(
        f"scientific records: {len(current)} current, {len(owed)} owed of {len(rows)} "
        f"({captured} with an execution capture)"
    )
    for item in owed:
        print(f"  owed: {item['selector']}")
    return int(args.check and bool(owed))


def python_environment_ok(root: Path) -> bool:
    for candidate in (root / "python/.venv/bin/python", root / "python/.venv/Scripts/python.exe"):
        if candidate.is_file():
            return subprocess.run([str(candidate), str(root / "scripts/native_build_identity.py"), "check"]).returncode == 0
    print("FAIL: the Python conformance environment is missing; run bash scripts/test.sh python")
    return False


def cmd_run(root: Path, args: argparse.Namespace) -> int:
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows, Fingerprinter(root))
    run_items = (owed + current) if force_all(args) else owed
    print(
        f"scientific records: running {len(run_items)} of {len(rows)} "
        f"({len(current) if not force_all(args) else 0} carried forward)",
        flush=True,
    )
    if not run_items:
        return 0
    if any(i["evidence_test"].endswith(".py") for i in run_items) and not python_environment_ok(root):
        return 1
    with tempfile.TemporaryDirectory(prefix="antecedent-scientific-") as tmp:
        ledger = Path(tmp) / "selection.toml"
        passed_out = Path(tmp) / "passed.json"
        ledger.write_text(
            "".join(
                "[[capabilities]]\n"
                f"id = {json.dumps(i['selector'])}\n"
                f"evidence_test = {json.dumps(i['evidence_test'])}\n"
                f"evidence_assertion = {json.dumps(i['evidence_assertion'])}\n\n"
                for i in run_items
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
    recorded = record_passes(root, [i for i in run_items if i["selector"] in passed])
    print(f"scientific records: recorded {recorded} pass(es); {len(run_items) - recorded} not recorded")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--check", action="store_true", help="exit 1 when any assertion is owed")
    run = sub.add_parser("run")
    run.add_argument("--all", action="store_true", help="rerun and re-record every assertion (release candidate)")
    capture = sub.add_parser("capture")
    capture.add_argument("--all", action="store_true", help="re-capture every Rust assertion")
    args = parser.parse_args()
    return {"status": cmd_status, "run": cmd_run, "capture": cmd_capture}[args.command](REPO_ROOT, args)


if __name__ == "__main__":
    raise SystemExit(main())
