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
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from _repo import REPO_ROOT, load_toml, scientific_target_feature_args
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

def record_passes(root: Path, items: list[dict], *, source_state: tuple[str, bool] | None = None) -> int:
    """Merge passing rows, each with the dependency manifest it earned, into the ledger."""
    if not items:
        return 0
    commit, dirty = source_state if source_state is not None else git_state(root)
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


class SelectorError(ValueError):
    """A scientific row does not name exactly one test: an infrastructure error, not a failure."""


def exact_test_name(root: Path, row: dict, info: dict) -> str:
    """The libtest name of a Rust row, including inline modules around its test function.

    The selector carries the file's module path; a test inside `mod tests { ... }` (or any
    inline module) is named through those modules too. Resolved from source before any
    build, so a row that names no unique test stops the run immediately."""
    import test_evidence as te

    base = info["selector"].split("::", 2)[2]
    prefix, _, function = base.rpartition("::")
    masked = te.source_masked(root / row["evidence_test"])
    found = [m.start() for m in re.finditer(rf"\bfn\s+{re.escape(function)}\s*[<(]", masked.skeleton)]
    if len(found) != 1:
        raise SelectorError(f"{info['selector']}: {len(found)} functions named {function} in {row['evidence_test']}")
    inner = [block.name for block in sorted(te.module_items(masked), key=lambda block: block.body[0])
             if block.body[0] < found[0] < block.body[1]]
    return "::".join([part for part in (prefix, *inner) if part] + [function])


def capture_one(root: Path, binary: str, info: dict, env: dict[str, str], run=subprocess.run,
                *, llvm_tools: dict[str, str] | None = None) -> dict[str, set[int]] | None:
    """Run one Rust assertion's instrumented binary; its line coverage, or `None` on failure.

    Running no test at all is an infrastructure error and raises: a wrong name or a missing
    test must stop the run, not be recorded as one more failure."""
    libtest = info.get("libtest") or info["selector"].split("::", 2)[2]
    with tempfile.TemporaryDirectory(prefix="antecedent-capture-") as tmp:
        raw, data = Path(tmp) / "run.profraw", Path(tmp) / "run.profdata"
        done = run(
            [binary, "--exact", libtest, "--include-ignored", "--test-threads", "1"],
            cwd=root / "crates" / info["crate"],
            env={**env, "LLVM_PROFILE_FILE": str(raw)},
            text=True,
            capture_output=True,
        )
        output = (done.stdout or "") + (done.stderr or "")
        if re.search(r"^running 0 tests$", output, re.M):
            raise SelectorError(f"{info['selector']}: the test binary has no test named {libtest}")
        if done.returncode != 0 or not PASSED_ONE.search(output):
            print(f"FAIL: {info['selector']}: not captured\n{((done.stdout or '') + (done.stderr or ''))[-400:]}")
            return None
        tool = (lambda name: llvm_tools[name]) if llvm_tools is not None else llvm_tool
        merged = run([tool("llvm-profdata"), "merge", "-sparse", str(raw), "-o", str(data)], text=True, capture_output=True)
        if merged.returncode != 0:
            print(f"FAIL: {info['selector']}: profile merge failed\n{merged.stderr[-400:]}")
            return None
        export = run(
            [tool("llvm-cov"), "export", f"-instr-profile={data}", binary],
            text=True,
            capture_output=True,
        )
        if export.returncode != 0:
            print(f"FAIL: {info['selector']}: llvm-cov export failed\n{export.stderr[-400:]}")
            return None
        data = json.loads(export.stdout)
        return line_coverage(data)


def earn_rust(root: Path, items: list[dict], run=subprocess.run, *, backend: str = "cargo",
              check_reproducible: bool = False) -> int:
    """Earn the given Rust rows: build each test target once with coverage instrumentation
    (through the compiler inventory driver when `rustc-dev` is installed), run each assertion
    exactly, and publish its manifest. The ledger is written after every target, so an
    interrupted run keeps what it finished. Returns the number of failures."""
    if backend == "nix":
        return earn_rust_nix(root, items, run, check_reproducible=check_reproducible)
    if backend != "cargo":
        raise ValueError("unknown Rust evidence build backend")
    from evidence_earning import require_compile_environment
    require_compile_environment(root)
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[dict, dict]]] = {}
    for row in items:
        info = locate(root, row)
        groups.setdefault((info["crate"], tuple(info["target"])), []).append((row, info))
    print(f"scientific records: earning {len(items)} Rust assertion(s) in {len(groups)} target(s)", flush=True)
    if not groups:
        return 0
    from evidence_earning import (
        Snapshot,
        binding,
        build_analyzer,
        compile_time_code,
        project_snapshot,
    )

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


def _remember_nix_output(root: Path, allowed: dict[str, str], relative: str) -> None:
    """Remember only this run's exact output bytes for subsequent source checks."""
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"evidence output is missing or linked: {relative}")
    allowed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()


def earn_rust_nix(root: Path, items: list[dict], run=subprocess.run, *,
                  scientific_targets: list[tuple[str, str, str]] | None = None,
                  check_reproducible: bool = False) -> int:
    """Build every owed target in one audited Nix derivation, then capture and record each."""
    from evidence_earning import Snapshot, binding, project_snapshot, require_compile_environment
    from nix_artifact_identity import target_binary
    from nix_evidence_build import build_committed_targets
    from nix_evidence_capture import capture_built_target
    from nix_evidence_policy import validate_build
    from nix_evidence_source import _clean_head

    root = root.resolve()
    require_compile_environment(root)
    allowed: dict[str, str] = {}
    _clean_head(root, allowed)  # Run outputs may be allowed only after starting clean.
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[dict, dict]]] = {}
    unresolved = []
    for row in items:
        info = locate(root, row)
        if info["lang"] != "rust":
            raise ValueError("Nix Rust earning requires Rust assertions")
        try:
            info["libtest"] = exact_test_name(root, row, info)
        except SelectorError as error:
            unresolved.append(str(error))
        groups.setdefault((info["crate"], tuple(info["target"])), []).append((row, info))
    if unresolved:
        # Before any build: a row that names no unique test cannot be earned.
        raise SelectorError("scientific rows name no unique test:\n  " + "\n  ".join(unresolved))
    if not groups:
        return 0
    # Before the full build: a disposable crate built through the same recipe must run
    # on the host, find its test by its exact name and read a committed file through
    # CARGO_MANIFEST_DIR. A boundary defect then stops the run in seconds, not after
    # every target has been compiled.
    from nix_evidence_engineering_fixture import run_fixture
    print("scientific records: boundary preflight", flush=True)
    run_fixture()
    fp = Fingerprinter(root, scientific_build=True)
    failed = recorded = 0

    def cargo_target(crate: str, target: tuple[str, ...]) -> tuple[str, str, str]:
        if target == ("--lib",):
            return crate, "lib", ""
        if len(target) == 2 and target[0] == "--test":
            return crate, "test", target[1]
        raise ValueError("unsupported Cargo target")

    _clean_head(root, allowed)
    print(f"scientific records: building {len(groups)} target(s) in one Nix derivation", flush=True)
    if scientific_targets is None:
        # The cached dependency build covers every scientific target, owed or not.
        located = (locate(root, row) for row in all_scientific_rows(root))
        scientific_targets = sorted({cargo_target(i["crate"], tuple(i["target"])) for i in located})
    builds = build_committed_targets(root, [cargo_target(c, t) for c, t in groups],
                                     deps_targets=scientific_targets, check_reproducible=check_reproducible,
                                     allowed_outputs=allowed)
    for (crate, target), todo in sorted(groups.items()):
        _clean_head(root, allowed)
        built = builds[cargo_target(crate, target)]
        # Audit and builder policy must be checked before withdrawing or running.
        validate_build(root, built)
        snapshots = {
            info["selector"]: Snapshot(root, row, binding("scientific", info["selector"], None), fp,
                                       nix_build=built, allowed_outputs=allowed)
            for row, info in todo
        }
        _clean_head(root, allowed)
        withdraw_records(root, [info for _, info in todo])
        ledger = "parity/scientific_records.toml"
        if (root / ledger).is_file():
            _remember_nix_output(root, allowed, ledger)
        result = capture_built_target(root, [row for row, _ in todo], built, run,
                                      continue_on_failure=True, allowed_outputs=allowed)
        failed += len(result["failures"])
        earned = []
        for row, info in todo:
            selector = info["selector"]
            if selector not in result["coverage"]:
                continue
            _clean_head(root, allowed)
            try:
                snapshots[selector].mark_nix_executed(str(target_binary(built["artifact"])))
                project_snapshot(snapshots[selector], Path(built["artifact"]["output"]) / "compiler-inventory",
                                 result["coverage"][selector], source_root=Path(built["source"]["store_path"]))
                ref = snapshots[selector].publish()
            except RuntimeError as error:
                print(f"FAIL: {selector}: {error}")
                failed += 1
                continue
            manifest = f"parity/evidence_manifests/{ref}.json"
            _remember_nix_output(root, allowed, manifest)
            # Manifest filenames are content addresses, not merely paths to trust.
            if allowed[manifest] != ref:
                raise RuntimeError("published evidence manifest does not match its content address")
            earned.append({**row, "selector": selector, "dependency_manifest": ref,
                           "fingerprint": snapshots[selector].manifest["record"]["fingerprint"]})
        source_commit, _ = _clean_head(root, allowed)
        if source_commit != built["source"]["commit"]:
            raise RuntimeError("checkout commit differs from Nix build source")
        recorded += record_passes(root, earned, source_state=(source_commit, False))
        if (root / ledger).is_file():
            _remember_nix_output(root, allowed, ledger)
        _clean_head(root, allowed)
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
    # Every row must name exactly one test, so a run never builds for a name it cannot find.
    unresolved = []
    for row in rows:
        try:
            exact_test_name(root, row, locate(root, row))
        except SelectorError as error:
            unresolved.append(str(error))
    for problem in unresolved:
        print(f"FAIL: {problem}")
    return int(bool(unresolved) or (args.check and bool(owed)))


def matches_only(item: dict, patterns: list[str]) -> bool:
    """Does a row match any `--only` pattern (substring or glob) on its selector, test path or id?"""
    if not patterns:
        return True
    fields = [str(item.get(k) or "") for k in ("selector", "evidence_test", "id")]
    return any(
        pattern in field or fnmatch.fnmatchcase(field, pattern) for pattern in patterns for field in fields
    )


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
    # Each target is recorded as soon as it finishes, so an interrupted or partly failing
    # run keeps every pass it earned. Scientific assertions are Rust tests (evidence_audit).
    failed = earn_rust(root, run_items, backend=getattr(args, "rust_build_backend", "nix"),
                       check_reproducible=bool(getattr(args, "check_reproducible", False)))
    recorded, status = len(run_items) - failed, int(bool(failed))
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
    run.add_argument("--rust-build-backend", choices=("cargo", "nix"), default="nix",
                     help="Rust build backend: nix (default) builds each target from committed HEAD under the "
                          "audited sandbox policy and projects unexecuted bodies; cargo builds on the host "
                          "and keeps complete source")
    run.add_argument("--check-reproducible", action="store_true",
                     help="also rebuild the Nix evidence derivation and require identical output, once per "
                          "recipe and lock")
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
