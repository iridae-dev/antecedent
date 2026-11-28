#!/usr/bin/env python3
"""Recorded passes of the scientific assertions, current while their dependency manifests match.

    scientific_records.py status [--check]   flagged versus current records; --check exits 1 when any
                                             is flagged and executes nothing (CI and releases)
    scientific_records.py run [--all] [--carry] [--only PATTERN]
                                             earn the flagged assertions and record their passes
    scientific_records.py justify --only PATTERN --reason TEXT [--author NAME]
                                             carry flagged passes forward by a written judgment

`parity/scientific_records.toml` holds one row per assertion that passed. Its
`dependency_manifest` is `pass:scientific:<selector>` (`scripts/claim_passes.py`): one file,
updated in place. `status` checks that file with `scripts/evidence_manifests.py` only: no
compiler, Cargo or test runs. A row without a pass, or whose pass no longer matches, is
flagged: it needs a new run or a written justification. The recorded pass stays in the file.

`run` earns flagged Rust assertions: every flagged test target is built once with coverage
instrumentation in one Nix derivation, and each assertion then runs exactly, the assertions
of all targets concurrently. Scientific assertions are Rust tests. A rerun withdraws the old pass first, so a failed or interrupted run never leaves a
superseded pass standing. `--all` or ANTECEDENT_SCIENTIFIC_ALL=1 reruns everything.

A flagged pass is carried forward by the cheapest sufficient step (ADR 0031). A facade
assertion runs with its results traced (`evidence-trace`), and its pass records their
`output_digest`. When a rerun reproduces the recorded digest, the new pass extends the old
record's `carry` chain with an `output_equivalent` entry; any other pass starts a new
chain. `run --carry` reruns only flagged rows with a recorded digest. `justify` appends a
`judgment` from the last carried fingerprint to the current one; the row is current while
its fingerprint stays there. `status` reports chain lengths and every judged record.

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
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from _repo import REPO_ROOT, load_toml
from evidence_fingerprint import Fingerprinter, UnresolvedInput, locate

RECORDS = "parity/scientific_records.toml"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

HEADER = """# Recorded passes of the scientific assertions (see scripts/scientific_records.py).
#
# One [[record]] per scientific assertion that passed, written by
# `python3 scripts/scientific_records.py run`, the one place scientific assertions
# run. A record is current while its pass matches; a flagged assertion is rerun or
# justified in that same file, and a current one carries forward. A recorded pass is evidence for
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
# dependency_manifest  pass:scientific:<selector>, the standing pass file
# output_digest  digest of the results the assertion's test was handed (facade tests only)
# carry        how the pass was carried forward since it was earned, oldest first: kind
#              output_equivalent (a rerun reproduced output_digest) or judgment
#              (`scientific_records.py justify`), from_fingerprint, to_fingerprint,
#              reason, author
"""


CARRY_FIELDS = ("kind", "from_fingerprint", "to_fingerprint", "reason", "author")


def render_carry(entry: dict) -> str:
    return "{ " + ", ".join(f"{field} = {json.dumps(entry[field])}" for field in CARRY_FIELDS) + " }"


def carry_entry(kind: str, old: str, new: str, reason: str, author: str) -> dict:
    return dict(zip(CARRY_FIELDS, (kind, old, new, reason, author), strict=True))


def carried_from(record: dict) -> str:
    """The fingerprint a record's pass was last carried to (its own when never carried)."""
    chain = record.get("carry") or []
    return chain[-1]["to_fingerprint"] if chain else record["fingerprint"]


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
            + (f"output_digest = {json.dumps(r['output_digest'])}\n" if r.get("output_digest") else "")
            + ("carry = [\n" + "".join(f"  {render_carry(entry)},\n" for entry in r["carry"]) + "]\n"
               if r.get("carry") else "")
        )
    (root / RECORDS).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORDS).write_text("".join(out), encoding="utf-8", newline="\n")


def git_state(root: Path) -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    return head.stdout.strip() or "unknown", bool(dirty.stdout.strip())


def classify(root: Path, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(owed, current) rows, each annotated with its selector. A row is current only when its
    recorded pass has a dependency manifest whose inputs and declarations still match, or a
    judgment carried it to the row's current fingerprint."""
    from evidence_earning import binding, declarations
    from evidence_manifests import Checker

    records = load_records(root)
    checker = fp = None
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
        chain = (rec or {}).get("carry") or []
        if chain and chain[-1]["kind"] == "judgment":
            fp = fp or current_fingerprinter(root)
            if fp.fingerprint(row) == chain[-1]["to_fingerprint"]:
                item.update(fingerprint=rec.get("fingerprint"), manifest_reasons=[*item["manifest_reasons"], "justified"])
                current.append(item)
                continue
        owed.append(item)
    return owed, current


def current_fingerprinter(root: Path) -> Fingerprinter:
    """The fingerprints judgments are made against: the workspace as the scientific build sees it."""
    return Fingerprinter(root, scientific_build=True)


def carry_chain(old: dict | None, item: dict) -> list[dict]:
    """A new pass's chain: the old one extended when the rerun reproduced its output, else none."""
    digest = item.get("output_digest")
    if not old or not digest or old.get("output_digest") != digest:
        return []
    return [*(old.get("carry") or []), carry_entry("output_equivalent", old["fingerprint"], item["fingerprint"],
                                                   "output digest unchanged", "scientific_records.py")]


def record_passes(root: Path, items: list[dict], *, source_state: tuple[str, bool] | None = None,
                  previous: dict[str, dict] | None = None) -> int:
    """Merge passing rows, each with the dependency manifest it earned, into the ledger.
    `previous` holds the records as they stood before the run withdrew them."""
    if not items:
        return 0
    commit, dirty = source_state if source_state is not None else git_state(root)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = load_records(root)  # reread: another gate may have written meanwhile
    for item in items:
        old = (previous or {}).get(item["selector"]) or records.get(item["selector"])
        record = {
            "roles": item.get("roles") or ["scientific"],
            "commit": commit,
            "dirty": dirty,
            "fingerprint": item["fingerprint"],
            "recorded_at": now,
            "dependency_manifest": item["dependency_manifest"],
        }
        if item.get("output_digest"):
            record["output_digest"] = item["output_digest"]
        if chain := carry_chain(old, item):
            record["carry"] = chain
        records[item["selector"]] = record
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
    """The scientific assertions: the same set the engineering filter leaves out."""
    import evidence_audit

    return evidence_audit.all_scientific_rows(root)


# --------------------------------------------------------------------------
# Execution capture
# --------------------------------------------------------------------------

PASSED_ONE = re.compile(r"test result: ok\. 1 passed")

from evidence_capture import (  # noqa: E402  (shared with calibration)
    region_coverage,  # noqa: F401  (re-exported)
)


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
    # The module path is where the file is declared, which a `#[path]` attribute can make
    # differ from where it sits; the selector keeps the file-location form as its identity.
    target = te.target_root(root / row["evidence_test"])
    if target is not None:
        prefix = te.module_path(root / row["evidence_test"], target[0]).rstrip(":")
    inner = [block.name for block in sorted(te.module_items(masked), key=lambda block: block.body[0])
             if block.body[0] < found[0] < block.body[1]]
    return "::".join([part for part in (prefix, *inner) if part] + [function])


def rust_execution_info(root: Path, row: dict) -> dict:
    """Resolve the exact executable test name separately from its stable evidence selector."""
    info = locate(root, row)
    if info["lang"] != "rust":
        raise ValueError("Rust earning requires Rust assertions")
    info["libtest"] = exact_test_name(root, row, info)
    return info


FACADE = "antecedent"
TRACES = Path("target") / "scientific-traces"


def trace_file(root: Path, selector: str) -> Path:
    """Where one assertion's run writes its evidence trace."""
    return root / TRACES / f"{hashlib.sha256(selector.encode()).hexdigest()[:16]}.tsv"


def trace_digest(text: str, test: str) -> str | None:
    """Digest of the results handed to libtest test `test`, in any order; `None` when it was
    handed none, so its pass cannot be carried by output equivalence."""
    found = sorted(fields[3] for line in text.splitlines()
                   if len(fields := line.split("\t")) == 4 and fields[0] == "output" and fields[2] == test)
    return hashlib.sha256("\n".join(found).encode()).hexdigest()[:16] if found else None


def output_digest(root: Path, info: dict) -> str | None:
    """The output digest of an assertion that just passed (`capture_one`)."""
    path = trace_file(root, info["selector"])
    if info.get("crate") != FACADE or not path.is_file():
        return None
    return trace_digest(path.read_text(encoding="utf-8"), info.get("libtest") or info["selector"].split("::", 2)[2])


def capture_one(root: Path, binary: str, info: dict, env: dict[str, str], run=subprocess.run,
                *, llvm_tools: dict[str, str]) -> dict[str, set[int]] | None:
    """Run one Rust assertion's instrumented binary; its executed regions, or `None` on failure.

    Running no test at all is an infrastructure error and raises: a wrong name or a missing
    test must stop the run, not be recorded as one more failure. A facade assertion writes
    a fresh evidence trace (`output_digest`). The tools are the pinned `llvm-profdata` and
    `llvm-cov` from the Nix build that produced `binary`."""
    from evidence_capture import instrumented_run
    libtest = info.get("libtest") or info["selector"].split("::", 2)[2]
    if info.get("crate") == FACADE:
        trace = trace_file(root, info["selector"])
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.unlink(missing_ok=True)
        env = {**env, "ANTECEDENT_EVIDENCE_TRACE": str(trace)}
    tools = llvm_tools
    status, output, coverage, problem = instrumented_run(
        binary, ["--exact", libtest, "--include-ignored", "--test-threads", "1"],
        cwd=root / "crates" / info["crate"], env=env, tools=tools, run=run)
    if re.search(r"^running 0 tests$", output, re.M):
        raise SelectorError(f"{info['selector']}: the test binary has no test named {libtest}")
    if status != 0 or not PASSED_ONE.search(output):
        print(f"FAIL: {info['selector']}: not captured\n{output[-400:]}")
        return None
    if coverage is None:
        print(f"FAIL: {info['selector']}: {problem}")
        return None
    return coverage


def earn_rust(root: Path, items: list[dict], run=subprocess.run, *,
              check_reproducible: bool = False, previous: dict[str, dict] | None = None) -> int:
    """Earn the given Rust rows from committed HEAD.

    One Nix derivation builds the owed targets. Each assertion then runs exactly.
    The ledger is written as each target finishes, so an interrupted run keeps what
    it finished. Returns the number of failures. `previous` (default: the ledger now)
    holds the records whose carry chains the new passes extend."""
    root = root.resolve()
    from evidence_earning import require_compile_environment
    require_compile_environment(root)
    return earn_rust_nix(root, items, run, check_reproducible=check_reproducible, previous=previous)


def _remember_nix_output(root: Path, allowed: dict[str, str], relative: str) -> None:
    """Remember only this run's exact output bytes for subsequent source checks."""
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"evidence output is missing or linked: {relative}")
    allowed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()


def earn_rust_nix(root: Path, items: list[dict], run=subprocess.run, *,
                  scientific_targets: list[tuple[str, str, str]] | None = None,
                  check_reproducible: bool = False, previous: dict[str, dict] | None = None) -> int:
    """Build every owed target in one audited Nix derivation, then capture and record each."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from evidence_earning import Snapshot, binding, publish_executed, require_compile_environment
    from evidence_platforms import earning_platform, validate_build
    from nix_evidence_build import build_committed_targets
    from nix_evidence_capture import CHECKOUT_LOCK, capture_built_target
    from nix_evidence_source import _clean_head

    root = root.resolve()
    previous = load_records(root) if previous is None else previous
    # Refuse before any work on a host that cannot earn; a platform that prepares
    # itself realises its build inputs here on first use. The sandbox rule is
    # checked here, before the boundary fixture or the derivation.
    from evidence_platforms import require_sandbox
    require_sandbox(earning_platform(root, prepare=True))
    require_compile_environment(root)
    allowed: dict[str, str] = {}
    _clean_head(root, allowed)  # Run outputs may be allowed only after starting clean.
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[dict, dict]]] = {}
    unresolved = []
    for row in items:
        try:
            info = rust_execution_info(root, row)
        except SelectorError as error:
            unresolved.append(str(error))
            continue
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
    ledger = "parity/scientific_records.toml"
    prepared = []
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
        prepared.append((crate, target, todo, built, snapshots))
    # Every build is validated and every snapshot taken before any pass is withdrawn,
    # and the ledger is rewritten once rather than once per target.
    _clean_head(root, allowed)
    withdraw_records(root, [info for _, _, todo, _, _ in prepared for _, info in todo])
    if (root / ledger).is_file():
        _remember_nix_output(root, allowed, ledger)
    # Longest first, by the last recorded wall times: targets by their slowest assertion,
    # assertions within a target by their own, so a long one never starts last.
    from evidence_capture import last_durations, longest_first
    from nix_evidence_capture import SCIENTIFIC_TIMINGS
    seconds = last_durations(root / SCIENTIFIC_TIMINGS)
    prepared = [(crate, target, longest_first(todo, lambda pair: seconds.get(pair[1]["selector"])), built, snapshots)
                for crate, target, todo, built, snapshots in prepared]
    prepared = longest_first(prepared, lambda item: max(
        (seconds[info["selector"]] for _, info in item[2] if info["selector"] in seconds), default=None))
    # Every target's tests run at once within the shared capture slots, so one slow
    # assertion no longer holds back the targets after it. Results are recorded here as
    # each target completes, under the lock the capture threads' checkout checks take.
    with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
        futures = {
            pool.submit(capture_built_target, root, [row for row, _ in todo], built, run,
                        continue_on_failure=True, allowed_outputs=allowed): (crate, target, todo, built, snapshots)
            for crate, target, todo, built, snapshots in prepared
        }
        for future in as_completed(futures):
            crate, target, todo, built, snapshots = futures[future]
            result = future.result()
            with CHECKOUT_LOCK:
                failed += len(result["failures"])
                earned = []
                for row, info in todo:
                    selector = info["selector"]
                    if selector not in result["coverage"]:
                        continue
                    _clean_head(root, allowed)
                    try:
                        ref = publish_executed(snapshots[selector], built, result["coverage"][selector])
                    except RuntimeError as error:
                        print(f"FAIL: {selector}: {error}")
                        failed += 1
                        continue
                    # The pass and the objects it names are this run's own outputs.
                    for output in snapshots[selector].outputs:
                        _remember_nix_output(root, allowed, output)
                    from claim_passes import pass_ref, read_stored_pass
                    from evidence_manifests import manifest_id
                    if manifest_id(read_stored_pass(root, pass_ref("scientific", selector))) != ref:
                        raise RuntimeError("published evidence pass does not match its content address")
                    earned.append({**row, "selector": selector,
                                   "dependency_manifest": pass_ref("scientific", selector),
                                   "fingerprint": snapshots[selector].manifest["record"]["fingerprint"],
                                   "output_digest": output_digest(root, info)})
                source_commit, _ = _clean_head(root, allowed)
                if source_commit != built["source"]["commit"]:
                    raise RuntimeError("checkout commit differs from Nix build source")
                recorded += record_passes(root, earned, source_state=(source_commit, False),
                                          previous=previous)
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
    return report_status(root, owed, current, check=args.check)


def report_status(root: Path, owed: list[dict], current: list[dict], *, check: bool) -> int:
    rows = owed + current
    print(f"scientific records: {len(current)} current, {len(owed)} flagged of {len(rows)}")
    for item in owed:
        print(f"  flagged: {item['selector']} ({', '.join(item['manifest_reasons'][:3])})")
    # Every row must name exactly one test, so a run never builds for a name it cannot find.
    unresolved = []
    for row in rows:
        try:
            exact_test_name(root, row, locate(root, row))
        except SelectorError as error:
            unresolved.append(str(error))
    for problem in unresolved:
        print(f"FAIL: {problem}")
    report_carry(load_records(root))
    # Evidence no record reaches is never committed: a run prunes it as it finishes.
    from evidence_manifests import unreachable_evidence
    orphans = unreachable_evidence(root)
    if orphans:
        print(f"FAIL: {len(orphans)} stored evidence file(s) no record reaches; "
              "run `python3 scripts/scientific_records.py prune`")
    return int(bool(unresolved) or (check and bool(owed or orphans)))


def report_carry(records: dict[str, dict]) -> None:
    """Carried passes by chain length, and every record a judgment carries."""
    lengths: dict[int, int] = {}
    for record in records.values():
        if record.get("carry"):
            lengths[len(record["carry"])] = lengths.get(len(record["carry"]), 0) + 1
    if not lengths:
        return
    print("scientific records: carried passes by chain length: "
          + ", ".join(f"{length}: {lengths[length]}" for length in sorted(lengths)))
    for selector in sorted(records):
        chain = records[selector].get("carry") or []
        if any(entry["kind"] == "judgment" for entry in chain):
            print(f"  judged: {selector} (chain {len(chain)})")


def cmd_justify(root: Path, args: argparse.Namespace) -> int:
    """Carry flagged passes forward by a written judgment, to their current fingerprints."""
    owed, _ = classify(root, all_scientific_rows(root))
    selected = [item for item in owed if matches_only(item, args.only)]
    if not selected:
        print(f"FAIL: --only {args.only!r} matches no flagged scientific assertion")
        return 2
    records = load_records(root)
    unrecorded = [item["selector"] for item in selected if item["selector"] not in records]
    if unrecorded:
        print("FAIL: no recorded pass to carry forward; run these instead:\n  " + "\n  ".join(unrecorded))
        return 1
    fp = current_fingerprinter(root)
    for item in selected:
        record = records[item["selector"]]
        row = {k: v for k, v in item.items() if k not in ("selector", "fingerprint", "manifest_reasons")}
        entry = carry_entry("judgment", carried_from(record), fp.fingerprint(row), args.reason, args.author)
        record["carry"] = [*(record.get("carry") or []), entry]
        print(f"  justified: {item['selector']} (chain {len(record['carry'])})")
    write_records(root, records)
    print(f"scientific records: justified {len(selected)} pass(es)")
    return 0


def cmd_refine(root: Path, args: argparse.Namespace) -> int:
    """Narrow earned records' inputs by the current precision rules, running nothing
    (`evidence_refine`), then prune what no record reaches."""
    from evidence_manifests import Checker
    from evidence_refine import refine_records
    from nix_evidence_source import _clean_head
    _clean_head(root)
    rows = {locate(root, row)["selector"]: row for row in all_scientific_rows(root)}
    records = load_records(root)
    checker = Checker(root)
    work = []
    for selector, record in records.items():
        ref = record.get("dependency_manifest")
        if ref and selector in rows:
            work.append((record, checker.load(ref), rows[selector]))
    # Calibration measurements and probe attestations are bound by the same rules, from the
    # row of the measurement's test. A manifest is content-addressed and named nowhere but
    # its record, so the reference is rewritten in place.
    import _repo
    from calibration_status import load_attestations
    from evidence_fingerprint import coverage_row
    coverage = {str(r.get("id")): r for r in _repo.load_toml(root / "parity/coverage_records.toml").get("record", [])}
    stores = {"parity/coverage_records.toml": list(coverage.values()),
              "parity/calibration_probe_attestations.toml":
                  [r for r in load_attestations(root / "parity/calibration_probe_attestations.toml")
                   if str(r.get("record_id")) in coverage]}
    for rows_ in stores.values():
        for row in rows_:
            measured = coverage[str(row.get("record_id", row.get("id")))]
            if row.get("dependency_manifest") and measured.get("test"):
                work.append((row, checker.load(row["dependency_manifest"]),
                             coverage_row(measured["test"], measured.get("dgp", ""))))
    moved = refine_records(root, work)
    if moved:
        for record in records.values():
            if record.get("dependency_manifest") in moved:
                record["dependency_manifest"] = moved[record["dependency_manifest"]]
        write_records(root, records)
        for rel in stores:
            path = root / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for old, new in moved.items():
                text = text.replace(f'dependency_manifest = "{old}"', f'dependency_manifest = "{new}"')
            path.write_text(text, encoding="utf-8")
    print(f"evidence records: refined {len(moved)} manifest(s) without running anything")
    return cmd_prune(root, args)


def cmd_prune(root: Path, args: argparse.Namespace) -> int:
    from evidence_manifests import prune_evidence
    removed = prune_evidence(root)
    print(f"scientific records: pruned {len(removed)} stored evidence file(s) no record reaches")
    return 0


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
    carry = bool(getattr(args, "carry", False))
    candidates = (owed + current) if force_all(args) and not carry else owed
    previous = load_records(root)
    run_items = [i for i in candidates if matches_only(i, only)
                 and (not carry or previous.get(i["selector"], {}).get("output_digest"))]
    print(
        f"scientific records: running {len(run_items)} of {len(rows)} "
        f"({len(current) if not force_all(args) or carry else 0} carried forward"
        + (f", filtered by --only {only}" if only else "")
        + (", carry candidates only" if carry else "")
        + ")",
        flush=True,
    )
    if not run_items:
        return 0
    # Each target is recorded as soon as it finishes, so an interrupted or partly failing
    # run keeps every pass it earned. Scientific assertions are Rust tests (evidence_audit).
    failed = earn_rust(root, run_items,
                       check_reproducible=bool(getattr(args, "check_reproducible", False)), previous=previous)
    recorded, status = len(run_items) - failed, int(bool(failed))
    print(f"scientific records: recorded {recorded} pass(es); {len(run_items) - recorded} not recorded")
    # Withdrawn passes leave their evidence behind; nothing reaches it any more.
    cmd_prune(root, args)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prune", help="delete stored evidence no record reaches (a run does this as it finishes)")
    sub.add_parser("refine", help="narrow earned records' inputs by the current precision rules; runs nothing")
    status = sub.add_parser("status")
    status.add_argument(
        "--check", action="store_true", help="exit 1 when any assertion is owed or stored evidence is unreachable; executes nothing"
    )
    run = sub.add_parser("run")
    run.add_argument("--all", action="store_true", help="rerun and re-record every assertion (manual)")
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
    run.add_argument("--carry", action="store_true",
                     help="rerun only flagged rows whose record has an output digest a rerun may reproduce")
    justify = sub.add_parser("justify", help="carry flagged passes forward by a written judgment")
    justify.add_argument("--only", action="append", required=True, metavar="PATTERN",
                         help="flagged rows to justify, matched as by run --only (repeatable)")
    justify.add_argument("--reason", required=True, help="why the change leaves the passes standing")
    justify.add_argument("--author", default="charles")
    args = parser.parse_args()
    try:
        return {"status": cmd_status, "run": cmd_run, "prune": cmd_prune,
            "refine": cmd_refine, "justify": cmd_justify}[args.command](REPO_ROOT, args)
    except UnresolvedInput as problem:
        # An input the fingerprint cannot account for owes its record; it is never silently current.
        print(f"FAIL: a scientific assertion has an undeclared input: {problem}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
