"""Execute registry evidence rows and prove each one ran.

    run_evidence_rows.py ROOT LEDGER REPO [TABLE] [GATE] [--static]

ROOT holds the workspace the rows point into, LEDGER is the TOML registry
relative to ROOT, and REPO supplies the Python project. TABLE names the array
of rows (`id`, `evidence_test`, `evidence_assertion`); GATE labels the summary
lines. Every cited Rust test is resolved statically to its exact libtest name
(declared `#[test]`, not ignored, compiled into its cargo target) and every
cited Python test to its exact node. All Rust rows then run in one `cargo
nextest run --ignore-default-filter` invocation and all Python rows in one
pytest batch; a row counts only if its exact test is reported passed.
Simulation evidence is incomplete unless ANTECEDENT_RUN_CALIBRATION_SIMULATIONS=1
explicitly opts in. `--static` resolves the rows and executes nothing.

Callers: scripts/scientific_records.py run (the one scientific entry point)
selects the owed scientific rows, and gates use `--static` to keep their cited
tests honest while the engineering lane executes the ordinary ones.
ANTECEDENT_EVIDENCE_PASSED_OUT=FILE writes the ids that were proven to run.
"""

import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

from _repo import WORKSPACE_TEST_FEATURES, cargo_target_feature_args
import route_behaviour
from test_evidence import (
    python_test_markers,
    registered_calibration_simulations,
    registered_rust_assertions,
    static_python_test,
)

_argv = [a for a in sys.argv[1:] if a != "--static"]
static_only = len(_argv) != len(sys.argv) - 1
root = Path(_argv[0]).resolve()
ledger = Path(_argv[1])
repo = Path(_argv[2]).resolve()
table = _argv[3] if len(_argv) > 3 else "capabilities"
gate = _argv[4] if len(_argv) > 4 else "gate_composition"
simulation_rows = (registered_calibration_simulations(root)
                   if (root / "parity/test_registry.toml").exists() else [])
simulations = {(r["evidence_test"], r["evidence_assertion"]) for r in simulation_rows}
run_simulations = os.environ.get("ANTECEDENT_RUN_CALIBRATION_SIMULATIONS") == "1"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# `PASS [   0.012s] ( 3/10) antecedent-design::suite_01 module::name`; the counter is optional.
STATUS_LINE = re.compile(
    r"^\s*(PASS|FAIL|LEAK-FAIL|LEAK|TIMEOUT|EXECFAIL|SIGABRT|SIGSEGV|SIGBUS|SIGILL|SIGFPE|SIGKILL|SIGTERM|SIGTRAP)"
    r"\s+\[[^\]]*\]\s+(?:\(\s*\d+/\s*\d+\)\s+)?(\S+)\s+(\S.*?)\s*$",
    re.M,
)


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    def once(env: dict[str, str] | None = None) -> tuple[int, str]:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        since = time.monotonic()
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                    return proc.returncode, stdout + stderr
                except subprocess.TimeoutExpired:
                    print(
                        f"{gate}: still running {' '.join(cmd[:8])} "
                        f"({time.monotonic() - since:.0f}s)",
                        flush=True,
                    )
        except BaseException:
            proc.kill()
            proc.communicate()
            raise

    code, log = once()
    if (
        code != 0
        and cmd[0] == "cargo"
        and os.environ.get("RUSTC_WRAPPER", "").endswith("sccache")
        and "sccache: error: Timed out waiting for server startup" in log
    ):
        print(f"{gate}: sccache server unavailable; retrying without wrapper", flush=True)
        return once({**os.environ, "RUSTC_WRAPPER": ""})
    return code, log


failures: list[str] = []
passed_labels: set[str] = set()


def workspace_features() -> list[str]:
    """The feature set `scripts/test.sh` builds the workspace with, for the packages this root has."""
    present = []
    for spec in WORKSPACE_TEST_FEATURES:
        package, _, feature = spec.partition("/")
        manifest = root / "crates" / package / "Cargo.toml"
        if manifest.is_file() and feature in tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {}):
            present.append(spec)
    return present


def nextest_expression(targets: dict[tuple[str, str], set[str]]) -> str:
    clauses = []
    for (crate, target), names in sorted(targets.items()):
        where = "kind(lib)" if target == "lib" else f"binary(={target})"
        clauses.append(
            f"(package(={crate}) and {where} and ("
            + " | ".join(f"test(={name})" for name in sorted(names))
            + "))"
        )
    return " or ".join(clauses)


def run_rust(expected: dict[tuple[str, str, str], list[str]], features: set[str]) -> None:
    """One nextest invocation for every cited Rust test; each must be reported passed."""
    if shutil.which("cargo-nextest") is None:
        for labels in expected.values():
            failures.extend(labels)
        print(f"FAIL: cargo-nextest is required to run {len(expected)} cited Rust test(s)")
        return
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for crate, target, name in expected:
        targets[(crate, target)].add(name)
    command = [
        "cargo", "nextest", "run",
        *(["--locked"] if (root / "Cargo.lock").is_file() else []),
        "--no-tests", "fail", "--no-fail-fast", "--color", "never",
        "--status-level", "pass", "--final-status-level", "none",
        # The default filter is the engineering lane; this run selects exact scientific tests.
        "--ignore-default-filter",
        "--workspace",
    ]
    all_features = sorted({*workspace_features(), *features})
    if all_features:
        command += ["--features", ",".join(all_features)]
    command += ["-E", nextest_expression(targets)]
    print(f"{gate}: running {len(expected)} cited Rust tests in {len(targets)} cargo targets (one nextest run)", flush=True)
    code, log = run(command, root)
    seen: dict[tuple[str, str], str] = {}
    for status, binary_id, name in STATUS_LINE.findall(ANSI_ESCAPE.sub("", log)):
        seen[(binary_id, name)] = status
    bad = 0
    for (crate, target, name), labels in expected.items():
        identity = crate + ("" if target == "lib" else f"::{target}")
        status = seen.get((identity, name))
        if status == "PASS":
            passed_labels.update(labels)
            for label in labels:
                print(f"ok: {label} (cited Rust test executed)")
        else:
            bad += 1
            failures.extend(labels)
            print(f"FAIL: {', '.join(labels)}: {identity} {name} " + (f"reported {status}" if status else "did not run"))
    if code != 0 or bad:
        print(f"FAIL: nextest exit={code}, {bad} of {len(expected)} cited Rust tests not passed\n  $ {' '.join(command[:12])} ...\n{log[-3000:]}")
        if code != 0 and not bad:
            failures.append("rust-batch")


def python_marker() -> list[str]:
    # Scientific assertions are deselected by the default lane; this run selects them by node id.
    return ["-m", "not requires_calibration or calibration_simulation"
            if run_simulations else "not requires_calibration and not calibration_simulation"]


rows = tomllib.loads((root / ledger).read_text(encoding="utf-8")).get(table, [])
if table == "routes" and ledger.name == "transport_stages.toml":
    # A record-owned route's stage, status and reason code are generated, not in the row.
    rows = route_behaviour.load_stage_routes(root, root / ledger)
    # A record-owned route that cites no test of its own is evidenced by its record's executed
    # fixtures (scripts/check_capability_records.py), as in scripts/gate_support.py.
    _owned = {
        r["route"]
        for entry in (route_behaviour.load(root / route_behaviour.GENERATED) or {}).values()
        for r in entry["routes"]
    }
    rows = [r for r in rows if not (r.get("route") in _owned and not r.get("evidence_test"))]
if not rows:
    print(f"FAIL: {ledger} has no [[{table}]] rows")
    sys.exit(1)

total_rows = len(rows)
by_id = {}
for r in rows:
    by_id.setdefault(r.get("id") or r.get("route") or "<no id>", []).append(r)

started = time.monotonic()
python_nodes: dict[str, list[str]] = defaultdict(list)
rust_cited: list[tuple[str, str, list[str]]] = []  # (evidence_test, assertion, labels)
for row in rows:
    cid = row.get("id") or row.get("route") or "<no id>"
    if row.get("status") == "closed" and not row.get("evidence_test"):
        continue
    test_rel = row.get("evidence_test")
    assertion = row.get("evidence_assertion")
    if not test_rel or not assertion:
        failures.append(cid)
        print(f"FAIL: {cid}: no evidence_test/evidence_assertion; every {gate} row must execute")
        continue
    if row.get("composition_filter") is not None:
        failures.append(cid)
        print(f"FAIL: {cid}: composition_filter is not supported; cite each test as its own row")
        continue

    python_simulation = (test_rel.endswith(".py") and (root / test_rel).is_file()
                         and "calibration_simulation" in python_test_markers(root / test_rel).get(assertion, set()))
    if not run_simulations and not static_only and ((test_rel, assertion) in simulations or python_simulation):
        failures.append(cid)
        print(f"INCOMPLETE: {cid}: cited calibration simulation was not run")
        continue

    if test_rel.endswith(".py"):
        if not test_rel.startswith("python/"):
            failures.append(cid)
            print(f"FAIL: {cid}: Python evidence {test_rel} is not under python/")
            continue
        problems = static_python_test(root / test_rel, assertion)
        if problems:
            failures.append(cid)
            print(f"FAIL: {cid}: {'; '.join(problems)}")
            continue
        python_nodes[f"{test_rel.removeprefix('python/')}::{assertion}"].append(cid)
        continue

    if not re.fullmatch(r"crates/([^/]+)/(tests|src)/(.+)\.rs", test_rel):
        failures.append(cid)
        print(f"FAIL: {cid}: evidence_test {test_rel} is not a known layout")
        continue
    rust_cited.append((test_rel, assertion, [cid]))

# Resolve every Rust citation to its exact libtest identity; attribute a problem to its row.
expected: dict[tuple[str, str, str], list[str]] = defaultdict(list)
features: set[str] = set()
if rust_cited:
    try:
        resolved = registered_rust_assertions([(t, a) for t, a, _ in rust_cited], root)
        groups = [(t, a, labels, [r for r in resolved if (r["evidence_test"], r["evidence_assertion"]) == (t, a)])
                  for t, a, labels in rust_cited]
    except ValueError:
        groups = []
        for t, a, labels in rust_cited:
            try:
                groups.append((t, a, labels, registered_rust_assertions([(t, a)], root)))
            except ValueError as problem:
                failures.extend(labels)
                print(f"FAIL: {', '.join(labels)}: {problem}")
    for _t, _a, labels, found in groups:
        for r in found:
            target_args = ["--lib"] if r["cargo_target"] == "lib" else ["--test", r["cargo_target"]]
            required = cargo_target_feature_args(root / "crates" / r["crate"] / "Cargo.toml", target_args)
            qualified = {f"{r['crate']}/{f}" for f in [*r.get("features", []), *(required[1].split(",") if required else [])]}
            features |= qualified
            expected[(r["crate"], r["cargo_target"], r["libtest_name"])].extend(labels)

print(
    f"{gate}: resolved {len(rows)} rows into {len(expected)} exact Rust tests "
    f"and {len(python_nodes)} Python nodes in {time.monotonic() - started:.1f}s",
    flush=True,
)

if static_only:
    for labels in [*expected.values(), *python_nodes.values()]:
        passed_labels.update(labels)
elif expected:
    run_rust(expected, features)

if python_nodes and not static_only:
    command = [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(repo / "python"),
        "pytest",
        "-q",
        "-rA",
        "-p",
        "no:cacheprovider",
        *python_marker(),
        *python_nodes,
    ]
    print(f"{gate}: running {len(python_nodes)} distinct Python nodes", flush=True)
    code, log = run(command, root / "python")
    passed_nodes = {m.group(1) for m in re.finditer(r"^PASSED\s+(\S+)", log, re.M)}
    if code != 0:
        print(f"{gate}: Python batch failed; isolating cited nodes", flush=True)
    isolated_failures = 0
    for node, labels in python_nodes.items():
        node_passed = code == 0 and any(
            found == node or found.startswith(node + "[") for found in passed_nodes
        )
        detail = log[-3000:]
        if code != 0:
            exact = [
                "uv",
                "run",
                "--no-sync",
                "--project",
                str(repo / "python"),
                "pytest",
                "-q",
                "-rA",
                "-p",
                "no:cacheprovider",
                *python_marker(),
                node,
            ]
            exact_code, detail = run(exact, root / "python")
            exact_passes = {m.group(1) for m in re.finditer(r"^PASSED\s+(\S+)", detail, re.M)}
            node_passed = exact_code == 0 and any(
                found == node or found.startswith(node + "[") for found in exact_passes
            )
        if node_passed:
            for label in labels:
                passed_labels.add(label)
                print(f"ok: {label} (cited Python test executed)")
        else:
            isolated_failures += 1
            failures.extend(labels)
            print(f"FAIL: {', '.join(labels)}: Python node {node} did not pass\n{detail[-3000:]}")
    if code != 0 and not isolated_failures:
        failures.append("python-batch")
        print(f"FAIL: Python batch failed although cited nodes passed alone:\n{log[-3000:]}")

if os.environ.get("ANTECEDENT_EVIDENCE_PASSED_OUT"):
    import json

    Path(os.environ["ANTECEDENT_EVIDENCE_PASSED_OUT"]).write_text(
        json.dumps(sorted(passed_labels - set(failures))), encoding="utf-8"
    )
if failures:
    print(f"{gate}: {len(failures)} failing row(s): {', '.join(failures)}")
    sys.exit(1)
print(f"{gate} rows: ok ({total_rows} rows{', resolved only' if static_only else ''})")
print(f"{gate}: total evidence runtime {time.monotonic() - started:.1f}s")
