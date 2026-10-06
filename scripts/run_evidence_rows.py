"""Execute registry evidence rows and prove each one ran.

    run_evidence_rows.py ROOT LEDGER REPO [TABLE] [GATE]

ROOT holds the workspace the rows point into, LEDGER is the TOML registry
relative to ROOT, and REPO supplies the Python project. TABLE names the array
of rows (`id`, `evidence_test`, `evidence_assertion`, optional
`composition_filter`); GATE labels the summary lines.
"""

import re
import shutil
import subprocess
import sys
import tomllib
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ledger = Path(sys.argv[2])
repo = Path(sys.argv[3]).resolve()
table = sys.argv[4] if len(sys.argv) > 4 else "capabilities"
gate = sys.argv[5] if len(sys.argv) > 5 else "gate_composition"

CARGO_RESULT = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", re.M)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def cargo_counts(log: str) -> tuple[int, int]:
    """(passed, failed) summed over every test binary cargo ran."""
    results = CARGO_RESULT.findall(log)
    passed = sum(int(p) for _, p, _ in results)
    failed = max(sum(int(f) for _, _, f in results), sum(s == "FAILED" for s, _, _ in results))
    return passed, failed


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    since = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=30)
                return proc.returncode, stdout + stderr
            except subprocess.TimeoutExpired:
                print(f"{gate}: still running {' '.join(cmd[:8])} ({time.monotonic() - since:.0f}s)", flush=True)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise


failures: list[str] = []


def check(label: str, cmd: list[str], cwd: Path, counter, exact: bool) -> None:
    code, log = run(cmd, cwd)
    passed, failed = counter(log)
    wanted = "exactly 1" if exact else "at least 1"
    ok = code == 0 and failed == 0 and (passed == 1 if exact else passed >= 1)
    if not ok:
        tail = "\n".join(log.splitlines()[-40:])
        failures.append(label)
        print(
            f"FAIL: {label}: exit={code} passed={passed} failed={failed} "
            f"(want {wanted} passed, 0 failed)\n  $ {' '.join(cmd)}\n{tail}"
        )
    else:
        print(f"ok: {label} ({passed} passed)")


def lib_module(test_rel: str) -> str:
    """`crates/x/src/a/b.rs` -> `a::b::`; `src/lib.rs` / `mod.rs` map to their parent."""
    parts = Path(test_rel).parts
    rel = list(parts[parts.index("src") + 1 :])
    rel[-1] = rel[-1].removesuffix(".rs")
    if rel[-1] in ("lib", "mod"):
        rel = rel[:-1]
    return "".join(f"{p}::" for p in rel)


def child_module_dir(test_rel: str) -> Path:
    """Directory holding the file modules declared by `test_rel`."""
    path = root / test_rel
    if path.name in ("lib.rs", "mod.rs"):
        return path.parent
    return path.with_suffix("")


@lru_cache(maxsize=None)
def rust_listing(crate: str, target: tuple[str, ...], ignored: bool = False):
    command = ["cargo", "test", "-p", crate, *target, "--", "--list"]
    if ignored:
        command.append("--ignored")
    return run(command, root)


def resolve_rust(crate: str, target: list[str], assertion: str, test_rel: str, module: str | None):
    code, log = rust_listing(crate, tuple(target))
    if code != 0:
        return None, f"`cargo test -p {crate} {' '.join(target)} -- --list` failed:\n{log[-2000:]}"
    names = [ln[: -len(": test")] for ln in log.splitlines() if ln.endswith(": test")]
    hits = [n for n in names if n.rsplit("::", 1)[-1] == assertion]
    if module is not None:
        # Keep tests defined in this file: inside its module, and not inside a
        # child module that lives in its own file.
        children = child_module_dir(test_rel)

        def in_file(name: str) -> bool:
            if not name.startswith(module):
                return False
            first = name[len(module):].split("::", 1)[0]
            return not ((children / f"{first}.rs").is_file() or (children / first / "mod.rs").is_file())

        hits = [n for n in hits if in_file(n)]
    if len(hits) != 1:
        return None, f"assertion {assertion!r} resolves to {hits or 'no test'} in {crate} {' '.join(target)}"
    ignored_code, ignored_log = rust_listing(crate, tuple(target), True)
    if ignored_code != 0:
        return None, f"ignored-test listing failed in {crate} {' '.join(target)}: {ignored_log[-2000:]}"
    if f"{hits[0]}: test" in ignored_log.splitlines():
        return None, f"assertion {assertion!r} is ignored in {crate} {' '.join(target)}"
    return hits[0], None


def filters(value) -> list[str]:
    if value is None:
        return []
    parts = value if isinstance(value, list) else str(value).split("|")
    return [p.strip() for p in parts if isinstance(p, str) and p.strip()]


rows = tomllib.loads((root / ledger).read_text()).get(table, [])
if not rows:
    print(f"FAIL: {ledger} has no [[{table}]] rows")
    sys.exit(1)

rust_batches: dict[tuple[str, tuple[str, ...]], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
python_nodes: dict[str, list[str]] = defaultdict(list)
filter_batches: dict[tuple[str, tuple[str, ...], str], list[str]] = defaultdict(list)
started = time.monotonic()
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

    if test_rel.endswith(".py"):
        if not test_rel.startswith("python/"):
            failures.append(cid)
            print(f"FAIL: {cid}: Python evidence {test_rel} is not under python/")
            continue
        node = f"{test_rel.removeprefix('python/')}::{assertion}"
        python_nodes[node].append(cid)
        for part in filters(row.get("composition_filter")):
            failures.append(f"{cid}.filter")
            print(f"FAIL: {cid}: composition_filter {part!r} is not supported on a Python row")
        continue

    m = re.fullmatch(r"crates/([^/]+)/(tests|src)/(.+)\.rs", test_rel)
    if not m:
        failures.append(cid)
        print(f"FAIL: {cid}: evidence_test {test_rel} is not a known layout")
        continue
    crate, kind, stem = m.groups()
    if kind == "tests":
        target = ["--test", stem.split("/")[0]]
        module = None
    else:
        target = ["--lib"]
        module = lib_module(test_rel)
    name, problem = resolve_rust(crate, target, assertion, test_rel, module)
    if problem:
        failures.append(cid)
        print(f"FAIL: {cid}: {problem}")
        continue
    rust_batches[(crate, tuple(target))][name].append(cid)
    for part in filters(row.get("composition_filter")):
        filter_batches[(crate, tuple(target), part)].append(f"{cid}.filter[{part}]")

print(
    f"{gate}: resolved {len(rows)} rows into {len(rust_batches)} Rust targets, "
    f"{len(python_nodes)} Python nodes and {len(filter_batches)} filters "
    f"in {time.monotonic() - started:.1f}s",
    flush=True,
)

for (crate, target), members in rust_batches.items():
    has_nextest = shutil.which("cargo-nextest") is not None
    expression = " | ".join(f"test(={name})" for name in members)
    command = (["cargo", "nextest", "run", "--no-tests", "fail", "--ignore-default-filter",
                "-p", crate, *target, "-E", expression]
               if has_nextest else [])
    print(f"{gate}: running Rust target {crate} {' '.join(target)} ({len(members)} cited tests)", flush=True)
    if has_nextest:
        code, log = run(command, root)
        # CI requests colored Cargo output. Nextest colors the summary's
        # numbers as well as "passed", so strip terminal codes before parsing.
        counts = re.findall(r"(\d+) tests? run: (\d+) passed", ANSI_ESCAPE.sub("", log))
        passed = int(counts[-1][1]) if counts else 0
        failed = 0 if code == 0 else 1
    else:
        # Local installations need no nextest dependency; CI installs it and
        # runs each target's cited tests in one process-isolated invocation.
        code, log, passed, failed = 0, "", 0, 0
        for name in members:
            exact_code, exact_log = run(["cargo", "test", "-p", crate, *target, "--", "--exact", name], root)
            exact_passed, exact_failed = cargo_counts(exact_log)
            passed += exact_passed
            failed += exact_failed or (exact_code != 0 or exact_passed != 1)
            if exact_code != 0:
                code = exact_code
                log += exact_log[-1000:]
    if code != 0 or failed or passed != len(members):
        print(f"FAIL: Rust target {crate} {' '.join(target)}: exit={code}, passed={passed}, failed={failed}\n{log[-3000:]}")
        # Attribute a failed target to its exact cited tests as well.
        before = len(failures)
        for name, labels in members.items():
            check(", ".join(labels), ["cargo", "test", "-p", crate, *target,
                                      "--", "--exact", name], root, cargo_counts, exact=True)
        if len(failures) == before:
            failures.append(f"{crate} {' '.join(target)} batch")
    else:
        for labels in members.values():
            for label in labels:
                print(f"ok: {label} (cited Rust test executed)")

if python_nodes:
    command = ["uv", "run", "--no-sync", "--project", str(repo / "python"),
               "pytest", "-q", "-rA", "-p", "no:cacheprovider", *python_nodes]
    print(f"{gate}: running {len(python_nodes)} distinct Python nodes", flush=True)
    code, log = run(command, root / "python")
    passed_nodes = {m.group(1) for m in re.finditer(r"^PASSED\s+(\S+)", log, re.M)}
    if code != 0:
        print(f"{gate}: Python batch failed; isolating cited nodes", flush=True)
    isolated_failures = 0
    for node, labels in python_nodes.items():
        node_passed = code == 0 and any(found == node or found.startswith(node + "[") for found in passed_nodes)
        detail = log[-3000:]
        if code != 0:
            exact = ["uv", "run", "--no-sync", "--project", str(repo / "python"),
                     "pytest", "-q", "-rA", "-p", "no:cacheprovider", node]
            exact_code, detail = run(exact, root / "python")
            exact_passes = {m.group(1) for m in re.finditer(r"^PASSED\s+(\S+)", detail, re.M)}
            node_passed = exact_code == 0 and any(
                found == node or found.startswith(node + "[") for found in exact_passes)
        if node_passed:
            for label in labels:
                print(f"ok: {label} (cited Python test executed)")
        else:
            isolated_failures += 1
            failures.extend(labels)
            print(f"FAIL: {', '.join(labels)}: Python node {node} did not pass\n{detail[-3000:]}")
    if code != 0 and not isolated_failures:
        failures.append("python-batch")
        print(f"FAIL: Python batch failed although cited nodes passed alone:\n{log[-3000:]}")

for (crate, target, part), labels in filter_batches.items():
    check(", ".join(labels), ["cargo", "test", "-p", crate, *target, "--", part], root, cargo_counts, exact=False)

if failures:
    print(f"{gate}: {len(failures)} failing row(s): {', '.join(failures)}")
    sys.exit(1)
print(f"{gate} rows: ok ({len(rows)} rows)")
print(f"{gate}: total evidence runtime {time.monotonic() - started:.1f}s")
