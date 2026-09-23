"""Execute registry evidence rows and prove each one ran.

    run_evidence_rows.py ROOT LEDGER REPO [TABLE] [GATE]

ROOT holds the workspace the rows point into, LEDGER is the TOML registry
relative to ROOT, and REPO supplies the Python project. TABLE names the array
of rows (`id`, `evidence_test`, `evidence_assertion`, optional
`composition_filter`); GATE labels the summary lines.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ledger = Path(sys.argv[2])
repo = Path(sys.argv[3]).resolve()
table = sys.argv[4] if len(sys.argv) > 4 else "capabilities"
gate = sys.argv[5] if len(sys.argv) > 5 else "gate_composition"

CARGO_RESULT = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", re.M)
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?)\b")


def cargo_counts(log: str) -> tuple[int, int]:
    """(passed, failed) summed over every test binary cargo ran."""
    results = CARGO_RESULT.findall(log)
    passed = sum(int(p) for _, p, _ in results)
    failed = max(sum(int(f) for _, _, f in results), sum(s == "FAILED" for s, _, _ in results))
    return passed, failed


def pytest_counts(log: str) -> tuple[int, int]:
    """(passed, failed+errors) from pytest's final summary line."""
    summary = [ln for ln in log.splitlines() if PYTEST_COUNT.search(ln) and " in " in ln]
    if not summary:
        return 0, 0
    counts = {"passed": 0, "failed": 0}
    for n, kind in PYTEST_COUNT.findall(summary[-1]):
        key = "passed" if kind == "passed" else "failed"
        counts[key] += int(n)
    return counts["passed"], counts["failed"]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


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


def resolve_rust(crate: str, target: list[str], assertion: str, test_rel: str, module: str | None):
    code, log = run(["cargo", "test", "-p", crate, *target, "--", "--list"], root)
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
        cmd = ["uv", "run", "--project", str(repo / "python"), "pytest", "-q", "-p", "no:cacheprovider", node]
        check(cid, cmd, root / "python", pytest_counts, exact=False)
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
    check(cid, ["cargo", "test", "-p", crate, *target, "--", "--exact", name], root, cargo_counts, exact=True)
    for part in filters(row.get("composition_filter")):
        check(f"{cid}.filter[{part}]", ["cargo", "test", "-p", crate, *target, "--", part], root, cargo_counts, exact=False)

if failures:
    print(f"{gate}: {len(failures)} failing row(s): {', '.join(failures)}")
    sys.exit(1)
print(f"{gate} rows: ok ({len(rows)} rows)")
