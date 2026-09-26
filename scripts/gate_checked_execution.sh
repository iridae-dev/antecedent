#!/usr/bin/env bash
# Checked execution: every licensed cell runs each of its licensed estimators
# from a retained checked operation with no builder alive.
#
# parity/support_licensed.toml cites, per cell and estimator, the executing
# test that drops its builder, executes the retained plan and inspects it
# (`checked_execution`). scripts/gate_support_matrix.sh resolves those
# citations statically; this gate executes them. Each distinct (test,
# assertion) pair runs once: Rust citations run grouped by Cargo target, Python
# citations in one pytest process. A failed group is re-run per cited test so
# the failure names its licensed cells.
#
# Public transport stage routes (parity/transport_stages.toml) are executed by
# scripts/gate_transport.sh and are not repeated here.
#
# Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python citations are not checked-execution evidence"
  exit 1
fi

python3 - <<'PY'
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "scripts"))
import test_evidence  # noqa: E402

cells = tomllib.loads((root / "parity/support_licensed.toml").read_text()).get("cell", [])


def coordinate(cell: dict) -> str:
    return ":".join(str(cell.get(k)) for k in ("query", "graph_class", "structure", "inference", "validation"))


issues: list[str] = []
# (test, assertion) -> the "<cell>::estimator=<id>" labels that cite it.
citations: dict[tuple[str, str], list[str]] = {}
for cell in cells:
    for entry in cell.get("checked_execution") or []:
        key = (str(entry.get("test")), str(entry.get("assertion")))
        citations.setdefault(key, []).append(f"{coordinate(cell)}::estimator={entry.get('estimator')}")
if not citations:
    print("FAIL: parity/support_licensed.toml cites no checked_execution evidence")
    sys.exit(1)

# batch key -> [(label, test name)]
batches: dict[tuple[str, ...], list[tuple[str, str]]] = {}
for (test_rel, assertion), labels in citations.items():
    label = f"{labels[0]}" + (f" (+{len(labels) - 1} more)" if len(labels) > 1 else "")
    path = root / test_rel
    if path.suffix == ".py":
        problems = test_evidence.static_python_test(path, assertion)
        if problems:
            issues.append(f"{label}: {test_rel}::{assertion} could not be resolved: {'; '.join(problems)}")
            continue
        rel = path.resolve().relative_to((root / "python").resolve())
        batches.setdefault(("python",), []).append((label, f"{rel}::{assertion}"))
    else:
        target_root = test_evidence.target_root(path)
        if target_root is None:
            issues.append(f"{label}: {test_rel} has no Cargo target")
            continue
        _, crate, target = target_root
        full_name, problems = test_evidence.resolve_rust_test(path, assertion, root)
        if problems or full_name is None:
            issues.append(f"{label}: {test_rel}::{assertion} could not be resolved: {'; '.join(problems)}")
            continue
        batches.setdefault(("rust", crate, *target), []).append((label, full_name))

for batch_key, members in batches.items():
    if batch_key[0] == "python":
        command = ["uv", "run", "--quiet", "--project", ".", "pytest", "-q",
                   "-p", "no:cacheprovider", *(name for _, name in members)]
        cwd = root / "python"
    else:
        command = ["cargo", "test", "-q", "-p", batch_key[1], *batch_key[2:]]
        cwd = root
    print(f"== {' '.join(command)} ({len(members)} cited test(s))", flush=True)
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if not result.returncode:
        continue
    detail = (result.stdout + result.stderr)[-2500:]
    # A failed group must not obscure which licensed cell failed: re-run each
    # cited member alone with an exact filter for attribution.
    isolated_failures = 0
    for label, name in members:
        if batch_key[0] == "python":
            exact = ["uv", "run", "--quiet", "--project", ".", "pytest", "-q",
                     "-p", "no:cacheprovider", name]
        else:
            exact = ["cargo", "test", "-q", "-p", batch_key[1], *batch_key[2:], "--", name, "--exact"]
        isolated = subprocess.run(exact, cwd=cwd, capture_output=True, text=True)
        if isolated.returncode:
            isolated_failures += 1
            issues.append(f"{label}: checked execution failed ({' '.join(exact)}):\n{(isolated.stdout + isolated.stderr)[-2500:]}")
    if not isolated_failures:
        issues.append(f"target failed outside cited tests ({' '.join(command)}):\n{detail}")

if issues:
    print("checked execution gate FAILED:")
    for issue in issues[:80]:
        print(f" - {issue}")
    if len(issues) > 80:
        print(f" - ... and {len(issues) - 80} more")
    sys.exit(1)

entries = sum(len(cell.get("checked_execution") or []) for cell in cells)
print(
    f"checked execution: ok ({len(cells)} licensed cells; {entries} estimator citations; "
    f"{len(citations)} distinct tests across {len(batches)} targets)"
)
PY
