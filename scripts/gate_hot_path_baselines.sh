#!/usr/bin/env bash
# Hot-path baseline metadata (and optional local Criterion-mean) gate.
#
# Always: every `docs/hot_paths.md` Baseline link exists, and that file records
# either a numeric wall-time (µs/ms/s) or an explicit "none published" waiver.
# Optional: GATE_CRITERION_MEANS=1 compares `target/criterion/**/estimates.json`
# means against parseable gates. Never enabled in CI — Ubuntu runners cannot
# enforce Apple M1 Max wall times.
#
# Run directly, or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

root = Path(".")
fail: list[str] = []

hot = (root / "docs/hot_paths.md").read_text()
linked = re.findall(r"\]\(\.\./benches/baselines/([^)]+?\.md)\)", hot)
if not linked:
    fail.append("docs/hot_paths.md has no Baseline links to benches/baselines/*.md")

TIME = re.compile(
    r"(?:\*\*[0-9.]+\s*(?:µs|ms|s)\*\*)"
    r"|(?:Gate:\s*mean\s*≤\s*\*\*[0-9.]+\s*(?:µs|ms|s)\*\*)"
    r"|(?:<\s*[0-9.]+\s*(?:ms|s))"
    r"|(?:none published)",
    re.I,
)

gates: list[tuple[str, str, float, str]] = []
# (baseline_file, workload_or_*, value, unit)
FILE_GATE = re.compile(
    r"Gate:\s*mean\s*≤\s*\*\*([0-9.]+)\s*(µs|ms|s)\*\*",
    re.I,
)
ROW_GATE = re.compile(
    r"\|\s*`?([A-Za-z0-9_]+)`?\s*\|[^|]*≤\s*\*\*([0-9.]+)\s*(µs|ms|s)\*\*",
)
DSEP_GATE = re.compile(
    r"\|\s*`([A-Za-z0-9_]+)`\s*\|[^|]*\|\s*\*\*([0-9.]+)\s*(µs|ms|s)\*\*\s*\|",
)

for name in sorted(set(linked)):
    path = root / "benches" / "baselines" / name
    if not path.is_file():
        fail.append(f"hot_paths Baseline link missing: benches/baselines/{name}")
        continue
    text = path.read_text()
    if not TIME.search(text):
        fail.append(
            f"benches/baselines/{name} has no numeric wall-time (µs/ms/s) and no "
            "'none published' waiver"
        )
    for m in FILE_GATE.finditer(text):
        gates.append((name, "*", float(m.group(1)), m.group(2).lower()))
    for m in ROW_GATE.finditer(text):
        gates.append((name, m.group(1), float(m.group(2)), m.group(3).lower()))
    for m in DSEP_GATE.finditer(text):
        gates.append((name, m.group(1), float(m.group(2)), m.group(3).lower()))

UNIT_NS = {"µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}

def mean_ns(estimates: Path) -> float | None:
    try:
        data = json.loads(estimates.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mean = data.get("mean") or {}
    val = mean.get("point_estimate")
    return float(val) if isinstance(val, (int, float)) else None

if os.environ.get("GATE_CRITERION_MEANS") == "1":
    criterion = root / "target" / "criterion"
    if not criterion.is_dir():
        fail.append("GATE_CRITERION_MEANS=1 but target/criterion is missing; run cargo bench first")
    else:
        estimates = list(criterion.glob("**/new/estimates.json"))
        if not estimates:
            fail.append("GATE_CRITERION_MEANS=1 but no target/criterion/**/new/estimates.json")
        for est in estimates:
            fn = est.parent.parent.name
            measured = mean_ns(est)
            if measured is None:
                continue
            matched = False
            for _base, workload, limit, unit in gates:
                if workload != "*" and workload not in fn and fn not in workload:
                    continue
                if workload == "*":
                    # File-level gate: bind only when the function name appears in that file.
                    base_text = (root / "benches" / "baselines" / _base).read_text()
                    if fn not in base_text:
                        continue
                ns_limit = limit * UNIT_NS[unit]
                matched = True
                if measured > ns_limit * 1.001:
                    fail.append(
                        f"{fn}: Criterion mean {measured/1e3:.2f} µs exceeds gate "
                        f"{limit} {unit} ({_base})"
                    )
            # Unmapped functions are skipped: CI Ubuntu must not invent M1 gates.

if fail:
    print("Hot-path baseline gate FAILED:")
    for f in fail:
        print(" -", f)
    sys.exit(1)

print(
    f"Hot-path baseline metadata OK ({len(set(linked))} linked files, "
    f"{len(gates)} parseable numeric gates)"
)
PY

echo "Hot-path baseline gate PASSED"
