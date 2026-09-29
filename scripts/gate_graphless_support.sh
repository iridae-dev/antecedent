#!/usr/bin/env bash
# Verify generated graphless licenses and that each cited test is in its harness.
#
# `cargo test --workspace` on the Rust job already runs these tests. This gate
# lists each Cargo target once and fails if a cited function is missing or ignored.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/generate_graphless_support.py --check

python3 - <<'PY'
import subprocess
import sys
from collections import OrderedDict

rows = subprocess.run(
    [sys.executable, "scripts/generate_graphless_support.py", "--evidence-tests"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

groups: OrderedDict[tuple[str, str, str], list[str]] = OrderedDict()
for line in rows:
    if not line.strip():
        continue
    package, kind, target, function = line.split("\t")
    groups.setdefault((package, kind, target), []).append(function)

if not groups:
    sys.exit("graphless support evidence list is empty")


def listed(output: str, function: str) -> bool:
    for line in output.splitlines():
        if ": " not in line:
            continue
        name, kind = line.rsplit(": ", 1)
        if name.rsplit("::", 1)[-1] == function and kind.split()[0] == "test":
            return True
    return False


for (package, kind, target), functions in groups.items():
    if kind == "lib":
        command = ["cargo", "test", "-p", package, "--lib", "--", "--list"]
    elif kind == "test":
        command = ["cargo", "test", "-p", package, "--test", target, "--", "--list"]
    else:
        sys.exit(f"unsupported graphless evidence target: {kind}")
    print("==", " ".join(command), f"({len(functions)} cited test(s))", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    output = completed.stdout + completed.stderr
    missing = [function for function in functions if not listed(output, function)]
    if completed.returncode != 0 or missing:
        if missing:
            print(
                "graphless support evidence is not a runnable test: " + ", ".join(missing),
                file=sys.stderr,
            )
        sys.exit(completed.returncode or 1)
PY
