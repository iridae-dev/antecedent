#!/usr/bin/env bash
# Every licensed estimator must execute from a retained checked operation after
# its builder is gone. Resolve every citation to an exact non-ignored libtest
# identity, run each distinct assertion once, and attribute failures to every
# licensing cell that cites it. The support-matrix gate validates the citation
# schema and estimator coverage separately.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

shard=all
if [[ "${1:-}" == --shard && $# -eq 2 && "$2" =~ ^[0-3]/4$ ]]; then
  shard="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--shard 0/4|1/4|2/4|3/4]" >&2
  exit 2
fi

EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 - "$EVIDENCE" "$shard" <<'PY'
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

cells = tomllib.loads(Path("parity/support_licensed.toml").read_text()).get("cell", [])
rows = []
for cell in cells:
    coordinate = ":".join(str(cell.get(key)) for key in
                          ("query", "graph_class", "structure", "inference", "validation"))
    for entry in cell.get("checked_execution") or []:
        rows.append((f"{coordinate}::estimator={entry.get('estimator')}",
                     str(entry.get("test", "")), str(entry.get("assertion", ""))))
if not rows:
    raise SystemExit("FAIL: support_licensed.toml has no checked-execution citations")
shard = sys.argv[2]
if shard != "all":
    # Keep all citations for one test target together, so one assertion is
    # executed by exactly one shard. Balance by distinct cited assertions.
    tests_by_path = defaultdict(set)
    for _, path, assertion in rows:
        if not path.startswith("crates/"):
            raise SystemExit(f"FAIL: checked shard needs Rust evidence; found {path}")
        tests_by_path[path].add(assertion)
    loads = [0] * 4
    assignment = {}
    for path, tests in sorted(tests_by_path.items(), key=lambda item: (-len(item[1]), item[0])):
        index = min(range(4), key=lambda i: (loads[i], i))
        assignment[path] = index
        loads[index] += len(tests) + 1
    index = int(shard.split("/")[0])
    total = len(rows)
    rows = [row for row in rows if assignment[row[1]] == index]
    if not rows:
        raise SystemExit(f"FAIL: checked shard {shard} has no citations")
    print(f"checked shard {shard}: {len(rows)}/{total} estimator citations", flush=True)
with Path(sys.argv[1]).open("w") as out:
    for label, path, assertion in rows:
        out.write("[[fixture_evidence]]\n")
        out.write(f"id = {json.dumps(label)}\n")
        out.write(f"evidence_test = {json.dumps(path)}\n")
        out.write(f"evidence_assertion = {json.dumps(assertion)}\n\n")
print(f"checked execution: {len(cells)} licensed cells, {len(rows)} estimator citations", flush=True)
PY

python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_checked_execution
