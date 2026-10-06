#!/usr/bin/env bash
# Every licensed estimator must execute from a retained checked operation after
# its builder is gone. Resolve every citation to an exact non-ignored libtest
# identity, run each distinct assertion once, and attribute failures to every
# licensing cell that cites it. The support-matrix gate validates the citation
# schema and estimator coverage separately.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 - "$EVIDENCE" <<'PY'
import json
import sys
import tomllib
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
with Path(sys.argv[1]).open("w") as out:
    for label, path, assertion in rows:
        out.write("[[fixture_evidence]]\n")
        out.write(f"id = {json.dumps(label)}\n")
        out.write(f"evidence_test = {json.dumps(path)}\n")
        out.write(f"evidence_assertion = {json.dumps(assertion)}\n\n")
print(f"checked execution: {len(cells)} licensed cells, {len(rows)} estimator citations", flush=True)
PY

python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_checked_execution
