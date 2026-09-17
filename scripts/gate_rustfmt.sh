#!/usr/bin/env bash
# rustfmt check for CI. Same surface as `cargo fmt --all -- --check`, except the
# three measured pin suites: rustfmt would collapse their n-grid arrays and
# those suite facets would owe a remesure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export ROOT
export ALLOWED='
crates/antecedent/tests/v110_calibration_bayesian_static.rs
crates/antecedent/tests/v110_calibration_counterfactual.rs
crates/antecedent/tests/v19_temporal_response_calibration.rs
'

out="$(mktemp)"
trap 'rm -f "$out"' EXIT

if cargo fmt --all -- --check >"$out" 2>&1; then
  echo "rustfmt gate OK"
  exit 0
fi

python3 - "$out" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT"]).resolve()
allowed = {line.strip() for line in os.environ["ALLOWED"].splitlines() if line.strip()}
diffs = set()
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.startswith("Diff in "):
        continue
    rest = line[len("Diff in ") :].rstrip(":")
    path = Path(rest.rsplit(":", 1)[0]).resolve()
    try:
        diffs.add(str(path.relative_to(root)))
    except ValueError:
        diffs.add(str(path))
extra = sorted(diffs - allowed)
if extra:
    print("FAIL: rustfmt diffs outside measured pin suites:", file=sys.stderr)
    print("\n".join(extra), file=sys.stderr)
    print(Path(sys.argv[1]).read_text(), file=sys.stderr)
    sys.exit(1)
print("rustfmt gate OK (3 measured pin suites skipped)")
PY
