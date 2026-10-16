#!/usr/bin/env bash
# Parity manifest schema gate: every [[capabilities]] row carries its required keys.
#
# Why this exists: a row missing `status` (or any required key) matches none of
# the feature gates' honesty checks and would pass them forever without ever being
# marked done/pending. The gates read rows through scripts/parity_rows.py
# (tomllib), and this gate is the schema-completeness check on top of that.
#
# Run standalone, or via any feature gate / gate_release.sh, which all invoke it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --self-test: each broken ledger, applied alone to an overlay of the repo,
# must fail this gate with the expected message (scripts/selftest_cases.py).
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" schema
fi

python3 "$ROOT/scripts/gate_parity_schema.py"

python3 "$ROOT/scripts/gate_parity_schema_2.py" "$@"
