#!/usr/bin/env bash
# Every supported estimator must execute from a retained checked operation after
# its builder is gone. The Python gate rejects a cell with no citation, a
# citation that does not run the retained program, and a citation that builds
# a public Study again before that run. Public behavioral assertions additionally
# exercise Gaussian resource/design/refresh guarantees and dose-grid/cyclic
# retained-fit reuse, invalidation and replay. Every supported route must dispatch
# a lowered program: the Python gate fails statically when a supported route records
# no registered `lowering` cell (or the record is not all 723 routes), when a
# partial lowering (`LoweringDepth::Generic`, `WidthSource::SchemaBound`, a
# `generic` lowering) survives, when library code under analysis/{execute,
# prepared,lowering} or planner re-enters the public builder outside test items,
# or when code outside antecedent-core writes an `ExecutionContext` literal. It
# then adds the compiler-path, lowered-dispatch and lowering-registry assertions
# to the evidence rows. Resolve every citation to a non-ignored libtest identity
# without running it: the engineering lane runs the ordinary assertions and
# scripts/scientific_records.py run (owed only) runs the scientific ones.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 "$ROOT/scripts/gate_checked_execution.py" "$EVIDENCE" all

python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_checked_execution --static
