#!/usr/bin/env bash
# Verify generated graphless licenses and that each cited test is executing evidence.
#
# `cargo test --workspace` on the Rust job already runs these tests. This gate
# resolves every cited `path::function` to its full libtest name (module path
# derived from the cited file) through scripts/test_evidence.py: the name must be
# listed by `cargo test -- --list` and absent from `-- --list --ignored`, and the
# function must be a compiled, non-ignored #[test]. Each Cargo target is listed once.
#   bash scripts/gate_graphless_support.sh --self-test   # stubbed listings, no compile
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "--self-test" ]]; then
  python3 scripts/generate_graphless_support.py --check
fi

python3 "$ROOT/scripts/gate_graphless_support.py" "$@"
