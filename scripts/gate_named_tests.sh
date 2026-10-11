#!/usr/bin/env bash
# Named-test existence: every filter the feature gates used to run through
# scripts/counted_cargo.sh still selects at least one non-ignored test.
#
# Those gates no longer re-run Rust suites (`cargo nextest run --workspace` in the
# Rust job runs them), so a renamed test or module would silently drop out of what
# the filter names. This reads `cargo nextest list`, which reuses the Rust job's
# build, and applies libtest's substring filter to each suite's test names.
# Run it after the Rust job's test steps (same profile and features):
#   bash scripts/gate_named_tests.sh
#   bash scripts/gate_named_tests.sh --self-test   # canned listings, no compile
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_named_tests.py" "$@"
