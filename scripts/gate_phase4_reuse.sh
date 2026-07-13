#!/usr/bin/env bash
# Phase 4 matching/index + bootstrap workspace reuse gate (DESIGN §14.6 / Phase 4 exit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p causal-estimate --lib matching_index_reused_across_compatible_point_fits -- --nocapture
cargo test -p causal-estimate --lib bootstrap_reuses_propensity_workspace_buffers -- --nocapture
cargo test -p causal-stats --lib matching::tests -- --nocapture

echo "phase4 reuse gate: ok"
