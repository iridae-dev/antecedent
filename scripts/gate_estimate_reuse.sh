#!/usr/bin/env bash
# Matching/index + bootstrap workspace reuse gate (DESIGN §14.6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p causal-estimate --lib matching_index_reused_across_compatible_point_fits -- --nocapture
cargo test -p causal-estimate --lib bootstrap_reuses_propensity_workspace_buffers -- --nocapture
cargo test -p causal-stats --lib matching::tests -- --nocapture

echo "estimate_reuse reuse gate: ok"
