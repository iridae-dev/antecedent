#!/usr/bin/env bash
# Matching/index + bootstrap workspace reuse gate .
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p antecedent-estimate --lib matching_index_reused_across_compatible_point_fits -- --nocapture
cargo test -p antecedent-estimate --lib bootstrap_reuses_propensity_workspace_buffers -- --nocapture
cargo test -p antecedent-stats --lib matching::tests -- --nocapture

echo "estimate_reuse reuse gate: ok"
