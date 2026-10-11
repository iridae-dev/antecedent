#!/usr/bin/env bash
# Hot-path baseline metadata (and optional local Criterion-mean) gate.
#
# Always: every `docs/hot_paths.md` Baseline link exists, and every
# `benches/baselines/*.md` file (linked or not) records either a numeric
# wall-time (µs/ms/s) or an explicit "none published" waiver, plus its
# Established / Machine class / Commit lines.
# Optional: GATE_CRITERION_MEANS=1 compares `target/criterion/**/estimates.json`
# means against parseable gates. Never enabled in CI — Ubuntu runners cannot
# enforce Apple M1 Max wall times.
#
# Run directly, or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_hot_path_baselines.py"

echo "Hot-path baseline gate PASSED"
