#!/usr/bin/env bash
# Bayesian gate: inventory honesty + Python smoke. Rust fixtures run in the Rust job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 "$ROOT/scripts/gate_bayesian.py"

# Rust suites and the Criterion smoke run in the Rust job and gate_release.sh.
echo "== Python panel Bayesian facade smoke =="
python_smoke tests/test_panel_bayesian.py tests/test_temporal_bayesian_pulse.py tests/test_prior_bank.py tests/test_temporal_prior_transfer.py tests/test_bayesian_estimator_lifecycle.py tests/test_bayesian_likelihood.py tests/test_attribution_lifecycle.py tests/test_transport_interference_lifecycle.py tests/test_transport_statistical.py

echo "Bayesian gate PASSED"
