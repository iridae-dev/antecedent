#!/usr/bin/env bash
# Response/observation/transport/interference calibration and conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/counted_cargo.sh test -p antecedent-identify --lib matches_frozen_bpbounds_table1_oracle
bash scripts/counted_cargo.sh test -p antecedent-identify --test causaleffect_transport_subset matches_frozen_causaleffect_supported_sid_subset
bash scripts/counted_cargo.sh test -p antecedent-stats --lib observation_primitives_match_frozen_paper_equation_fixture
bash scripts/counted_cargo.sh test -p antecedent-estimate --lib matches_frozen_trial_transport_equation_fixture
bash scripts/counted_cargo.sh test -p antecedent-estimate --lib matches_frozen_exact_design_calibration_fixture
bash scripts/counted_cargo.sh test -p antecedent --test response_facade \
  two_point_curve_contrast_conforms_to_average_effect_under_shared_linear_contract
bash scripts/counted_cargo.sh test -p antecedent --test temporal_response_facade

bash scripts/counted_cargo.sh test -p antecedent-stats --lib cox_ipcw
bash scripts/counted_cargo.sh test -p antecedent --test staged_derivatives
bash scripts/counted_cargo.sh test -p antecedent --test staged_static_kinds
bash scripts/counted_cargo.sh test -p antecedent --test conditional_ipcw

echo "gate_response_calibration: ok"
