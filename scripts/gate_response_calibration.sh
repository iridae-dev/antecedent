#!/usr/bin/env bash
# 0.5.0 response/observation/transport/interference calibration and conformance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo test -p antecedent-identify --lib matches_frozen_bpbounds_table1_oracle
cargo test -p antecedent-identify --test causaleffect_transport_subset matches_frozen_causaleffect_supported_sid_subset
cargo test -p antecedent-stats --lib observation_primitives_match_frozen_paper_equation_fixture
cargo test -p antecedent-estimate --lib matches_frozen_trial_transport_equation_fixture
cargo test -p antecedent-estimate --lib matches_frozen_exact_design_calibration_fixture
cargo test -p antecedent --test response_facade \
  two_point_curve_contrast_conforms_to_average_effect_under_shared_linear_contract
cargo test -p antecedent --test temporal_response_facade

echo "gate_response_calibration: ok"
