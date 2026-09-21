#!/usr/bin/env bash
# Estimate/CI parity gate: inventory honesty + conformance + calibration.
# black-box / Exact pins diverge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

EVIDENCE = {
    "estimate.temporal_sequential": "crates/antecedent/tests/temporal_response_facade.rs",

    # Estimate inventory
    "estimate.identify.efficient_backdoor": "conformance/estimate/efficient_backdoor",
    "estimate.glm": "conformance/estimate/glm_adjustment",
    "estimate.propensity": "conformance/estimate/propensity_ipw",
    "estimate.matching": "conformance/estimate/distance_matching",
    "estimate.doubly_robust": "conformance/estimate/aipw",
    "estimate.iv": "conformance/estimate/iv_2sls",
    "estimate.rd": "conformance/estimate/rd_sharp",
    "estimate.two_stage": "conformance/estimate/frontdoor",
    "estimate.refute.placebo": "conformance/estimate/refuters",
    "estimate.refute.random_common_cause": "conformance/estimate/refuters",
    "estimate.refute.bootstrap": "conformance/estimate/refuters",
    "estimate.refute.unobserved_common_cause": "conformance/estimate/refuters",
    "estimate.refute.overlap": "conformance/estimate/refuters",
    "estimate.refute.data_subset": "conformance/estimate/refuters",
    "estimate.refute.dummy_outcome": "conformance/estimate/refuters",
    "estimate.refute.evalue": "conformance/estimate/refuters",
    "estimate.refute.graph": "conformance/estimate/refuters",
    "estimate.refute.sensitivity": "conformance/estimate/refuters",
    # Discovery / CI inventory
    "discovery.data.transforms": "crates/antecedent-data/src/transforms.rs",
    "discovery.ci.multivariate_partial_corr": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.weighted_partial_corr": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.robust_partial_corr": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.regression": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.knn_dependence": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.mixed_knn_dependence": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.symbolic_cmi": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.gpdc": "crates/antecedent-stats/src/ci/advanced.rs",
    "discovery.ci.gsquared": "crates/antecedent-stats/src/ci/calibration.rs",
    "discovery.ci.oracle": "crates/antecedent-discovery/src/engine_tests.rs",
    "discovery.pcmci_plus": "conformance/discovery/pcmci_plus_lag0",
    "discovery.graphs.endpoints": "crates/antecedent-graph/src/cpdag.rs",
    "discovery.data.masks": "conformance/discovery/masked_mci_lag1",
    "discovery.data.vector_variables": "conformance/discovery/vector_vars_pcmci",
    "discovery.temporal.max_cond_size": "python/tests/test_discovery_provenance.py",
}

# Only the estimate/CI evidence set is gated here (not every inventory row).
problems = pr.evidence_map_problems(EVIDENCE, ["parity/estimate.toml", "parity/discovery.toml"])
pr.finish("parity inventory", problems, f"parity inventory evidence map: ok ({len(EVIDENCE)} estimate/CI rows)")
PY

echo "== conformance / calibration =="
bash scripts/counted_cargo.sh test -p antecedent --test estimate_conformance --test estimate_linear_gaussian_ate
bash scripts/counted_cargo.sh test -p antecedent-validate --test refuters
bash scripts/counted_cargo.sh test -p antecedent-discovery --test discovery_pcmci_lag1 --test discovery_pcmci_plus_lag0 --test discovery_masked_mci_lag1 --test discovery_vector_vars_pcmci --test discovery_notears_chain
bash scripts/counted_cargo.sh test -p antecedent-stats --lib ci::calibration
bash scripts/counted_cargo.sh test -p antecedent-stats --test foundations_oracle
bash scripts/counted_cargo.sh test -p antecedent-stats --test uncertainty_routing_contract
bash scripts/counted_cargo.sh test -p antecedent-stats --test advanced_ci_oracle
bash scripts/counted_cargo.sh test -p antecedent-stats --test bayesian_ci_oracle
bash scripts/counted_cargo.sh test -p antecedent-discovery --test multiplicity_oracle
bash scripts/gate_estimate_reuse.sh

echo "== Python max_cond_size (PCMCI family) facade smoke =="
python_smoke tests/test_discovery_provenance.py

echo "estimate_ci parity gate: ok"
