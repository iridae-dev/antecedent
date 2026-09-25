#!/usr/bin/env bash
# Bayesian gate: inventory honesty + fixtures + benches.
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
    "estimate.bayesian_conditional": "crates/antecedent/tests/prepared_analysis.rs",
    "estimate.bayesian_temporal_mediation": "python/tests/test_release_12.py",
    "validate.dbn_posterior": "crates/antecedent/tests/manufacturing_temporal.rs",

    "bayes.prob.columnar_posteriors": "crates/antecedent-prob/src/posterior.rs",
    "bayes.prob.priors": "crates/antecedent-prob/src/prior.rs",
    "bayes.backend.conjugate_gaussian": "crates/antecedent/tests/bayesian.rs",
    "bayes.backend.laplace_glm": "crates/antecedent/tests/bayesian.rs",
    "bayes.estimate.gcomp": "crates/antecedent/tests/bayesian.rs",
    "bayes.estimate.temporal_gcomp": "crates/antecedent/tests/bayesian.rs",
    "bayes.estimate.graph_envelopes": "crates/antecedent/tests/bayesian.rs",
    "bayes.estimate.basis_gcomp_all_observed": "crates/antecedent-learn/tests/bayesian_basis.rs",
    "bayes.estimate.robust_ate_modular": "crates/antecedent/tests/bayesian_robust_ate.rs",
    "bayes.estimate.iv_joint_linear": "crates/antecedent/tests/bayesian_iv_rd_staged.rs",
    "bayes.estimate.sharp_rd_local_linear": "crates/antecedent/tests/bayesian_iv_rd_staged.rs",
    "bayes.estimate.interference_neighbor_count": "crates/antecedent/tests/staged_attribution_transport_interference.rs",
    "bayes.estimate.trial_to_target": "crates/antecedent/tests/bayesian_trial_transport.rs",
    "bayes.transport.empirical_and_state_space_laws": "crates/antecedent/src/analysis/statistical.rs",
    "bayes.transport.z_cited_laws": "crates/antecedent-estimate/tests/z_transport_cited_evidence.rs",
    "bayes.response.simultaneous_joint_band": "crates/antecedent-estimate/src/response/mod.rs",
    "bayes.validate.sbc_glm_families": "crates/antecedent-validate/src/bayesian_checks.rs",
    "bayes.validate.ppc": "crates/antecedent-validate/src/bayesian_checks.rs",
    "bayes.validate.prior_sensitivity": "crates/antecedent-validate/src/bayesian_checks.rs",
    "bayes.data.bayesian_bootstrap": "provenance/data.bayesian_bootstrap.toml",
    "bayes.io.posterior_artifact": "crates/antecedent-io/src/posterior.rs",
    "bayes.io.posterior_artifact_summary_only": "crates/antecedent-io/src/posterior.rs",
    "bayes.facade.inference_mode": "crates/antecedent/src/inference.rs",
    "bayes.model.pcm_scm_registry": "crates/antecedent-model/src/lib.rs",
    "bayes.discovery.dag_posterior": "crates/antecedent-discovery/tests/dag_posterior_conformance.rs",
    "bayes.backend.hierarchical_bvar_gp": "crates/antecedent-model/src/registry.rs",
    "bayes.validate.mcmc_diagnostics": "crates/antecedent-prob/tests/mcmc_arviz_oracle.rs",
    "bayes.ci.tests": "crates/antecedent-stats/src/ci/bayes.rs",
    "bayes.prior_bank.temporal_transfer": "crates/antecedent/tests/temporal_prior_transfer.rs",
    "bayes.prior_bank.catalog": "crates/antecedent-io/src/prior_bank.rs",
    "bayes.panel.random_intercept_non_gaussian": "crates/antecedent-estimate/src/bayesian.rs",
    "bayes.gcomp.negative_binomial_se": "crates/antecedent-estimate/src/glm_adjustment.rs",
    "bayes.prior_bank.effect_map": "crates/antecedent-estimate/src/bayesian.rs",
    "bayes.prior_bank.power_mixture": "crates/antecedent-prob/src/external_prior.rs",
    "bayes.prior_bank.conflict": "crates/antecedent-validate/src/conflict.rs",
    "bayes.prior_bank.transport": "crates/antecedent-prob/src/transport.rs",
    "bayes.prior_bank.ess_accounting": "crates/antecedent-prob/src/external_prior.rs",
    "bayes.prior_bank.conjugate_moment_match": "crates/antecedent-prob/src/conjugate_moment_match.rs",
}

EXIT_ARTIFACTS = [
    "conformance/bayesian/shared_functional_ate/expected.json",
    "conformance/bayesian/nonidentified_prior/expected.json",
    "conformance/bayesian/laplace_glm/expected.json",
    "conformance/bayesian/dag_posterior/expected.json",
    "conformance/bayesian/temporal_pulse/expected.json",
    "conformance/bayesian/temporal_prior_transfer/expected.json",
    "conformance/bayesian/prior_bank_catalog/expected.json",
    "conformance/bayesian/prior_bank_effect_map/expected.json",
    "conformance/bayesian/prior_bank_power_mixture/expected.json",
    "conformance/bayesian/prior_bank_conflict_shrink/expected.json",
    "conformance/bayesian/prior_bank_transport/expected.json",
    "conformance/bayesian/prior_bank_alpha_sensitivity/expected.json",
    "conformance/bayesian/prior_bank_ess/expected.json",
    "conformance/bayesian/prior_conjugate_moment_match/expected.json",
    "conformance/validate/bayesian_checks/expected.json",
    "conformance/validate/bayesian_checks/diagnostics_oracle.json",
    "crates/antecedent-prob/benches/laplace_glm.rs",
    "crates/antecedent-prob/benches/hmc.rs",
    "crates/antecedent-prob/benches/mcmc_stats.rs",
    "crates/antecedent-estimate/benches/posterior_functional.rs",
]

problems = pr.honesty_problems("parity/bayesian.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
pr.finish("Bayesian", problems, "Bayesian inventory evidence map OK")
PY

echo "== cargo test antecedent-prob / estimate bayesian / io posterior / bayesian conformance =="
bash scripts/counted_cargo.sh test -p antecedent-prob --lib
bash scripts/counted_cargo.sh test -p antecedent-prob --test prior_support_oracle
bash scripts/counted_cargo.sh test -p antecedent-prob --test mcmc_arviz_oracle
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib graph_posterior::
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib exact_enumeration::
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib structure_mcmc::
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib order_mcmc::
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib ci_screened_posterior::
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib dbn_posterior::
bash scripts/counted_cargo.sh test -p antecedent-discovery --test graph_mcmc_oracle
bash scripts/counted_cargo.sh test -p antecedent-estimate --lib bayesian
bash scripts/counted_cargo.sh test -p antecedent-estimate --lib envelope
bash scripts/counted_cargo.sh test -p antecedent-estimate --test bayesian_robust_ate_calibration
bash scripts/counted_cargo.sh test -p antecedent-learn --test bayesian_basis
bash scripts/counted_cargo.sh test -p antecedent-validate --lib bayesian_checks
bash scripts/counted_cargo.sh test -p antecedent-io --lib posterior
bash scripts/counted_cargo.sh test -p antecedent-io --lib prior_bank
bash scripts/counted_cargo.sh test -p antecedent-data --lib resample
bash scripts/counted_cargo.sh test -p antecedent --test prepared_analysis
bash scripts/counted_cargo.sh test -p antecedent --test bayesian
bash scripts/counted_cargo.sh test -p antecedent --test bayesian_iv_rd_staged
bash scripts/counted_cargo.sh test -p antecedent --test bayesian_robust_ate
bash scripts/counted_cargo.sh test -p antecedent --test bayesian_trial_transport
bash scripts/counted_cargo.sh test -p antecedent --test staged_bayesian_basis_gcomp
bash scripts/counted_cargo.sh test -p antecedent --test staged_attribution_transport_interference
bash scripts/counted_cargo.sh test -p antecedent --test class_posterior_ate
bash scripts/counted_cargo.sh test -p antecedent --lib bayesian_transport_providers_match_independent_dirichlet_moments
bash scripts/counted_cargo.sh test -p antecedent --test temporal_prior_transfer
bash scripts/counted_cargo.sh test -p antecedent --test manufacturing_temporal

echo "== criterion smoke (reuse gates) =="
cargo bench -p antecedent-prob --bench laplace_glm -- --test
cargo bench -p antecedent-prob --bench hmc -- --test
cargo bench -p antecedent-prob --bench mcmc_stats -- --test
cargo bench -p antecedent-estimate --bench posterior_functional -- --test

echo "== Python panel Bayesian facade smoke =="
python_smoke tests/test_panel_bayesian.py tests/test_temporal_bayesian_pulse.py tests/test_prior_bank.py tests/test_temporal_prior_transfer.py tests/test_bayesian_estimator_lifecycle.py tests/test_bayesian_likelihood.py tests/test_attribution_lifecycle.py tests/test_transport_interference_lifecycle.py tests/test_transport_statistical.py

echo "Bayesian gate PASSED"
