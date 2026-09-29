#!/usr/bin/env bash
# Bayesian gate: inventory honesty + Python smoke. Rust fixtures run in the Rust job.
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

# Rust suites and the Criterion smoke run in the Rust job and gate_release.sh.
echo "== Python panel Bayesian facade smoke =="
python_smoke tests/test_panel_bayesian.py tests/test_temporal_bayesian_pulse.py tests/test_prior_bank.py tests/test_temporal_prior_transfer.py tests/test_bayesian_estimator_lifecycle.py tests/test_bayesian_likelihood.py tests/test_attribution_lifecycle.py tests/test_transport_interference_lifecycle.py tests/test_transport_statistical.py

echo "Bayesian gate PASSED"
