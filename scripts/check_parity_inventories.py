#!/usr/bin/env python3
"""Parity inventory honesty, run once.

Reads every parity inventory (`parity/<feature>.toml`) and checks that each `done` row has its
evidence on disk, that the exit artifacts exist and are non-empty, that no row is left in a retired
status, and that the release inventory names real evidence. Static: no build, no test execution.

    python3 scripts/check_parity_inventories.py
"""

import sys
from pathlib import Path

import gate_hot_path_baselines as hp
import parity_rows as pr
from _rust_source import has_lint_attribute

FORBID_CRATES = [
    "crates/antecedent-core",
    "crates/antecedent-contract",
    "crates/antecedent-graph",
    "crates/antecedent-expr",
    "crates/antecedent-identify",
    "crates/antecedent-io",
    "crates/antecedent-stats",
    "crates/antecedent-prob",
    "crates/antecedent-estimate",
    "crates/antecedent-validate",
    "crates/antecedent-model",
    "crates/antecedent-counterfactual",
    "crates/antecedent-attribution",
    "crates/antecedent-design",
    "crates/antecedent-state",
    "crates/antecedent-discovery",
    "crates/antecedent-learn",
    "crates/antecedent-learn-burn",
    "crates/antecedent-test-support",
    "crates/antecedent",
]
def forbid_unsafe_problems(root: Path) -> list[str]:
    problems = []
    for crate in FORBID_CRATES:
        lib = root / crate / "src" / "lib.rs"
        if not lib.is_file():
            problems.append(f"{crate} missing src/lib.rs")
            continue
        if not has_lint_attribute(lib.read_text(encoding="utf-8"), "forbid", "unsafe_code", inner=True):
            problems.append(f"{crate} missing #![forbid(unsafe_code)]")
    return problems


def release_problems() -> list[str]:
    root = Path(".")
    missing: list[str] = []

    # Inventories: forbid the retired waiver status and rows that never closed.
    for manifest in [
        "parity/estimate.toml",
        "parity/discovery.toml",
        "parity/bayesian.toml",
        "parity/pag.toml",
        "parity/context.toml",
        "parity/design_state.toml",
        "parity/gcm.toml",
        "parity/attribution.toml",
        "parity/response.toml",
        "parity/compiler.toml",
    ]:
        for c in pr.rows(manifest):
            if c.get("status") == "intentional_deviation":
                missing.append(f"{manifest}: {c.get('id')} still intentional_deviation (retired)")
            if c.get("status") in ("planned", "blocked"):
                missing.append(f"{manifest}: {c.get('id')} still {c.get('status')}")

    # Evidence is a TOML (or code) record. ADR and doc markdown are not evidence.
    EVIDENCE = {
        "release.parity_closure": "parity/release.toml",
        "release.graph_dot_json": "crates/antecedent-io-core/src/graph_gml.rs",
        "release.artifact_schema": "crates/antecedent-io-core/src/migrate.rs",
        "release.artifact_mmap_stream_skip": "crates/antecedent-io-core/src/reader.rs",
        "release.wheel_matrix": ".github/workflows/ci.yml",
        "release.conformance_docs": "parity/oracle_closure.toml",
        "release.hot_path_baselines": "benches/baselines/baselines.toml",
        "release.dependency_policy": "deny.toml",
        "release.ci_required_jobs": ".github/workflows/ci.yml",
    }

    missing += pr.honesty_problems("parity/release.toml", EVIDENCE, all_done=True)

    for path in [
        "parity/release.toml",
        "parity/support_axes.toml",
        "parity/support_cells.toml",
        "parity/support_n_a.toml",
        "parity/support_refusal_reasons.toml",
        "parity/oracle_closure.toml",
        "benches/baselines/baselines.toml",
        "deny.toml",
        "conformance/interchange/graph_dot_json/expected.json",
        "conformance/interchange/graph_gml_networkx/expected.json",
        "conformance/interchange/artifact_migrate/expected.json",
        "crates/antecedent-io-core/src/graph_dot.rs",
        "crates/antecedent-io-core/src/graph_gml.rs",
        "crates/antecedent-io-core/src/graph_networkx.rs",
        "crates/antecedent-io-core/src/graph_json.rs",
        "crates/antecedent-io-core/src/migrate.rs",
        "crates/antecedent-io/src/model_bundle.rs",
        "scripts/generate_conformance_docs.py",
    ]:
        if not (root / path).exists():
            missing.append(f"required exit artifact missing: {path}")

    # Semantic crates: forbid unsafe by default. antecedent-data / antecedent-io keep
    # #![deny(unsafe_code)] with scoped allows (Arrow FFI / foreign buffers / mmap, which
    # lives in antecedent-io-core; antecedent-io itself forbids unsafe).
    deny_escape_crates = {
        "crates/antecedent-data": ("buffer.rs", "arrow_ffi.rs"),
        "crates/antecedent-io-core": ("mmap_file.rs",),
    }
    missing += forbid_unsafe_problems(root)
    for crate, allow_mods in deny_escape_crates.items():
        lib = root / crate / "src" / "lib.rs"
        text = lib.read_text(encoding="utf-8")
        if not has_lint_attribute(text, "deny", "unsafe_code", inner=True):
            missing.append(f"{crate} missing #![deny(unsafe_code)] (scoped unsafe escape)")
        if not has_lint_attribute(text, "allow", "unsafe_code") and not any(
            has_lint_attribute(
                (root / crate / "src" / m).read_text(encoding="utf-8"),
                "allow",
                "unsafe_code",
            )
            for m in allow_mods
            if (root / crate / "src" / m).exists()
        ):
            missing.append(f"{crate} missing allow(unsafe_code) for scoped escape modules")
        for mod_name in allow_mods:
            if not (root / crate / "src" / mod_name).exists():
                missing.append(f"{crate} expected unsafe escape module missing: {mod_name}")

    # Required baselines are [[baseline]] rows with required=true. The name list
    # is the registry, not a second copy here, and not a scrape of hot_paths.md.
    missing += hp.registry_problems()

    return missing


ESTIMATE_CI_EVIDENCE = {
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


def estimate_ci_problems() -> list[str]:
    problems = pr.evidence_map_problems(ESTIMATE_CI_EVIDENCE, ["parity/estimate.toml", "parity/discovery.toml"])
    return problems


BAYESIAN_EVIDENCE = {
    "estimate.bayesian_conditional": "crates/antecedent/tests/prepared_analysis.rs",
    "estimate.bayesian_temporal_mediation": "python/tests/test_posterior_moments.py",
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
    "bayes.estimate.panel_did": "crates/antecedent/tests/bayesian_design_routes.rs",
    "bayes.estimate.synthetic_control": "crates/antecedent/tests/bayesian_design_routes.rs",
    "bayes.estimate.local_ratio": "crates/antecedent/tests/bayesian_design_routes.rs",
    "bayes.estimate.interference_neighbor_count": "crates/antecedent/tests/staged_attribution_transport_interference.rs",
    "bayes.estimate.trial_to_target": "crates/antecedent/tests/bayesian_trial_transport.rs",
    "bayes.transport.empirical_and_state_space_laws": "crates/antecedent/src/analysis/statistical.rs",
    "bayes.transport.z_cited_laws": "crates/antecedent-estimate/tests/z_transport_cited_evidence.rs",
    "bayes.response.simultaneous_joint_band": "crates/antecedent-estimate/src/response/tests.rs",
    "bayes.validate.sbc_glm_families": "crates/antecedent-validate/src/bayesian_checks/tests.rs",
    "bayes.validate.ppc": "crates/antecedent-validate/src/bayesian_checks/tests.rs",
    "bayes.validate.prior_sensitivity": "crates/antecedent-validate/src/bayesian_checks/tests.rs",
    "bayes.data.bayesian_bootstrap": "provenance/data.bayesian_bootstrap.toml",
    "bayes.io.posterior_artifact": "crates/antecedent-io-core/src/posterior.rs",
    "bayes.io.posterior_artifact_summary_only": "crates/antecedent-io-core/src/posterior.rs",
    "bayes.facade.inference_mode": "crates/antecedent/src/inference.rs",
    "bayes.model.pcm_scm_registry": "crates/antecedent-model/src/lib.rs",
    "bayes.discovery.dag_posterior": "crates/antecedent-discovery/tests/dag_posterior_conformance.rs",
    "bayes.backend.hierarchical_bvar_gp": "crates/antecedent-model/src/registry/mod.rs",
    "bayes.validate.mcmc_diagnostics": "crates/antecedent-prob/tests/mcmc_arviz_oracle.rs",
    "bayes.validate.full_pair_ess": "crates/antecedent-prob/tests/mcmc_arviz_oracle.rs",
    "bayes.ci.tests": "crates/antecedent-stats/src/ci/bayes.rs",
    "bayes.prior_bank.temporal_transfer": "crates/antecedent/tests/temporal_prior_transfer.rs",
    "bayes.prior_bank.catalog": "crates/antecedent-io-core/src/prior_bank.rs",
    "bayes.panel.random_intercept_non_gaussian": "crates/antecedent-estimate/src/bayesian/tests.rs",
    "bayes.gcomp.negative_binomial_se": "crates/antecedent-estimate/src/glm_adjustment.rs",
    "bayes.prior_bank.effect_map": "crates/antecedent-estimate/src/bayesian/hydrate.rs",
    "bayes.prior_bank.power_mixture": "crates/antecedent-prob/src/external_prior.rs",
    "bayes.prior_bank.conflict": "crates/antecedent-validate/src/conflict.rs",
    "bayes.prior_bank.transport": "crates/antecedent-prob/src/transport.rs",
    "bayes.prior_bank.ess_accounting": "crates/antecedent-prob/src/external_prior.rs",
    "bayes.prior_bank.conjugate_moment_match": "crates/antecedent-prob/src/conjugate_moment_match.rs",
}

BAYESIAN_EXIT_ARTIFACTS = [
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
    "conformance/validate/bayesian_checks/independent_ips.json",
    "conformance/validate/bayesian_checks/reference.py",
    "conformance/validate/bayesian_checks/reference.py.lock",
    "crates/antecedent-prob/benches/laplace_glm.rs",
    "crates/antecedent-prob/benches/hmc.rs",
    "crates/antecedent-prob/benches/mcmc_stats.rs",
    "crates/antecedent-estimate/benches/posterior_functional.rs",
]



def bayesian_problems() -> list[str]:
    problems = pr.honesty_problems("parity/bayesian.toml", BAYESIAN_EVIDENCE)
    problems += pr.exit_artifact_problems(BAYESIAN_EXIT_ARTIFACTS)
    return problems


GCM_EVIDENCE = {
    "gcm.model.compiled_plans": "crates/antecedent-model/src/compile.rs",
    "gcm.model.mechanisms": "crates/antecedent-model/src/mechanism.rs",
    "gcm.model.registry_fit": "crates/antecedent-model/src/registry/mod.rs",
    "gcm.model.mechanism_selection": "crates/antecedent-model/tests/mechanism_selection.rs",
    "gcm.model.sampling": "crates/antecedent-model/src/sample.rs",
    "gcm.model.noise_replay": "crates/antecedent-model/tests/noise_replay.rs",
    "gcm.model.intervention_programs": "crates/antecedent-model/tests/intervention_programs.rs",
    "gcm.model.policy_validation": "crates/antecedent-model/tests/intervention_programs.rs",
    "gcm.model.policy_workflows": "crates/antecedent/tests/policy_python_parity.rs",
    "gcm.model.prepared_policy": "crates/antecedent/tests/policy_python_parity.rs",
    "gcm.do_sampling": "crates/antecedent/tests/gcm.rs",
    # Additive shift do(X := X + delta): overlay accumulation is the primary
    # implementation. Harness: crates/antecedent-model/src/overlay.rs
    # (`overlay::tests::shift_overlay_accumulates_and_is_independent_of_hard_set`,
    # via `cargo test -p antecedent-model --lib` in the Rust job) plus
    # crates/antecedent/tests/gcm.rs::gcm_shift_intervention_differs_from_hard_set
    # (via `cargo test -p antecedent --test gcm` in the Rust job), which shows shift and hard
    # set are observably different on a linear-Gaussian fixture.
    "gcm.do_sampling.shift": "crates/antecedent-model/src/overlay.rs",
    "gcm.model.falsification": "crates/antecedent-model/src/evaluate.rs",
    "gcm.counterfactual.aap": "crates/antecedent/tests/gcm.rs",
    "gcm.attribution.basic": "crates/antecedent/tests/gcm.rs",
}

GCM_EXIT_ARTIFACTS = [
    "conformance/gcm/gcm_fit_intervene/expected.json",
    "conformance/gcm/gcm_anomaly/expected.json",
    "conformance/gcm/gcm_cf_ite/expected.json",
    "conformance/gcm/do_sampling_weighting/expected.json",
    "conformance/gcm/do_sampling_kde/expected.json",
    "conformance/gcm/do_sampling_mcmc/expected.json",
    "crates/antecedent-model/benches/sample_overlay.rs",
    "crates/antecedent-counterfactual/benches/counterfactual_batch.rs",
    "parity/gcm.toml",
]

# Only the GCM/CF evidence set is gated here (the attribution inventory is gated separately).


def gcm_problems() -> list[str]:
    problems = pr.evidence_map_problems(GCM_EVIDENCE, ["parity/gcm.toml"])
    problems += pr.exit_artifact_problems(GCM_EXIT_ARTIFACTS)
    problems += pr.require_done("parity/estimate.toml", ("gcm.surface", "gcm.do_sampling"), "GCM")
    problems += pr.require_done("parity/bayesian.toml", ("bayes.model.pcm_scm_registry",), "GCM")
    return problems


PAG_EVIDENCE = {
    "pag.graph.admg": "crates/antecedent-graph/src/admg.rs",
    "pag.graph.pag_temporal": "crates/antecedent-graph/src/pag.rs",
    "pag.graph.m_separation": "crates/antecedent-graph/src/msep.rs",
    "pag.graph.latent_projection": "crates/antecedent/tests/pag.rs",
    "pag.graph.completions_streamed": "crates/antecedent-graph/src/completion.rs",
    "pag.graph.cpdag_mec_completions": "crates/antecedent-graph/src/cpdag_completion.rs",
    "pag.identify.generalized_adjustment": "crates/antecedent/tests/pag.rs",
    "pag.identify.full_id_idc": "crates/antecedent-identify/src/id.rs",
    "pag.identify.response_visibility_id": "crates/antecedent-identify/src/mag_id_bruteforce.rs",
    "pag.discovery.lpcmci": "crates/antecedent/tests/pag.rs",
    "pag.discovery.fci_rfci": "crates/antecedent-discovery/src/fci.rs",
    "pag.facade.dag_only_reject": "crates/antecedent/tests/pag.rs",
}

PAG_EXIT_ARTIFACTS = [
    "conformance/graph/cpdag_operations/expected.json",
    "conformance/graph/pag_operations/expected.json",
    "conformance/graph/definite_status_separation/expected.json",
    "conformance/graph/latent_projection/expected.json",
    "conformance/graph/pag_mag_completion/expected.json",
    "conformance/identify/efficient_adjustment/expected.json",
    "conformance/identify/id_hedge/expected.json",
    "conformance/identify/idc/expected.json",
    "conformance/identify/generalized_adjustment/expected.json",
    "conformance/identify/mag_visibility_id/expected.json",
    "conformance/identify/path_specific/expected.json",
    "conformance/identify/auto_envelopes/expected.json",
    "conformance/discovery/rfci/expected.json",
    "conformance/pag/lpcmci_chain/expected.json",
    "conformance/pag/latent_projection_msep/expected.json",
    "conformance/pag/envelope_unidentified_mass/expected.json",
    "conformance/pag/dag_only_pag_reject/expected.json",
    "crates/antecedent-graph/benches/mseparation.rs",
    "crates/antecedent-discovery/benches/pag_orientation.rs",
    "benches/baselines/baselines.toml",
    "parity/pag.toml",
]



def pag_problems() -> list[str]:
    problems = pr.honesty_problems("parity/pag.toml", PAG_EVIDENCE)
    problems += pr.exit_artifact_problems(PAG_EXIT_ARTIFACTS)
    problems += hp.missing_ids(("pag",))
    return problems


CONTEXT_EVIDENCE = {
    "validate.functional": "crates/antecedent-validate/src/functional.rs",
    "validate.temporal_mediation": "crates/antecedent/tests/prepared_analysis.rs",
    "context.panel_data": "crates/antecedent-data/src/panel.rs",
    "context.context_graph": "crates/antecedent-graph/src/cpdag.rs",
    "context.jpcmci_plus": "crates/antecedent/tests/context_effects.rs",
    "context.rpcmci": "crates/antecedent/tests/context_effects.rs",
    "context.mediation": "crates/antecedent/tests/context_effects.rs",
    "context.mediation.nonparametric": "crates/antecedent-identify/src/path_specific.rs",
    "context.conditional": "crates/antecedent/tests/context_effects.rs",
    "context.prediction": "crates/antecedent/tests/context_effects.rs",
    "context.query_model_planned_variants": "crates/antecedent-io-core/src/query_wire.rs",
}

CONTEXT_EXIT_ARTIFACTS = [
    "conformance/discovery/jpcmci_plus_two_env/expected.json",
    "conformance/discovery/jpcmci_plus_two_env_space_dummy_mv/expected.json",
    "conformance/discovery/rpcmci_two_regime/expected.json",
    "conformance/context/temporal_mediation/expected.json",
    "conformance/context/conditional_effect/expected.json",
    "conformance/context/prediction_smoke/expected.json",
    "crates/antecedent-discovery/benches/rpcmci.rs",
    "crates/antecedent-estimate/benches/temporal_mediation.rs",
    "benches/baselines/baselines.toml",
    "parity/context.toml",
]



def context_problems() -> list[str]:
    problems = pr.honesty_problems("parity/context.toml", CONTEXT_EVIDENCE)
    problems += pr.exit_artifact_problems(CONTEXT_EXIT_ARTIFACTS)
    problems += hp.missing_ids(("regime_mediation",))
    problems += pr.require_done(
        "parity/discovery.toml",
        ("discovery.jpcmci_plus", "discovery.rpcmci", "discovery.effects"),
        "Context",
    )
    problems += pr.require_done("parity/estimate.toml", ("estimate.conditional",), "Context")
    return problems


ATTRIBUTION_EVIDENCE = {
    "attribution.shapley": "crates/antecedent-attribution/src/shapley.rs",
    "attribution.coalition_cache": "crates/antecedent-attribution/src/coalition.rs",
    "attribution.distribution_change": "crates/antecedent-attribution/src/distribution_change.rs",
    "attribution.distribution_change_robust": "crates/antecedent-attribution/src/robust.rs",
    "attribution.mechanism_change_detection": "crates/antecedent-attribution/src/mechanism_change.rs",
    "attribution.mechanism_change_kernel": "crates/antecedent-stats/src/divergence.rs",
    "attribution.mechanism_change_change_point": "crates/antecedent-stats/src/divergence.rs",
    "attribution.unit_change": "crates/antecedent-attribution/src/unit_change.rs",
    "attribution.path_decompose": "crates/antecedent-attribution/src/path.rs",
    "attribution.feature_relevance": "crates/antecedent-attribution/src/feature_relevance.rs",
    "attribution.root_cause": "crates/antecedent-attribution/src/root_cause.rs",
    "attribution.structure": "crates/antecedent-attribution/src/structure_change.rs",
    "attribution.facade": "crates/antecedent/tests/attribution.rs",
}

ATTRIBUTION_EXIT_ARTIFACTS = [
    "conformance/attribution/path_allocation/expected.json",
    "conformance/attribution/distribution_change_grid/expected.json",
    "conformance/attribution/structure_change_grid/expected.json",
    "conformance/attribution/mechanism_unit_change/expected.json",
    "conformance/attribution/anomaly_root_cause/expected.json",
    "conformance/attribution/arrow_strength/expected.json",
    "conformance/validate/refuters/expected.json",
    "conformance/validate/confounding_sensitivity/expected.json",
    "conformance/validate/riesz_sensitivity/expected.json",
    "conformance/validate/overlap_graph_refutation/expected.json",
    "conformance/attribution/distribution_change_y_shift/expected.json",
    "conformance/attribution/structure_change_parent_swap/expected.json",
    "conformance/attribution/mechanism_change_detect/expected.json",
    "conformance/attribution/mechanism_change_kernel_shift/expected.json",
    "conformance/attribution/mechanism_change_change_point/expected.json",
    "crates/antecedent-attribution/benches/shapley.rs",
    "benches/baselines/baselines.toml",
    "parity/attribution.toml",
]



def attribution_problems() -> list[str]:
    problems = pr.honesty_problems("parity/attribution.toml", ATTRIBUTION_EVIDENCE)
    problems += pr.exit_artifact_problems(ATTRIBUTION_EXIT_ARTIFACTS)
    problems += hp.missing_ids(("shapley",))
    problems += pr.require_done(
        "parity/gcm.toml",
        (
            "gcm.attribution.shapley",
            "gcm.attribution.distribution_change",
            "gcm.attribution.robust",
            "gcm.attribution.unit_change",
            "gcm.attribution.feature_relevance",
            "gcm.attribution.structure",
        ),
        "Attribution",
    )
    return problems


DESIGN_STATE_EVIDENCE = {
    "design_state.candidate_types": "crates/antecedent-design/src/candidate.rs",
    "design_state.eig": "crates/antecedent-design/src/ranker.rs",
    "design_state.id_probability": "crates/antecedent-design/src/ranker.rs",
    "design_state.effect_width": "crates/antecedent-design/src/ranker.rs",
    "design_state.decision_utility": "crates/antecedent-design/src/decision.rs",
    "design_state.design_ranker": "crates/antecedent-design/src/ranker.rs",
    "design_state.antecedent_state": "crates/antecedent-state/src/state.rs",
    "design_state.incremental_ols": "crates/antecedent-state/src/suff_stats.rs",
    "design_state.streaming_cov": "crates/antecedent-state/src/suff_stats.rs",
    "design_state.cache_budget": "crates/antecedent-state/src/store.rs",
    "design_state.facade": "crates/antecedent/tests/design_state.rs",
    "design_state.incremental.particle_graph_score": "crates/antecedent-state/src/graph_score.rs",
    "design_state.rolling_mechanism_diagnostics": "crates/antecedent-state/src/mechanism_diag.rs",
}

DESIGN_STATE_EXIT_ARTIFACTS = [
    "conformance/design_state/rank_candidates_eig/expected.json",
    "conformance/design_state/incremental_ols_match/expected.json",
    "conformance/design_state/incremental_graph_score_match/expected.json",
    "conformance/design_state/incremental_particle_filter_match/expected.json",
    "conformance/design_state/rolling_mechanism_diag_match/expected.json",
    "crates/antecedent-design/benches/design_rank.rs",
    "crates/antecedent-state/benches/state_append.rs",
    "benches/baselines/baselines.toml",
    "parity/design_state.toml",
    "provenance/design.eig.toml",
    "provenance/state.incremental_ols.toml",
    "provenance/state.incremental.particle_graph_score.toml",
]



def design_state_problems() -> list[str]:
    problems = pr.honesty_problems("parity/design_state.toml", DESIGN_STATE_EVIDENCE)
    problems += pr.exit_artifact_problems(DESIGN_STATE_EXIT_ARTIFACTS)
    problems += hp.missing_ids(("design_state",))
    return problems


INVENTORIES = (
    ("release", release_problems),
    ("estimate and CI", estimate_ci_problems),
    ("Bayesian", bayesian_problems),
    ("GCM", gcm_problems),
    ("PAG", pag_problems),
    ("context", context_problems),
    ("attribution", attribution_problems),
    ("design state", design_state_problems),
)


def main() -> int:
    failed = False
    for label, check in INVENTORIES:
        problems = check()
        if problems:
            failed = True
            print(f"{label} inventory FAILED:")
            for problem in problems:
                print(" -", problem)
    if failed:
        return 1
    print(f"parity inventories OK ({len(INVENTORIES)} inventories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
