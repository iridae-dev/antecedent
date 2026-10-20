//! Posterior of a fixed binary policy's value under known randomized assignment.
//!
//! The Bayesian counterpart of the retained policy-value operation answers the same target as
//! its frequentist row: the net value of the fixed policy, of the fixed reference, and their
//! paired difference, under the declared assignment probabilities. The posterior is the
//! Bayesian-bootstrap functional of the per-subject augmented score. The declared probabilities
//! identify the value; the prior on the subject law does not. The route is unlicensed and its
//! calibration is unmeasured.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    assemble_posterior, effect_quantity, model_assumption, require_built_in_model, scalar_quantity,
    unread_option,
};
use super::super::{
    BayesianConfig, Diagnostic, DiagnosticKind, DiagnosticSeverity, InferenceDiagnostics,
};
use super::{CausalError, CheckedPolicyValueOperation, ExecutionContext};
use antecedent_estimate::policy_value::BinaryPolicyDesign;
use antecedent_estimate::policy_value_posterior::bayesian_binary_policy_value;
use antecedent_estimate::score_posterior::ScorePosteriorOptions;

/// Posterior backend of the policy-value posterior.
const BACKEND: &str = "bayesian_bootstrap.known_randomization_policy_value";

/// Uncertainty label of a policy-value result that carries a posterior.
pub(super) const BAYESIAN_POLICY_UNCERTAINTY: &str =
    "bayesian_bootstrap_posterior_known_randomization_no_interval";

/// Central coverage of the posterior's equal-tailed interval.
const COVERAGE: f64 = 0.95;

/// Refuse the policy shapes and options the posterior does not read, naming the next attempt.
pub(super) fn admit_bayesian_policy(
    query: &antecedent_core::PolicyValueQuery,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    const ROUTE: &str = "the Bayesian policy value";
    require_built_in_model(config, ROUTE)?;
    let next = "the frequentist binary policy value, which reads this option";
    if query.multi_action.is_some() {
        return Err(unread_option(ROUTE, "a multi-action policy", next));
    }
    if query.regret.is_some() || query.uplift_bin_count > 0 {
        return Err(unread_option(ROUTE, "finite-class regret or uplift bins", next));
    }
    if query.global_constraints_present {
        return Err(unread_option(ROUTE, "global capacity or budget constraints", next));
    }
    if !(query.mu0.is_empty()
        || query.disjoint_training_subjects
        || query.crossfit_fold_ownership_valid)
    {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the Bayesian policy value needs subject-excluded outcome predictions: declare \
             disjoint training subjects or valid cross-fit fold ownership. Next attempt: refit \
             the predictions on rows disjoint from the scored subjects, or omit them to use the \
             inverse-probability score"
        ));
    }
    Ok(())
}

/// The standing diagnostic of every policy-value posterior.
pub(super) fn unlicensed_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.policy_value.bayesian.unlicensed",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "unlicensed and uncalibrated Bayesian route: the Bayesian-bootstrap posterior of the \
         augmented score has no coverage evidence, the declared assignment probabilities (not \
         the prior) identify the value, and outcome predictions must be subject-excluded",
    )
}

impl CheckedPolicyValueOperation {
    /// Bayesian-bootstrap posterior of the policy value, its reference and their difference.
    pub(super) fn bayesian_policy_posterior(
        &self,
        y: &[f64],
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_estimate::CausalPosterior, CausalError> {
        let query = &self.query;
        let design = BinaryPolicyDesign {
            assignment: &query.assignment,
            actions: &query.actions,
            propensity: &query.propensity,
            reference: &query.reference,
            costs: &query.costs,
            reference_costs: &query.reference_costs,
        };
        let nuisance = if query.mu0.is_empty() { None } else { Some((&*query.mu0, &*query.mu1)) };
        let options = ScorePosteriorOptions { draws: config.n_draws, coverage: COVERAGE };
        let fit = bayesian_binary_policy_value(y, &design, nuisance, options, ctx)
            .map_err(CausalError::from)?;
        let columns = vec![
            (effect_quantity("incremental_value"), fit.incremental.draws.clone()),
            (scalar_quantity("policy_value"), fit.policy.draws.clone()),
            (scalar_quantity("reference_value"), fit.reference.draws.clone()),
        ];
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(model_assumption(
            BACKEND,
            "bayesian_bootstrap_known_randomization",
            "the subject law has a vanishing-concentration Dirichlet-process prior and the \
             posterior of each value is that of the mean augmented score under the declared \
             randomization; outcome predictions are frozen and subject-excluded",
        ));
        let mut inference = InferenceDiagnostics::analytic(BACKEND);
        let note = format!(
            "score={} draws={} interval=equal_tailed_{}",
            if query.mu0.is_empty() { "inverse_probability" } else { "augmented" },
            config.n_draws,
            (COVERAGE * 100.0).round()
        );
        inference.notes =
            [note].iter().map(|text| std::sync::Arc::<str>::from(text.as_str())).collect();
        assemble_posterior(&columns, &self.identification, assumptions, inference)
    }
}
