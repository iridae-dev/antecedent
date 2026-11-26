//! Posterior of a prespecified sequential regime's value under known sequential randomization.
//!
//! The Bayesian counterpart of the sequential doubly robust regime value answers the same target:
//! the mean terminal outcome if every subject followed the prespecified regime. The posterior is
//! the Bayesian-bootstrap functional of the backward-recursive subject scores, computed from the
//! declared sequential probabilities and frozen subject-excluded Q predictions. The declared
//! probabilities identify the value; the prior on the subject law does not.
//!
//! A two-period fit under known fair-coin randomization without censoring, with Q predictions
//! declared subject-excluded under the subjects' own folds, enough subjects and enough draws
//! states its support row in the design-posterior diagnostic. The row was measured with Q
//! predictions that are fixed functions of each subject's own history; whether declared
//! subject-excluded predictions were instead fitted on the other folds cannot be seen here,
//! and that construction is not covered. Every other fit runs and states no support.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    assemble_posterior, effect_quantity, model_assumption, require_built_in_model, unread_option,
};
use super::super::{
    BayesianConfig, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceDiagnostics,
    OverlapPolicy, Strategy, StudyResult, TabularData, finish_identified_execute_with_context,
};
use super::{
    CausalError, CheckedLongitudinalRegimeOperation, ExecutionContext, LongitudinalRegimeMethod,
    RegimeIntervals,
};
use antecedent_estimate::policy_value_posterior::bayesian_sequential_regime_value;
use antecedent_estimate::score_posterior::ScorePosteriorOptions;

/// Posterior backend of the sequential regime posterior.
const BACKEND: &str = "bayesian_bootstrap.known_sequential_randomization_regime_value";

/// Uncertainty label of a regime result that carries a posterior.
const BAYESIAN_REGIME_UNCERTAINTY: &str =
    "bayesian_bootstrap_posterior_known_sequential_randomization_no_interval";

/// Why a Bayesian regime result publishes no interval.
const BAYESIAN_REGIME_INTERVAL_REASON: &str = "bayesian_posterior_reports_no_interval";

/// Central coverage of the posterior's equal-tailed interval.
const COVERAGE: f64 = 0.95;

/// Code of the standing note of a posterior that states no support row.
const NO_ROW_CODE: &str = "estimate.longitudinal.bayesian_sequential_regime_value.unsupported";

/// Refuse the methods and options the posterior does not read, naming the next attempt.
pub(super) fn admit_bayesian_regime(
    query: &antecedent_core::LongitudinalRegimeQuery,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    const ROUTE: &str = "the Bayesian sequential regime value";
    require_built_in_model(config, ROUTE)?;
    if query.method != LongitudinalRegimeMethod::SequentialDoublyRobust {
        return Err(unread_option(
            ROUTE,
            "the IPW, g-formula and marginal structural model methods",
            "the sequential doubly robust method, which has a posterior, or the frequentist \
             method, which reads this one",
        ));
    }
    if !query.excluded_fold_predictions {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the Bayesian sequential regime value needs subject-excluded Q predictions: declare \
             excluded_fold_predictions. Next attempt: refit the Q predictions with each \
             subject's whole history held out of its own fit"
        ));
    }
    Ok(())
}

impl CheckedLongitudinalRegimeOperation {
    /// Execute the retained sequential regime as a Bayesian-bootstrap posterior.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        outcome: &[f64],
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let estimation = self.estimate_regime_value(outcome, data, ctx)?;
        let (Some(observation_history), Some(q_predictions)) =
            (self.query.observation_history.as_deref(), self.query.q_predictions.as_deref())
        else {
            return Err(CausalError::Unsupported {
                message: "sequential doubly robust regime value requires observation history and Q predictions",
            });
        };
        let histories = self.subject_histories();
        let sequential = antecedent_estimate::longitudinal_regime::SequentialDrInputs {
            actions: &self.query.regime_actions,
            observation_history,
            q_predictions,
        };
        let options = ScorePosteriorOptions { draws: config.n_draws, coverage: COVERAGE };
        let fit = bayesian_sequential_regime_value(outcome, &histories, &sequential, options, ctx)
            .map_err(CausalError::from)?;
        let columns = vec![(effect_quantity("regime_value"), fit.value.draws.clone())];
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(model_assumption(
            BACKEND,
            "bayesian_bootstrap_known_sequential_randomization",
            "the subject law has a vanishing-concentration Dirichlet-process prior and the \
             posterior of the regime value is that of the mean backward-recursive augmented \
             score under the declared sequential probabilities; Q predictions are frozen and \
             subject-excluded, so subject scores are only approximately independent",
        ));
        let mut inference = InferenceDiagnostics::analytic(BACKEND);
        let note = format!("draws={} interval=equal_tailed_{}", config.n_draws, COVERAGE * 100.0);
        inference.notes =
            [note].iter().map(|text| std::sync::Arc::<str>::from(text.as_str())).collect();
        let posterior = assemble_posterior(&columns, &self.identification, assumptions, inference)?;
        let n_draws = u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX);
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.identification.clone(),
                self.estimand.clone(),
                EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                Strategy::new(
                    IdentifierId::RandomizedDesign,
                    EstimatorId::BayesianSequentialRegimeValue,
                ),
                self.query.outcome,
                self.query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(vec![Diagnostic::new(
                NO_ROW_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian route with calibration status `not_measured`: the Bayesian-bootstrap posterior of \
                 the backward-recursive augmented score has no coverage record (calibration `not_measured`), the declared \
                 sequential probabilities (not the prior) identify the value, and Q predictions \
                 must be subject-excluded",
            )])
            .extras(IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(n_draws),
                ..IdentifiedExecuteExtras::default()
            }),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.support_status = None;
        let no_intervals = RegimeIntervals { g_formula: None, ipw: None, msm: None, dr: None };
        let mut payload = self.regime_payload(&estimation, &no_intervals, false);
        payload.value_standard_error = None;
        payload.uncertainty = std::sync::Arc::from(BAYESIAN_REGIME_UNCERTAINTY);
        payload.interval_reason = Some(std::sync::Arc::from(BAYESIAN_REGIME_INTERVAL_REASON));
        result.payload = crate::result::ResultPayload::LongitudinalRegime(payload);
        self.state_regime_posterior_row(&mut result);
        Ok(result)
    }

    /// State the support row of the posterior when the fit is the measured design: two
    /// periods of known randomization, no censoring, and Q predictions declared
    /// subject-excluded under the subjects' own folds. The row's thresholds then require the
    /// fair coin, the subjects and the draws.
    fn state_regime_posterior_row(&self, result: &mut StudyResult) {
        let query = &self.query;
        let uncensored = query.censoring_probabilities.iter().all(|kept| *kept >= 1.0)
            && query.outcome_observed.iter().all(|seen| *seen)
            && query
                .observation_history
                .as_deref()
                .is_some_and(|history| history.iter().all(|seen| *seen));
        let owned = query.excluded_fold_predictions
            && query.prediction_fold_ids.as_deref() == Some(query.fold_ids.as_ref());
        if query.periods != 2 || !query.probabilities_known_by_design || !uncensored || !owned {
            return;
        }
        // Smaller of the two action probabilities at any decision: 0.5 only for a fair coin.
        let floor = query
            .treatment_probabilities
            .iter()
            .map(|probability| probability.min(1.0 - probability))
            .fold(f64::INFINITY, f64::min);
        let draws = result.posterior.as_ref().map_or(0, |posterior| posterior.draws.n_draws);
        super::super::bayesian_quasi_common::state_row_with_resamples(
            result,
            crate::support::GraphlessSupportKey {
                family: "longitudinal_regime",
                design: "known_sequential_randomized_two_period",
                method: "subject_excluded_q_sequential_dr_bayesian_bootstrap",
                inference_claim: "conditional_q_pointwise_95_posterior_quantiles",
            },
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "unit",
                rows: query.outcome_observed.len(),
                min_probability: if floor.is_finite() { floor } else { 0.0 },
                interval_95_published: true,
                reported_intervals: 1,
                all_reported_intervals: true,
                ..Default::default()
            },
            antecedent_io::graphless_support::GraphlessResampleSupport {
                interval_resamples: draws,
                ..Default::default()
            },
        );
        if result.diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_ref() == antecedent_io::graphless_support::DESIGN_POSTERIOR_ROW_CODE
        }) {
            result.diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != NO_ROW_CODE);
        }
    }
}
