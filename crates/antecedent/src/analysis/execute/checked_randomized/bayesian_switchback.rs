//! Bayesian switchback effect under a declared carryover and within-sequence dependence.
//!
//! The design-based switchback row is a sequence-level Horvitz-Thompson average whose interval is
//! a design-based statement and which identifies the direct effect only under no carryover and no
//! between-sequence interference. A posterior needs a likelihood for the dependence inside a
//! sequence, so this route requires one, declared with `BayesianConfig::design_model`:
//!
//! * the carryover (`None` or `FirstOrder`); unrestricted carryover has no likelihood in which
//!   the direct effect is identified and is refused with `cell_not_licensed`, naming first-order
//!   carryover and washout periods as the next attempts;
//! * the within-sequence dependence (independent, compound symmetric or autoregressive with a
//!   stated correlation); an unspecified dependence is refused with `cell_not_licensed`, naming
//!   the declared families and a sensitivity grid as the next attempt;
//! * the period adjustment and the diagonal Normal-Inverse-Gamma prior.
//!
//! The estimand is the constant direct effect of the declared model, with the carryover, the
//! sustained effect and the period contrasts where declared; it is not the Horvitz-Thompson
//! schedule average, which it equals when the effect is constant. Rows of a sequence must be in
//! period order. The route is explicitly unlicensed and uncalibrated, and its draws are the
//! exact Student-t marginals' quantile grid.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
    require_built_in_model, unread_option,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, IdentifierId,
    Instant, StudyResult, TabularData,
};
use crate::DesignPosteriorModel;
use antecedent_core::{RandomizationDesign, RandomizedEffectQuery};
use antecedent_estimate::gaussian_linear_posterior::StudentPosterior;
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, SequenceDependence, SwitchbackMarginals, SwitchbackModel, SwitchbackRows,
    posterior_switchback,
};
use antecedent_stats::student_t_ppf;

const ROUTE: &str = "the Bayesian switchback effect";
const CREDIBLE_LEVEL: f64 = 0.95;

/// Whether the query and configuration can run, and the declared model when they can.
pub(super) fn admit(
    query: &RandomizedEffectQuery,
    config: &super::super::BayesianConfig,
) -> Result<SwitchbackModel, CausalError> {
    require_built_in_model(config, ROUTE)?;
    if query.received_treatment.is_some()
        || query.fixed_cuped.is_some()
        || !query.ancova_covariates.is_empty()
        || query.estimand != antecedent_core::RandomizedEstimand::IntentionToTreat
        || query.exact_randomization_test
    {
        return Err(unread_option(
            ROUTE,
            "receipt adjustment, CUPED, ANCOVA or the exact randomization test",
            "the design-based switchback estimator under Frequentist inference",
        ));
    }
    let Some(DesignPosteriorModel::Switchback { model }) = &config.design_model else {
        return Err(crate::compile_reason!(
            "required_option_missing",
            "{ROUTE} needs a declared carryover, within-sequence dependence and prior: the \
             design-based sandwich has no likelihood and none is defaulted. Next attempt: \
             BayesianConfig::design_model(DesignPosteriorModel::Switchback {{ model }}) declaring \
             independent, compound-symmetric or autoregressive periods"
        ));
    };
    if model.carryover == CarryoverDeclaration::Unrestricted {
        return Err(crate::compile_reason!(
            "cell_not_licensed",
            "switchback_posterior.carryover_unrestricted: unrestricted treatment carryover has \
             no likelihood in which the direct effect is identified. Next attempt: declare \
             first-order carryover, or randomize with washout periods that make the no-carryover \
             declaration true"
        ));
    }
    if model.dependence == SequenceDependence::Unspecified {
        return Err(crate::compile_reason!(
            "cell_not_licensed",
            "switchback_posterior.dependence_undeclared: a posterior over switchback effects \
             needs a stated within-sequence dependence model; the design-based sandwich has no \
             likelihood. Next attempt: declare independent, compound-symmetric or autoregressive \
             periods and read the sensitivity over a grid of correlations"
        ));
    }
    Ok(*model)
}

/// Quantile-grid draws of a Student-t marginal.
fn grid(marginal: &StudentPosterior, draws: usize) -> Vec<f64> {
    (0..draws)
        .map(|i| {
            let u = (i as f64 + 0.5) / draws as f64;
            marginal.location + marginal.scale * student_t_ppf(u, marginal.degrees_of_freedom)
        })
        .collect()
}

/// The effect columns of the quantile-grid draws: direct, then the declared carryover, sustained
/// and period contrasts.
fn effect_columns(
    marginals: &SwitchbackMarginals,
    draws: usize,
) -> Vec<(antecedent_prob::PosteriorQuantityKind, Vec<f64>)> {
    let mut columns =
        vec![(effect_quantity("direct_effect"), grid(&marginals.direct_effect, draws))];
    if let Some(carryover) = &marginals.carryover_effect {
        columns.push((effect_quantity("carryover_effect"), grid(carryover, draws)));
    }
    if let Some(sustained) = &marginals.sustained_effect {
        columns.push((effect_quantity("sustained_effect"), grid(sustained, draws)));
    }
    for (j, period) in marginals.period_contrasts.iter().enumerate() {
        columns.push((effect_quantity(format!("period_{}_vs_0", j + 1)), grid(period, draws)));
    }
    columns
}

/// The retained identification states no carryover; a declared first-order carryover replaces
/// that premise with the declared one.
fn declared_identification(
    retained: &super::super::IdentificationResult,
    carryover: CarryoverDeclaration,
) -> super::super::IdentificationResult {
    let mut identification = retained.clone();
    if carryover == CarryoverDeclaration::FirstOrder {
        for record in &mut identification.required_assumptions.entries {
            let no_carryover = matches!(
                &record.assumption,
                antecedent_core::Assumption::Custom { id, .. }
                    if id.as_ref() == "switchback_no_carryover"
            );
            if no_carryover {
                record.assumption = antecedent_core::Assumption::Custom {
                    id: std::sync::Arc::from("switchback_first_order_carryover"),
                    description: std::sync::Arc::from(
                        "a unit-period outcome depends on its current assignment and the \
                         previous period's assignment through one coefficient; the first \
                         period of a sequence starts from control",
                    ),
                };
            }
        }
    }
    identification
}

impl super::CheckedRandomizedOperation {
    /// Execute the retained switchback design as a Bayesian posterior on complete outcomes.
    pub(super) fn execute_bayesian_switchback(
        &self,
        data: &TabularData,
        config: &super::super::BayesianConfig,
    ) -> Result<StudyResult, CausalError> {
        let model = admit(&self.query, config)?;
        if !matches!(self.query.design, RandomizationDesign::Switchback { .. }) {
            return Err(CausalError::Compile {
                message: "the Bayesian switchback posterior requires a switchback design".into(),
            });
        }
        if data.storage().weights().is_some() {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the Bayesian switchback posterior reads no sampling weights. Next attempt: \
                     the design-based switchback estimator, which reads them",
                ),
            ));
        }
        let outcomes = super::bayesian_arms::complete_outcomes(data, self.query.outcome)?;
        let sequence_ids: Vec<&str> =
            self.query.assignment_units.iter().map(AsRef::as_ref).collect();
        let started = Instant::now();
        let posterior = posterior_switchback(
            &SwitchbackRows {
                outcomes: &outcomes,
                assignment: &self.query.realized_assignment,
                sequence_ids: &sequence_ids,
            },
            &model,
            CREDIBLE_LEVEL,
        )?;
        let identity = DesignPosteriorModel::Switchback { model }.identity();
        let identification = declared_identification(&self.identification, model.carryover);
        let fit = DesignPosterior {
            columns: effect_columns(&posterior.marginals, config.n_draws),
            point: posterior.direct_effect.mean,
            point_sd: posterior.direct_effect.standard_deviation,
            n_obs: outcomes.len() as u64,
            interval_withheld: false,
            backend: "bayesian.switchback_effect",
            model: format!(
                "Gaussian independent sequences with the declared carryover and dependence; \
                 diagonal Normal-Inverse-Gamma prior; {} sequences",
                posterior.evidence.sequences
            ),
            assumptions: vec![
                (
                    "bayesian_switchback_dependence_declared",
                    "within-sequence errors are Gaussian with the declared correlation (declared, \
                     not estimated) and sequences are independent; the carryover is the declared \
                     one; rows of a sequence are in period order; there is no interference \
                     between sequences"
                        .to_owned(),
                ),
                (
                    "bayesian_switchback_prior_declared",
                    "the prior is declared with the study and retained in its inference identity; \
                     a prior alone does not identify the effect"
                        .to_owned(),
                ),
            ],
            diagnostics: vec![Diagnostic::new(
                "estimate.randomized.bayesian_switchback.estimand",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "estimand=constant direct effect of the declared carryover and dependence model; \
                 with heterogeneous effects it is a generalized-least-squares weighted average, \
                 not the Horvitz-Thompson schedule average; the posterior is conditional on the \
                 randomized assignment",
            )],
            notes: vec![
                format!("declared model {identity}"),
                "draws are the exact Student-t marginals' quantile grid, not Monte Carlo samples"
                    .to_owned(),
            ],
            sampler: None,
        };
        finish_design_posterior_as(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            fit,
            super::super::Strategy::new(
                IdentifierId::RandomizedDesign,
                EstimatorId::BayesianSwitchback,
            ),
        )
    }
}
