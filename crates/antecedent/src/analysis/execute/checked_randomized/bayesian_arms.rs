//! Bayesian arm contrasts of the randomized designs (two arms, many arms, the 2x2 factorial).
//!
//! The design-based rows estimate the average of the unit-level potential outcomes over the
//! trial's own units, with a randomization-distribution interval that is not a posterior. This
//! route states a likelihood (independent Bernoulli or Gaussian arms) and a declared prior and
//! answers one of two estimands, chosen by the estimator id: the trial's own units
//! (`bayesian.randomized_finite_population`, the same target as the design-based row) or the
//! population the units are sampled from (`bayesian.randomized_superpopulation`). The prior is
//! declared with `BayesianConfig::design_model`; it has no default, because a prior alone never
//! identifies a contrast. Assignment probabilities must be known and identical across units.
//!
//! The route is explicitly unlicensed and uncalibrated: no coverage evidence covers the credible
//! interval, and the draws are the exact posterior's quantile grid, not Monte Carlo samples.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
    require_built_in_model, unread_option,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EstimatorId, ExecutionContext,
    IdentifierId, InferenceMode, Instant, StudyResult, TableView, TabularData,
};
use crate::DesignPosteriorModel;
use antecedent_core::{RandomizationDesign, RandomizedEffectQuery};
use antecedent_estimate::randomized_posterior::{
    ArmContrastRequest, ArmOutcomeModel, RandomizedEstimand, RandomizedPosterior,
    posterior_arm_contrasts_with_draws,
};

const ROUTE: &str = "the Bayesian randomized arm contrast";
const CREDIBLE_LEVEL: f64 = 0.95;

/// The Bayesian estimator a randomized query runs under Bayesian inference: the superpopulation
/// posterior when it was selected, otherwise the finite-population posterior, which answers the
/// same target as the design-based row.
pub(in crate::analysis::execute) fn bayesian_randomized_estimator(
    selected: Option<EstimatorId>,
    design: &RandomizationDesign,
) -> EstimatorId {
    if matches!(design, RandomizationDesign::Switchback { .. }) {
        return EstimatorId::BayesianSwitchback;
    }
    match selected {
        Some(EstimatorId::BayesianRandomizedSuperpopulation) => {
            EstimatorId::BayesianRandomizedSuperpopulation
        }
        _ => EstimatorId::BayesianRandomizedFinitePopulation,
    }
}

/// Whether the query and configuration can run, and the declared arm model when they can.
pub(super) fn admit(
    query: &RandomizedEffectQuery,
    config: &super::super::BayesianConfig,
) -> Result<ArmOutcomeModel, CausalError> {
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
            "the design-based randomized estimators, which read them, under Frequentist inference",
        ));
    }
    if !matches!(
        query.design,
        RandomizationDesign::Bernoulli
            | RandomizationDesign::Complete { .. }
            | RandomizationDesign::Factorial2x2 { .. }
            | RandomizationDesign::MultiArm { .. }
    ) {
        return Err(unread_option(
            ROUTE,
            "a cluster, stratified or switchback assignment design",
            "a Bernoulli, complete, multi-arm or 2x2 factorial design, or the design-based \
             estimator of this design under Frequentist inference",
        ));
    }
    match &config.design_model {
        Some(DesignPosteriorModel::ArmOutcome { model }) => Ok(*model),
        _ => Err(crate::compile_reason!(
            "required_option_missing",
            "{ROUTE} needs a declared arm outcome model and prior: a prior alone does not \
             identify a contrast and none is defaulted. Next attempt: \
             BayesianConfig::design_model(DesignPosteriorModel::ArmOutcome {{ model }}) with a \
             Beta prior for 0/1 outcomes or a Normal-Inverse-Gamma prior for real outcomes"
        )),
    }
}

/// Assigned arms, known probabilities and declared contrasts of a query.
struct ArmRows {
    assignment: Vec<usize>,
    probabilities: Vec<Vec<f64>>,
    labels: Vec<String>,
    contrasts: Vec<(String, Vec<f64>)>,
}

fn arm_rows(query: &RandomizedEffectQuery) -> Result<ArmRows, CausalError> {
    match &query.design {
        RandomizationDesign::Bernoulli | RandomizationDesign::Complete { .. } => Ok(ArmRows {
            assignment: query.realized_assignment.iter().map(|t| usize::from(*t)).collect(),
            probabilities: query
                .assignment_probabilities
                .iter()
                .map(|p| vec![1.0 - p, *p])
                .collect(),
            labels: vec![query.treatment_arms.0.to_string(), query.treatment_arms.1.to_string()],
            contrasts: vec![("itt".to_owned(), vec![-1.0, 1.0])],
        }),
        RandomizationDesign::MultiArm { assignment, probabilities, arms } => {
            let labels: Vec<String> = arms.iter().map(ToString::to_string).collect();
            let reference = labels.first().cloned().unwrap_or_default();
            let contrasts = labels
                .iter()
                .enumerate()
                .skip(1)
                .map(|(k, label)| {
                    let mut weights = vec![0.0; labels.len()];
                    if let Some(w) = weights.first_mut() {
                        *w = -1.0;
                    }
                    if let Some(w) = weights.get_mut(k) {
                        *w = 1.0;
                    }
                    (format!("{label}_vs_{reference}"), weights)
                })
                .collect();
            Ok(ArmRows {
                assignment: assignment.to_vec(),
                probabilities: probabilities.to_vec(),
                labels,
                contrasts,
            })
        }
        RandomizationDesign::Factorial2x2 { second_factor_assignment, cell_counts, .. } => {
            let n = query.realized_assignment.len();
            // Declared cell order is (A=0,B=0), (A=1,B=0), (A=0,B=1), (A=1,B=1); the arm index
            // is 2 a + b and a unit's arm probability is its cell's share of the trial.
            let share = |count: usize| count as f64 / n as f64;
            let row = vec![
                share(cell_counts[0]),
                share(cell_counts[2]),
                share(cell_counts[1]),
                share(cell_counts[3]),
            ];
            Ok(ArmRows {
                assignment: query
                    .realized_assignment
                    .iter()
                    .zip(second_factor_assignment.iter())
                    .map(|(a, b)| 2 * usize::from(*a) + usize::from(*b))
                    .collect(),
                probabilities: vec![row; n],
                labels: ["a0b0", "a0b1", "a1b0", "a1b1"].map(str::to_owned).to_vec(),
                contrasts: vec![
                    ("factor_a_main_effect".to_owned(), vec![-0.5, -0.5, 0.5, 0.5]),
                    ("factor_b_main_effect".to_owned(), vec![-0.5, 0.5, -0.5, 0.5]),
                    ("interaction".to_owned(), vec![1.0, -1.0, -1.0, 1.0]),
                ],
            })
        }
        _ => Err(unread_option(ROUTE, "this assignment design", "a modelled design")),
    }
}

fn prior_text(model: ArmOutcomeModel) -> String {
    match model {
        ArmOutcomeModel::Bernoulli(prior) => {
            format!("Bernoulli arms, Beta({}, {}) prior", prior.alpha, prior.beta)
        }
        ArmOutcomeModel::Gaussian(prior) => format!(
            "Gaussian arms, Normal-Inverse-Gamma prior (mean {}, pseudo-units {}, shape {}, scale {})",
            prior.mean, prior.pseudo_units, prior.variance_shape, prior.variance_scale
        ),
    }
}

/// The outcome column when every row is observed and finite.
pub(super) fn complete_outcomes(
    data: &TabularData,
    outcome: super::super::VariableId,
) -> Result<Vec<f64>, CausalError> {
    let antecedent_data::ColumnView::Float64(column) =
        data.column(outcome).map_err(CausalError::from)?
    else {
        return Err(CausalError::Unsupported { message: "randomized outcome must be continuous" });
    };
    let outcomes = column.values.as_slice();
    if (0..outcomes.len()).any(|i| !column.validity.is_valid(i))
        || outcomes.iter().any(|y| !y.is_finite())
    {
        return Err(unread_option(
            ROUTE,
            "missing or non-finite outcomes",
            "complete the outcome rows, or the design-based responder estimators under \
             Frequentist inference",
        ));
    }
    Ok(outcomes.to_vec())
}

/// The posterior columns, point, premises and diagnostics of one arm-contrast fit.
fn arm_design_posterior(
    rows: &ArmRows,
    posterior: &RandomizedPosterior,
    model: ArmOutcomeModel,
    estimand: RandomizedEstimand,
    n_obs: usize,
) -> Result<DesignPosterior, CausalError> {
    let Some(primary) = posterior.contrasts.first() else {
        return Err(CausalError::Compile { message: "no contrast was requested".into() });
    };
    let estimand_name = match estimand {
        RandomizedEstimand::Superpopulation => "superpopulation",
        RandomizedEstimand::FinitePopulation => "finite_population",
    };
    let statement = match estimand {
        RandomizedEstimand::Superpopulation => {
            "contrasts of the arm outcome means in the population the trial's units are \
             sampled from"
        }
        RandomizedEstimand::FinitePopulation => {
            "contrasts of the arm means of the potential outcomes over the trial's own units, \
             the unobserved share of each arm drawn from the arm's posterior predictive"
        }
    };
    Ok(DesignPosterior {
        columns: rows
            .contrasts
            .iter()
            .zip(&posterior.contrasts)
            .map(|((name, _), c)| (effect_quantity(name), c.draws.clone()))
            .collect(),
        point: primary.mean,
        point_sd: primary.standard_deviation,
        n_obs: n_obs as u64,
        interval_withheld: false,
        backend: "bayesian.randomized_arm_contrast",
        model: format!("{}; {}", posterior.likelihood, prior_text(model)),
        assumptions: vec![
            (
                "bayesian_arm_exchangeability",
                "outcomes within an arm are exchangeable draws from the arm's law under the \
                 declared likelihood and prior; assignment probabilities are known and \
                 identical for every unit, so the posterior conditions on the outcomes by arm"
                    .to_owned(),
            ),
            (
                "bayesian_arm_prior_declared",
                "the arm prior is declared with the study and retained in its inference \
                 identity; a prior alone does not identify a contrast"
                    .to_owned(),
            ),
        ],
        diagnostics: vec![Diagnostic::new(
            "estimate.randomized.bayesian_arm_contrast.estimand",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "estimand={estimand_name}: {statement}; arms {:?}; the credible interval is a \
                 posterior statement, not the design-based randomization interval",
                rows.labels
            ),
        )],
        notes: vec![
            format!("estimand={estimand_name}"),
            format!("prior {}", prior_text(model)),
            "draws are the exact posterior's quantile grid, not Monte Carlo samples".to_owned(),
        ],
        sampler: None,
    })
}

impl super::CheckedRandomizedOperation {
    /// Whether the compiled plan selected the superpopulation posterior.
    fn selected_estimator(&self) -> EstimatorId {
        let superpopulation = self.physical.logical.record.estimator.as_deref()
            == Some(EstimatorId::BayesianRandomizedSuperpopulation.as_str());
        if superpopulation {
            EstimatorId::BayesianRandomizedSuperpopulation
        } else {
            EstimatorId::BayesianRandomizedFinitePopulation
        }
    }

    /// Execute the retained design as a Bayesian posterior on complete, finite outcomes.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        _ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let InferenceMode::Bayesian(config) = &self.result_context.inference else {
            return Err(CausalError::unsupported_inference(
                "Bayesian randomized execution requires Bayesian inference",
            ));
        };
        if matches!(self.query.design, RandomizationDesign::Switchback { .. }) {
            return self.execute_bayesian_switchback(data, config);
        }
        let model = admit(&self.query, config)?;
        if data.storage().weights().is_some() {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the Bayesian arm-contrast posterior reads no sampling weights. Next \
                     attempt: the design-based estimator of this design, which reads them",
                ),
            ));
        }
        let outcomes = complete_outcomes(data, self.query.outcome)?;
        let estimator = self.selected_estimator();
        let estimand = if estimator == EstimatorId::BayesianRandomizedSuperpopulation {
            RandomizedEstimand::Superpopulation
        } else {
            RandomizedEstimand::FinitePopulation
        };
        let rows = arm_rows(&self.query)?;
        let weights: Vec<Vec<f64>> = rows.contrasts.iter().map(|(_, w)| w.clone()).collect();
        let started = Instant::now();
        let posterior = posterior_arm_contrasts_with_draws(
            &ArmContrastRequest {
                outcome: &outcomes,
                assignment: &rows.assignment,
                probabilities: &rows.probabilities,
                model,
                contrasts: &weights,
                estimand,
                credible_level: CREDIBLE_LEVEL,
            },
            config.n_draws,
        )?;
        let fit = arm_design_posterior(&rows, &posterior, model, estimand, outcomes.len())?;
        finish_design_posterior_as(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            fit,
            super::super::Strategy::new(IdentifierId::RandomizedDesign, estimator),
        )
    }
}
