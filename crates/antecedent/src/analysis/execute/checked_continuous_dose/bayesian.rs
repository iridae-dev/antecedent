//! Bayesian posterior of the fixed group-to-dose policy value.
//!
//! The target is the frequentist row's: for fixed baseline groups with observed fractions and a
//! fixed group-to-dose rule, the average response over the prespecified Epanechnikov window of
//! each group's rule dose, against the reference rule, paired. It is not an exact-dose
//! intervention. The likelihood is stated: within a group the outcome is a polynomial of the
//! dose, rescaled to `[-1, 1]` on the declared support, plus independent normal noise, with the
//! coefficient prior and residual prior declared on the study's Bayesian configuration. The
//! dose density does not enter a likelihood for the outcome given the dose, so it is not read.
//! Conditional exchangeability within groups and local positivity stay declared premises; the
//! prior supplies none. The result is an unlicensed, uncalibrated posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
    require_built_in_model, scalar_quantity,
};
use super::{
    CausalError, CheckedContinuousDoseOperation, EstimatorId, ExecutionContext, IdentifierId,
    Strategy, StudyResult, TableView, TabularData,
};
use crate::{BayesianConfig, DesignPosteriorModel};
use antecedent_estimate::EstimationError;
use antecedent_estimate::continuous_dose::DoseRules;
use antecedent_estimate::continuous_dose_posterior::{
    DoseCurvePrior, DoseResponseRows, PolicyWindow, fixed_dose_policy_posterior,
};

fn refuse(code: &'static str, detail: &str, message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::refused_detail(code, detail, message))
}

/// Refuse what this posterior does not model, before any row is read.
pub(super) fn require_scope(
    query: &antecedent_core::ContinuousDoseResponseQuery,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    require_built_in_model(config, "the Bayesian dose-policy posterior")?;
    if query.fixed_policy.is_none() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "continuous_dose_policy.policy_required",
            "the posterior answers the fixed group-dose policy value, not the pointwise response \
             grid. Next attempt: declare a fixed policy and reference on the query, or use \
             frequentist inference for the response grid",
        ));
    }
    if !query.target_doses.is_empty() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "continuous_dose_policy.grid_not_read",
            "the posterior reads the fixed policy and no pointwise response grid. Next attempt: \
             omit the target doses, or use frequentist inference for the grid",
        ));
    }
    if !matches!(config.design_model, Some(DesignPosteriorModel::DosePolynomial { .. })) {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "continuous_dose_policy.model_not_declared",
            "the dose-policy posterior needs its dose support, coefficient prior scales and \
             residual prior declared. Next attempt: BayesianConfig::design_model with \
             DesignPosteriorModel::DosePolynomial; the prior is part of the answer and is never \
             selected from the data",
        ));
    }
    Ok(())
}

fn numeric(data: &TabularData, id: antecedent_core::VariableId) -> Result<Vec<f64>, CausalError> {
    match data.column(id).map_err(CausalError::from)? {
        antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice().to_vec()),
        _ => Err(CausalError::Unsupported {
            message: "continuous-dose outcome and dose columns must be numeric",
        }),
    }
}

impl CheckedContinuousDoseOperation {
    /// Execute the retained design as the posterior of the fixed group-dose policy value.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = std::time::Instant::now();
        let (
            Some(rule),
            Some(DesignPosteriorModel::DosePolynomial { support, coefficient_scales, residual }),
        ) = (&self.query.fixed_policy, &config.design_model)
        else {
            return Err(CausalError::Compile {
                message: "the Bayesian dose policy lost its policy or declared model".into(),
            });
        };
        let outcome = numeric(data, self.query.outcome)?;
        let dose = numeric(data, self.query.dose)?;
        let groups: Vec<String> =
            self.query.baseline_groups.iter().map(ToString::to_string).collect();
        let policy: Vec<(String, f64)> =
            rule.policy_doses.iter().map(|(group, dose)| (group.to_string(), *dose)).collect();
        let reference: Vec<(String, f64)> =
            rule.reference_doses.iter().map(|(group, dose)| (group.to_string(), *dose)).collect();
        let posterior = fixed_dose_policy_posterior(
            &DoseResponseRows { outcome: &outcome, dose: &dose, groups: &groups },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseCurvePrior { support: *support, coefficient_scales, residual: *residual },
            &PolicyWindow {
                bandwidth: self.query.bandwidth,
                min_local_support: self.query.min_local_support,
                draws: config.n_draws,
                seed: ctx.rng.master_seed(),
            },
        )
        .map_err(CausalError::from)?;
        let row = antecedent_io::bayesian_posterior_row_artifact::policy_row(&posterior);
        let incremental: Vec<f64> = posterior
            .policy_draws
            .iter()
            .zip(&posterior.reference_draws)
            .map(|(policy, reference)| policy - reference)
            .collect();
        let design = DesignPosterior {
            columns: vec![
                (effect_quantity("incremental_value"), incremental),
                (scalar_quantity("policy_value"), posterior.policy_draws.clone()),
                (scalar_quantity("reference_value"), posterior.reference_draws.clone()),
            ],
            point: posterior.incremental_mean,
            point_sd: f64::NAN,
            n_obs: self.rows as u64,
            interval_withheld: false,
            backend: "bayesian_dose_policy.group_polynomial_conjugate",
            model: format!(
                "per-group polynomial dose response of {} coefficients on [{}, {}] with \
                 independent zero-centered normal coefficient priors; Epanechnikov window of \
                 half-width {}; groups independent a posteriori with their observed fractions",
                coefficient_scales.len(),
                support.0,
                support.1,
                self.query.bandwidth
            ),
            assumptions: vec![(
                "group_polynomial_dose_response_model",
                "within each baseline group the outcome is a polynomial of the dose on the \
                 declared support plus independent normal noise; the window average of that curve \
                 is the policy value, not an exact-dose potential outcome"
                    .to_string(),
            )],
            diagnostics: Vec::new(),
            notes: vec![format!(
                "bandwidth={} minimum_local_rows={} probability_incremental_positive={:.6}",
                self.query.bandwidth,
                posterior.minimum_local_rows,
                posterior.probability_incremental_positive
            )],
            sampler: None,
        };
        let mut result = finish_design_posterior_as(
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
            design,
            Strategy::new(
                IdentifierId::ContinuousDoseExchangeability,
                EstimatorId::BayesianDosePolicy,
            ),
        )?;
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.payload = crate::result::ResultPayload::PosteriorRow(row);
        Ok(result)
    }
}
