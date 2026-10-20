//! Bayesian fuzzy-jump and regression-kink local ratio.
//!
//! The estimand is the frequentist row's ratio: the outcome's jump (or slope change) at the
//! cutoff over the received treatment's jump (or slope change). Continuity at the cutoff,
//! exclusion of the cutoff from the outcome except through treatment and, for the jump, monotone
//! compliance stay declared; the prior supplies none of them. A ratio of two Normal posteriors has
//! no finite mean, so the reported point is the posterior median, and the credible interval is
//! withheld, with a weak-first-stage diagnostic, when the denominator has posterior mass near
//! zero.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior, scalar_quantity,
};
use super::{
    BayesianConfig, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    StudyResult, TabularData,
};
use antecedent_estimate::DrawSpec;
use antecedent_estimate::bayesian_local_ratio::{
    BayesianLocalRatioEstimate, LocalRatioKind, LocalRatioSample, LocalRatioSpec,
    fit_bayesian_local_ratio,
};
use std::time::Instant;

/// Fraction of the window treatment spread below which the first stage is treated as unidentified.
const FIRST_STAGE_FLOOR_FRACTION: f64 = 0.05;

/// Plan id, identifier and estimator a compiled Bayesian local-ratio plan records.
pub(in crate::analysis::execute) fn bayesian_local_polynomial_ratio_plan_ids(
    kink: bool,
) -> (String, String, String) {
    let label = if kink { "regression_kink" } else { "fuzzy_rd" };
    (
        format!("bayesian.{label}"),
        format!("quasi.{label}.local_ratio"),
        crate::strategy_table::EstimatorId::BayesianLocalRatio.as_str().to_string(),
    )
}

/// Sample standard deviation of the treatment inside the window, or `None` when it has fewer than
/// two finite rows.
fn window_treatment_sd(sample: &LocalRatioSample<'_>, cutoff: f64, bandwidth: f64) -> Option<f64> {
    let window: Vec<f64> = sample
        .running
        .iter()
        .zip(sample.treatment)
        .filter(|(running, _)| (**running - cutoff).abs() <= bandwidth)
        .map(|(_, treatment)| *treatment)
        .collect();
    if window.len() < 2 {
        return None;
    }
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    let variance =
        window.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
    Some(variance.sqrt())
}

/// Diagnostics of the local-ratio posterior: first-stage summary, declared premises, and the
/// weak-first-stage warning when the denominator has mass near zero.
fn local_ratio_diagnostics(
    fit: &BayesianLocalRatioEstimate,
    kind: &str,
    floor: f64,
) -> Vec<Diagnostic> {
    let mut diagnostics = vec![
        Diagnostic::new(
            "estimate.quasi.bayesian_local_ratio.first_stage",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "denominator {kind} posterior mean {:.6} sd {:.6}, probability positive \
                 {:.4}, posterior mass within the floor {floor:.6} of zero {:.4}; \
                 {} window rows; fixed bandwidth {}",
                fit.first_stage_mean,
                fit.first_stage_sd,
                fit.first_stage_positive_probability,
                fit.first_stage_near_zero_mass,
                fit.n_window,
                fit.bandwidth
            ),
        ),
        Diagnostic::new(
            "identification.quasi.bayesian.local_ratio_premises_declared",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "local continuity at the cutoff, exclusion of the cutoff from the outcome except \
             through treatment, and monotone compliance (for the jump) stay declared \
             premises; the prior supplies none of them",
        ),
        Diagnostic::new(
            "estimate.quasi.bayesian_local_ratio.moments_undefined",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "a ratio of Normal posteriors has no finite mean or variance; the reported point \
             is the posterior median and any sample moment of the ratio draws is not a \
             posterior summary",
        ),
    ];
    if fit.weak_first_stage {
        diagnostics.push(Diagnostic::new(
            "estimate.quasi.bayesian_local_ratio.weak_first_stage",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the denominator posterior has {:.4} of its mass within the floor of zero, so \
                 the ratio has no credible interval and its draws are withheld; the posteriors \
                 of the outcome and treatment {kind} are retained. Next attempt: a wider \
                 bandwidth or a stronger discontinuity, or report the outcome {kind} alone as \
                 the intent-to-treat effect",
                fit.first_stage_near_zero_mass
            ),
        ));
    }
    diagnostics
}

impl super::CheckedLocalPolynomialRatioOperation {
    /// Execute the retained design as a Bayesian posterior of the local ratio.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        sample: &LocalRatioSample<'_>,
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let query = &self.query;
        let scale = if query.kink { query.bandwidth } else { 1.0 };
        let floor = window_treatment_sd(sample, query.cutoff, query.bandwidth)
            .map_or(f64::NAN, |sd| FIRST_STAGE_FLOOR_FRACTION * sd / scale);
        if !(floor.is_finite() && floor > 0.0) {
            return Err(CausalError::Estimate(antecedent_estimate::EstimationError::refused(
                antecedent_core::reason_code!("treatment_support_too_discrete"),
                "the received treatment does not vary inside the window, so no first stage can \
                 exist. Next attempt: a wider bandwidth, or a running variable and cutoff at \
                 which the treatment changes",
            )));
        }
        let spec = LocalRatioSpec {
            kind: if query.kink { LocalRatioKind::Kink } else { LocalRatioKind::FuzzyJump },
            cutoff: query.cutoff,
            bandwidth: query.bandwidth,
            prior_sd: config.prior_scale,
            first_stage_floor: floor,
        };
        let fit = fit_bayesian_local_ratio(
            sample,
            &spec,
            DrawSpec::new(config.n_draws, ctx.rng.master_seed()),
        )?;
        let kind = if query.kink { "slope change" } else { "jump" };
        let diagnostics = local_ratio_diagnostics(&fit, kind, floor);
        let mut columns = Vec::with_capacity(3);
        if !fit.weak_first_stage {
            columns.push((effect_quantity("ratio"), fit.ratio_draws));
        }
        columns.push((scalar_quantity("reduced_form"), fit.reduced_form_draws));
        columns.push((scalar_quantity("first_stage"), fit.first_stage_draws));
        let interval = fit.ratio_interval;
        let design = DesignPosterior {
            columns,
            point: fit.ratio_median,
            point_sd: f64::NAN,
            n_obs: fit.n_window as u64,
            interval_withheld: fit.weak_first_stage,
            backend: "bayesian_local_ratio.joint_local_linear",
            model: fit.model.to_string(),
            assumptions: vec![(
                "bayesian_joint_local_linear_gaussian_model",
                "inside the fixed window the outcome and the treatment are Gaussian and linear on \
                 each side of the cutoff with independent zero-centered Normal coefficient priors \
                 and a plug-in residual covariance"
                    .to_string(),
            )],
            diagnostics,
            notes: vec![
                match interval {
                    Some([lower, upper]) => {
                        format!(
                            "interval=equal_tailed_95 ratio_lower={lower:.9} ratio_upper={upper:.9}"
                        )
                    }
                    None => "interval=withheld weak_first_stage=true".to_string(),
                },
                format!(
                    "ratio_median={:.9} first_stage_near_zero_mass={:.6} first_stage_floor={floor:.9}",
                    fit.ratio_median, fit.first_stage_near_zero_mass
                ),
            ],
            sampler: None,
        };
        finish_design_posterior(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: query.outcome,
                config,
                started,
            },
            design,
        )
    }
}
