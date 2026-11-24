//! Sensitivity ranges of a compiled result, read on the observations that produced it.
//!
//! [`PreparedStudy::sensitivity_range`] runs one confounding-sensitivity analysis on a result
//! the prepared handle produced. The result must be the unmodified engine-produced result of
//! the handle's lowered program and the observations its retained table; every analysis reads
//! the treatment, outcome, mediator and adjustment set the compiled study identified, so no
//! caller supplies an estimate, a standard error or a column. Every range is bound to the
//! result's refutation target (its family, quadrant and coordinate labels, defined by the
//! identity of the lowered program).
//!
//! Each analysis calls the same kernel function as the direct path it replaces, on the study's
//! own quantities:
//!
//! * [`SensitivityRangeRequest::EValue`]: [`evalue_range`] on the result's effect as a
//!   standardized mean difference `d = effect / s` with standard error `se / s`, where `s` is
//!   the residual standard deviation `sqrt(RSS / (n - p))` of the outcome regressed on a
//!   constant, the treatment and the adjustment set (the scale the compiled `sensitivity.evalue`
//!   refuter reads). The effect and standard error are the result's own (the bootstrap standard
//!   error when present, else the analytic one). A 0/1 outcome is refused: its effect is a risk
//!   difference, not a standardized difference.
//! * [`SensitivityRangeRequest::OmittedVariableBias`]: [`linear_adjustment_fit`] of
//!   `outcome ~ 1 + treatment + adjustment set`, [`ovb_range`] on its treatment coefficient and
//!   [`benchmark_bound`] against every adjustment covariate (Cinelli and Hazlett 2020).
//! * [`SensitivityRangeRequest::Mediation`]: [`mediation_sensitivity_fit`] and
//!   [`mediation_range`] on the treatment, the single mediator, the outcome and the adjustment
//!   set (Imai, Keele and Yamamoto 2010), the working model the compiled
//!   `target.sequential_ignorability` refuter reads.
//! * [`SensitivityRangeRequest::MediationComponent`]: [`mediation_components_fit`] and
//!   [`mediation_vector_range`] on the same columns, for the indirect or the direct effect.
//! * [`SensitivityRangeRequest::MediationPath`]: [`mediation_paths_fit`] and
//!   [`mediation_path_ovb_range`] on the same columns.
//! * [`SensitivityRangeRequest::MarginalSensitivity`]: [`msm_ate_sensitivity`] on the empirical
//!   strata of the adjustment set (mass, propensity and the outcome law of each arm), for a
//!   two-level treatment and the mean of the outcome in all observed units.
//! * The posterior analyses ([`SensitivityRangeRequest::EValuePosterior`],
//!   [`SensitivityRangeRequest::OmittedVariableBiasPosterior`] and the mediation posterior
//!   analyses) read the study's own posterior draws or the reference posterior of the working
//!   model; see the `posterior` submodule.
//! * The marginal structural model analyses ([`SensitivityRangeRequest::MsmParameters`],
//!   [`SensitivityRangeRequest::MsmParameterEValue`] and
//!   [`SensitivityRangeRequest::MsmParametersRefittingWeights`]) read the weighted design of a
//!   longitudinal regime study fitted with the marginal structural model method;
//!   [`SensitivityRangeRequest::MsmParameterEValuePosterior`] and
//!   [`SensitivityRangeRequest::MsmParametersOvbPosterior`] read the conjugate posterior of that
//!   weighted design ([`MsmParameterPosterior`]) under keyed streams of the context seed.
//! * [`sensitivity_vector_range`] reads several compiled results as the coordinates of one
//!   vector target.
//!
//! The linear working models are the models the ranges are defined for; on a route whose
//! estimator is not linear adjustment the range describes the linear working model on the same
//! observations, as the compiled mediation refuter does. Rows outside the analysis mask, and
//! rows with a missing or non-finite value in a column the analysis reads, are dropped
//! (complete cases); a weighted table is refused. Every number is an assumption range or a
//! diagnostic; calibration is not measured.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, OutcomeFunctional, TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::EffectEstimate;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::target_native::{RefutationTarget, TargetFamily};
use antecedent_validate::{
    AssumptionRange, BenchmarkBound, EValueBounds, EffectInterval, EffectScale,
    LinearAdjustmentFit, LinearAdjustmentSummary, MediationComponent, MediationComponents,
    MediationPath, MediationPaths, MediationSensitivitySummary, MsmOutcomeLaw, MsmRefitFit,
    MsmSensitivityResult, MsmSensitivitySpec, MsmStratum, PosteriorEValue, ValidationError,
    benchmark_bound, evalue_range, linear_adjustment_fit, mediation_components_fit,
    mediation_path_ovb_range, mediation_paths_fit, mediation_range, mediation_sensitivity_fit,
    mediation_vector_range, msm_ate_sensitivity, ovb_range,
};

use super::builder::DataInput;
use super::prepared::PreparedStudy;
use super::target_validation::{
    PreparedTargetData, checked_refutation_input, result_quadrant, result_refutation_target,
};
use crate::error::CausalError;
use crate::result::StudyResult;

mod msm_parameters;
mod msm_posterior;
mod posterior;
mod vector;

pub use msm_posterior::{MSM_POSTERIOR_STREAM, MsmParameterPosterior};
pub use vector::{SensitivityVectorOutcome, SensitivityVectorRequest, sensitivity_vector_range};

/// Most strata the empirical marginal sensitivity model reads.
const MSM_MAX_STRATA: usize = 4096;
/// Most distinct outcome values per arm and stratum of the empirical marginal sensitivity model.
const MSM_MAX_OUTCOMES: usize = 256;

/// Which sensitivity analysis to run on a compiled result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SensitivityRangeRequest {
    /// E-values of the result's effect and its interval limit nearest the null, read as a
    /// standardized mean difference (`VanderWeele` and Ding 2017).
    EValue,
    /// Partial-R² sensitivity of the linear-adjustment coefficient of the treatment.
    OmittedVariableBias {
        /// Share of the estimate a confounder must remove to explain it away (`1` = all).
        q: f64,
        /// Level of the significance robustness value and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
        /// Treatment-side multiplier of the benchmark bounds.
        kd: f64,
        /// Outcome-side multiplier of the benchmark bounds (`kd` when absent).
        ky: Option<f64>,
    },
    /// The indirect effect as a function of the mediator-outcome error correlation `rho`.
    Mediation,
    /// The indirect or the direct effect under the mediator-outcome error correlation `rho`.
    MediationComponent(MediationComponent),
    /// Partial-R² sensitivity of one path of the mediation analysis.
    MediationPath {
        /// The path.
        path: MediationPath,
        /// Share of the estimate a confounder must remove to explain it away.
        q: f64,
        /// Level of the significance robustness value and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
    /// Sharp bounds of the average effect under the marginal sensitivity model.
    MarginalSensitivity(MsmSensitivitySpec),
    /// E-values of the posterior of a Bayesian average effect fitted under a Gaussian outcome
    /// model: each draw of the effect over the residual standard deviation of the same draw.
    EValuePosterior {
        /// Credible level of the limit E-value.
        level: f64,
    },
    /// Partial-R² sensitivity of the posterior of a Bayesian average effect fitted under a
    /// Gaussian outcome model.
    OmittedVariableBiasPosterior {
        /// Credible level of the adjusted limit.
        level: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
    /// The mediation component against the mediator-outcome error correlation, over the
    /// reference posterior of the linear working model (see [`MediationReferencePosterior`]).
    MediationPosterior {
        /// The component.
        component: MediationComponent,
        /// Credible level of the significance break-even.
        level: f64,
        /// Reference posterior draws (at least two).
        draws: usize,
    },
    /// E-values of the mediation component over the reference posterior of the working model,
    /// each draw standardized by the residual standard deviation of the total-effect regression.
    MediationComponentEValuePosterior {
        /// The component.
        component: MediationComponent,
        /// Credible level of the limit E-value.
        level: f64,
        /// Reference posterior draws (at least two).
        draws: usize,
    },
    /// Partial-R² sensitivity of the mediator-outcome coefficient over the reference posterior
    /// of the working model.
    MediationComponentOvbPosterior {
        /// Credible level of the adjusted limit.
        level: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
        /// Reference posterior draws (at least two).
        draws: usize,
    },
    /// Partial-R² sensitivity of the period parameters of the marginal structural model a
    /// longitudinal regime study fitted, with its stabilized weights held fixed.
    MsmParameters {
        /// Share of the estimate a confounder must remove to explain it away.
        q: f64,
        /// Level of the significance robustness value and of the adjusted limits.
        alpha: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
    },
    /// E-values of the period parameters of the marginal structural model, each standardized
    /// by the weighted residual standard deviation of the fit.
    MsmParameterEValue,
    /// Partial-R² sensitivity of the period parameters with weight models refitted in a unit
    /// bootstrap, so their standard errors carry the cost of estimating the weights.
    MsmParametersRefittingWeights(MsmRefitRequest),
    /// E-values of the period parameters over the posterior of the marginal structural model's
    /// weighted design ([`MsmParameterPosterior`]): each parameter draw over the residual
    /// standard deviation of the same draw.
    MsmParameterEValuePosterior {
        /// Credible level of the limit E-values.
        level: f64,
        /// Posterior draws (between 2 and 100000).
        draws: usize,
    },
    /// Partial-R² sensitivity of the period parameters over the posterior of the marginal
    /// structural model's weighted design.
    MsmParametersOvbPosterior {
        /// Credible level of the adjusted limits.
        level: f64,
        /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
        r2yz_extreme: f64,
        /// Posterior draws (between 2 and 100000).
        draws: usize,
    },
}

/// The weight models and bootstrap of [`SensitivityRangeRequest::MsmParametersRefittingWeights`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MsmRefitRequest {
    /// Covariate columns of each period's numerator model (empty: the marginal probability).
    pub numerator: Vec<Vec<VariableId>>,
    /// Covariate columns of each period's denominator model (the measured history).
    pub denominator: Vec<Vec<VariableId>>,
    /// Bootstrap replicates (at least `MIN_MSM_REPLICATES`).
    pub replicates: usize,
    /// Seed of the bootstrap.
    pub seed: u64,
    /// Share of the estimate a confounder must remove to explain it away.
    pub q: f64,
    /// Level of the significance robustness value and of the adjusted limits.
    pub alpha: f64,
    /// Outcome partial R² of the extreme-scenario curve, in `(0, 1]`.
    pub r2yz_extreme: f64,
}

impl MsmRefitRequest {
    /// A refit request with conventional settings: `q = 1`, `alpha = 0.05`, the most extreme
    /// outcome partial R² and 200 replicates.
    #[must_use]
    pub const fn new(
        numerator: Vec<Vec<VariableId>>,
        denominator: Vec<Vec<VariableId>>,
        seed: u64,
    ) -> Self {
        Self {
            numerator,
            denominator,
            replicates: 200,
            seed,
            q: 1.0,
            alpha: 0.05,
            r2yz_extreme: 1.0,
        }
    }
}

impl SensitivityRangeRequest {
    /// The omitted-variable-bias analysis at its conventional settings: `q = 1`,
    /// `alpha = 0.05`, the most extreme outcome partial R² and benchmarks as strong as each
    /// covariate.
    #[must_use]
    pub const fn omitted_variable_bias() -> Self {
        Self::OmittedVariableBias { q: 1.0, alpha: 0.05, r2yz_extreme: 1.0, kd: 1.0, ky: None }
    }

    /// The path analysis of `path` at its conventional settings: `q = 1`, `alpha = 0.05` and
    /// the most extreme outcome partial R².
    #[must_use]
    pub const fn mediation_path(path: MediationPath) -> Self {
        Self::MediationPath { path, q: 1.0, alpha: 0.05, r2yz_extreme: 1.0 }
    }
}

/// One sensitivity analysis of a compiled result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SensitivityRangeOutcome {
    /// [`SensitivityRangeRequest::EValue`].
    EValue {
        /// Effect divided by the residual standard deviation.
        standardized_effect: f64,
        /// Standard error divided by the residual standard deviation, when the result has a
        /// usable one.
        standardized_standard_error: Option<f64>,
        /// Residual standard deviation of the outcome regression.
        residual_sd: f64,
        /// The E-values.
        bounds: EValueBounds,
        /// The range over confounder strengths.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::OmittedVariableBias`].
    OmittedVariableBias {
        /// The linear-adjustment fit and each covariate's partial R²s.
        fit: LinearAdjustmentFit,
        /// The adjustment covariates, in the order of the fit's partial R²s.
        covariates: Vec<VariableId>,
        /// Bounds of a confounder `kd`/`ky` times as strong as each covariate, in that order.
        benchmarks: Vec<BenchmarkBound>,
        /// The extreme-scenario range.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::Mediation`].
    Mediation {
        /// The regression summaries the curve is read from.
        summary: MediationSensitivitySummary,
        /// The curve.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MediationComponent`].
    MediationComponent {
        /// The regression summaries and the total effect.
        components: MediationComponents,
        /// The range of the requested component.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MediationPath`].
    MediationPath {
        /// The three path coefficients.
        paths: MediationPaths,
        /// The range of the requested path.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MarginalSensitivity`].
    MarginalSensitivity {
        /// The empirical strata of the adjustment set, in order of first appearance.
        strata: Vec<MsmStratum>,
        /// Sharp bounds, grid and tipping point.
        result: MsmSensitivityResult,
    },
    /// [`SensitivityRangeRequest::EValuePosterior`].
    EValuePosterior {
        /// Draws of the effect over the residual standard deviation of the same draw.
        draws: Vec<f64>,
        /// Posterior medians, credible limits and E-values.
        summary: PosteriorEValue,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::OmittedVariableBiasPosterior`].
    OmittedVariableBiasPosterior {
        /// Draws of the effect.
        coefficients: Vec<f64>,
        /// Per draw, `|active - control|` times the residual standard deviation of the outcome
        /// over the residual standard deviation of the treatment given the adjustment set.
        bias_scale: Vec<f64>,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MediationPosterior`],
    /// [`SensitivityRangeRequest::MediationComponentEValuePosterior`] and
    /// [`SensitivityRangeRequest::MediationComponentOvbPosterior`].
    MediationPosterior {
        /// The reference posterior of the working model.
        posterior: MediationReferencePosterior,
        /// E-values of the component, for the E-value analysis.
        evalue: Option<PosteriorEValue>,
        /// The range.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MsmParameters`].
    MsmParameters {
        /// One summary per period parameter.
        summaries: Vec<LinearAdjustmentSummary>,
        /// The stabilized weight of each observed subject, in row order.
        weights: Vec<f64>,
        /// The range, read at the weakest parameter.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MsmParameterEValue`].
    MsmParameterEValue {
        /// One summary per period parameter.
        summaries: Vec<LinearAdjustmentSummary>,
        /// Weighted residual standard deviation the parameters are standardized by.
        residual_sd: f64,
        /// The E-values of each parameter.
        bounds: Vec<EValueBounds>,
        /// The range, read at the weakest parameter.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MsmParametersRefittingWeights`].
    MsmParametersRefittingWeights {
        /// Fixed-weight and refitted summaries and the replicate counts.
        fit: MsmRefitFit,
        /// The range of the refitted summaries, read at the weakest parameter.
        range: AssumptionRange,
    },
    /// [`SensitivityRangeRequest::MsmParameterEValuePosterior`] and
    /// [`SensitivityRangeRequest::MsmParametersOvbPosterior`].
    MsmParameterPosterior {
        /// The posterior draws of the parameters and of the residual standard deviation.
        posterior: MsmParameterPosterior,
        /// E-values of the parameters, for the E-value analysis.
        evalue: Option<PosteriorEValue>,
        /// The range, read at the weakest parameter.
        range: AssumptionRange,
    },
}

impl SensitivityRangeOutcome {
    /// The assumption range, for every analysis that reports one (all but the marginal
    /// sensitivity model, which reports its sharp bounds).
    #[must_use]
    pub const fn range(&self) -> Option<&AssumptionRange> {
        match self {
            Self::EValue { range, .. }
            | Self::OmittedVariableBias { range, .. }
            | Self::Mediation { range, .. }
            | Self::MediationComponent { range, .. }
            | Self::MediationPath { range, .. }
            | Self::EValuePosterior { range, .. }
            | Self::OmittedVariableBiasPosterior { range, .. }
            | Self::MediationPosterior { range, .. }
            | Self::MsmParameters { range, .. }
            | Self::MsmParameterEValue { range, .. }
            | Self::MsmParametersRefittingWeights { range, .. }
            | Self::MsmParameterPosterior { range, .. } => Some(range),
            Self::MarginalSensitivity { .. } => None,
        }
    }
}

/// Per-draw regression summaries of the linear working model of a mediation, drawn from its
/// reference posterior (`p(beta, sigma^2) ∝ 1 / sigma^2`, independent across the mediator
/// regression `M ~ 1 + T + X` and the outcome regression `Y ~ 1 + T + M + X`) on the retained
/// table, under keyed streams of the execution context's seed. The total-effect regression's
/// summaries follow from the two: `total = c + gamma a`,
/// `sigma_total = sqrt(sigma_outcome^2 + gamma^2 sigma_mediator^2)` and
/// `rho_tilde = gamma sigma_mediator / sigma_total`, so at `rho = 0` every draw's indirect effect
/// is the product `a gamma`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MediationReferencePosterior {
    /// Treatment coefficient of the mediator regression, `a`.
    pub beta_mediator: Vec<f64>,
    /// Residual standard deviation of the total-effect regression.
    pub sigma_total: Vec<f64>,
    /// Residual standard deviation of the mediator regression.
    pub sigma_mediator: Vec<f64>,
    /// Correlation of the total-effect and mediator residuals.
    pub rho_tilde: Vec<f64>,
    /// Treatment coefficient of the total-effect regression.
    pub total_effect: Vec<f64>,
    /// Mediator coefficient of the outcome regression, `gamma`.
    pub mediator_outcome: Vec<f64>,
    /// Residual standard deviation of the outcome regression.
    pub sigma_outcome: Vec<f64>,
    /// Residual standard deviation of the mediator given the treatment and the adjustment set,
    /// on the outcome regression's residual degrees of freedom (the scale of the
    /// mediator-outcome coefficient's bias).
    pub mediator_scale: f64,
}

/// An effect and its standard error over a residual standard deviation.
#[derive(Clone, Copy, Debug)]
struct Standardized {
    effect: f64,
    standard_error: Option<f64>,
    residual_sd: f64,
}

/// Treatment, mediator, outcome and adjustment columns of a mediation study.
type MediationColumns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

/// What every analysis reads from the result and the retained table.
struct Bound<'a> {
    data: &'a TabularData,
    result: &'a StudyResult,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: Vec<VariableId>,
    program: [u8; 32],
}

impl PreparedStudy {
    /// Run `request` on `result`, a result this handle produced from its retained table.
    ///
    /// # Errors
    ///
    /// A result that another program produced, that was modified after it was produced or that
    /// was computed on other observations; a retained input that is not one table; a result
    /// without a treatment; an analysis that does not apply to the result (a non-mediation
    /// query for the mediation analyses, a 0/1 outcome for the E-value, a treatment with other
    /// than two levels for the marginal sensitivity model); a column that is not numeric; a
    /// weighted table; and the refusals of the kernel function the analysis calls.
    pub fn sensitivity_range(
        &self,
        result: &StudyResult,
        request: &SensitivityRangeRequest,
        ctx: &ExecutionContext,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        let (data, program) = self.checked_result(result, ctx)?;
        if let Some(outcome) = self.msm_parameter_range(data, result, program, request, ctx)? {
            return Ok(outcome);
        }
        let bound = Self::bound_from(data, result, program)?;
        match request {
            SensitivityRangeRequest::EValue => bound.evalue(),
            SensitivityRangeRequest::OmittedVariableBias { q, alpha, r2yz_extreme, kd, ky } => {
                bound.omitted_variable_bias(*q, *alpha, *r2yz_extreme, *kd, ky.unwrap_or(*kd))
            }
            SensitivityRangeRequest::Mediation => {
                let (t, m, y, covariates) = self.mediation_columns(&bound)?;
                let summary = mediation_sensitivity_fit(&t, &m, &y, &covariates)?;
                let range = mediation_range(result_refutation_target(result)?, &summary)?;
                Ok(SensitivityRangeOutcome::Mediation { summary, range })
            }
            SensitivityRangeRequest::MediationComponent(component) => {
                let (t, m, y, covariates) = self.mediation_columns(&bound)?;
                let components = mediation_components_fit(&t, &m, &y, &covariates)?;
                let label = match component {
                    MediationComponent::Indirect => "indirect",
                    MediationComponent::Direct => "direct",
                };
                let range =
                    mediation_vector_range(bound.coordinate(label)?, &[components], *component)?;
                Ok(SensitivityRangeOutcome::MediationComponent { components, range })
            }
            SensitivityRangeRequest::MediationPath { path, q, alpha, r2yz_extreme } => {
                let (t, m, y, covariates) = self.mediation_columns(&bound)?;
                let paths = mediation_paths_fit(&t, &m, &y, &covariates)?;
                let label = match path {
                    MediationPath::TreatmentMediator => "treatment_mediator",
                    MediationPath::MediatorOutcome => "mediator_outcome",
                    MediationPath::TreatmentOutcome => "treatment_outcome",
                    MediationPath::Indirect => "indirect",
                };
                let range = mediation_path_ovb_range(
                    bound.coordinate(label)?,
                    &[paths],
                    *path,
                    *q,
                    *alpha,
                    *r2yz_extreme,
                )?;
                Ok(SensitivityRangeOutcome::MediationPath { paths, range })
            }
            SensitivityRangeRequest::MarginalSensitivity(spec) => {
                self.marginal_sensitivity(&bound, spec)
            }
            SensitivityRangeRequest::EValuePosterior { level } => {
                self.evalue_posterior(&bound, *level)
            }
            SensitivityRangeRequest::OmittedVariableBiasPosterior { level, r2yz_extreme } => {
                self.ovb_posterior(&bound, *level, *r2yz_extreme)
            }
            SensitivityRangeRequest::MediationPosterior { .. }
            | SensitivityRangeRequest::MediationComponentEValuePosterior { .. }
            | SensitivityRangeRequest::MediationComponentOvbPosterior { .. } => {
                self.mediation_posterior(&bound, request, ctx)
            }
            SensitivityRangeRequest::MsmParameters { .. }
            | SensitivityRangeRequest::MsmParameterEValue
            | SensitivityRangeRequest::MsmParametersRefittingWeights(_)
            | SensitivityRangeRequest::MsmParameterEValuePosterior { .. }
            | SensitivityRangeRequest::MsmParametersOvbPosterior { .. } => {
                Err(CausalError::Unsupported {
                    message: "the marginal structural model parameters are read from a \
                              longitudinal regime study fitted with the marginal structural \
                              model method",
                })
            }
        }
    }

    /// The retained table and the producing program of `result`, after checking that this
    /// handle's lowered program produced it, unmodified, from these exact observations.
    fn checked_result(
        &self,
        result: &StudyResult,
        ctx: &ExecutionContext,
    ) -> Result<(&TabularData, [u8; 32]), CausalError> {
        let DataInput::Tabular(data) = &self.study().data else {
            return Err(CausalError::Unsupported {
                message: "a sensitivity range reads independent rows of one table; next \
                          attempt: a study on a tabular input",
            });
        };
        // The result must be the sealed engine-produced result of these exact observations.
        let _checked = checked_refutation_input(result, PreparedTargetData::Tabular(data), ctx)?;
        let program = result.lowered_program_identity().ok_or(CausalError::Unsupported {
            message: "a sensitivity range is bound to the lowered program that produced the \
                      result; next attempt: execute through a prepared handle",
        })?;
        if let Some(lowered) = self.plan().lowered()
            && lowered.semantic_identity() != Some(program)
        {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "the result was not produced by this prepared handle's lowered program",
            });
        }
        Ok((data, *program.as_bytes()))
    }

    /// What every analysis of an effect reads: [`Self::checked_result`], the treatment, the
    /// outcome and the identified adjustment set.
    fn bind<'a>(
        &'a self,
        result: &'a StudyResult,
        ctx: &ExecutionContext,
    ) -> Result<Bound<'a>, CausalError> {
        let (data, program) = self.checked_result(result, ctx)?;
        Self::bound_from(data, result, program)
    }

    /// [`Self::bind`] on an already checked result.
    fn bound_from<'a>(
        data: &'a TabularData,
        result: &'a StudyResult,
        program: [u8; 32],
    ) -> Result<Bound<'a>, CausalError> {
        let treatment = result.treatment.ok_or(CausalError::Unsupported {
            message: "a sensitivity range needs a result with a treatment",
        })?;
        Ok(Bound {
            data,
            result,
            treatment,
            outcome: result.outcome,
            adjustment: result.estimand.adjustment_set.iter().copied().collect(),
            program,
        })
    }

    /// Treatment, mediator, outcome and adjustment columns (complete cases) of a mediation
    /// study with one mediator.
    fn mediation_columns(&self, bound: &Bound<'_>) -> Result<MediationColumns, CausalError> {
        let CausalQuery::Mediation(query) = &self.study().query else {
            return Err(CausalError::Unsupported {
                message: "the mediation sensitivity ranges read a mediation query; next \
                          attempt: a study with a MediationQuery",
            });
        };
        let [mediator] = &query.mediators[..] else {
            return Err(CausalError::Unsupported {
                message: "the mediation sensitivity ranges read one mediator; next attempt: a \
                          study with a single mediator",
            });
        };
        let mut ids = vec![bound.treatment, *mediator, bound.outcome];
        ids.extend(
            bound
                .adjustment
                .iter()
                .copied()
                .filter(|&id| id != bound.treatment && id != *mediator && id != bound.outcome),
        );
        let mut columns = complete_cases(bound.data, &ids)?.into_iter();
        let (Some(t), Some(m), Some(y)) = (columns.next(), columns.next(), columns.next()) else {
            return Err(CausalError::Unsupported {
                message: "the mediation sensitivity ranges read a treatment, a mediator and an \
                          outcome column",
            });
        };
        Ok((t, m, y, columns.collect()))
    }

    /// The marginal sensitivity model on the empirical strata of the adjustment set.
    fn marginal_sensitivity(
        &self,
        bound: &Bound<'_>,
        spec: &MsmSensitivitySpec,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        let CausalQuery::AverageEffect(query) = &self.study().query else {
            return Err(CausalError::Unsupported {
                message: "the marginal sensitivity model bounds an average effect; next \
                          attempt: a study with an AverageEffectQuery",
            });
        };
        if !matches!(query.target_population, TargetPopulation::AllObserved)
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
            || !query.effect_modifiers.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "the marginal sensitivity model bounds the mean outcome contrast in all \
                          observed units; next attempt: an untransformed average effect without \
                          effect modifiers",
            });
        }
        let level = |intervention: &Intervention| match intervention {
            Intervention::Set { value, .. } => value.as_f64().filter(|v| v.is_finite()),
            _ => None,
        };
        let (Some(control), Some(active)) = (level(&query.control), level(&query.active)) else {
            return Err(CausalError::Unsupported {
                message: "the marginal sensitivity model reads two set treatment levels",
            });
        };
        let mut ids = vec![bound.treatment, bound.outcome];
        ids.extend(bound.adjustment.iter().copied());
        let columns = complete_cases(bound.data, &ids)?;
        let strata = empirical_strata(&columns, control, active)?;
        let result = msm_ate_sensitivity(&strata, spec)
            .map_err(|error| crate::compile_reason!("route_not_supported", "{error}"))?;
        Ok(SensitivityRangeOutcome::MarginalSensitivity { strata, result })
    }
}

impl Bound<'_> {
    /// A one-coordinate mediation target of the result's program, labelled `label`.
    fn coordinate(&self, label: &str) -> Result<RefutationTarget, CausalError> {
        RefutationTarget::new(
            TargetFamily::Mediation,
            result_quadrant(self.result),
            [label],
            self.program,
        )
        .map_err(CausalError::from)
    }

    #[allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        reason = "exact 0/1 coding is the binary-outcome test; the residual degrees of freedom are \
                  far below 2^53"
    )]
    fn residual_sd(&self) -> Result<f64, CausalError> {
        let mut ids = vec![self.outcome, self.treatment];
        ids.extend(self.adjustment.iter().copied());
        let columns = complete_cases(self.data, &ids)?;
        let outcome = &columns[0];
        if !outcome.is_empty() && outcome.iter().all(|&y| y == 0.0 || y == 1.0) {
            return Err(CausalError::Unsupported {
                message: "the effect on a 0/1 outcome is a risk difference, not a standardized \
                          mean difference; next attempt: the compiled sensitivity.evalue refuter \
                          (RefuteSuite::Cheap), which reads the arm risks",
            });
        }
        let n = outcome.len();
        let p = columns.len();
        if n <= p {
            return Err(CausalError::Unsupported {
                message: "the E-value needs more complete rows than outcome-regression parameters",
            });
        }
        let mut design = vec![1.0; n];
        for column in &columns[1..] {
            design.extend_from_slice(column);
        }
        let fit = FaerBackend
            .least_squares(&design, n, p, outcome, &mut LeastSquaresWorkspace::default())
            .map_err(ValidationError::from)?;
        if fit.rank < p {
            return Err(CausalError::Unsupported {
                message: "the E-value's outcome regression needs a full-rank design",
            });
        }
        let residual_sd = (fit.rss / (n - p) as f64).sqrt();
        if !(residual_sd.is_finite() && residual_sd > 0.0) {
            return Err(CausalError::Unsupported {
                message: "the outcome regression leaves no residual variation to standardize by",
            });
        }
        Ok(residual_sd)
    }

    /// `effect` over the residual standard deviation of the outcome regression, with its
    /// standard error (the bootstrap one when present, else the analytic one) on that scale.
    fn standardized(&self, effect: &EffectEstimate) -> Result<Standardized, CausalError> {
        let residual_sd = self.residual_sd()?;
        let se = effect.se_bootstrap().unwrap_or(effect.se_analytic());
        Ok(Standardized {
            effect: effect.ate / residual_sd,
            standard_error: (se.is_finite() && se >= 0.0).then_some(se / residual_sd),
            residual_sd,
        })
    }

    fn evalue(&self) -> Result<SensitivityRangeOutcome, CausalError> {
        let effect = self.result.estimate.as_effect().ok_or(CausalError::Unsupported {
            message: "the E-value reads the result's scalar effect",
        })?;
        let Standardized {
            effect: standardized_effect,
            standard_error: standardized_standard_error,
            residual_sd,
        } = self.standardized(effect)?;
        let (bounds, range) = evalue_range(
            result_refutation_target(self.result)?,
            EffectInterval::Standardized {
                estimate: standardized_effect,
                standard_error: standardized_standard_error,
            },
            EffectScale::StandardizedMeanDifference,
        )?;
        Ok(SensitivityRangeOutcome::EValue {
            standardized_effect,
            standardized_standard_error,
            residual_sd,
            bounds,
            range,
        })
    }

    fn omitted_variable_bias(
        &self,
        q: f64,
        alpha: f64,
        r2yz_extreme: f64,
        kd: f64,
        ky: f64,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        let (fit, covariates) = self.linear_fit()?;
        let range = ovb_range(
            result_refutation_target(self.result)?,
            &fit.summary,
            q,
            alpha,
            r2yz_extreme,
        )?;
        let benchmarks = fit
            .r2_treatment
            .iter()
            .zip(&fit.r2_outcome)
            .map(|(&r2d, &r2y)| benchmark_bound(r2d, r2y, kd, ky))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SensitivityRangeOutcome::OmittedVariableBias { fit, covariates, benchmarks, range })
    }

    /// The adjustment covariates (treatment and outcome left out) and their columns.
    fn covariates(&self) -> Vec<VariableId> {
        self.adjustment
            .iter()
            .copied()
            .filter(|&id| id != self.treatment && id != self.outcome)
            .collect()
    }

    /// [`linear_adjustment_fit`] of `outcome ~ 1 + treatment + adjustment set`, and the
    /// covariates in the order of its partial R²s.
    fn linear_fit(&self) -> Result<(LinearAdjustmentFit, Vec<VariableId>), CausalError> {
        let covariates = self.covariates();
        let mut ids = vec![self.outcome, self.treatment];
        ids.extend(covariates.iter().copied());
        let mut columns = complete_cases(self.data, &ids)?;
        let adjustment = columns.split_off(2);
        let fit = linear_adjustment_fit(&columns[0], &columns[1], &adjustment)?;
        Ok((fit, covariates))
    }
}

/// The columns of `ids` on the rows inside the analysis mask with a finite value in every one.
fn complete_cases(data: &TabularData, ids: &[VariableId]) -> Result<Vec<Vec<f64>>, CausalError> {
    let storage = data.storage();
    if storage.weights().is_some() {
        return Err(CausalError::Unsupported {
            message: "the sensitivity ranges read unweighted rows; next attempt: an unweighted \
                      table",
        });
    }
    let columns: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| data.float64_values(id))
        .collect::<Result<_, _>>()
        .map_err(|_| CausalError::Unsupported {
            message: "a column the sensitivity range reads is not numeric; next attempt: encode \
                      the treatment, mediator, outcome and adjustment set as numbers",
        })?;
    let mask = storage.analysis_mask();
    let rows: Vec<usize> = (0..data.row_count())
        .filter(|&i| mask.is_none_or(|m| m.is_valid(i)) && columns.iter().all(|c| c[i].is_finite()))
        .collect();
    Ok(columns.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect())
}

/// Per-arm outcome counts of one stratum, each outcome in order of first appearance.
#[derive(Default)]
struct ArmCounts {
    values: Vec<f64>,
    counts: Vec<u64>,
    index: HashMap<u64, usize>,
    total: u64,
}

impl ArmCounts {
    fn add(&mut self, value: f64) -> Result<(), CausalError> {
        // -0.0 and 0.0 are one outcome value.
        let value = value + 0.0;
        let slot = if let Some(&slot) = self.index.get(&value.to_bits()) {
            slot
        } else {
            if self.values.len() == MSM_MAX_OUTCOMES {
                return Err(CausalError::Unsupported {
                    message: "the empirical marginal sensitivity model reads at most 256 \
                              distinct outcomes per arm and stratum",
                });
            }
            self.index.insert(value.to_bits(), self.values.len());
            self.values.push(value);
            self.counts.push(0);
            self.values.len() - 1
        };
        self.counts[slot] += 1;
        self.total += 1;
        Ok(())
    }

    /// The empirical law: values ascending, probability `count / total`.
    #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^53")]
    fn law(&self) -> MsmOutcomeLaw {
        let mut order: Vec<usize> = (0..self.values.len()).collect();
        order.sort_by(|&a, &b| self.values[a].total_cmp(&self.values[b]));
        MsmOutcomeLaw {
            values: order.iter().map(|&i| self.values[i]).collect(),
            probabilities: order
                .iter()
                .map(|&i| self.counts[i] as f64 / self.total as f64)
                .collect(),
        }
    }
}

/// The empirical strata of `columns` (treatment, outcome, then the adjustment set): one per
/// distinct adjustment value, in order of first appearance, with mass `count / rows`,
/// propensity `treated / count` and the empirical outcome law of each arm.
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "row counts are far below 2^53 and a treatment level is matched exactly"
)]
fn empirical_strata(
    columns: &[Vec<f64>],
    control: f64,
    active: f64,
) -> Result<Vec<MsmStratum>, CausalError> {
    let rows = columns.first().map_or(0, Vec::len);
    if rows == 0 {
        return Err(CausalError::Unsupported {
            message: "the empirical marginal sensitivity model needs complete rows",
        });
    }
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut arms: Vec<(ArmCounts, ArmCounts)> = Vec::new();
    for row in 0..rows {
        let treatment = columns[0][row];
        let key: Vec<u64> = columns[2..].iter().map(|c| (c[row] + 0.0).to_bits()).collect();
        let slot = if let Some(&slot) = index.get(&key) {
            slot
        } else {
            if arms.len() == MSM_MAX_STRATA {
                return Err(CausalError::Unsupported {
                    message: "the empirical marginal sensitivity model reads at most 4096 \
                              strata; next attempt: an adjustment set of finite levels",
                });
            }
            index.insert(key, arms.len());
            arms.push((ArmCounts::default(), ArmCounts::default()));
            arms.len() - 1
        };
        let (untreated, treated) = &mut arms[slot];
        if treatment == active {
            treated.add(columns[1][row])?;
        } else if treatment == control {
            untreated.add(columns[1][row])?;
        } else {
            return Err(CausalError::Unsupported {
                message: "the marginal sensitivity model reads a treatment with exactly the \
                          query's two levels",
            });
        }
    }
    arms.iter()
        .map(|(untreated, treated)| {
            if untreated.total == 0 || treated.total == 0 {
                return Err(CausalError::Unsupported {
                    message: "every stratum of the empirical marginal sensitivity model needs \
                              both treatment arms",
                });
            }
            let count = untreated.total + treated.total;
            Ok(MsmStratum {
                mass: count as f64 / rows as f64,
                propensity: treated.total as f64 / count as f64,
                treated: treated.law(),
                control: untreated.law(),
            })
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "the empirical masses and laws are exact ratios")]
mod tests {
    use super::*;

    #[test]
    fn empirical_strata_follow_first_appearance_and_sort_each_law() {
        let columns = vec![
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![2.0, 1.0, -0.0, 0.0, 2.0, 3.0],
            vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0],
        ];
        let strata = empirical_strata(&columns, 0.0, 1.0).unwrap();
        assert_eq!(strata.len(), 2);
        assert_eq!(strata[0].mass, 4.0 / 6.0);
        assert_eq!(strata[0].propensity, 0.5);
        assert_eq!(strata[0].treated.values, vec![0.0, 2.0]);
        assert_eq!(strata[0].treated.probabilities, vec![0.5, 0.5]);
        assert_eq!(strata[0].control.values, vec![0.0, 1.0]);
        assert_eq!(strata[1].mass, 2.0 / 6.0);
        assert_eq!(strata[1].treated.values, vec![2.0]);
        assert_eq!(strata[1].control.values, vec![3.0]);
    }

    #[test]
    fn empirical_strata_refuse_a_third_level_and_a_missing_arm() {
        let third = vec![vec![1.0, 0.0, 2.0], vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]];
        assert!(empirical_strata(&third, 0.0, 1.0).is_err());
        let one_arm = vec![vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 1.0]];
        assert!(empirical_strata(&one_arm, 0.0, 1.0).is_err());
    }
}
