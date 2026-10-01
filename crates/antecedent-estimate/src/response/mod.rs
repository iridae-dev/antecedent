//! Complete-observation estimators for continuous causal responses.
//!
//! The scalar curve path uses deterministic cross-fitting, Gaussian additive
//! nuisance regressions, the Kennedy-style doubly robust pseudo-outcome, and a
//! Gaussian local-polynomial smoother. Least-squares nuisances require finite
//! outcome moments; a heavy-tailed outcome is reported on the support diagnostic
//! axis rather than by demoting overlap or matrix-cell licensing. Average
//! derivatives use the corresponding Gaussian treatment-score Riesz representer.
//! Multivariate derivatives are deliberately low-dimensional, model-dependent
//! plug-in functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments, clippy::too_many_lines)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, CausalRng, CredibleDraws, DerivativeScale,
    DerivativeWeighting, Diagnostic, DiagnosticKind, DiagnosticSeverity, IdentificationStatus,
    Intervention, MAX_NONPARAMETRIC_RESPONSE_DIM, ObservationSpec, ParametricAssumption,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    StreamDomain, SupportDiagnostic, SupportRegion, SupportReport, SupportStatus, TargetPopulation,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_stats::{
    AdditiveDesign, DenseLinearAlgebra, FaerBackend, GamOptions, GamWorkspace,
    GaussianMixtureDensity, LeastSquaresWorkspace, LocalQuadraticWorkspace, QuantileRule,
    SmoothSpec, StatsError, equal_tail_interval_sorted, fit_gam, fit_gam_weighted_design,
    gaussian_density, gaussian_local_quadratic_influence_prechecked, normal_ppf,
    silverman_bandwidth,
};

use crate::EstimationError;
use crate::util::range;

mod band;
mod crossfit;
mod policy;
mod support;
mod transform;
use band::{simultaneous_multiplier_band, simultaneous_posterior_band};
use crossfit::{
    CrossFitFold, ensure_crossfit_size, fit_additive, fit_additive_design, require_converged_gam,
    treatment_sigma, treatment_sigma_train_weighted,
};
use policy::{
    DiscreteAtom, additive_policy_rows, exact_discrete_intervention_rows,
    intervention_needs_quadrature, policy_support, static_bayesian_policy,
};
#[cfg(test)]
use support::outcome_tail_ratio;
use support::{
    multivariate_support, push_outcome_tail_diagnostic, push_pseudo_outcome_winsor_shift,
    support_report,
};
use transform::{
    bias_corrected_interval_note, delta_method_interval_note, delta_method_standard_error,
    fieller_elasticity_interval, fieller_interval_note, transform_derivative,
    transform_point_derivative, transform_point_derivative_gradient,
};

/// Refuse a local-polynomial response whose design is singular at `at`.
///
/// With a Gaussian kernel every row carries some weight, so the local design is
/// singular only when fewer distinct treatment values than the polynomial's
/// `order + 1` coefficients carry appreciable weight at `at`: a binary or
/// few-level treatment, or levels spaced far beyond the bandwidth. That is a
/// property of the treatment, not a numerical accident, so it is refused with a
/// reason code that names the queries that answer a discrete treatment. Every
/// other error passes through unchanged.
fn local_design_refusal(
    err: StatsError,
    treatments: &[f64],
    at: f64,
    bandwidth: f64,
) -> EstimationError {
    let StatsError::SingularLocalDesign { order } = err else {
        return err.into();
    };
    let mut levels: Vec<f64> = treatments.iter().copied().filter(|v| v.is_finite()).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    EstimationError::refused(
        antecedent_core::reason_code!("treatment_support_too_discrete"),
        format!(
            "a local-polynomial response of order {order} needs at least {} distinct treatment \
             values carrying kernel weight at each evaluation point; at treatment = {at} \
             (bandwidth {bandwidth:.4}) fewer do, and the treatment takes {} distinct values in \
             all. A binary treatment's contrast is AverageEffect; each level of a discrete \
             treatment is InterventionResponse",
            order + 1,
            levels.len(),
        ),
    )
}

/// Lower clamp on the fitted conditional treatment density in the Kennedy weight.
///
/// A clamped row has an unbounded inverse weight, so every clamp is counted and
/// surfaced as a positivity diagnostic rather than absorbed silently.
const CONDITIONAL_DENSITY_FLOOR: f64 = 1e-8;

/// `max |Y − median| / (1.4826 MAD)` above which least-squares Kennedy nuisances
/// are outside their regularity. Gaussian samples stay near sqrt(2 log n)
/// (about 5 at n = 1e6).
const OUTCOME_TAIL_RATIO_BOUND: f64 = 20.0;

/// Finite stand-in when MAD is zero but the outcome is not constant. Diagnostic
/// values must be finite (the Python view rejects infinities).
const OUTCOME_TAIL_RATIO_UNSCALED: f64 = 1e12;

/// Lower/upper percentile used to winsorize the Kennedy pseudo-outcome when
/// measuring whether extreme φ rows move the fitted curve.
const PSEUDO_OUTCOME_WINSOR_P: f64 = 0.01;

/// Relative curve movement after that winsorization that indicates extreme
/// pseudo-outcome rows are driving the fit.
const PSEUDO_OUTCOME_WINSOR_SHIFT_BOUND: f64 = 0.25;

/// Largest cartesian-product support the exact discrete intervention mixture will evaluate.
///
/// Every combination costs one GAM prediction per row, so this bounds the work at roughly
/// the same order as the Monte-Carlo path it replaced. Joint interventions on a handful of
/// binary or small-categorical variables stay well inside it.
const MAX_EXACT_MIXTURE_COMBINATIONS: usize = 4096;

/// Roughness penalty for the additive-GAM plug-in *target* (Jacobian /
/// directional derivative). Zero: an unpenalized regression spline, so the
/// published gradient carries only sieve-approximation bias, not shrinkage.
const PLUGIN_TARGET_LAMBDA: f64 = 0.0;

/// Cross-fitted Kennedy pseudo-outcome with its positivity accounting.
struct PseudoOutcome {
    values: Vec<f64>,
    density_floor_rows: usize,
    /// Per-row covariate centering `μ̂(A_i, X_i) − ∫μ̂(A_i, x) dP_n(x)` from the
    /// row's own cross-fit fold. The additive outcome nuisance makes this
    /// `ĥ(X_i) − mean ĥ(X)`, constant in the treatment level.
    covariate_centered: Vec<f64>,
}

/// Cross-fitted average-derivative scores with the Riesz representer that built them.
struct AverageDerivativeScores {
    scores: Vec<f64>,
    riesz_weights: Vec<f64>,
}

/// Refusal of an unfinished plug-in target (unpenalized treatment smooths) backfit.
const GAM_TARGET_NOT_CONVERGED: &str =
    "additive GAM target did not converge; refuse rather than publish an unfinished fit";

/// What a caller can do when a plug-in response Jacobian names more than
/// [`MAX_NONPARAMETRIC_RESPONSE_DIM`] treatments.
const JACOBIAN_TREATMENT_LIMIT_REMEDY: &str = "query the Jacobian over at most two treatments \
     at a time (one ResponseJacobian per pair, with the remaining treatments in the adjustment \
     set where the graph licenses it), or query an AverageDerivative per treatment";

/// What a caller can do when a plug-in directional derivative names more than
/// [`MAX_NONPARAMETRIC_RESPONSE_DIM`] treatments.
const DIRECTIONAL_TREATMENT_LIMIT_REMEDY: &str = "restrict the direction to at most two \
     treatments, or query a ResponseJacobian per pair of at most two treatments and take the \
     inner product of the gradient with the direction yourself";

/// Configuration for [`ContinuousResponseEstimator`].
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousResponseOptions {
    /// Deterministic row-index folds used for nuisance cross-fitting.
    pub folds: usize,
    /// Cubic B-spline basis count for each additive nuisance term.
    pub nuisance_basis: usize,
    /// Roughness penalty for nuisance smooths.
    pub nuisance_lambda: f64,
    /// Optional response-kernel bandwidth; normal-reference bandwidth when absent.
    pub bandwidth: Option<f64>,
    /// Local ESS below which support is classified as weak.
    pub minimum_local_ess: f64,
    /// Pointwise confidence level.
    pub confidence_level: f64,
    /// Wild-multiplier replicates for a frequentist fixed-grid simultaneous band.
    /// For Bayesian Gaussian response, this requests a joint-posterior credible
    /// band and is the minimum number of posterior draws; all fitted draws are
    /// used and the result reports their actual count.
    /// `None` preserves pointwise-band behavior.
    pub simultaneous_replicates: Option<u32>,
    /// Deterministic seed for simultaneous-band multipliers.
    pub multiplier_seed: u64,
    /// Export per-row pseudo-outcomes and influence values as support diagnostics.
    ///
    /// Off by default; the estimate is identical either way. When set, three
    /// channels are appended to `support.diagnostics`, for `N` retained rows
    /// and `G` grid points:
    ///
    /// - `response.row_index` — 0-based positions of retained rows in the data
    ///   as received (after caller preprocessing, before the all-finite row
    ///   filter); exact integers stored as `f64`.
    /// - `response.row_pseudo_outcome` — cross-fitted Kennedy pseudo-outcomes
    ///   `φ_i`, outcome units, aligned with `row_index`. Fold = retained-row
    ///   position mod `folds`; nuisances for row `i` exclude `i`'s fold.
    /// - `response.row_influence` — `G * N`, grid-major (`value[g*N + i]`):
    ///   the influence of row `i` on the fitted level at grid point `g`:
    ///   the local-WLS term `w_i · [(XᵀWX)⁻¹]₀ · x_i · (φ_i − x_iᵀβ̂)` plus the
    ///   covariate-marginalization term `(c_i − c̄)/N` (Kennedy et al. 2017,
    ///   Thm. 3; `c_i = μ̂(A_i, X_i) − ∫μ̂(A_i, x) dP_n(x)`). Sums to zero per
    ///   grid point; reported pointwise `SE(g) = √(Σ_i ψ²)`, and both bands
    ///   are `m̂ ± c·SE` with these values.
    ///
    /// These are diagnostics conditional on the fitted nuisances and bandwidth:
    /// second-order nuisance error and bandwidth selection are NOT inside the
    /// influences. Channel ids,
    /// alignment, and layout are stable; values may change when the internal
    /// construction changes. Full contract:
    /// `docs/causal-responses.md#row-diagnostic-export-contract`.
    pub export_row_diagnostics: bool,
}

impl Default for ContinuousResponseOptions {
    fn default() -> Self {
        Self {
            folds: 5,
            nuisance_basis: 6,
            nuisance_lambda: 1.0,
            bandwidth: None,
            minimum_local_ess: 20.0,
            confidence_level: 0.95,
            simultaneous_replicates: None,
            multiplier_seed: 0xA17E_CEDE_0500,
            export_row_diagnostics: false,
        }
    }
}

/// Per-row influence columns for a fitted response functional.
///
/// One column per reported coordinate (a scalar intervention, or each
/// `MeanCurve` grid point). Rows are complete-case rows in `row_index`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseInfluence {
    /// Influence of each retained row on each reported coordinate.
    pub columns: Vec<Vec<f64>>,
    /// Original data-frame row index of each complete-case row.
    pub row_index: Arc<[u32]>,
}

/// Continuous-response estimator with a caller-supplied valid adjustment set.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousResponseEstimator {
    /// Adjustment variables licensed by identification.
    pub adjustment_set: Arc<[VariableId]>,
    /// Numerical and diagnostic options.
    pub options: ContinuousResponseOptions,
}

impl ContinuousResponseEstimator {
    /// Construct with default numerical options.
    #[must_use]
    pub fn new(adjustment_set: impl Into<Arc<[VariableId]>>) -> Self {
        Self {
            adjustment_set: adjustment_set.into(),
            options: ContinuousResponseOptions::default(),
        }
    }

    /// Estimate a response already licensed by the identification layer.
    ///
    /// Only complete observations and the all-observed target are supported here.
    /// Missing rows in any used column are dropped jointly. The estimator never
    /// chooses or validates the causal adequacy of `adjustment_set`.
    ///
    /// # Errors
    ///
    /// Invalid/unsupported queries, non-point identification, insufficient complete
    /// observations, invalid options, or nuisance/smoothing failures.
    pub fn estimate_identified(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
    ) -> Result<CausalResponse, EstimationError> {
        self.estimate_identified_scored(data, query, identification_status, assumptions)
            .map(|(response, _)| response)
    }

    /// Estimate and retain the response-functional influence columns.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate_identified`].
    pub fn estimate_identified_scored(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
    ) -> Result<(CausalResponse, Option<ResponseInfluence>), EstimationError> {
        self.validate(query, identification_status)?;
        let mut influence = None;
        let (value, uncertainty, support, provenance_id) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                let (value, uncertainty, support, scores) =
                    self.mean_curve(data, *outcome, treatment.variable, &treatment.grid.values()?)?;
                influence = Some(scores);
                let provenance =
                    if matches!(uncertainty, ResponseUncertainty::SimultaneousBand { .. }) {
                        "estimate.response.kennedy_dr_simultaneous"
                    } else {
                        "estimate.response.kennedy_dr"
                    };
                (value, uncertainty, support, provenance)
            }
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
                let (value, uncertainty, support) =
                    self.point_derivative(data, *outcome, *treatment, *at, *order, *scale)?;
                (value, uncertainty, support, "estimate.response.point_derivative")
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                let (value, uncertainty, support) =
                    self.average_derivative(data, *outcome, *treatment, weighting)?;
                (value, uncertainty, support, "estimate.response.riesz_ade")
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                let (value, uncertainty, support) =
                    self.jacobian(data, outcomes, treatments, at, *scale)?;
                (value, uncertainty, support, "estimate.response.gam_derivative")
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                let (value, uncertainty, support) =
                    self.directional_derivative(data, outcomes, treatments, at, direction)?;
                (value, uncertainty, support, "estimate.response.gam_derivative")
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                let (value, uncertainty, support, scores) =
                    self.intervention_response(data, *outcome, interventions)?;
                influence = Some(scores);
                (value, uncertainty, support, "estimate.response.intervention_gcomp")
            }
        };
        let assumptions = with_estimation_assumptions(assumptions, &query.functional);
        let interaction_structurally_zero = matches!(
            &query.functional,
            ResponseFunctional::InterventionResponse { interventions, .. } if interventions.len() > 1
        );
        Ok((
            CausalResponse {
                estimand: query.functional.clone(),
                identification_status,
                estimate: ResponseIdentification::PointIdentified(value),
                uncertainty,
                support,
                assumptions,
                provenance_id: Arc::from(provenance_id),
                horizon_identification: None,
                interaction_structurally_zero,
            },
            influence,
        ))
    }

    /// Bayesian generalized-linear response levels, with posterior
    /// coefficient uncertainty propagated through every intervention coordinate.
    /// This estimator is parametric; it makes no doubly robust claim.
    pub fn estimate_bayesian(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        mut assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if matches!(
            &query.functional,
            ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. }
        ) {
            return self.estimate_bayesian_derivative(
                data,
                query,
                identification_status,
                assumptions,
                estimator,
                ctx,
            );
        }
        self.validate(query, identification_status)?;
        if self.options.export_row_diagnostics {
            return Err(EstimationError::unsupported(
                "Bayesian response does not export frequentist row influence diagnostics",
            ));
        }
        let mut support_grid = None;
        let mut joint_levels = Vec::new();
        let mut primary_shift = None;
        let mut stochastic = false;
        let (outcome, treatment, mut grid, scalar) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                (*outcome, treatment.variable, treatment.grid.values()?, false)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                if interventions.iter().any(|iv| matches!(iv, Intervention::Sequence(_))) {
                    return Err(EstimationError::unsupported(
                        "static Bayesian response does not accept temporal Sequence",
                    ));
                }
                stochastic =
                    interventions.iter().any(|iv| matches!(iv, Intervention::Stochastic { .. }));
                let (t, level, shift) = static_bayesian_policy(&interventions[0])?;
                for intervention in &interventions[1..] {
                    let (target, level, shift) = static_bayesian_policy(intervention)?;
                    joint_levels.push((target, level, shift));
                }
                let sample = CompleteSample::read(data, *outcome, &[t], &self.adjustment_set)?;
                if level.is_none() {
                    primary_shift = Some(shift);
                    let (lo, hi) = range(&sample.treatments);
                    support_grid = Some(vec![lo + shift, hi + shift]);
                }
                let level = level.unwrap_or_else(|| {
                    sample.treatments.iter().sum::<f64>() / sample.len() as f64 + shift
                });
                (*outcome, t, vec![level], true)
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "Bayesian response supports MeanCurve, InterventionResponse, and derivative functionals",
                ));
            }
        };
        let treatments: Vec<_> = std::iter::once(treatment)
            .chain(joint_levels.iter().map(|(target, _, _)| *target))
            .collect();
        let sample = CompleteSample::read(data, outcome, &treatments, &self.adjustment_set)?;
        let family = estimator.glm_family();
        if stochastic && !matches!(family, antecedent_stats::GlmFamily::GaussianIdentity) {
            return Err(EstimationError::unsupported(
                "non-Gaussian Bayesian response does not integrate stochastic policies by their mean",
            ));
        }
        let n = sample.len();
        if let Some(shift) = primary_shift {
            grid[0] = sample.treatments.iter().sum::<f64>() / n as f64 + shift;
            let (lo, hi) = range(&sample.treatments);
            support_grid = Some(vec![lo + shift, hi + shift]);
        }

        let mut covs: Vec<_> = treatments
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &id)| (id, &sample.treatment_matrix[i * n..(i + 1) * n]))
            .collect();
        covs.extend(
            self.adjustment_set
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, &sample.adjustment[i * n..(i + 1) * n])),
        );
        let design = antecedent_stats::CompiledDesign::linear_adjustment(
            &sample.treatments,
            &covs,
            &sample.outcome,
            &sample.keep,
        )?;
        let prep = crate::PreparedBayesianProblem {
            design,
            method: Arc::from("response.backdoor"),
            adjustment_set: self.adjustment_set.clone(),
            active: 1.0,
            control: 0.0,
            overlap: crate::OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: crate::SerialDependence::Iid,
        };
        let posterior = estimator.fit(
            &prep,
            identification_status,
            &mut crate::BayesianGCompWorkspace::default(),
            ctx,
        )?;
        let mut weights: Vec<_> =
            prep.design.matrix.chunks(n).map(|c| c.iter().sum::<f64>() / n as f64).collect();
        for (i, (_, level, shift)) in joint_levels.iter().enumerate() {
            weights[i + 2] = level.unwrap_or(weights[i + 2] + shift);
        }
        let mut means = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut sds = Vec::new();
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut ess = Vec::new();
        let mut density = Vec::new();
        // The draw vector each grid point's interval is the quantiles of is
        // retained on the published uncertainty (`CredibleDraws`).
        let mut retained: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
        let coefficient_columns: Vec<&[f64]> = (0..prep.design.ncols)
            .map(|index| {
                let column = posterior
                    .draws
                    .schema
                    .quantities
                    .iter()
                    .position(|quantity| matches!(quantity,
                        antecedent_prob::PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index))
                    .ok_or_else(|| EstimationError::stats_msg("Bayesian response missing coefficient draw"))?;
                posterior.draws.column(column).map_err(EstimationError::from)
            })
            .collect::<Result<_, _>>()?;
        for &dose in &grid {
            weights[1] = dose;
            let values = if matches!(family, antecedent_stats::GlmFamily::GaussianIdentity) {
                crate::bayesian::linear_response_draws(&posterior, &weights)?
            } else {
                let mut values = vec![0.0; posterior.draws.n_draws];
                for (draw, value) in values.iter_mut().enumerate() {
                    for row in 0..n {
                        let mut eta = 0.0;
                        for (column, coefficients) in coefficient_columns.iter().enumerate() {
                            let design_value = if column == 1 {
                                primary_shift.map_or(dose, |shift| sample.treatments[row] + shift)
                            } else if column >= 2 && column < 1 + treatments.len() {
                                let (level, shift) =
                                    (joint_levels[column - 2].1, joint_levels[column - 2].2);
                                level.unwrap_or(
                                    sample.treatment_matrix[(column - 1) * n + row] + shift,
                                )
                            } else {
                                prep.design.matrix[column * n + row]
                            };
                            eta += coefficients[draw] * design_value;
                        }
                        *value += family.mean_from_eta(eta);
                    }
                    *value /= n as f64;
                }
                values
            };
            let (mean, lo, hi, sd) = crate::bayesian::summarize_linear_response_draws(
                values.clone(),
                self.options.confidence_level,
            )?;
            retained.push(values);
            means.push(mean);
            lower.push(lo);
            upper.push(hi);
            sds.push(sd);
        }
        let draws = Some(CredibleDraws::columns(posterior.draws.n_draws, &retained));
        if self
            .options
            .simultaneous_replicates
            .is_some_and(|minimum| posterior.draws.n_draws < minimum as usize)
        {
            return Err(EstimationError::unsupported(
                "Bayesian simultaneous band requires at least the requested number of posterior draws",
            ));
        }
        let support_points = support_grid.as_deref().unwrap_or(&grid);
        for &dose in support_points {
            let kernels: Vec<_> = sample
                .treatments
                .iter()
                .map(|t| (-0.5 * ((t - dose) / bandwidth).powi(2)).exp())
                .collect();
            let sum = kernels.iter().sum::<f64>();
            ess.push(sum * sum / kernels.iter().map(|k| k * k).sum::<f64>().max(f64::MIN_POSITIVE));
            density.push(sum / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()));
        }
        let mut support = support_report(
            support_points,
            &sample.treatments,
            &ess,
            density,
            self.options.minimum_local_ess,
            0,
        );
        if treatments.len() > 1 {
            support.status = SupportStatus::Extrapolative;
            support.point_status = None;
            support.query_region = SupportRegion {
                minima: (0..treatments.len()).map(|i| sample.treatment_column_range(i).0).collect(),
                maxima: (0..treatments.len()).map(|i| sample.treatment_column_range(i).1).collect(),
            };
            support.warnings.push(Diagnostic::new(
                "response.joint_support_unverified", DiagnosticKind::Support, DiagnosticSeverity::Warning,
                "per-treatment bounds do not certify joint policy support; posterior uncertainty conditions on the declared additive outcome model and empirical covariate distribution",
            ));
        }
        if stochastic {
            support.status = SupportStatus::Extrapolative;
            support.point_status = None;
            support.warnings.push(Diagnostic::new(
                "response.stochastic_policy_support_unverified", DiagnosticKind::Support, DiagnosticSeverity::Warning,
                "the Gaussian additive model integrates stochastic policies by their exact means; local support at the mean does not certify support over the policy distribution; intervals describe the policy mean, not a predictive draw",
            ));
        }
        assumptions.entries.extend(posterior.assumptions.entries);
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("bayesian.response.linear_additive"), description: Arc::from(format!("{family:?} additive outcome mechanism on its link scale; each posterior draw is evaluated on the outcome scale over the empirical covariate distribution; credible intervals use coherent joint draws over the requested grid, without nuisance-distribution uncertainty")),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("response.bayesian") },
            scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
        });
        let value = if scalar {
            ResponseValue::Scalar(means[0])
        } else {
            ResponseValue::Surface {
                dimension: 1,
                grid: Arc::from(grid),
                mean: Arc::from(means.clone()),
            }
        };
        let uncertainty = if scalar {
            ResponseUncertainty::Scalar {
                standard_error: sds[0],
                level: self.options.confidence_level,
                lower: lower[0],
                upper: upper[0],
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                draws,
            }
        } else if self.options.simultaneous_replicates.is_some() {
            simultaneous_posterior_band(
                &means,
                &retained,
                &lower,
                &upper,
                &sds,
                self.options.confidence_level,
            )?
        } else {
            ResponseUncertainty::PointwiseBand {
                level: self.options.confidence_level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                draws,
            }
        };
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(value),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from("estimate.response.bayesian"),
            horizon_identification: None,
            interaction_structurally_zero: matches!(
                &query.functional,
                ResponseFunctional::InterventionResponse { interventions, .. } if interventions.len() > 1
            ),
        })
    }

    fn estimate_bayesian_derivative(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if estimator.prior.is_some() {
            return Err(EstimationError::unsupported(
                "coefficient priors require a mapping to the Riesz/local-polynomial/GAM parameterization; a linear response prior cannot be silently ignored",
            ));
        }
        query.validate()?;
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "Bayesian derivatives require point identification",
            });
        }
        self.validate(query, identification_status)?;
        let mut assumptions = with_estimation_assumptions(assumptions, &query.functional);
        let level = self.options.confidence_level;
        let draws_n = crate::require_bayesian_n_draws(estimator.n_draws)?;
        let (value, uncertainty, support, assumption_id, assumption_text, provenance) = match &query
            .functional
        {
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                if !matches!(weighting, DerivativeWeighting::Observed) {
                    return Err(EstimationError::unsupported(
                        "Bayesian ADE currently supports observed-law weighting only",
                    ));
                }
                let sample =
                    CompleteSample::read(data, *outcome, &[*treatment], &self.adjustment_set)?;
                let (_, _, support) =
                    self.average_derivative(data, *outcome, *treatment, weighting)?;
                // Folds, their training designs and knots are fixed across draws:
                // expanded once here, refit under each draw's weights.
                let folds = self.cross_fit_folds(&sample)?;
                // One RNG stream per draw: draws are independent, so they run across
                // the context's thread budget and return in draw order.
                let values = ctx.map_indexed::<_, EstimationError, _>(draws_n, |draw, ctx| {
                    if ctx.cancellation.is_cancelled() {
                        return Err(EstimationError::unsupported("Bayesian ADE cancelled"));
                    }
                    let mut rng = ctx
                        .rng
                        .stream_for(StreamDomain::Bayesian, (0xADEB_0001_u64 << 32) | draw as u64);
                    let weights = bootstrap_weights(sample.len(), &mut rng);
                    self.weighted_cross_fitted_ade(&sample, &folds, &weights)
                })?;
                let (mean, lo, hi, sd) = summarize_scalar_draws(&values, level)?;
                (
                    ResponseValue::Scalar(mean),
                    credible_scalar_uncertainty(sd, level, lo, hi, values),
                    support,
                    "bayesian.derivative.riesz_weighted_cross_fit",
                    "Each Rubin Bayesian-bootstrap draw cross-fits the additive-GAM outcome μ and Gaussian treatment law α under Dirichlet(1,...,1)/Exp(1) row weights: every fold refits both nuisances on its weighted training rows and scores its held-out rows with the Riesz ADE φ = ∂_a μ̂_w + α_w (Y − μ̂_w), and the draw is the weighted mean of the held-out scores. This is not a frozen-score reweight of the first-fit φ_i. Held fixed: fold assignment, spline knots, and penalty. Estimator identity stays response.riesz_ade",
                    "estimate.response.riesz_ade",
                )
            }
            ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => {
                self.validate(query, identification_status)?;
                let (point, _, mut support, provenance) = match &query.functional {
                    ResponseFunctional::PointDerivative {
                        outcome,
                        treatment,
                        at,
                        order,
                        scale,
                    } => {
                        let (value, uncertainty, support) =
                            self.point_derivative(data, *outcome, *treatment, *at, *order, *scale)?;
                        (value, uncertainty, support, "estimate.response.point_derivative")
                    }
                    ResponseFunctional::DirectionalDerivative {
                        outcomes,
                        treatments,
                        at,
                        direction,
                    } => {
                        let (value, uncertainty, support) =
                            self.directional_derivative(data, outcomes, treatments, at, direction)?;
                        (value, uncertainty, support, "estimate.response.gam_derivative")
                    }
                    ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                        let (value, uncertainty, support) =
                            self.jacobian(data, outcomes, treatments, at, *scale)?;
                        (value, uncertainty, support, "estimate.response.gam_derivative")
                    }
                    _ => unreachable!(),
                };
                // The Frequentist withheld / bias-corrected notes describe the
                // analytic interval; the draws below supply their own interval and
                // their own disclosure.
                support.warnings.retain(|warning| {
                    !matches!(
                        warning.code.as_ref(),
                        "response.derivative_interval_withheld"
                            | "response.derivative_interval_bias_corrected"
                            | "response.derivative_interval_delta_method"
                            | "response.derivative_interval_fieller"
                            | "response.derivative_interval_unbounded"
                    )
                });
                let point_derivative =
                    matches!(query.functional, ResponseFunctional::PointDerivative { .. });
                // Every draw refits its nuisances: point draws rebuild the
                // cross-fitted Kennedy pseudo-outcome under the draw's row weights
                // (the bandwidth is the caller's fixed value, never data-selected);
                // GAM draws refit coefficients with the same continuous weights.
                let (samples, bandwidth) = match &query.functional {
                    ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
                        let sample = CompleteSample::read(
                            data,
                            *outcome,
                            &[*treatment],
                            &self.adjustment_set,
                        )?;
                        let bandwidth = self
                            .options
                            .bandwidth
                            .unwrap_or(silverman_bandwidth(&sample.treatments)?);
                        (vec![sample], bandwidth)
                    }
                    ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
                    | ResponseFunctional::Jacobian { outcomes, treatments, .. } => (
                        read_shared_complete_samples(
                            data,
                            outcomes,
                            treatments,
                            &self.adjustment_set,
                        )?,
                        0.0,
                    ),
                    _ => unreachable!(),
                };
                let n = samples[0].len();
                // Fold assignment, knots and bases are fixed across draws: the point
                // route's cross-fitting folds and the GAM routes' plug-in target
                // designs are expanded once here and refit under each draw's weights.
                let folds =
                    if point_derivative { self.cross_fit_folds(&samples[0])? } else { vec![] };
                let target_designs: Vec<AdditiveDesign> = if point_derivative {
                    vec![]
                } else {
                    samples
                        .iter()
                        .map(|sample| self.outcome_target_design(sample))
                        .collect::<Result<_, _>>()?
                };
                // One RNG stream per draw: draws are independent, so they run across the
                // context's thread budget and return in draw order. Each draw yields its
                // point coordinates (local-quadratic, bias-corrected) or its GAM vector,
                // plus the GAM fits' effective degrees of freedom for the HC1-type spread
                // correction (`inflate_draws_by_edf`).
                let draws = ctx.map_indexed::<_, EstimationError, _>(draws_n, |draw, ctx| {
                    if ctx.cancellation.is_cancelled() {
                        return Err(EstimationError::unsupported("Bayesian derivative cancelled"));
                    }
                    let mut rng = ctx
                        .rng
                        .stream_for(StreamDomain::Bayesian, (0xADEB_0002_u64 << 32) | draw as u64);
                    let weights = bootstrap_weights(n, &mut rng);
                    let mut out = DerivativeDraw {
                        scalar: 0.0,
                        corrected: 0.0,
                        vector: Vec::new(),
                        edf: vec![0.0; samples.len()],
                    };
                    match &query.functional {
                        ResponseFunctional::PointDerivative { at, order, scale, .. } => {
                            let pseudo = self
                                .cross_fitted_pseudo_outcome_weighted(
                                    &samples[0],
                                    &folds,
                                    Some(&weights),
                                )?
                                .values;
                            let p = antecedent_stats::gaussian_local_quadratic_weighted(
                                &samples[0].treatments,
                                &pseudo,
                                *at,
                                bandwidth,
                                &weights,
                            )
                            .map_err(|err| {
                                local_design_refusal(err, &samples[0].treatments, *at, bandwidth)
                            })?;
                            out.scalar = transform_point_derivative(
                                p.value,
                                p.first_derivative,
                                p.second_derivative,
                                *at,
                                *order,
                                *scale,
                            )?;
                            let c = antecedent_stats::gaussian_local_quadratic_bias_corrected(
                                &samples[0].treatments,
                                &pseudo,
                                *at,
                                bandwidth,
                                Some(&weights),
                            )
                            .map_err(|err| {
                                local_design_refusal(err, &samples[0].treatments, *at, bandwidth)
                            })?;
                            out.corrected = transform_point_derivative(
                                c.value,
                                c.first_derivative,
                                c.second_derivative,
                                *at,
                                *order,
                                *scale,
                            )?;
                        }
                        ResponseFunctional::DirectionalDerivative { at, direction, .. } => {
                            for (s, sample) in samples.iter().enumerate() {
                                let fit = Self::fit_outcome_target_weighted(
                                    sample,
                                    &target_designs[s],
                                    Some(&weights),
                                )?;
                                out.edf[s] += fit.edf_approx;
                                let (_, gradient) =
                                    Self::plugin_gradient_at_fit(&fit, sample, at, Some(&weights))?;
                                out.vector.push(
                                    gradient.iter().zip(direction.iter()).map(|(a, b)| a * b).sum(),
                                );
                            }
                        }
                        ResponseFunctional::Jacobian { at, scale, .. } => {
                            for (s, sample) in samples.iter().enumerate() {
                                let fit = Self::fit_outcome_target_weighted(
                                    sample,
                                    &target_designs[s],
                                    Some(&weights),
                                )?;
                                out.edf[s] += fit.edf_approx;
                                let (level, gradient) =
                                    Self::plugin_gradient_at_fit(&fit, sample, at, Some(&weights))?;
                                for (j, raw) in gradient.into_iter().enumerate() {
                                    out.vector
                                        .push(transform_derivative(raw, at[j], level, *scale)?);
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    Ok(out)
                })?;
                let scalars: Vec<f64> = draws.iter().map(|d| d.scalar).collect();
                let corrected_scalars: Vec<f64> = draws.iter().map(|d| d.corrected).collect();
                let mut edf_sum = vec![0.0; samples.len()];
                for draw in &draws {
                    for (total, edf) in edf_sum.iter_mut().zip(&draw.edf) {
                        *total += edf;
                    }
                }
                let vectors: Vec<Vec<f64>> = draws.into_iter().map(|d| d.vector).collect();
                let (value, uncertainty) = if point_derivative {
                    // Reported value: posterior mean of the local-quadratic estimator.
                    // Interval and SD: the bias-corrected draws on the same weights.
                    let (mean, _, _, _) = summarize_scalar_draws(&scalars, level)?;
                    let (corrected_mean, lo, hi, sd) =
                        summarize_scalar_draws(&corrected_scalars, level)?;
                    support.warnings.push(bias_corrected_interval_note(true, mean, corrected_mean));
                    // The retained draws are the bias-corrected ones: the interval's.
                    (
                        ResponseValue::Scalar(mean),
                        credible_scalar_uncertainty(sd, level, lo, hi, corrected_scalars),
                    )
                } else {
                    let dim = vectors.first().map_or(0, Vec::len);
                    let mut means = vec![0.0; dim];
                    let mut lower = vec![0.0; dim];
                    let mut upper = vec![0.0; dim];
                    // Retained per coordinate *after* the edf inflation: the
                    // vector the band's quantiles were taken from.
                    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(dim);
                    for j in 0..dim {
                        let mut col: Vec<f64> = vectors.iter().map(|row| row[j]).collect();
                        // Columns are outcome-major (Jacobian) or one per outcome
                        // (directional); either way the outcome index is j·S/dim.
                        let edf = edf_sum[j * samples.len() / dim] / draws_n as f64;
                        inflate_draws_by_edf(&mut col, n, edf);
                        let (mean, lo, hi, _) = summarize_scalar_draws(&col, level)?;
                        means[j] = mean;
                        lower[j] = lo;
                        upper[j] = hi;
                        columns.push(col);
                    }
                    let draws = Some(CredibleDraws::columns(draws_n, &columns));
                    let value = match &point {
                        ResponseValue::Jacobian { outcomes, treatments, .. } => {
                            ResponseValue::Jacobian {
                                outcomes: *outcomes,
                                treatments: *treatments,
                                values: Arc::from(means),
                            }
                        }
                        _ => ResponseValue::Vector(Arc::from(means)),
                    };
                    (
                        value,
                        ResponseUncertainty::PointwiseBand {
                            level,
                            lower: Arc::from(lower),
                            upper: Arc::from(upper),
                            interpretation: antecedent_core::IntervalInterpretation::Credible,
                            draws,
                        },
                    )
                };
                (
                    value,
                    uncertainty,
                    support,
                    "bayesian.derivative.estimator_bootstrap",
                    if point_derivative {
                        "Dirichlet(1,...,1)/Exp(1) row-weight posterior of the Kennedy-DR point derivative: every draw refits the cross-fitted additive-GAM outcome and Gaussian treatment nuisances on its weighted training folds, rebuilds the pseudo-outcome, and evaluates the weighted local quadratic (posterior mean = reported value) and its robust bias-corrected coordinate at the same caller-fixed bandwidth (local-cubic slope for a first derivative, local-quartic level and curvature otherwise) (quantiles = credible interval, SD = standard_error). Held fixed: the caller bandwidth, fold assignment, spline knots, and penalty. Not a frozen-pseudo-outcome reweight. Estimator identity stays estimate.response.point_derivative"
                    } else {
                        "Dirichlet(1,...,1)/Exp(1) row-weight posterior of the additive-GAM plug-in gradient: every draw refits the GAM coefficients under the draw's row weights with fixed knots and penalty; the draw spread is inflated around the posterior mean by sqrt(n/(n − edf)) (the row-weight spread is an HC0 sandwich, which understates a fitted-coefficient functional's variance by the fit's leverage; edf is the penalized fit's effective degrees of freedom), and the pointwise band takes exchangeable-rank (type-6) quantiles of the draws. The band inherits the additive-surface restriction and any penalized-spline smoothing bias at the evaluation point; no nuisance-selection uncertainty"
                    },
                    provenance,
                )
            }
            _ => {
                return Err(EstimationError::unsupported("not a derivative functional"));
            }
        };
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from(assumption_id),
                description: Arc::from(assumption_text),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("response.bayesian.derivative"),
            },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(value),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from(provenance),
            horizon_identification: None,
            interaction_structurally_zero: false,
        })
    }

    /// Linear Gaussian design used by Bayesian response and derivative transfer.
    ///
    /// # Errors
    ///
    /// Incomplete rows or a singular design.
    pub fn prepare_linear_response_problem(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatments: &[VariableId],
    ) -> Result<crate::PreparedBayesianProblem, EstimationError> {
        let sample = CompleteSample::read(data, outcome, treatments, &self.adjustment_set)?;
        let n = sample.len();
        let mut covs: Vec<_> = treatments
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &id)| (id, &sample.treatment_matrix[i * n..(i + 1) * n]))
            .collect();
        covs.extend(
            self.adjustment_set
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, &sample.adjustment[i * n..(i + 1) * n])),
        );
        let design = antecedent_stats::CompiledDesign::linear_adjustment(
            &sample.treatments,
            &covs,
            &sample.outcome,
            &sample.keep,
        )?;
        Ok(crate::PreparedBayesianProblem {
            design,
            method: Arc::from("response.linear"),
            adjustment_set: self.adjustment_set.clone(),
            active: 1.0,
            control: 0.0,
            overlap: crate::OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: crate::SerialDependence::Iid,
        })
    }

    fn validate(
        &self,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
    ) -> Result<(), EstimationError> {
        query.validate()?;
        if query.temporal.is_some() {
            return Err(EstimationError::unsupported(
                "static continuous-response estimator does not execute temporal response queries; use TemporalResponseEstimator",
            ));
        }
        if query.observation != ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "continuous-response estimator currently requires complete observations",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "continuous-response estimation requires point identification",
            });
        }
        let o = &self.options;
        if matches!(query.functional, ResponseFunctional::PointDerivative { .. })
            && o.bandwidth.is_none()
        {
            // Silverman's rule is a level/KDE rate. Using it for m'/m'' oversmooths the
            // derivative toward zero while still publishing Identity-scale sandwich SEs —
            // the same silent-undersmoothing hole simultaneous bands already refuse.
            return Err(EstimationError::unsupported(
                "point derivatives require an explicit bandwidth; Silverman's rule is not an undersmoothing rule for m' or m''",
            ));
        }
        if o.folds < 2
            || o.nuisance_basis < 4
            || !o.nuisance_lambda.is_finite()
            || o.nuisance_lambda < 0.0
            || o.bandwidth.is_some_and(|h| !h.is_finite() || h <= 0.0)
            || !o.minimum_local_ess.is_finite()
            || o.minimum_local_ess <= 0.0
            || !o.confidence_level.is_finite()
            // A level of exactly 0 yields z = 0 and a zero-width interval that would be
            // reported as a band at level 0 rather than refused.
            || o.confidence_level <= 0.0
            || o.confidence_level >= 1.0
            || o.simultaneous_replicates.is_some_and(|replicates| replicates < 100)
            || (o.simultaneous_replicates.is_some() && o.bandwidth.is_none())
        {
            return Err(EstimationError::unsupported("invalid continuous-response options"));
        }
        Ok(())
    }

    fn mean_curve(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        grid: &[f64],
    ) -> Result<
        (ResponseValue, ResponseUncertainty, SupportReport, ResponseInfluence),
        EstimationError,
    > {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let PseudoOutcome { values: pseudo, density_floor_rows, covariate_centered } =
            self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut mean = Vec::with_capacity(grid.len());
        let mut lower = Vec::with_capacity(grid.len());
        let mut upper = Vec::with_capacity(grid.len());
        let mut ess = Vec::with_capacity(grid.len());
        let mut density = Vec::with_capacity(grid.len());
        let n = sample.len();
        let g = grid.len();
        // One G×n buffer: local-polynomial IFs plus the grid-constant covariate
        // term written in place (ENG-012). The additive-μ claim that makes the
        // extra term the same at every `a` is disclosed below (SUS-002).
        let mut influences = vec![0.0; g * n];
        let mut robust_se = Vec::with_capacity(g);
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        if sample.treatments.iter().chain(&pseudo).any(|v| !v.is_finite()) {
            return Err(
                StatsError::Shape { message: "local quadratic inputs must be finite" }.into()
            );
        }
        // Per-row marginalization contribution `(c_i − c̄) / n` (see the loop).
        // Centering over rows keeps every grid column summing to zero exactly,
        // as the local WLS term does, because the cross-fit folds center `c`
        // on their own training rows.
        let covariate_term: Vec<f64> = {
            let n_rows = n as f64;
            let center = covariate_centered.iter().sum::<f64>() / n_rows;
            covariate_centered.iter().map(|c| (c - center) / n_rows).collect()
        };
        let mut local = LocalQuadraticWorkspace::default();
        for (g_idx, &at) in grid.iter().enumerate() {
            let fit = gaussian_local_quadratic_influence_prechecked(
                &mut local,
                &sample.treatments,
                &pseudo,
                at,
                bandwidth,
            )
            .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
            let point = fit.point;
            mean.push(point.value);
            // Kennedy et al. (2017, Thm. 3): the pseudo-outcome's marginalization
            // `∫μ̂(a, x) dP_n(x)` averages every row's covariates, so each row also
            // moves the fitted level through `e₀ᵀD⁻¹ P_n[g K {μ̂(·, X_i) − m̂(·)}]`.
            // With an additive μ̂ that bracket is the constant `ĥ(X_i) − mean ĥ`
            // and `P_n[g K] = D e₀`, so the term is exactly `covariate_term[i]`,
            // the same at every grid point. Treating the pseudo-outcomes as fixed
            // data drops it; without it, calibration measured 0.80–0.86 pointwise
            // coverage at nominal 0.90 without it.
            let row = &mut influences[g_idx * n..(g_idx + 1) * n];
            for (slot, (local_if, centered)) in
                row.iter_mut().zip(fit.influences.iter().zip(&covariate_term))
            {
                *slot = local_if + centered;
            }
            let standard_error = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            // The pointwise band uses the same influence-based standard error the
            // simultaneous band standardizes by, so the two are nested by
            // construction rather than being two different variance estimates at
            // the same nominal level.
            lower.push(point.value - z * standard_error);
            upper.push(point.value + z * standard_error);
            ess.push(point.local_ess);
            density.push(
                point.weight_sum / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            );
            robust_se.push(standard_error);
        }
        let mut support = support_report(
            grid,
            &sample.treatments,
            &ess,
            density,
            self.options.minimum_local_ess,
            density_floor_rows,
        );
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        push_pseudo_outcome_winsor_shift(
            &mut support,
            &sample.treatments,
            &pseudo,
            grid,
            &mean,
            bandwidth,
            &mut local,
        );
        support.warnings.push(antecedent_core::Diagnostic::new(
            "response.kennedy_dr.additive_covariate_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "the grid-constant Kennedy IF covariate term (c_i − c̄)/n assumes the fitted \
             additive outcome restriction; it is not re-derived for treatment–covariate \
             interactions in μ̂",
        ));
        if self.options.export_row_diagnostics {
            let flat_influences = influences.clone();
            // Downstream result layers require finite diagnostic values; refuse
            // rather than publish a non-finite influence into the export channel.
            if flat_influences.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::unsupported(
                    "row-diagnostic export encountered a non-finite influence value",
                ));
            }
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_index"),
                values: Arc::from(
                    sample.keep.iter().map(|&index| index as f64).collect::<Vec<_>>(),
                ),
                detail: Arc::from("original dataframe row index of each retained complete row"),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_pseudo_outcome"),
                values: Arc::from(pseudo),
                detail: Arc::from("cross-fitted Kennedy pseudo-outcome per retained row"),
            });
            support.diagnostics.push(SupportDiagnostic {
                id: Arc::from("response.row_influence"),
                values: Arc::from(flat_influences),
                detail: Arc::from(format!(
                    "row-major by grid point, grid_len={}, n={n}: value[g*N + i]",
                    grid.len()
                )),
            });
        }
        let uncertainty = if let Some(replicates) = self.options.simultaneous_replicates {
            simultaneous_multiplier_band(
                &mean,
                &influences,
                n,
                &robust_se,
                self.options.confidence_level,
                replicates,
                self.options.multiplier_seed,
            )?
        } else {
            ResponseUncertainty::PointwiseBand {
                level: self.options.confidence_level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        };
        let row_index = row_ids_u32(&sample.keep)?;
        // Local-polynomial exports are contributions to the estimate (order 1/n).
        // Shared IF covariance expects unnormalised row scores (order 1).
        let scale = n as f64;
        let columns = (0..g)
            .map(|g_idx| influences[g_idx * n..(g_idx + 1) * n].iter().map(|v| v * scale).collect())
            .collect();
        let scores = ResponseInfluence { columns, row_index };
        Ok((
            ResponseValue::Surface {
                grid: Arc::from(grid.to_vec()),
                dimension: 1,
                mean: Arc::from(mean),
            },
            uncertainty,
            support,
            scores,
        ))
    }

    fn intervention_response(
        &self,
        data: &TabularData,
        outcome: VariableId,
        interventions: &[Intervention],
    ) -> Result<
        (ResponseValue, ResponseUncertainty, SupportReport, ResponseInfluence),
        EstimationError,
    > {
        let mut treatments = Vec::with_capacity(interventions.len());
        for intervention in interventions {
            let Some(variable) = intervention.primary_variable() else {
                return Err(EstimationError::unsupported(
                    "intervention-response estimation requires one target per intervention",
                ));
            };
            if treatments.contains(&variable) {
                return Err(EstimationError::unsupported(
                    "intervention-response targets must be unique",
                ));
            }
            if matches!(intervention, Intervention::Soft { .. } | Intervention::Sequence(_)) {
                return Err(EstimationError::unsupported(
                    "soft and sequenced intervention responses require a structural model",
                ));
            }
            treatments.push(variable);
        }
        let sample = CompleteSample::read(data, outcome, &treatments, &self.adjustment_set)?;
        let (fit, target_fallback) = self.fit_outcome_target(&sample)?;
        // Discrete policies (Set/Shift/Bernoulli/Categorical) are integrated exactly as a
        // finite mixture. Monte Carlo through a continuous spline would treat categorical
        // codes as ordered coordinates and approximate a sum that has a closed form.
        // Gaussian policies are integrated by Gauss–Hermite quadrature: the outcome
        // model is additive, so E[μ(A_1, …, A_k, X)] depends only on each policy's own
        // marginal law and the same rule serves every row, with no sampling error.
        let row_means = if interventions.iter().any(intervention_needs_quadrature) {
            additive_policy_rows(&fit, &sample, interventions)?
        } else {
            exact_discrete_intervention_rows(&fit, &sample, interventions)?
        };
        let n = row_means.len() as f64;
        let estimate = row_means.iter().sum::<f64>() / n;
        let psi = intervention_plugin_influence(&fit, &sample, interventions, &row_means)?;
        let se = if n > 1.0 {
            (psi.iter().map(|x| x * x).sum::<f64>() / (n * (n - 1.0))).sqrt()
        } else {
            f64::NAN
        };
        let level = self.options.confidence_level;
        let z = normal_ppf(0.5 + level / 2.0);
        let scores =
            ResponseInfluence { columns: vec![psi], row_index: row_ids_u32(&sample.keep)? };
        let minima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).0).collect();
        let maxima: Vec<f64> =
            (0..treatments.len()).map(|column| sample.treatment_column_range(column).1).collect();
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::Scalar {
                standard_error: se,
                level,
                lower: estimate - z * se,
                upper: estimate + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
            SupportReport {
                status: SupportStatus::Extrapolative,
                query_region: SupportRegion {
                    minima: Arc::from(minima.clone()),
                    maxima: Arc::from(maxima.clone()),
                },
                diagnostics: vec![SupportDiagnostic {
                    id: Arc::from("response.intervention_observed_bounds"),
                    values: Arc::from(minima.into_iter().chain(maxima).collect::<Vec<_>>()),
                    detail: Arc::from(
                        "observed minima followed by maxima; policy support is not certified",
                    ),
                }],
                warnings: intervention_plugin_warnings(target_fallback.as_deref()),
                point_status: None,
            },
            scores,
        ))
    }

    fn point_derivative(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        at: f64,
        order: u8,
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let PseudoOutcome { values: pseudo, density_floor_rows, .. } =
            self.cross_fitted_pseudo_outcome(&sample)?;
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let local = antecedent_stats::gaussian_local_quadratic_influence(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
        )
        .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
        let point = local.point;
        let estimate = transform_point_derivative(
            point.value,
            point.first_derivative,
            point.second_derivative,
            at,
            order,
            scale,
        )?;
        // The published interval is robust bias-corrected (CCT, pilot bandwidth
        // = h): centered at the higher-order coordinate (local-cubic slope for
        // order 1, local-quartic curvature for order 2) and studentized by its
        // own sandwich. The conventional local-quadratic interval ignores the
        // O(h²·m''') smoothing bias of m̂' (and the O(h²·m'''') bias of m̂'')
        // and under-covers the true derivative at an MSE-sized bandwidth
        // (tests/v19_derivative_calibration.rs).
        let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
            None,
        )
        .map_err(|err| local_design_refusal(err, &sample.treatments, at, bandwidth))?;
        let corrected_estimate = transform_point_derivative(
            corrected.value,
            corrected.first_derivative,
            corrected.second_derivative,
            at,
            order,
            scale,
        )?;
        let derivative_se = if order == 1 {
            corrected.robust_first_derivative_standard_error
        } else {
            corrected.robust_second_derivative_standard_error
        };
        // A directly published coordinate SE (identity level/curvature, or the
        // log-treatment order-1 scale-up `|at|·SE(m')`) versus a nonlinear
        // transform of the coordinates that takes the full delta method.
        let delta_transformed = !matches!(
            (order, scale),
            (1 | 2, DerivativeScale::Identity) | (1, DerivativeScale::LogTreatment)
        );
        let standard_error = match (order, scale) {
            (1 | 2, DerivativeScale::Identity) => derivative_se,
            (1, DerivativeScale::LogTreatment) => at.abs() * derivative_se,
            // Every other order/scale is a nonlinear transform of the local
            // coordinates θ = (m, m', m''); its interval is the full delta method
            // on the joint coordinate covariance Σ_θ, not a partial one. The
            // gradient is evaluated at the bias-corrected coordinates the interval
            // is centred on; Σ_θ uses the same bias-corrected local-cubic
            // slope and local-quartic level/curvature as that center.
            _ => {
                let gradient = transform_point_derivative_gradient(
                    corrected.value,
                    corrected.first_derivative,
                    corrected.second_derivative,
                    at,
                    order,
                    scale,
                );
                delta_method_standard_error(&corrected.coefficient_covariance, &gradient)
            }
        };
        // A ratio of estimated coordinates needs denominator uncertainty in
        // the interval itself. A symmetric delta interval can badly under-cover
        // when the response level is small even though that level is positive.
        // Fieller inverts the joint normal test for a*m'/m. If its confidence
        // set is unbounded, a finite scalar interval cannot represent it.
        let fieller = (order == 1 && scale == DerivativeScale::LogLog).then(|| {
            fieller_elasticity_interval(
                corrected.value,
                corrected.first_derivative,
                at,
                &corrected.coefficient_covariance,
                normal_ppf(0.5 + self.options.confidence_level / 2.0),
            )
        });
        let mut support = support_report(
            &[at],
            &sample.treatments,
            &[point.local_ess],
            vec![
                point.weight_sum
                    / (sample.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()),
            ],
            self.options.minimum_local_ess,
            density_floor_rows,
        );
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        if !standard_error.is_finite() {
            // The interval is published for every order and scale now; a
            // non-finite delta-method variance is a degenerate local fit (a
            // rank-deficient covariance or a transform singularity), not a
            // deliberate withholding. Say why the one interval is absent.
            support.warnings.push(Diagnostic::new(
                "response.derivative_interval_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "no interval is reported: the delta-method variance of this transformed derivative is not finite at this coordinate (a degenerate local covariance or a transform singularity)",
            ));
        }
        let uncertainty = if standard_error.is_finite()
            && fieller.as_ref().is_some_and(Option::is_none)
        {
            support.warnings.push(Diagnostic::new(
                "response.derivative_interval_unbounded",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "the Fieller confidence set for elasticity is unbounded because the response level is not separated from zero by its joint-covariance confidence region; no finite scalar interval is reported",
            ));
            ResponseUncertainty::None
        } else if standard_error.is_finite() {
            if fieller.is_some() {
                support.warnings.push(fieller_interval_note(estimate, corrected_estimate));
            } else if delta_transformed {
                support.warnings.push(delta_method_interval_note(estimate, corrected_estimate));
            } else {
                support.warnings.push(bias_corrected_interval_note(
                    false,
                    estimate,
                    corrected_estimate,
                ));
            }
            let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
            let (lower, upper) = fieller.flatten().unwrap_or((
                corrected_estimate - z * standard_error,
                corrected_estimate + z * standard_error,
            ));
            ResponseUncertainty::Scalar {
                standard_error,
                level: self.options.confidence_level,
                lower,
                upper,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            }
        } else {
            ResponseUncertainty::None
        };
        Ok((ResponseValue::Scalar(estimate), uncertainty, support))
    }

    fn average_derivative(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        weighting: &DerivativeWeighting,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        if !matches!(weighting, DerivativeWeighting::Observed) {
            return Err(EstimationError::unsupported(
                "Gaussian-score Riesz ADE currently supports observed-law weighting only",
            ));
        }
        let sample = CompleteSample::read(data, outcome, &[treatment], &self.adjustment_set)?;
        let AverageDerivativeScores { scores, riesz_weights } =
            self.cross_fitted_ade_scores(&sample)?;
        let rows = scores.len() as f64;
        let estimate = scores.iter().sum::<f64>() / rows;
        let variance = scores.iter().map(|v| (v - estimate).powi(2)).sum::<f64>() / (rows - 1.0);
        let se = (variance / rows).sqrt();
        let z = normal_ppf(0.5 + self.options.confidence_level / 2.0);
        let (minimum, maximum) = range(&sample.treatments);
        // Kish effective sample size of the Riesz representer. The raw row count is
        // not a weighted ESS: it cannot fall when the representer concentrates on a
        // handful of treatment-tail rows, which is exactly the failure this reports.
        let absolute_sum = riesz_weights.iter().map(|weight| weight.abs()).sum::<f64>();
        let square_sum = riesz_weights.iter().map(|weight| weight * weight).sum::<f64>();
        let effective_n =
            if square_sum > 0.0 { absolute_sum * absolute_sum / square_sum } else { 0.0 };
        let weak = effective_n < self.options.minimum_local_ess;
        let mut support = SupportReport {
            status: if weak { SupportStatus::WeakOverlap } else { SupportStatus::Supported },
            query_region: SupportRegion {
                minima: Arc::from([minimum]),
                maxima: Arc::from([maximum]),
            },
            diagnostics: vec![SupportDiagnostic {
                id: Arc::from("response.weighted_effective_sample_size"),
                values: Arc::from([effective_n, rows]),
                detail: Arc::from(
                    "Kish effective sample size of the Riesz representer, then complete rows",
                ),
            }],
            warnings: if weak {
                vec![Diagnostic::new(
                    "response.weak_riesz_overlap",
                    DiagnosticKind::Support,
                    DiagnosticSeverity::Warning,
                    "the average-derivative Riesz representer concentrates on few rows; the estimate is driven by the treatment tail",
                )]
            } else {
                Vec::new()
            },
            point_status: None,
        };
        push_outcome_tail_diagnostic(&mut support, &sample.outcome);
        Ok((
            ResponseValue::Scalar(estimate),
            ResponseUncertainty::Scalar {
                standard_error: se,
                level: self.options.confidence_level,
                lower: estimate - z * se,
                upper: estimate + z * se,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
            support,
        ))
    }

    fn jacobian(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        scale: DerivativeScale,
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let run = self.plugin_gradient_run(
            data,
            outcomes,
            treatments,
            at,
            PluginGradientKind::Jacobian,
            false,
        )?;
        let mut values = Vec::with_capacity(outcomes.len() * treatments.len());
        for (level, gradient) in run.levels.iter().zip(&run.gradients) {
            for (j, &raw) in gradient.iter().enumerate() {
                values.push(transform_derivative(raw, at[j], *level, scale)?);
            }
        }
        let mut support = multivariate_support(at, &run.all_treatments, treatments.len());
        support.warnings.push(plugin_gradient_interval_withheld());
        Ok((
            ResponseValue::Jacobian {
                outcomes: outcomes.len(),
                treatments: treatments.len(),
                values: Arc::from(values),
            },
            ResponseUncertainty::None,
            support,
        ))
    }

    fn directional_derivative(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        direction: &[f64],
    ) -> Result<(ResponseValue, ResponseUncertainty, SupportReport), EstimationError> {
        let run = self.plugin_gradient_run(
            data,
            outcomes,
            treatments,
            at,
            PluginGradientKind::Directional,
            false,
        )?;
        let values: Vec<f64> = run
            .gradients
            .iter()
            .map(|gradient| gradient.iter().zip(direction).map(|(a, b)| a * b).sum())
            .collect();
        let mut support = multivariate_support(at, &run.all_treatments, treatments.len());
        support.warnings.push(plugin_gradient_interval_withheld());
        Ok((ResponseValue::Vector(Arc::from(values)), ResponseUncertainty::None, support))
    }

    /// The plug-in level and treatment gradient of every outcome at `at`, from one
    /// unpenalized-treatment additive-GAM target fit per outcome on the shared
    /// complete-case rows, with each gradient coordinate's coefficient-sandwich
    /// influence column when `with_influence`.
    fn plugin_gradient_run(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        limit: PluginGradientKind,
        with_influence: bool,
    ) -> Result<PluginGradientRun, EstimationError> {
        if treatments.len() > MAX_NONPARAMETRIC_RESPONSE_DIM {
            let (message, remedy) = limit.refusal();
            return Err(EstimationError::unsupported_with_remedy(message, remedy));
        }
        let samples =
            read_shared_complete_samples(data, outcomes, treatments, &self.adjustment_set)?;
        let all_treatments =
            samples.first().map(|sample| sample.treatment_matrix.clone()).unwrap_or_default();
        let mut run = PluginGradientRun {
            levels: Vec::with_capacity(samples.len()),
            gradients: Vec::with_capacity(samples.len()),
            influence: Vec::new(),
            all_treatments,
            n: samples.first().map_or(0, CompleteSample::len),
        };
        for sample in &samples {
            let design = self.outcome_target_design(sample)?;
            let fit = Self::fit_outcome_target_weighted(sample, &design, None)?;
            let (level, gradient) = Self::plugin_gradient_at_fit(&fit, sample, at, None)?;
            if with_influence {
                run.influence.push(plugin_gradient_influence(&fit, sample, at)?);
            }
            run.levels.push(level);
            run.gradients.push(gradient);
        }
        Ok(run)
    }

    /// Pointwise confidence band of the Frequentist additive-GAM plug-in gradient
    /// (a [`ResponseFunctional::Jacobian`] on the identity scale, or a
    /// [`ResponseFunctional::DirectionalDerivative`]): the closed interval route the
    /// calibration harness measures.
    ///
    /// Each coordinate is a linear functional `c'β̂` of the target fit's
    /// coefficients (`c` the treatment-basis derivatives at `at`), so its
    /// influence is the fixed-basis penalized least-squares sandwich already used
    /// for the intervention-response level, with no covariate-average term (the
    /// additive gradient does not depend on the adjustment covariates). The
    /// variance takes the HC1 factor `n / (n − edf)` of the fit, as the licensed
    /// Bayesian band inflates its draws. The band conditions on the fixed knots and
    /// adjustment penalty and excludes sieve-approximation bias of a non-additive
    /// or rough outcome surface. The public route withholds this band
    /// (`response.derivative_interval_withheld`) until its coverage records are
    /// measured; the returned value is the published point, bit for bit.
    ///
    /// # Errors
    ///
    /// The public route's refusals, a transformed Jacobian scale (no interval is
    /// constructed for it), invalid options, or a design whose penalized Gram is
    /// rank deficient.
    #[cfg(feature = "calibration-internal")]
    #[doc(hidden)]
    pub fn plugin_gradient_interval_internal(
        &self,
        data: &TabularData,
        functional: &ResponseFunctional,
    ) -> Result<(ResponseValue, ResponseUncertainty), EstimationError> {
        let level = self.options.confidence_level;
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(EstimationError::unsupported("invalid continuous-response options"));
        }
        let (outcomes, treatments, at, direction) = match functional {
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if *scale != DerivativeScale::Identity {
                    return Err(EstimationError::unsupported(
                        "the plug-in gradient interval is constructed on the identity scale only",
                    ));
                }
                (outcomes, treatments, at, None)
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                (outcomes, treatments, at, Some(direction))
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "the plug-in gradient interval serves Jacobian and directional derivatives",
                ));
            }
        };
        let kind = if direction.is_some() {
            PluginGradientKind::Directional
        } else {
            PluginGradientKind::Jacobian
        };
        let run = self.plugin_gradient_run(data, outcomes, treatments, at, kind, true)?;
        let nf = run.n as f64;
        let z = normal_ppf(0.5 + level / 2.0);
        let mut values = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for (gradient, influence) in run.gradients.iter().zip(&run.influence) {
            let coordinates: Vec<(f64, Vec<f64>)> = match direction {
                None => gradient.iter().copied().zip(influence.columns.iter().cloned()).collect(),
                Some(direction) => {
                    let value = gradient.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
                    let mut psi = vec![0.0; run.n];
                    for (column, &weight) in influence.columns.iter().zip(direction.iter()) {
                        for (out, v) in psi.iter_mut().zip(column) {
                            *out += weight * v;
                        }
                    }
                    vec![(value, psi)]
                }
            };
            for (value, psi) in coordinates {
                let se = plugin_sandwich_se(&psi, nf, influence.edf);
                values.push(value);
                lower.push(value - z * se);
                upper.push(value + z * se);
            }
        }
        let value = match direction {
            None => ResponseValue::Jacobian {
                outcomes: outcomes.len(),
                treatments: treatments.len(),
                values: Arc::from(values),
            },
            Some(_) => ResponseValue::Vector(Arc::from(values)),
        };
        Ok((
            value,
            ResponseUncertainty::PointwiseBand {
                level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
        ))
    }

    fn cross_fitted_pseudo_outcome(
        &self,
        sample: &CompleteSample,
    ) -> Result<PseudoOutcome, EstimationError> {
        let folds = self.cross_fit_folds(sample)?;
        self.cross_fitted_pseudo_outcome_weighted(sample, &folds, None)
    }

    /// The cross-fitting folds of `sample`, with each fold's training designs
    /// expanded once (see [`CrossFitFold`]).
    ///
    /// # Errors
    ///
    /// Too few rows for the folds and basis, or a basis expansion failure.
    fn cross_fit_folds(
        &self,
        sample: &CompleteSample,
    ) -> Result<Vec<CrossFitFold>, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        (0..self.options.folds)
            .map(|fold| {
                let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
                let valid: Vec<usize> = (0..n).filter(|i| i % self.options.folds == fold).collect();
                let outcome_design = AdditiveDesign::expand(
                    &sample.raw_subset(&train),
                    train.len(),
                    sample.raw_cols,
                    &self.nuisance_specs(sample.raw_cols),
                )?;
                let treatment_design = if sample.adjustment_cols == 0 {
                    None
                } else {
                    Some(AdditiveDesign::expand(
                        &sample.adjustment_subset(&train),
                        train.len(),
                        sample.adjustment_cols,
                        &self.nuisance_specs(sample.adjustment_cols),
                    )?)
                };
                Ok(CrossFitFold {
                    outcome: train.iter().map(|&i| sample.outcome[i]).collect(),
                    treatment: train.iter().map(|&i| sample.treatments[i]).collect(),
                    train,
                    valid,
                    outcome_design,
                    treatment_design,
                })
            })
            .collect()
    }

    /// Penalized nuisance smooths over `ncols` raw columns, knots at the sample
    /// quantiles of the rows they are expanded over.
    fn nuisance_specs(&self, ncols: usize) -> Vec<SmoothSpec> {
        (0..ncols)
            .map(|col| {
                SmoothSpec::new(col, self.options.nuisance_basis, self.options.nuisance_lambda)
            })
            .collect()
    }

    /// Cross-fitted Kennedy pseudo-outcome, optionally under row weights.
    ///
    /// With `weights`, every fold refits the outcome and treatment nuisances on
    /// its weighted training rows, and the treatment scale, marginal density, and
    /// covariate offset become weighted training-row averages. This is one
    /// Bayesian-bootstrap draw of the whole nuisance stage, not a reweight of the
    /// first-fit pseudo-outcome. `None` is the unweighted estimator (weights of
    /// one, bit-identical). `folds` is [`Self::cross_fit_folds`] of `sample`,
    /// built once by a draw loop and shared by every draw.
    fn cross_fitted_pseudo_outcome_weighted(
        &self,
        sample: &CompleteSample,
        folds: &[CrossFitFold],
        weights: Option<&[f64]>,
    ) -> Result<PseudoOutcome, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        if weights.is_some_and(|w| w.len() != n) {
            return Err(EstimationError::stats_msg("pseudo-outcome weights must match rows"));
        }
        let row_weight = |i: usize| weights.map_or(1.0, |w| w[i]);
        let mut pseudo = vec![0.0; n];
        let mut covariate_centered = vec![0.0; n];
        let mut density_floor_rows = 0usize;
        let mut gam_ws = GamWorkspace::default();
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let mut raw_row = vec![0.0; sample.raw_cols];
        let mut train_weights_buf: Vec<f64> = Vec::new();
        for fold in folds {
            let (train, valid) = (fold.train.as_slice(), fold.valid.as_slice());
            let train_weights: Option<&[f64]> = weights.map(|_| {
                train_weights_buf.clear();
                train_weights_buf.extend(train.iter().map(|&i| row_weight(i)));
                train_weights_buf.as_slice()
            });
            let train_weight_sum = match train_weights {
                Some(w) => w.iter().sum::<f64>(),
                None => train.len() as f64,
            };
            if !train_weight_sum.is_finite() || train_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("fold training weights are degenerate"));
            }
            let (outcome_fit, treatment_fit, sigma) =
                Self::fit_train_nuisances(sample, fold, train_weights, &mut gam_ws)?;
            // The training-row treatment means do not depend on the validation row;
            // computing them once per fold avoids |valid| x |train| spline expansions.
            let constant_treatment_mean = match train_weights {
                None => sample.train_treatment_mean(train),
                Some(w) => {
                    train.iter().zip(w).map(|(&i, wi)| wi * sample.treatments[i]).sum::<f64>()
                        / train_weight_sum
                }
            };
            let train_treatment_means: Vec<f64> = match treatment_fit.as_ref() {
                Some(fit) => {
                    let mut means = Vec::with_capacity(train.len());
                    for &j in train {
                        sample.write_adjustment_row(j, &mut adj_row);
                        means.push(predict_one(fit, &adj_row)?);
                    }
                    means
                }
                None => vec![constant_treatment_mean; train.len()],
            };
            // The outcome nuisance is additive, so a counterfactual prediction
            // decomposes exactly: μ̂(a, X_j) = μ̂(A_j, X_j) − f_T(A_j) + f_T(a),
            // where f_T is the centered treatment smooth, giving one smooth
            // evaluation per row instead of a |valid|×|train| full-prediction
            // double loop.
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            // Kennedy et al. (2017, Thm. 3): the marginalization `∫μ̂(a, x) dP_n(x)`
            // is the empirical mean over the *full* sample `P_n`, not the training
            // fold alone — μ̂ is trained out-of-fold, but every row's covariates
            // still contribute to the average that estimates the population
            // marginal (see the identical claim in `mean_curve`'s covariate-term
            // comment). Averaging only the training rows makes the offset the
            // training fold's own covariate mean rather than the population's:
            // unbiased when the held-out rows are an unbiased subsample of every
            // raw column's marginal, which a representative hold-out normally is,
            // but is not guaranteed in general — a fold that happens to omit a
            // covariate's extreme rows shifts its empirical mean away from the
            // population's, biasing every pseudo-outcome the fold produces, and
            // cross-fitting `μ̂` out-of-fold does not fix that shift because the
            // shift is in which rows are averaged, not in which rows trained the
            // model.
            let mut covariate_offset = 0.0;
            let mut covariate_weight_sum = 0.0;
            for (position, &j) in train.iter().enumerate() {
                let treat_partial =
                    outcome_fit.smooth_partial(treat_smooth, sample.treatments[j])?;
                let w = train_weights.map_or(1.0, |tw| tw[position]);
                covariate_offset += w * (outcome_fit.fitted[position] - treat_partial);
                covariate_weight_sum += w;
            }
            for &j in valid {
                sample.write_raw_row(j, &mut raw_row);
                let mu_j = predict_one(&outcome_fit, &raw_row)?;
                let treat_partial =
                    outcome_fit.smooth_partial(treat_smooth, sample.treatments[j])?;
                let w = row_weight(j);
                covariate_offset += w * (mu_j - treat_partial);
                covariate_weight_sum += w;
            }
            if !covariate_weight_sum.is_finite() || covariate_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("row weights are degenerate"));
            }
            covariate_offset /= covariate_weight_sum;
            // The marginal treatment density is the (weighted) Gaussian mixture of the
            // training-row means at bandwidth σ; without a treatment model every
            // mean is the same constant and the mixture is one Gaussian.
            let marginal_mixture = if treatment_fit.is_none() {
                GaussianMixtureDensity::new(&[constant_treatment_mean], None, sigma)?
            } else {
                GaussianMixtureDensity::new(&train_treatment_means, train_weights, sigma)?
            };
            for &i in valid {
                sample.write_raw_row(i, &mut raw_row);
                let mu_observed = predict_one(&outcome_fit, &raw_row)?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => constant_treatment_mean,
                };
                let raw_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                if !raw_density.is_finite() || raw_density <= CONDITIONAL_DENSITY_FLOOR {
                    density_floor_rows += 1;
                }
                let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                let marginal_density = marginal_mixture.density(sample.treatment_matrix[i]);
                let marginal_mu = covariate_offset
                    + outcome_fit.smooth_partial(treat_smooth, sample.treatment_matrix[i])?;
                pseudo[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
                covariate_centered[i] = mu_observed - marginal_mu;
            }
        }
        Ok(PseudoOutcome { values: pseudo, density_floor_rows, covariate_centered })
    }

    fn cross_fitted_ade_scores(
        &self,
        sample: &CompleteSample,
    ) -> Result<AverageDerivativeScores, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut scores = vec![0.0; n];
        let mut riesz_weights = vec![0.0; n];
        let mut gam_ws = GamWorkspace::default();
        let mut row = vec![0.0; sample.raw_cols];
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        for fold in 0..self.options.folds {
            let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
            let outcome_fit = self.fit_outcome(sample, &train, &mut gam_ws)?;
            let treatment_fit = self.fit_treatment(sample, &train, &mut gam_ws)?;
            let sigma = treatment_sigma(sample, &train, treatment_fit.as_ref())?;
            let treatment_mean_constant = sample.train_treatment_mean(&train);
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            for i in (0..n).filter(|i| i % self.options.folds == fold) {
                sample.write_raw_row(i, &mut row);
                let mu = predict_one(&outcome_fit, &row)?;
                // Additive μ(a, x) = α + f_T(a) + Σ g_k(x_k), so ∂μ/∂a = f_T'(a)
                // and is identical for every covariate row. The clamped evaluation
                // is constant outside the knot interior; the analytic derivative
                // is therefore exactly 0 there (`response.clamped_basis_derivative`).
                let derivative =
                    outcome_fit.smooth_derivative(treat_smooth, sample.treatments[i])?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => treatment_mean_constant,
                };
                let riesz = (sample.treatments[i] - treatment_mean) / (sigma * sigma);
                riesz_weights[i] = riesz;
                scores[i] = derivative + riesz * (sample.outcome[i] - mu);
            }
        }
        Ok(AverageDerivativeScores { scores, riesz_weights })
    }

    /// Outcome GAM, treatment GAM (absent without adjusters) and treatment scale
    /// fitted on the fold's training rows, under their row weights `w` when given.
    fn fit_train_nuisances(
        sample: &CompleteSample,
        fold: &CrossFitFold,
        w: Option<&[f64]>,
        gam_ws: &mut GamWorkspace,
    ) -> Result<(antecedent_stats::GamFit, Option<antecedent_stats::GamFit>, f64), EstimationError>
    {
        let outcome_fit = fit_additive_design(&fold.outcome_design, &fold.outcome, w, gam_ws)?;
        let treatment_fit = match &fold.treatment_design {
            None => None,
            Some(design) => Some(fit_additive_design(design, &fold.treatment, w, gam_ws)?),
        };
        let sigma = match w {
            None => treatment_sigma(sample, &fold.train, treatment_fit.as_ref())?,
            Some(w) => {
                treatment_sigma_train_weighted(sample, &fold.train, w, treatment_fit.as_ref())?
            }
        };
        Ok((outcome_fit, treatment_fit, sigma))
    }

    /// One Bayesian-bootstrap draw of the cross-fitted Riesz ADE: every fold refits
    /// the outcome and treatment nuisances on its training rows under the draw's row
    /// weights and scores only its held-out rows, then the weighted mean of the
    /// held-out scores `∂_a μ̂ + α̂ (Y − μ̂)` is returned. Held-out scoring keeps the
    /// draws on the same footing as the cross-fitted point estimate, without the
    /// leverage shrinkage of in-sample residuals.
    fn weighted_cross_fitted_ade(
        &self,
        sample: &CompleteSample,
        folds: &[CrossFitFold],
        weights: &[f64],
    ) -> Result<f64, EstimationError> {
        let n = sample.len();
        if weights.len() != n {
            return Err(EstimationError::stats_msg("ADE weights length must match complete rows"));
        }
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        let mut gam_ws = GamWorkspace::default();
        let mut row = vec![0.0; sample.raw_cols];
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let mut train_weights: Vec<f64> = Vec::new();
        let mut num = 0.0;
        let mut den = 0.0;
        for fold in folds {
            let train = fold.train.as_slice();
            train_weights.clear();
            train_weights.extend(train.iter().map(|&i| weights[i]));
            let train_weight_sum: f64 = train_weights.iter().sum();
            if !train_weight_sum.is_finite() || train_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("fold training weights are degenerate"));
            }
            let (outcome_fit, treatment_fit, sigma) =
                Self::fit_train_nuisances(sample, fold, Some(&train_weights), &mut gam_ws)?;
            let treatment_mean_constant = train
                .iter()
                .zip(&train_weights)
                .map(|(&i, w)| w * sample.treatments[i])
                .sum::<f64>()
                / train_weight_sum;
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            for &i in &fold.valid {
                sample.write_raw_row(i, &mut row);
                let mu = predict_one(&outcome_fit, &row)?;
                let derivative =
                    outcome_fit.smooth_derivative(treat_smooth, sample.treatments[i])?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => treatment_mean_constant,
                };
                let alpha = (sample.treatments[i] - treatment_mean) / (sigma * sigma);
                num += weights[i] * (derivative + alpha * (sample.outcome[i] - mu));
                den += weights[i];
            }
        }
        if !den.is_finite() || den <= 0.0 || !num.is_finite() {
            return Err(EstimationError::stats_msg("weighted ADE draw was non-finite"));
        }
        Ok(num / den)
    }

    /// Unpenalized treatment / penalized adjustment specs for the plug-in target.
    ///
    /// Shared by [`Self::fit_outcome_target`] and the weighted Jacobian path so
    /// the unpenalized-treatment / penalized-adjustment split cannot drift
    /// between the two (ENG-009).
    fn outcome_target_specs(&self, sample: &CompleteSample) -> Vec<SmoothSpec> {
        (0..sample.raw_cols)
            .map(|col| {
                let lambda = if col < sample.treatment_cols {
                    PLUGIN_TARGET_LAMBDA
                } else {
                    self.options.nuisance_lambda
                };
                SmoothSpec::new(col, self.options.nuisance_basis, lambda)
            })
            .collect()
    }

    /// Full-sample outcome GAM whose treatment smooths are the plug-in *target*.
    ///
    /// Same split as [`Self::plugin_gradient_run`]: treatment smooths are
    /// unpenalized cubic regression splines (a roughness penalty with quantile
    /// knots shrinks even a linear dose effect, which biased the g-computation
    /// level by about half its SE at every n in calibration), and
    /// adjustment smooths keep `nuisance_lambda`.
    ///
    /// When the unpenalized target fit fails (a binary or low-cardinality
    /// treatment cannot support an unpenalized cubic basis), the penalized
    /// nuisance fit is used instead and the failure reason is returned so the
    /// caller can disclose that the reported level is the penalized fit.
    fn fit_outcome_target(
        &self,
        sample: &CompleteSample,
    ) -> Result<(antecedent_stats::GamFit, Option<String>), EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        let x = sample.raw_subset(&rows);
        let specs = self.outcome_target_specs(sample);
        let mut gam_ws = GamWorkspace::default();
        let target = fit_gam(
            &x,
            sample.len(),
            sample.raw_cols,
            &sample.outcome,
            &specs,
            &GamOptions { max_iter: 500, tol: 1e-6 },
            &FaerBackend,
            &mut gam_ws,
        );
        match target {
            Ok(fit) => Ok((require_converged_gam(fit, GAM_TARGET_NOT_CONVERGED)?, None)),
            // A binary or low-cardinality treatment cannot support an unpenalized
            // cubic basis (singular Gram); keep the penalized nuisance fit there
            // and report why.
            Err(error) => {
                let fit = self.fit_outcome(sample, &rows, &mut gam_ws)?;
                Ok((fit, Some(error.to_string())))
            }
        }
    }

    /// The plug-in target design of [`Self::outcome_target_specs`] expanded once
    /// over every complete row, for the weighted refits of a draw loop.
    fn outcome_target_design(
        &self,
        sample: &CompleteSample,
    ) -> Result<AdditiveDesign, EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        Ok(AdditiveDesign::expand(
            &sample.raw_subset(&rows),
            sample.len(),
            sample.raw_cols,
            &self.outcome_target_specs(sample),
        )?)
    }

    /// Fit the treatment-target outcome GAM of `design`
    /// ([`Self::outcome_target_design`]) once for this weight vector.
    fn fit_outcome_target_weighted(
        sample: &CompleteSample,
        design: &AdditiveDesign,
        weights: Option<&[f64]>,
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        let mut gam_ws = GamWorkspace::default();
        let fit = fit_gam_weighted_design(
            design,
            &sample.outcome,
            &GamOptions { max_iter: 500, tol: 1e-6 },
            weights,
            &mut gam_ws,
        )?;
        if !fit.converged {
            return Err(EstimationError::stats_msg("weighted GAM did not converge"));
        }
        Ok(fit)
    }

    /// Evaluate the plug-in level and treatment gradient at `at` on a fitted GAM.
    fn plugin_gradient_at_fit(
        fit: &antecedent_stats::GamFit,
        sample: &CompleteSample,
        at: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>), EstimationError> {
        let mut treat_smooths = Vec::with_capacity(at.len());
        for j in 0..at.len() {
            treat_smooths.push(fit.smooth_for_raw_col(j).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing a treatment smooth")
            })?);
        }
        // Empirical μ̂(at) = α + Σ_j f_j(at[j]) + mean_i Σ_k g_k(X_i[k]).
        // The covariate offset is recovered from the in-sample fitted values
        // so we do not re-evaluate every adjustment smooth.
        let n = sample.len() as f64;
        let mut observed_treat_partial = 0.0;
        for i in 0..sample.len() {
            for (j, &smooth) in treat_smooths.iter().enumerate() {
                let a_ij = sample.treatment_matrix[j * sample.len() + i];
                observed_treat_partial +=
                    weights.map_or(1.0, |w| w[i]) * fit.smooth_partial(smooth, a_ij)?;
            }
        }
        let fitted_mean = fit
            .fitted
            .iter()
            .enumerate()
            .map(|(i, v)| v * weights.map_or(1.0, |w| w[i]))
            .sum::<f64>()
            / n;
        let covariate_offset = fitted_mean - fit.intercept - observed_treat_partial / n;
        let mut treat_level = 0.0;
        let mut gradient = Vec::with_capacity(at.len());
        for (j, &smooth) in treat_smooths.iter().enumerate() {
            treat_level += fit.smooth_partial(smooth, at[j])?;
            gradient.push(fit.smooth_derivative(smooth, at[j])?);
        }
        Ok((fit.intercept + treat_level + covariate_offset, gradient))
    }

    fn fit_outcome(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        let x = sample.raw_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.outcome[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.raw_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
            workspace,
        )
    }

    fn fit_treatment(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
    ) -> Result<Option<antecedent_stats::GamFit>, EstimationError> {
        if sample.adjustment_cols == 0 {
            return Ok(None);
        }
        let x = sample.adjustment_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.treatments[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.adjustment_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
            workspace,
        )
        .map(Some)
    }
}

fn with_estimation_assumptions(
    mut assumptions: AssumptionSet,
    functional: &ResponseFunctional,
) -> AssumptionSet {
    let (id, description, algorithm) = match functional {
        ResponseFunctional::MeanCurve { .. } => (
            "response.kennedy_dr.nuisance_regularity",
            "Kennedy response estimation uses an additive-GAM outcome nuisance and a homoskedastic Gaussian treatment-density nuisance. Consistency requires at least one nuisance family to be adequate plus continuous-treatment smoothness/positivity; reported local-polynomial bands condition on the fitted nuisances and selected bandwidth.",
            "estimate.response.kennedy_dr",
        ),
        ResponseFunctional::PointDerivative { .. } => (
            "response.kennedy_dr.nuisance_regularity",
            "Kennedy response estimation uses an additive-GAM outcome nuisance and a homoskedastic Gaussian treatment-density nuisance. Consistency requires at least one nuisance family to be adequate plus continuous-treatment differentiability/positivity. The point value is the local-quadratic coordinate; the derivative interval is robust bias-corrected at the caller bandwidth (local-cubic slope for a first derivative, local-quartic curvature for a second derivative), targets the true derivative, and conditions on the fitted nuisances and that bandwidth.",
            "estimate.response.point_derivative",
        ),
        ResponseFunctional::AverageDerivative { .. } => (
            "response.riesz_ade.gaussian_score",
            "The average-derivative augmentation uses a homoskedastic Gaussian treatment-score representer and an additive-GAM outcome derivative. Consistency requires the treatment score or outcome-derivative nuisance to be adequate; the scalar interval conditions on those fitted nuisances.",
            "estimate.response.riesz_ade",
        ),
        ResponseFunctional::Jacobian { .. } | ResponseFunctional::DirectionalDerivative { .. } => (
            "response.additive_gam.plugin",
            "This response is an additive-GAM plug-in functional. Its numerical value depends on the additive outcome-surface restriction; treatment interactions are not learned unless represented by the fitted model.",
            "estimate.response.gam_derivative",
        ),
        ResponseFunctional::InterventionResponse { .. } => (
            "response.additive_gam.plugin",
            "This response is an additive-GAM plug-in functional. Its numerical value depends on the additive outcome-surface restriction; treatment interactions are not learned unless represented by the fitted model.",
            "estimate.response.intervention_gcomp",
        ),
    };
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from(id),
            description: Arc::from(description),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(algorithm) },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

fn bootstrap_weights(n: usize, rng: &mut CausalRng) -> Vec<f64> {
    let mut weights: Vec<f64> =
        (0..n).map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w *= n as f64 / total;
    }
    weights
}

/// Posterior mean, equal-tailed credible interval, and SD of a scalar's draws.
///
/// The interval uses *exchangeable-rank* quantiles (Hyndman–Fan type 6: the
/// `p`-quantile sits at rank `p·(D + 1)` of the `D` sorted draws), not the
/// `1 + p·(D − 1)` interpolation of a sample quantile. The reason is what a
/// credible interval built from finitely many draws claims: when the posterior
/// is calibrated the truth is exchangeable with the draws, and the interval
/// between order statistics `r` and `s` then covers it with probability exactly
/// `(s − r)/(D + 1)`, whatever the posterior's shape. Rank `p·(D + 1)` makes
/// that probability the level. The sample-quantile ranks sit about one order
/// statistic inside on each side, which at `D = 200` and a 90% level is about
/// one coverage point (0.890 expected) and at the interactive tier's `D = 64`
/// almost three (0.873); the exchangeable ranks give 0.900 at either count,
/// at the price of an interval that is wider on average by 1% (`D = 200`) or
/// 4% (`D = 64`) than the large-`D` limit. Ranks below 1 or above `D` clamp to
/// the extreme draws, which is the only case where the rule under-covers, and
/// a caller can widen it with more draws.
fn summarize_scalar_draws(
    values: &[f64],
    level: f64,
) -> Result<(f64, f64, f64, f64), EstimationError> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "derivative posterior needs at least two finite draws",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, level, QuantileRule::ExchangeableRank);
    Ok((mean, lower, upper, sd))
}

/// Inflate `draws` around their mean by `sqrt(n / (n − edf))`, the
/// degrees-of-freedom correction of a row-weight posterior spread.
///
/// A Dirichlet(1, …, 1) row-weight (Rubin Bayesian-bootstrap) posterior of a
/// fitted-coefficient functional `c'β̂` has, to first order in the weights,
/// the spread of `Σ_i w_i a_i e_i` with `a = B(B'B + λP)⁻¹c` the functional's
/// influence direction and `e` the full-sample residuals: its variance is the
/// HC0 sandwich `Σ_i a_i² e_i²`. Under homoskedastic errors `E[e_i²] =
/// σ²(1 − h_ii)`, so the sandwich understates the sampling variance
/// `σ² Σ a_i²` by the leverage `h_ii` of the rows that carry the functional;
/// replacing each row's leverage by the average `tr(H)/n = edf/n` gives the
/// HC1 factor `n/(n − edf)`, with `edf` the effective degrees of freedom of
/// the penalized fit (the penalty lowers the hat-matrix trace, so the
/// correction is smaller for a smoother fit). The correction is exact when
/// leverage is flat across the rows the functional loads on and conservative
/// when those rows sit at lower-than-average leverage (interior evaluation
/// points of a spline); it does not repair a mis-centred posterior. Its size
/// is bounded by the coefficient-to-row ratio: with the default six-function
/// bases on three columns (`edf ≈ 14`) and 1000 rows the factor is 1.007, so
/// the band widens by under one percent; the correction matters at small
/// samples or wide bases, not here.
fn inflate_draws_by_edf(draws: &mut [f64], n: usize, edf: f64) {
    let n = n as f64;
    if draws.len() < 2 || !(edf.is_finite() && edf >= 0.0 && edf < n) {
        return;
    }
    let factor = (n / (n - edf)).sqrt();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    for v in draws {
        *v = mean + (*v - mean) * factor;
    }
}

/// A posterior summary published as scalar uncertainty: `sd` is the posterior standard
/// deviation and `[lo, hi]` the `level` credible interval, tagged so a consumer cannot
/// read either as a frequentist standard error or confidence interval. `draws` is the
/// vector `[lo, hi]` are the quantiles of, retained for re-summarization.
fn credible_scalar_uncertainty(
    sd: f64,
    level: f64,
    lo: f64,
    hi: f64,
    draws: Vec<f64>,
) -> ResponseUncertainty {
    ResponseUncertainty::Scalar {
        standard_error: sd,
        level,
        lower: lo,
        upper: hi,
        interpretation: antecedent_core::IntervalInterpretation::Credible,
        draws: Some(CredibleDraws::scalar(draws)),
    }
}

/// Original row indices as the `u32` row ids influence scores carry; a row past the
/// `u32` range is an error rather than a saturated id that would alias another row.
fn row_ids_u32(rows: &[usize]) -> Result<Arc<[u32]>, EstimationError> {
    rows.iter()
        .map(|&i| {
            u32::try_from(i)
                .map_err(|_| EstimationError::data_msg("row index exceeds the u32 row-id capacity"))
        })
        .collect()
}

struct CompleteSample {
    /// Original dataframe row index of each retained complete row.
    keep: Vec<usize>,
    outcome: Vec<f64>,
    treatments: Vec<f64>,
    treatment_matrix: Vec<f64>,
    adjustment: Vec<f64>,
    treatment_cols: usize,
    adjustment_cols: usize,
    raw_cols: usize,
}

fn read_shared_complete_samples(
    data: &TabularData,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    adjustment: &[VariableId],
) -> Result<Vec<CompleteSample>, EstimationError> {
    let mut samples = Vec::with_capacity(outcomes.len());
    for &outcome in outcomes {
        samples.push(CompleteSample::read(data, outcome, treatments, adjustment)?);
    }
    if let Some(first) = samples.first() {
        if samples.iter().any(|sample| sample.keep != first.keep) {
            return Err(EstimationError::unsupported(
                "plug-in Jacobian and directional derivatives require a shared complete-case row set across outcomes",
            ));
        }
    }
    Ok(samples)
}

impl CompleteSample {
    fn read(
        data: &TabularData,
        outcome: VariableId,
        treatments: &[VariableId],
        adjustment: &[VariableId],
    ) -> Result<Self, EstimationError> {
        let duplicate_treatment =
            treatments.iter().enumerate().any(|(i, value)| treatments[i + 1..].contains(value));
        let duplicate_adjustment =
            adjustment.iter().enumerate().any(|(i, value)| adjustment[i + 1..].contains(value));
        if treatments.is_empty()
            || duplicate_treatment
            || duplicate_adjustment
            || adjustment.iter().any(|v| treatments.contains(v) || *v == outcome)
        {
            return Err(EstimationError::unsupported(
                "treatments and adjustments must be unique and adjustments distinct from outcome",
            ));
        }
        let y = data.float64_values(outcome)?;
        let treatment_values: Vec<Vec<f64>> =
            treatments.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let adjustment_values: Vec<Vec<f64>> =
            adjustment.iter().map(|&v| data.float64_values(v)).collect::<Result<_, _>>()?;
        let keep: Vec<usize> = (0..data.row_count())
            .filter(|&i| {
                y[i].is_finite()
                    && treatment_values.iter().all(|c| c[i].is_finite())
                    && adjustment_values.iter().all(|c| c[i].is_finite())
            })
            .collect();
        if keep.len() < 20 {
            return Err(EstimationError::unsupported(
                "continuous-response estimation requires at least 20 complete rows",
            ));
        }
        let outcome = keep.iter().map(|&i| y[i]).collect();
        let mut treatment_matrix = Vec::with_capacity(keep.len() * treatments.len());
        for column in &treatment_values {
            treatment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        let treatments = treatment_matrix[..keep.len()].to_vec();
        let mut adjustment_matrix = Vec::with_capacity(keep.len() * adjustment.len());
        for column in &adjustment_values {
            adjustment_matrix.extend(keep.iter().map(|&i| column[i]));
        }
        Ok(Self {
            keep,
            outcome,
            treatments,
            treatment_matrix,
            adjustment: adjustment_matrix,
            treatment_cols: treatment_values.len(),
            adjustment_cols: adjustment_values.len(),
            raw_cols: treatment_values.len() + adjustment_values.len(),
        })
    }

    fn len(&self) -> usize {
        self.outcome.len()
    }

    #[cfg(test)]
    fn raw_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.raw_cols];
        self.write_raw_row(row, &mut out);
        out
    }

    fn write_raw_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.raw_cols);
        // Column-major treatment/adjustment layouts: index by col deliberately.
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.treatment_cols {
            out[col] = self.treatment_matrix[col * self.len() + row];
        }
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.adjustment_cols {
            out[self.treatment_cols + col] = self.adjustment[col * self.len() + row];
        }
    }

    fn write_adjustment_row(&self, row: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.adjustment_cols);
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.adjustment_cols {
            out[col] = self.adjustment[col * self.len() + row];
        }
    }

    #[cfg(test)]
    fn adjustment_row(&self, row: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.adjustment_cols];
        self.write_adjustment_row(row, &mut out);
        out
    }

    fn raw_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.raw_cols);
        for col in 0..self.treatment_cols {
            out.extend(rows.iter().map(|&row| self.treatment_matrix[col * self.len() + row]));
        }
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    fn adjustment_subset(&self, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.adjustment_cols);
        for col in 0..self.adjustment_cols {
            out.extend(rows.iter().map(|&row| self.adjustment[col * self.len() + row]));
        }
        out
    }

    fn train_treatment_mean(&self, rows: &[usize]) -> f64 {
        rows.iter().map(|&i| self.treatments[i]).sum::<f64>() / rows.len() as f64
    }

    fn treatment_column_range(&self, col: usize) -> (f64, f64) {
        range(&self.treatment_matrix[col * self.len()..(col + 1) * self.len()])
    }
}

fn predict_one(fit: &antecedent_stats::GamFit, raw_row: &[f64]) -> Result<f64, EstimationError> {
    Ok(fit.predict_row(raw_row)?)
}

/// One Bayesian-bootstrap draw of a derivative functional.
struct DerivativeDraw {
    /// Local-quadratic point coordinate (point derivatives).
    scalar: f64,
    /// Robust bias-corrected point coordinate (point derivatives).
    corrected: f64,
    /// Plug-in gradient coordinates (directional derivative / Jacobian).
    vector: Vec<f64>,
    /// Effective degrees of freedom of each sample's GAM fit.
    edf: Vec<f64>,
}

/// Fixed-basis penalized least-squares sandwich of an additive-GAM fit on its own
/// rows. Dropping the last basis in each smooth removes the partition-of-unity
/// alias with the intercept; the D2 penalty is invariant to the corresponding
/// constant coefficient shift, so the reduced design spans the fitted space and
/// penalizes it the same way.
struct PluginSandwich {
    n: usize,
    p: usize,
    /// Column-major `n × p` design `[1 | B₁ (less its last basis) | …]`.
    design: Vec<f64>,
    /// `X'X + S`.
    gram: Vec<f64>,
    /// First design column of each raw column's smooth.
    offsets: Vec<usize>,
}

impl PluginSandwich {
    fn new(
        fit: &antecedent_stats::GamFit,
        sample: &CompleteSample,
    ) -> Result<Self, EstimationError> {
        let n = sample.len();
        let p = 1 + fit.smooths.iter().map(|s| s.n_basis - 1).sum::<usize>();
        let mut design = vec![1.0; n];
        let mut gram = vec![0.0; p * p];
        let mut offsets = Vec::with_capacity(sample.raw_cols);
        let mut offset = 1;
        for raw_col in 0..sample.raw_cols {
            let smooth = plugin_smooth(fit, raw_col)?;
            let (basis, _) = antecedent_stats::expand_bspline(
                &sample_raw_column(sample, raw_col),
                smooth.n_basis,
                Some(&smooth.knots),
            )?;
            design.extend_from_slice(&basis[..n * (smooth.n_basis - 1)]);
            for r in 0..smooth.n_basis - 2 {
                for (a, da) in [1.0, -2.0, 1.0].iter().enumerate() {
                    for (b, db) in [1.0, -2.0, 1.0].iter().enumerate() {
                        if r + a < smooth.n_basis - 1 && r + b < smooth.n_basis - 1 {
                            gram[(offset + r + b) * p + offset + r + a] += smooth.lambda * da * db;
                        }
                    }
                }
            }
            offsets.push(offset);
            offset += smooth.n_basis - 1;
        }
        for j in 0..p {
            for k in 0..p {
                gram[j * p + k] +=
                    (0..n).map(|i| design[j * n + i] * design[k * n + i]).sum::<f64>();
            }
        }
        Ok(Self { n, p, design, gram, offsets })
    }

    /// Per-row influence `n e_i x_i'(X'X + S)⁻¹ c` of the linear functional `c'β̂`.
    fn coefficient_influence(
        &self,
        residuals: &[f64],
        gradient: &[f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let (n, p) = (self.n, self.p);
        let solve = FaerBackend.least_squares(
            &self.gram,
            p,
            p,
            gradient,
            &mut LeastSquaresWorkspace::default(),
        )?;
        if solve.rank < p {
            return Err(EstimationError::unsupported(
                "response influence requires an identifiable penalized design",
            ));
        }
        let nf = n as f64;
        Ok((0..n)
            .map(|i| {
                nf * residuals[i]
                    * (0..p).map(|j| self.design[j * n + i] * solve.coefficients[j]).sum::<f64>()
            })
            .collect())
    }
}

fn plugin_smooth(
    fit: &antecedent_stats::GamFit,
    raw_col: usize,
) -> Result<&antecedent_stats::RecordedSmooth, EstimationError> {
    fit.smooth_for_raw_col(raw_col)
        .and_then(|index| fit.smooths.get(index))
        .ok_or_else(|| EstimationError::unsupported("missing GAM smooth in response influence"))
}

/// Raw column `raw_col` of the sample: treatments first, then adjustment.
fn sample_raw_column(sample: &CompleteSample, raw_col: usize) -> Vec<f64> {
    let n = sample.len();
    if raw_col < sample.treatment_cols {
        sample.treatment_matrix[raw_col * n..(raw_col + 1) * n].to_vec()
    } else {
        let col = raw_col - sample.treatment_cols;
        sample.adjustment[col * n..(col + 1) * n].to_vec()
    }
}

fn center_in_place(psi: &mut [f64]) {
    let center = psi.iter().sum::<f64>() / psi.len() as f64;
    for v in psi {
        *v -= center;
    }
}

// Fixed-basis penalized g-computation sandwich ([`PluginSandwich`]) plus the
// covariate-average term of the level.
fn intervention_plugin_influence(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    interventions: &[Intervention],
    row_means: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let n = sample.len();
    let nf = n as f64;
    let sandwich = PluginSandwich::new(fit, sample)?;
    let mut gradient = vec![1.0];
    for raw_col in 0..sample.raw_cols {
        let smooth = plugin_smooth(fit, raw_col)?;
        let observed = sample_raw_column(sample, raw_col);
        let (points, weights): (Vec<f64>, Vec<f64>) = if let Some(iv) = interventions.get(raw_col) {
            if let Intervention::Shift { delta, .. } = iv {
                let delta = delta.as_f64().filter(|d| d.is_finite()).ok_or_else(|| {
                    EstimationError::unsupported(
                        "intervention response requires finite numeric values",
                    )
                })?;
                (observed.iter().map(|&v| v + delta).collect(), vec![1.0 / nf; n])
            } else {
                policy_support(iv)?
                    .into_iter()
                    .map(|atom| match atom {
                        DiscreteAtom::Level { value, weight } => (value, weight),
                        DiscreteAtom::Shift { delta } => (delta, 1.0),
                    })
                    .unzip()
            }
        } else {
            (observed, vec![1.0 / nf; n])
        };
        let (counterfactual, _) =
            antecedent_stats::expand_bspline(&points, smooth.n_basis, Some(&smooth.knots))?;
        for j in 0..smooth.n_basis - 1 {
            gradient.push(
                counterfactual[j * points.len()..(j + 1) * points.len()]
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| v * w)
                    .sum(),
            );
        }
    }
    let coefficient = sandwich.coefficient_influence(&fit.residuals, &gradient)?;
    let mean = row_means.iter().sum::<f64>() / nf;
    let mut psi: Vec<_> =
        coefficient.iter().zip(row_means).map(|(c, row)| row - mean + c).collect();
    center_in_place(&mut psi);
    Ok(psi)
}

/// Which plug-in gradient query a run serves, for its refusal wording.
#[derive(Clone, Copy)]
enum PluginGradientKind {
    Jacobian,
    Directional,
}

impl PluginGradientKind {
    fn refusal(self) -> (&'static str, &'static str) {
        match self {
            Self::Jacobian => (
                "plug-in response Jacobian supports at most two treatments",
                JACOBIAN_TREATMENT_LIMIT_REMEDY,
            ),
            Self::Directional => (
                "plug-in directional derivative supports at most two treatments",
                DIRECTIONAL_TREATMENT_LIMIT_REMEDY,
            ),
        }
    }
}

/// Plug-in levels and gradients of every outcome (and their influence on request).
struct PluginGradientRun {
    levels: Vec<f64>,
    gradients: Vec<Vec<f64>>,
    /// One entry per outcome when influence was requested, else empty. Read only
    /// by the calibration-internal band.
    #[cfg_attr(not(feature = "calibration-internal"), allow(dead_code))]
    influence: Vec<PluginGradientInfluence>,
    all_treatments: Vec<f64>,
    /// Shared complete-case row count.
    #[cfg_attr(not(feature = "calibration-internal"), allow(dead_code))]
    n: usize,
}

/// Coefficient-sandwich influence of one outcome's gradient coordinates.
#[cfg_attr(not(feature = "calibration-internal"), allow(dead_code))]
struct PluginGradientInfluence {
    /// One centred column per treatment coordinate.
    columns: Vec<Vec<f64>>,
    /// Effective degrees of freedom of the target fit.
    edf: f64,
}

/// Influence of each plug-in gradient coordinate `∂μ̂/∂a_j (at)`.
///
/// The additive gradient is `f_j'(at_j) = Σ_b B_b'(at_j) β_b`, a linear functional
/// of the treatment smooth's coefficients alone: it carries no covariate-average
/// term. In the reduced design the dropped last basis contributes nothing,
/// because the basis derivatives sum to zero (partition of unity).
fn plugin_gradient_influence(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    at: &[f64],
) -> Result<PluginGradientInfluence, EstimationError> {
    let sandwich = PluginSandwich::new(fit, sample)?;
    let mut columns = Vec::with_capacity(at.len());
    for (j, &point) in at.iter().enumerate() {
        let index = fit.smooth_for_raw_col(j).ok_or_else(|| {
            EstimationError::unsupported("outcome nuisance is missing a treatment smooth")
        })?;
        let derivative = fit.smooth_basis_derivative(index, point)?;
        let mut gradient = vec![0.0; sandwich.p];
        let offset = sandwich.offsets[j];
        gradient[offset..offset + derivative.len() - 1]
            .copy_from_slice(&derivative[..derivative.len() - 1]);
        let mut psi = sandwich.coefficient_influence(&fit.residuals, &gradient)?;
        center_in_place(&mut psi);
        columns.push(psi);
    }
    Ok(PluginGradientInfluence { columns, edf: fit.edf_approx })
}

/// HC1 standard error of a centred influence column: `Σψ² / (n (n − edf))`, the
/// sandwich `Σ a_i² e_i²` with the leverage factor `n / (n − edf)` (see
/// `inflate_draws_by_edf`). Falls back to `n − 1` when `edf` is not usable.
#[cfg(feature = "calibration-internal")]
fn plugin_sandwich_se(psi: &[f64], n: f64, edf: f64) -> f64 {
    let dof = if edf.is_finite() && edf >= 1.0 && edf < n { n - edf } else { n - 1.0 };
    if dof <= 0.0 {
        return f64::NAN;
    }
    (psi.iter().map(|x| x * x).sum::<f64>() / (n * dof)).sqrt()
}

/// Why the Frequentist plug-in gradient publishes no interval.
fn plugin_gradient_interval_withheld() -> Diagnostic {
    Diagnostic::new(
        "response.derivative_interval_withheld",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "no Frequentist interval is reported for this additive-GAM plug-in gradient: its coefficient-sandwich confidence band is wired for calibration but its repeated-sampling coverage is not yet measured, so the interval route stays closed and only the point value is published; Bayesian inference publishes a measured pointwise credible band for the same plug-in gradient",
    )
}

/// Model-dependence disclosure for the intervention-response g-computation,
/// worded for the outcome fit that actually ran. `target_fallback` carries the
/// reason the unpenalized treatment-spline target fit failed, if it did.
fn intervention_plugin_warnings(target_fallback: Option<&str>) -> Vec<Diagnostic> {
    let Some(reason) = target_fallback else {
        return vec![Diagnostic::new(
            "response.intervention_plugin_model_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "intervention response uses additive-GAM g-computation with unpenalized treatment splines (the target) and penalized adjustment splines (nuisances); SE includes fitted-coefficient and covariate-average influence conditional on the fitted spline knots and adjustment penalty; it excludes knot-selection, sieve-approximation bias of a non-additive or rough outcome surface, and policy-integration error",
        )];
    };
    let mut fallback = Diagnostic::new(
        "response.intervention_target_penalized_fallback",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "the unpenalized treatment-spline target fit failed (typically a binary or low-cardinality treatment that cannot support an unpenalized cubic basis), so the level and SE come from the penalized nuisance outcome fit; roughness shrinkage of the treatment smooths can bias the level toward the observed mean by a fraction of its SE, and the interval does not account for that bias",
    );
    fallback.fields = Arc::from(vec![(Arc::from("target_fit_error"), Arc::from(reason))]);
    vec![
        Diagnostic::new(
            "response.intervention_plugin_model_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "intervention response uses additive-GAM g-computation with penalized treatment and adjustment splines (the unpenalized treatment-spline target fit failed); SE includes fitted-coefficient and covariate-average influence conditional on the fitted spline knots and penalties; it excludes penalty-induced shrinkage bias, knot-selection, sieve-approximation bias of a non-additive or rough outcome surface, and policy-integration error",
        ),
        fallback,
    ]
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        ContinuousDomain, GridSpec, Intervention, ResponseFunctional, ResponseQuery,
        StochasticPolicy, Value,
    };
    use antecedent_data::{TableView, TabularData};

    use super::*;

    /// The draws a Bayesian response retains on its credible uncertainty are
    /// exactly the vector its endpoints are the quantiles of: re-summarizing
    /// every coordinate at the published level under the estimator's own rule
    /// reproduces `[lower, upper]` bit for bit.
    fn assert_retained_draws_publish(uncertainty: &ResponseUncertainty, n_draws: usize) {
        let (level, published): (f64, Vec<(f64, f64)>) = match uncertainty {
            ResponseUncertainty::Scalar { level, lower, upper, .. } => {
                (*level, vec![(*lower, *upper)])
            }
            ResponseUncertainty::PointwiseBand { level, lower, upper, .. } => {
                (*level, lower.iter().copied().zip(upper.iter().copied()).collect())
            }
            other => panic!("expected a credible interval or band, got {other:?}"),
        };
        let draws = uncertainty.credible_draws().expect("credible uncertainty retains its draws");
        assert_eq!(draws.n_draws, n_draws);
        assert_eq!(draws.n_coordinates(), published.len());
        for (j, (lo, hi)) in published.into_iter().enumerate() {
            let (_, rebuilt_lo, rebuilt_hi, _) =
                summarize_scalar_draws(draws.column(j).unwrap(), level).unwrap();
            assert!(
                rebuilt_lo.to_bits() == lo.to_bits() && rebuilt_hi.to_bits() == hi.to_bits(),
                "coordinate {j}: [{lo}, {hi}] != retained-draw quantiles [{rebuilt_lo}, {rebuilt_hi}]"
            );
        }
    }

    #[test]
    fn bayesian_responses_retain_the_draws_their_bands_summarize() {
        let (data, a, y, x) = confounded_curve(240);
        let estimator = ContinuousResponseEstimator::new([x]);
        let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
        let ctx = antecedent_core::ExecutionContext::for_tests(18);
        let run = |functional| {
            estimator
                .estimate_bayesian(
                    &data,
                    &ResponseQuery::new(functional),
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &bayes,
                    &ctx,
                )
                .unwrap()
        };
        // Linear route: one column per grid point of the curve.
        let curve = run(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
        });
        assert_retained_draws_publish(&curve.uncertainty, bayes.n_draws);
        assert_eq!(curve.uncertainty.credible_draws().unwrap().n_coordinates(), 3);
        // Linear route, scalar.
        let level = run(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(a, Value::f64(0.25))]),
        });
        assert_retained_draws_publish(&level.uncertainty, bayes.n_draws);
        // Additive-GAM plug-in band: retained *after* the edf spread inflation.
        let jacobian = run(ResponseFunctional::Jacobian {
            outcomes: Arc::from([y]),
            treatments: Arc::from([a]),
            at: Arc::from([0.2]),
            scale: DerivativeScale::Identity,
        });
        assert_retained_draws_publish(&jacobian.uncertainty, bayes.n_draws);
    }

    #[test]
    fn bayesian_logit_response_evaluates_each_draw_on_probability_scale() {
        let n = 240;
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut covariate = Vec::with_capacity(n);
        for row in 0..n {
            let x = -1.0 + 2.0 * row as f64 / (n - 1) as f64;
            let a = -0.8 + 1.6 * ((row * 53 % n) as f64 / n as f64);
            let p =
                antecedent_stats::GlmFamily::BinomialLogit.mean_from_eta(-0.4 + 0.8 * a + 0.7 * x);
            let y = f64::from((row * 37 % 101) as f64 / 101.0 < p);
            covariate.push(x);
            treatment.push(a);
            outcome.push(y);
        }
        let data = TabularData::from_f64_columns([
            ("a", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let a = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let x = VariableId::from_raw(2);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let bayes = crate::BayesianGComputationAte::new()
            .with_likelihood(antecedent_prob::BayesLikelihood::BernoulliLogit)
            .with_n_draws(80)
            .with_seed(31);
        let result = ContinuousResponseEstimator::new([x])
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &antecedent_core::ExecutionContext::for_tests(31),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &result.estimate
        else {
            panic!("expected response surface")
        };
        assert!(mean.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!(mean[0] < mean[2]);
        assert_retained_draws_publish(&result.uncertainty, 80);
    }

    #[test]
    fn bayesian_curve_simultaneous_band_uses_joint_posterior_draws() {
        let (data, a, y, x) = confounded_curve(240);
        let bayes = crate::BayesianGComputationAte::new().with_n_draws(200).with_seed(31);
        let ctx = antecedent_core::ExecutionContext::for_tests(18);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
        });
        let mut pointwise = ContinuousResponseEstimator::new([x]);
        pointwise.options.bandwidth = Some(0.4);
        let ordinary = pointwise
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &ctx,
            )
            .unwrap();
        pointwise.options.simultaneous_replicates = Some(100);
        let joint = pointwise
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &ctx,
            )
            .unwrap();
        let ResponseUncertainty::PointwiseBand { lower: p_lo, upper: p_hi, .. } =
            ordinary.uncertainty
        else {
            panic!("expected pointwise credible band");
        };
        let ResponseUncertainty::SimultaneousBand {
            lower: s_lo,
            upper: s_hi,
            replicates,
            interpretation,
            ..
        } = joint.uncertainty
        else {
            panic!("expected simultaneous credible band");
        };
        assert_eq!(replicates, 200);
        assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Credible);
        for ((&lo, &hi), (&point_lo, &point_hi)) in
            s_lo.iter().zip(s_hi.iter()).zip(p_lo.iter().zip(p_hi.iter()))
        {
            assert!(lo <= point_lo && hi >= point_hi);
        }
    }

    /// Minimal deterministic uniform stream (`SplitMix64`) for exchangeability checks.
    fn uniform_stream(mut state: u64) -> impl FnMut() -> f64 {
        move || crate::splitmix::splitmix64_unit(&mut state)
    }

    #[test]
    fn exchangeable_rank_interval_covers_an_exchangeable_truth_at_the_level() {
        // With the truth exchangeable with the draws, the type-6 interval
        // covers at the level for any draw count; the type-7 interval of 20
        // draws covers about 0.81 at a 90% level.
        let mut u = uniform_stream(0x5EED);
        let trials = 20_000;
        for draws_n in [20usize, 64, 200] {
            let (mut covered, mut covered_type7) = (0u32, 0u32);
            for _ in 0..trials {
                let truth = u();
                let draws: Vec<f64> = (0..draws_n).map(|_| u()).collect();
                let (_, lo, hi, _) = summarize_scalar_draws(&draws, 0.9).unwrap();
                covered += u32::from(truth >= lo && truth <= hi);
                let mut sorted = draws;
                sorted.sort_by(f64::total_cmp);
                let type7 = |p: f64| {
                    let x = p * (sorted.len() - 1) as f64;
                    let (l, h) = (x.floor() as usize, x.ceil() as usize);
                    sorted[l] + (sorted[h] - sorted[l]) * (x - l as f64)
                };
                covered_type7 += u32::from(truth >= type7(0.05) && truth <= type7(0.95));
            }
            let rate = f64::from(covered) / f64::from(trials);
            assert!((rate - 0.9).abs() < 0.01, "draws={draws_n}: type-6 coverage {rate}");
            if draws_n == 20 {
                let rate7 = f64::from(covered_type7) / f64::from(trials);
                assert!(rate7 < 0.84, "draws=20: type-7 coverage {rate7} should be ≈ 0.81");
            }
        }
    }

    #[test]
    fn inflate_draws_by_edf_scales_spread_and_keeps_mean() {
        let mut draws = vec![1.0, 2.0, 4.0, 5.0];
        inflate_draws_by_edf(&mut draws, 100, 20.0);
        let factor = (100.0_f64 / 80.0).sqrt();
        let expect = |v: f64| 3.0 + (v - 3.0) * factor;
        for (got, raw) in draws.iter().zip([1.0, 2.0, 4.0, 5.0]) {
            assert!((got - expect(raw)).abs() < 1e-12);
        }
        assert!((draws.iter().sum::<f64>() / 4.0 - 3.0).abs() < 1e-12);
        // Degenerate edf (≥ n or non-finite) leaves the draws alone.
        for edf in [100.0, 150.0, f64::NAN, -1.0] {
            let mut untouched = vec![1.0, 2.0, 4.0, 5.0];
            inflate_draws_by_edf(&mut untouched, 100, edf);
            assert_eq!(untouched, vec![1.0, 2.0, 4.0, 5.0], "edf={edf}");
        }
    }

    #[test]
    fn bayesian_point_derivative_refits_nuisances_per_draw() {
        let (data, a, y, x) = confounded_curve(240);
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
        let frozen = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
        let folds = estimator.cross_fit_folds(&sample).unwrap();
        let ones = vec![1.0; sample.len()];
        let unit =
            estimator.cross_fitted_pseudo_outcome_weighted(&sample, &folds, Some(&ones)).unwrap();
        let unit_gap =
            frozen.iter().zip(&unit.values).map(|(f, u)| (f - u).abs()).fold(0.0, f64::max);
        assert!(unit_gap < 1e-9, "unit weights must reproduce the unweighted pseudo-outcome");
        let weights: Vec<f64> =
            (0..sample.len()).map(|i| if i % 3 == 0 { 2.2 } else { 0.4 }).collect();
        let refit = estimator
            .cross_fitted_pseudo_outcome_weighted(&sample, &folds, Some(&weights))
            .unwrap();
        let refit_gap =
            frozen.iter().zip(&refit.values).map(|(f, r)| (f - r).abs()).fold(0.0, f64::max);
        assert!(refit_gap > 1e-6, "weighted draws must refit the nuisance stage, not reuse φ");

        let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
        for scale in [DerivativeScale::Identity, DerivativeScale::LogLog] {
            let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: y,
                treatment: a,
                at: 0.2,
                order: 1,
                scale,
            });
            let response = estimator
                .estimate_bayesian(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &bayes,
                    &antecedent_core::ExecutionContext::for_tests(18),
                )
                .unwrap();
            assert_eq!(response.provenance_id.as_ref(), "estimate.response.point_derivative");
            // Posterior draws: the "standard error" is a posterior SD and the interval is
            // credible, and the tag says so.
            assert!(matches!(
                response.uncertainty,
                ResponseUncertainty::Scalar {
                    interpretation: antecedent_core::IntervalInterpretation::Credible,
                    ..
                }
            ));
            // The retained draws are the bias-corrected ones the interval is read from.
            assert_retained_draws_publish(&response.uncertainty, 8);
            assert!(response.assumptions.entries.iter().any(|record| match &record.assumption {
                Assumption::ParametricRestriction(restriction) =>
                    restriction.description.contains("Not a frozen-pseudo-outcome reweight"),
                _ => false,
            }));
            let codes: Vec<_> =
                response.support.warnings.iter().map(|w| w.code.as_ref().to_owned()).collect();
            assert!(
                codes.iter().any(|c| c == "response.derivative_interval_bias_corrected"),
                "{codes:?}"
            );
            assert!(!codes.iter().any(|c| c == "response.derivative_interval_withheld"));
            let note = response
                .support
                .warnings
                .iter()
                .find(|w| w.code.as_ref() == "response.derivative_interval_bias_corrected")
                .unwrap();
            assert!(note.message.contains("refits the cross-fitted"), "{}", note.message);
        }
    }

    fn confounded_curve(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
        let mut a = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let noise = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.3;
            let treatment = 0.7 * z + noise;
            x.push(z);
            a.push(treatment);
            y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        (data, VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2))
    }

    #[test]
    fn dr_curve_calibrates_linear_response_and_reports_support() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            response.estimate
        else {
            panic!("expected surface");
        };
        assert!((mean[2] - mean[0] - 1.6).abs() < 0.25, "means={mean:?}");
        assert_ne!(response.support.status, SupportStatus::OutsideEmpiricalSupport);
        assert_eq!(response.support.diagnostics[0].values.len(), 3);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.kennedy_dr");
        assert!(response.assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            Assumption::ParametricRestriction(restriction)
                if restriction.id.as_ref() == "response.kennedy_dr.nuisance_regularity"
        )));
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|d| d.code.as_ref() == "response.kennedy_dr.additive_covariate_if"),
            "Kennedy IF covariate term must disclose the additive-μ restriction"
        );
        let tail = response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.outcome_tail_ratio")
            .expect("outcome tail ratio");
        assert_eq!(tail.values.len(), 2);
        assert!(tail.values[0] < OUTCOME_TAIL_RATIO_BOUND, "ratio={}", tail.values[0]);
        assert!((tail.values[1] - OUTCOME_TAIL_RATIO_BOUND).abs() < 1e-12);
        assert!(
            response
                .support
                .warnings
                .iter()
                .all(|w| w.code.as_ref() != "response.heavy_tailed_outcome")
        );
        let winsor = response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.pseudo_outcome_winsor_shift")
            .expect("winsor shift");
        assert_eq!(winsor.values.len(), 3);
    }

    #[test]
    fn kennedy_curve_discloses_additive_if_on_a_treatment_covariate_product() {
        let n = 80usize;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64) * 2.0 - 1.0).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 17) % 11) as f64 / 5.0 - 1.0).collect();
        let y: Vec<f64> = a.iter().zip(&x).map(|(a, x)| a * x).collect();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
            ),
        });
        let response = ContinuousResponseEstimator::new([VariableId::from_raw(2)])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|d| d.code.as_ref() == "response.kennedy_dr.additive_covariate_if")
        );
        assert!(matches!(
            response.estimate,
            ResponseIdentification::PointIdentified(ResponseValue::Surface { .. })
        ));
    }

    fn overwrite_outcome(data: &TabularData, patches: &[(usize, f64)]) -> TabularData {
        let mut columns: Vec<Vec<f64>> =
            (0..3).map(|c| data.float64_values(VariableId::from_raw(c)).unwrap()).collect();
        for &(i, value) in patches {
            columns[1][i] = value;
        }
        TabularData::from_f64_columns([
            ("a", columns[0].as_slice()),
            ("y", columns[1].as_slice()),
            ("x", columns[2].as_slice()),
        ])
        .unwrap()
    }

    #[test]
    fn outcome_tail_ratio_is_zero_on_a_constant_and_explodes_on_a_spike() {
        assert!(outcome_tail_ratio(&[1.0, 1.0, 1.0, 1.0]).abs() < 1e-12);
        let mut y = vec![0.0; 21];
        y[0] = 500.0;
        assert!(outcome_tail_ratio(&y) > OUTCOME_TAIL_RATIO_BOUND);
    }

    #[test]
    fn heavy_tailed_outcome_warns_without_demoting_overlap() {
        let (data, a, y, x) = confounded_curve(240);
        let data = overwrite_outcome(&data, &[(120, 1_000.0), (121, -1_000.0)]);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(response.support.status, SupportStatus::Supported);
        let tail = response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.outcome_tail_ratio")
            .expect("outcome tail ratio");
        assert!(tail.values[0] > OUTCOME_TAIL_RATIO_BOUND, "ratio={}", tail.values[0]);
        let codes: Vec<&str> = response.support.warnings.iter().map(|w| w.code.as_ref()).collect();
        assert!(codes.contains(&"response.heavy_tailed_outcome"), "codes={codes:?}");
        let winsor = response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.pseudo_outcome_winsor_shift")
            .expect("winsor shift");
        assert_eq!(winsor.values.len(), 3);
        assert!(winsor.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pseudo_outcome_winsor_shift_warns_when_a_spike_moves_the_curve() {
        let n = 80usize;
        let treatments: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut phi: Vec<f64> = treatments.iter().map(|t| 2.0 * t).collect();
        phi[n / 2] = 400.0;
        let grid = [0.25, 0.5, 0.75];
        let mut workspace = LocalQuadraticWorkspace::default();
        let mean: Vec<f64> = grid
            .iter()
            .map(|&at| {
                gaussian_local_quadratic_influence_prechecked(
                    &mut workspace,
                    &treatments,
                    &phi,
                    at,
                    0.15,
                )
                .unwrap()
                .point
                .value
            })
            .collect();
        let mut support = SupportReport {
            status: SupportStatus::Supported,
            query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([1.0]) },
            diagnostics: Vec::new(),
            warnings: Vec::new(),
            point_status: None,
        };
        push_pseudo_outcome_winsor_shift(
            &mut support,
            &treatments,
            &phi,
            &grid,
            &mean,
            0.15,
            &mut workspace,
        );
        assert!(
            support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.pseudo_outcome_tail_sensitivity"),
            "codes={:?}",
            support.warnings.iter().map(|w| w.code.as_ref()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pseudo_outcome_additive_hoist_matches_brute_force_double_loop() {
        // The O(n) covariate-offset form must agree with the definitional
        // |valid|×n double loop (full counterfactual prediction per pair)
        // up to floating-point re-association. 160 rows sums the marginal density
        // term by term; 900 rows (720 training rows per fold) takes the binned
        // Gauss-transform path, which must agree with the same definitional sum.
        //
        // The covariate marginalization `∫μ̂(a, x) dP_n(x)` (Kennedy et al. 2017,
        // Thm. 3) is the empirical mean over the *full* sample, so `marginal_mu`
        // below averages the counterfactual prediction over every row `j` in
        // `0..n`, not only the fold's training rows — matching
        // `cross_fitted_pseudo_outcome_weighted`'s `covariate_offset`. The
        // marginal *density* mixture stays a training-row-only estimate, matching
        // the production `marginal_mixture`, which this test does not change.
        for rows in [160, 900] {
            let (data, a, y, x) = confounded_curve(rows);
            let estimator = ContinuousResponseEstimator::new([x]);
            let sample = CompleteSample::read(&data, y, &[a], &estimator.adjustment_set).unwrap();
            let fast = estimator.cross_fitted_pseudo_outcome(&sample).unwrap();

            let n = sample.len();
            let folds = estimator.options.folds;
            let mut brute = vec![0.0; n];
            for fold in 0..folds {
                let train: Vec<usize> = (0..n).filter(|i| i % folds != fold).collect();
                let valid: Vec<usize> = (0..n).filter(|i| i % folds == fold).collect();
                let mut gam_ws = GamWorkspace::default();
                let outcome_fit = estimator.fit_outcome(&sample, &train, &mut gam_ws).unwrap();
                let treatment_fit = estimator.fit_treatment(&sample, &train, &mut gam_ws).unwrap();
                let sigma = treatment_sigma(&sample, &train, treatment_fit.as_ref()).unwrap();
                let constant_mean = sample.train_treatment_mean(&train);
                for &i in &valid {
                    let mu_observed = predict_one(&outcome_fit, &sample.raw_row(i)).unwrap();
                    let treatment_mean = match treatment_fit.as_ref() {
                        Some(fit) => predict_one(fit, &sample.adjustment_row(i)).unwrap(),
                        None => constant_mean,
                    };
                    let raw_density =
                        gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                    let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                    let mut marginal_density = 0.0;
                    for &j in &train {
                        let mean_j = match treatment_fit.as_ref() {
                            Some(fit) => predict_one(fit, &sample.adjustment_row(j)).unwrap(),
                            None => constant_mean,
                        };
                        marginal_density +=
                            gaussian_density(sample.treatment_matrix[i], mean_j, sigma);
                    }
                    marginal_density /= train.len() as f64;
                    let mut marginal_mu = 0.0;
                    for j in 0..n {
                        let mut row = sample.raw_row(j);
                        row[0] = sample.treatment_matrix[i];
                        marginal_mu += predict_one(&outcome_fit, &row).unwrap();
                    }
                    marginal_mu /= n as f64;
                    brute[i] = marginal_mu
                        + (sample.outcome[i] - mu_observed) * marginal_density
                            / conditional_density;
                }
            }
            for (i, (&fast_i, &brute_i)) in fast.values.iter().zip(&brute).enumerate() {
                assert!(
                    (fast_i - brute_i).abs() <= 1e-9 * brute_i.abs().max(1.0),
                    "row {i}: fast={fast_i} brute={brute_i}"
                );
            }
        }
    }

    #[test]
    fn simultaneous_band_is_never_narrower_than_the_pointwise_band() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let run = |replicates: Option<u32>| {
            let mut estimator = ContinuousResponseEstimator::new([x]);
            estimator.options.bandwidth = Some(0.35);
            estimator.options.simultaneous_replicates = replicates;
            estimator
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let ResponseUncertainty::PointwiseBand { lower: p_lo, upper: p_hi, .. } =
            run(None).uncertainty
        else {
            panic!("expected pointwise band");
        };
        let ResponseUncertainty::SimultaneousBand { lower: s_lo, upper: s_hi, .. } =
            run(Some(400)).uncertainty
        else {
            panic!("expected simultaneous band");
        };
        // Both bands standardize by the same influence-based standard error, so the
        // simultaneous band can never be the tighter statement at the same level.
        for index in 0..3 {
            assert!(
                s_lo[index] <= p_lo[index] + 1e-12 && s_hi[index] >= p_hi[index] - 1e-12,
                "simultaneous band narrower than pointwise at {index}"
            );
        }
    }

    #[test]
    fn simultaneous_band_is_deterministic_and_contains_pointwise_curve() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
        });
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        estimator.options.simultaneous_replicates = Some(200);
        let first = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let second = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(first, second);
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &first.estimate
        else {
            panic!("expected surface");
        };
        let ResponseUncertainty::SimultaneousBand { lower, upper, replicates, .. } =
            &first.uncertainty
        else {
            panic!("expected simultaneous band");
        };
        assert_eq!(*replicates, 200);
        assert!(
            mean.iter().zip(lower.iter()).zip(upper.iter()).all(|((m, lo), hi)| lo < m && m < hi)
        );
        assert_eq!(first.provenance_id.as_ref(), "estimate.response.kennedy_dr_simultaneous");
    }

    #[test]
    fn intervention_response_executes_set_shift_and_stochastic_policies() {
        let (data, a, y, x) = confounded_curve(500);
        let estimate = |intervention| {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([intervention]),
            });
            ContinuousResponseEstimator::new([x])
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let set = estimate(Intervention::set(a, Value::f64(0.25)));
        let shift = estimate(Intervention::shift(a, Value::f64(0.25)));
        let stochastic =
            estimate(Intervention::stochastic(a, StochasticPolicy::gaussian(0.25, 0.01)));
        let scalar = |response: &CausalResponse| match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
            _ => panic!("expected scalar"),
        };
        assert!((scalar(&set) - 1.5).abs() < 0.2);
        assert!((scalar(&stochastic) - scalar(&set)).abs() < 0.1);
        assert!((scalar(&shift) - 1.5).abs() < 0.2);
        assert_eq!(set.support.status, SupportStatus::Extrapolative);
        assert_eq!(set.provenance_id.as_ref(), "estimate.response.intervention_gcomp");
        let codes: Vec<_> = set.support.warnings.iter().map(|w| w.code.as_ref()).collect();
        assert!(codes.contains(&"response.intervention_plugin_model_dependent"));
        assert!(!codes.contains(&"response.intervention_target_penalized_fallback"));
    }

    #[test]
    fn gaussian_policy_expectation_is_exact_for_a_quadratic_dose_effect() {
        // y = 1 + 2a + a² + 0.8z is a quadratic in the treatment, which the
        // unpenalized cubic treatment smooth reproduces exactly, and z is centred.
        // Under A ~ N(0.25, 0.05²): E[y] = 1 + 2(0.25) + (0.25² + 0.05²) = 1.565 —
        // no sampling error and no fixed policy-integration bias, PROVIDED the
        // treatment and the adjustment covariate are not concurved: `a` is a
        // permutation of the same grid as `z` (decorrelated by a coprime stride),
        // not `0.7*z + small noise` (|corr| < 0.98). With the near-collinear
        // original fixture, the additive GAM cannot uniquely identify the
        // quadratic-in-`a` term from the linear-in-`z` term off the observed
        // (a, z) manifold, and the estimate was biased to ~1.589 regardless of
        // how faithfully the policy integration itself was implemented.
        let n = 500;
        let mut a = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let treatment = -1.0 + 2.0 * ((i * 173 + 37) % n) as f64 / (n - 1) as f64;
            x.push(z);
            a.push(treatment);
            y.push(1.0 + 2.0 * treatment + treatment * treatment + 0.8 * z);
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let (a, y, x) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::stochastic(
                a,
                StochasticPolicy::gaussian(0.25, 0.0025),
            )]),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            &response.estimate
        else {
            panic!("expected scalar");
        };
        // Finite spline-basis / solver tolerance (GamOptions::tol = 1e-6) leaves a residual
        // ~1.5e-5 even once concurvity is removed; 2e-5 keeps the assertion meaningful
        // (two orders of magnitude tighter than the old 0.024 concurvity bias) without
        // chasing solver-noise digits.
        assert!((value - 1.565).abs() < 2e-5, "value={value}");
    }

    #[test]
    fn intervention_response_discloses_the_penalized_fallback_for_a_binary_treatment() {
        // A binary treatment cannot support the unpenalized cubic target basis;
        // the penalized nuisance fit runs instead, and the result must say so.
        let n = 400;
        let mut a = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let treatment = if (i * 37 % 101) as f64 / 100.0 < 0.5 + 0.3 * z { 1.0 } else { 0.0 };
            x.push(z);
            a.push(treatment);
            y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let (a, y, x) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(a, Value::f64(1.0))]),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let fallback = response
            .support
            .warnings
            .iter()
            .find(|w| w.code.as_ref() == "response.intervention_target_penalized_fallback")
            .expect("the penalized fallback must be disclosed");
        assert!(fallback.fields.iter().any(|(key, _)| key.as_ref() == "target_fit_error"));
        let model = response
            .support
            .warnings
            .iter()
            .find(|w| w.code.as_ref() == "response.intervention_plugin_model_dependent")
            .unwrap();
        assert!(model.message.contains("penalized treatment and adjustment splines"));
        assert!(!model.message.contains("unpenalized treatment splines (the target)"));
    }

    #[test]
    fn intervention_response_fails_closed_for_soft_interventions() {
        let (data, a, y, x) = confounded_curve(100);
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::soft(
                a,
                antecedent_core::MechanismOverride::named("replacement", Arc::from([])),
            )]),
        });
        let error = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("structural model"));
    }

    #[test]
    fn joint_discrete_mixture_beyond_the_budget_is_refused() {
        // The mixture is a cartesian product, so cost is exponential in the number of joined
        // discrete policies. Four 14-level categoricals is already ~38k combinations per row
        // — seconds of work — and nothing stops a caller going further. Refuse instead of
        // running for hours or silently falling back to Monte Carlo.
        let (data, a, y, x) = confounded_curve(200);
        let run = |levels: usize| {
            let probs: Arc<[f64]> = Arc::from(vec![1.0 / levels as f64; levels]);
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([Intervention::stochastic(
                    a,
                    StochasticPolicy::Categorical { probs },
                )]),
            });
            ContinuousResponseEstimator::new([x]).estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
        };
        assert!(run(MAX_EXACT_MIXTURE_COMBINATIONS).is_ok());
        let error = run(MAX_EXACT_MIXTURE_COMBINATIONS + 1).unwrap_err();
        assert!(error.to_string().contains("exact-mixture budget"), "got {error}");
    }

    #[test]
    fn categorical_intervention_is_an_exact_finite_mixture_not_monte_carlo() {
        // Under an additive model, a Categorical policy on codes {0,1} with weights
        // (0.25, 0.75) must equal 0.25·μ(0) + 0.75·μ(1). Monte Carlo through a continuous
        // spline only approximates that sum and would treat the codes as ordered coordinates.
        let (data, a, y, x) = confounded_curve(500);
        let estimate = |intervention| {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([intervention]),
            });
            let response = ContinuousResponseEstimator::new([x])
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap();
            match response.estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => value,
                _ => panic!("expected scalar"),
            }
        };
        let at0 = estimate(Intervention::set(a, Value::f64(0.0)));
        let at1 = estimate(Intervention::set(a, Value::f64(1.0)));
        let mixture =
            estimate(Intervention::stochastic(a, StochasticPolicy::categorical([0.25, 0.75])));
        let expected = 0.25 * at0 + 0.75 * at1;
        assert!(
            (mixture - expected).abs() < 1e-10,
            "categorical g-comp={mixture}, exact mixture={expected}"
        );
    }

    #[test]
    fn treatment_sigma_uses_gam_edf_not_n_minus_one() {
        // Discriminating residual-scale check: with edf > 1, dividing by n−1 understates σ.
        let n = 80usize;
        let mut treatments = Vec::with_capacity(n);
        let mut adjustment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            adjustment.push(z);
            treatments.push(0.8 * z + 0.1 * (i as f64 * 0.3).sin());
            outcome.push(1.0 + treatments[i] + 0.5 * z);
        }
        let sample = CompleteSample {
            keep: (0..n).collect(),
            outcome,
            treatments: treatments.clone(),
            treatment_matrix: treatments,
            adjustment,
            treatment_cols: 1,
            adjustment_cols: 1,
            raw_cols: 2,
        };
        let train: Vec<usize> = (0..n).collect();
        let mut gam_ws = GamWorkspace::default();
        let fit = ContinuousResponseEstimator::new([VariableId::from_raw(0)])
            .fit_treatment(&sample, &train, &mut gam_ws)
            .unwrap()
            .expect("adjustment present");
        assert!(fit.edf_approx > 1.0 + 1e-6, "edf={}", fit.edf_approx);
        let got = treatment_sigma(&sample, &train, Some(&fit)).unwrap();
        let rss: f64 = fit.residuals.iter().map(|r| r * r).sum();
        let wrong = (rss / (n - 1) as f64).sqrt();
        let right = (rss / (n as f64 - fit.edf_approx).max(1.0)).sqrt();
        assert!(
            (got - right).abs() < 1e-12,
            "treatment_sigma={got} should use edf denominator ({right})"
        );
        assert!(
            (got - wrong).abs() > 1e-6,
            "edf and n-1 denominators coincide; the test cannot discriminate"
        );
        assert!(got > wrong, "edf-aware σ must exceed the n-1 understatement");
    }

    #[test]
    fn row_diagnostics_export_is_opt_in_and_aligned_to_retained_rows() {
        let (data, a, y, x) = confounded_curve(200);
        // Poison one row so a retained-row index gap is observable in the export.
        let mut columns: Vec<Vec<f64>> =
            (0..3).map(|c| data.float64_values(VariableId::from_raw(c)).unwrap()).collect();
        columns[1][7] = f64::NAN;
        let data = TabularData::from_f64_columns([
            ("a", columns[0].as_slice()),
            ("y", columns[1].as_slice()),
            ("x", columns[2].as_slice()),
        ])
        .unwrap();
        let grid = [-0.4, 0.0, 0.4];
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from(grid))),
        });
        let run = |export: bool| {
            let mut estimator = ContinuousResponseEstimator::new([x]);
            estimator.options.export_row_diagnostics = export;
            estimator
                .estimate_identified(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                )
                .unwrap()
        };
        let off = run(false);
        assert!(
            off.support.diagnostics.iter().all(|d| !d.id.starts_with("response.row_")),
            "row diagnostics must be absent when the flag is off"
        );
        let on = run(true);
        let diagnostic = |id: &str| {
            on.support
                .diagnostics
                .iter()
                .find(|d| d.id.as_ref() == id)
                .unwrap_or_else(|| panic!("missing diagnostic {id}"))
        };
        let n = 199; // one NaN row dropped
        let row_index = diagnostic("response.row_index");
        assert_eq!(row_index.values.len(), n);
        let expected: Vec<f64> = (0..200).filter(|&i| i != 7).map(f64::from).collect();
        assert_eq!(row_index.values.as_ref(), expected.as_slice());
        let pseudo = diagnostic("response.row_pseudo_outcome");
        assert_eq!(pseudo.values.len(), n);
        assert!(pseudo.values.iter().all(|v| v.is_finite()));
        let influence = diagnostic("response.row_influence");
        assert_eq!(influence.values.len(), grid.len() * n);
        assert!(influence.values.iter().all(|v| v.is_finite()));
        assert!(influence.detail.contains("grid_len=3"));
        // The export must not perturb the estimate itself.
        assert_eq!(off.estimate, on.estimate);
        assert_eq!(off.uncertainty, on.uncertainty);
    }

    #[test]
    fn riesz_ade_calibrates_linear_slope() {
        let (data, a, y, x) = confounded_curve(600);
        let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
            outcome: y,
            treatment: a,
            weighting: DerivativeWeighting::Observed,
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("expected scalar");
        };
        assert!((value - 2.0).abs() < 0.2, "ade={value}");
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
        assert!(
            response
                .support
                .diagnostics
                .iter()
                .any(|d| d.id.as_ref() == "response.outcome_tail_ratio")
        );
    }

    #[test]
    fn bayesian_ade_refits_nuisances_not_frozen_scores() {
        let (data, a, y, x) = confounded_curve(240);
        let estimator = ContinuousResponseEstimator::new([x]);
        let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
        let frozen = estimator.cross_fitted_ade_scores(&sample).unwrap();
        let mut weights = vec![1.0; sample.len()];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = if i < sample.len() / 5 { 8.0 } else { 0.2 };
        }
        let folds = estimator.cross_fit_folds(&sample).unwrap();
        let plugin = estimator.weighted_cross_fitted_ade(&sample, &folds, &weights).unwrap();
        let frozen_mean =
            frozen.scores.iter().zip(&weights).map(|(score, weight)| score * weight).sum::<f64>()
                / weights.iter().sum::<f64>();
        assert!(
            (plugin - frozen_mean).abs() > 1e-3,
            "weighted-plugin ADE must move with refitted μ,α; plugin={plugin} frozen={frozen_mean}"
        );

        let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
            outcome: y,
            treatment: a,
            weighting: DerivativeWeighting::Observed,
        });
        let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
        let response = estimator
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &antecedent_core::ExecutionContext::for_tests(18),
            )
            .unwrap();
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
        assert_retained_draws_publish(&response.uncertainty, 8);
        assert!(
            response.assumptions.entries.iter().any(|record| match &record.assumption {
                Assumption::ParametricRestriction(restriction) =>
                    restriction.description.contains("not a frozen-score reweight"),
                _ => false,
            }),
            "Bayesian ADE must record the refit assumption: {:?}",
            response.assumptions
        );
        for n_draws in [0, 1] {
            let too_few = crate::BayesianGComputationAte::new().with_n_draws(n_draws);
            let err = estimator
                .estimate_bayesian(
                    &data,
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &too_few,
                    &antecedent_core::ExecutionContext::for_tests(18),
                )
                .unwrap_err();
            assert!(
                err.to_string().contains("n_draws >= 2"),
                "n_draws={n_draws} must refuse: {err}"
            );
        }
    }

    #[test]
    fn point_elasticity_applies_scale_transform() {
        let (data, a, y, x) = confounded_curve(500);
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at: 0.4,
            order: 1,
            scale: DerivativeScale::LogLog,
        });
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("expected scalar");
        };
        assert!(value.is_finite() && value > 0.2 && value < 0.7, "elasticity={value}");
        // The elasticity publishes a Fieller interval on the joint
        // local-coordinate covariance and a delta-method standard error.
        let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = response.uncertainty
        else {
            panic!("expected a published scalar interval, got {:?}", response.uncertainty);
        };
        assert!(standard_error.is_finite() && standard_error > 0.0, "se={standard_error}");
        assert!(lower <= upper, "interval must be ordered");
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.derivative_interval_fieller"),
            "log-scale elasticity must disclose the Fieller interval"
        );
        assert!(
            !response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.derivative_interval_withheld"),
            "log-scale elasticity no longer withholds its interval"
        );
    }

    #[test]
    fn fieller_elasticity_interval_accounts_for_denominator_uncertainty() {
        let covariance = [[0.01, 0.002, 0.0], [0.002, 0.04, 0.0], [0.0; 3]];
        let (lower, upper) = fieller_elasticity_interval(1.0, 2.0, 0.5, &covariance, 1.96).unwrap();
        assert!(lower < 1.0 && 1.0 < upper);
        assert!(upper - 1.0 > 1.0 - lower, "denominator uncertainty makes asymmetric bounds");
        assert!(fieller_elasticity_interval(0.1, 2.0, 0.5, &covariance, 1.96).is_none());
    }

    #[test]
    fn elasticity_refuses_nonpositive_fitted_response() {
        assert!(
            transform_derivative(1.0, 2.0, 0.0, DerivativeScale::LogLog)
                .unwrap_err()
                .to_string()
                .contains("positive fitted response")
        );
        let n: usize = 80;
        let a: Vec<f64> = (0..n).map(|i| -0.4 + i as f64 * 0.01).collect();
        let y: Vec<f64> = a.iter().map(|av| -4.0 - 2.0 * av).collect();
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: VariableId::from_raw(1),
            treatment: VariableId::from_raw(0),
            at: 0.2,
            order: 1,
            scale: DerivativeScale::LogLog,
        });
        let mut estimator = ContinuousResponseEstimator::new([VariableId::from_raw(2)]);
        estimator.options.bandwidth = Some(0.35);
        let error = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("positive fitted response"), "got {error}");
    }

    #[test]
    fn point_derivative_refuses_default_silverman_bandwidth() {
        let (data, a, y, x) = confounded_curve(200);
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at: 0.0,
            order: 1,
            scale: DerivativeScale::Identity,
        });
        let error = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("explicit bandwidth"), "got {error}");
    }

    #[test]
    fn static_estimator_refuses_temporal_response_attachment() {
        let (data, a, y, x) = confounded_curve(100);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.2, 0.2]))),
        })
        .with_temporal(
            antecedent_core::TemporalResponseSpec::new(
                [1u32],
                antecedent_core::TemporalPolicy::pulse(0),
                None,
            )
            .unwrap(),
        );
        let error = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("TemporalResponseEstimator"));
    }

    #[test]
    fn point_derivative_interval_is_robust_bias_corrected() {
        let (data, a, y, x) = confounded_curve(500);
        let at = 0.2;
        let bandwidth = 0.35;
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at,
            order: 1,
            scale: DerivativeScale::Identity,
        });
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(bandwidth);

        let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
        let pseudo = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
        let local = antecedent_stats::gaussian_local_quadratic_influence(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
        )
        .unwrap();
        let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
            None,
        )
        .unwrap();
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        // The point value stays the local-quadratic coordinate.
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("expected a scalar derivative");
        };
        assert!((value - local.point.first_derivative).abs() < 1e-12);
        // The interval is the RBC interval: the bias-corrected coordinate ± z·its
        // own robust SE, not the conventional local-quadratic interval.
        let ResponseUncertainty::Scalar {
            standard_error, lower, upper, level, interpretation, ..
        } = response.uncertainty
        else {
            panic!("expected scalar uncertainty");
        };
        assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Confidence);
        let z = normal_ppf(0.5 + level / 2.0);
        assert!((standard_error - corrected.robust_first_derivative_standard_error).abs() < 1e-12);
        assert!((0.5 * (lower + upper) - corrected.first_derivative).abs() < 1e-10);
        assert!((upper - lower - 2.0 * z * standard_error).abs() < 1e-10);
        assert!(
            standard_error > local.robust_first_derivative_standard_error,
            "the bias-corrected SE pays for the bias estimate's variance"
        );
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.derivative_interval_bias_corrected"),
            "a published derivative interval must say it is bias-corrected"
        );
    }

    #[test]
    fn second_derivative_interval_is_centered_at_the_local_quartic_curvature() {
        let (data, a, y, x) = confounded_curve(500);
        let (at, bandwidth) = (0.2, 0.6);
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at,
            order: 2,
            scale: DerivativeScale::Identity,
        });
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(bandwidth);
        let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
        let pseudo = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
        let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
            &sample.treatments,
            &pseudo,
            at,
            bandwidth,
            None,
        )
        .unwrap();
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = response.uncertainty
        else {
            panic!("expected scalar uncertainty");
        };
        assert!((standard_error - corrected.robust_second_derivative_standard_error).abs() < 1e-12);
        assert!((0.5 * (lower + upper) - corrected.second_derivative).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_uses_chain_rule_on_log_log_scale() {
        // m(a)=a² at a=2: d² log(m)/d(log(a))² = 0.
        let value =
            transform_point_derivative(4.0, 4.0, 2.0, 2.0, 2, DerivativeScale::LogLog).unwrap();
        assert!(value.abs() < 1e-12);
    }

    #[test]
    fn plugin_jacobian_recovers_low_dimensional_slopes() {
        let n = 400;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut y1 = Vec::with_capacity(n);
        let mut y2 = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let av = z + 0.2 * (i as f64 * 0.7).sin();
            let bv = -0.4 * z + 0.3 * (i as f64 * 1.1).cos();
            a.push(av);
            b.push(bv);
            x.push(z);
            y1.push(1.0 + 2.0 * av - 0.5 * bv + z);
            y2.push(-1.0 + 0.25 * av + 1.5 * bv - 0.7 * z);
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y1", y1.as_slice()),
            ("y2", y2.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([0.0, 0.0]),
            scale: DerivativeScale::Identity,
        });
        let response = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) =
            response.estimate
        else {
            panic!("expected Jacobian");
        };
        for (got, expected) in values.iter().zip([2.0, -0.5, 0.25, 1.5]) {
            assert!((got - expected).abs() < 0.25, "got={got}, expected={expected}");
        }
        assert_eq!(response.support.status, SupportStatus::Extrapolative);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.gam_derivative");
    }

    #[test]
    fn plugin_jacobian_refuses_mismatched_complete_cases() {
        let n: usize = 80;
        let a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 / n as f64).collect();
        let y1: Vec<f64> = a.iter().zip(&b).map(|(av, bv)| 1.0 + 2.0 * av - 0.5 * bv).collect();
        let mut y2 = y1.iter().map(|v| -1.0 + 0.25 * v).collect::<Vec<_>>();
        for value in y2.iter_mut().skip(n - 25) {
            *value = f64::NAN;
        }
        let x: Vec<f64> = a.clone();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y1", y1.as_slice()),
            ("y2", y2.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([0.5, 0.5]),
            scale: DerivativeScale::Identity,
        });
        let error = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("shared complete-case"), "got {error}");
    }

    #[test]
    fn plugin_jacobian_warns_when_clamped_outside_support() {
        let n: usize = 120;
        let a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let b: Vec<f64> = a.clone();
        let y1: Vec<f64> = a.iter().map(|av| 1.0 + 2.0 * av).collect();
        let y2: Vec<f64> = a.iter().map(|av| -1.0 + 0.25 * av).collect();
        let x: Vec<f64> = a.clone();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y1", y1.as_slice()),
            ("y2", y2.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([8.0, 8.0]),
            scale: DerivativeScale::Identity,
        });
        let response = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(response.support.status, SupportStatus::OutsideEmpiricalSupport);
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.clamped_basis_derivative"),
            "outside-support Jacobian must emit the clamped-basis warning"
        );
    }

    /// The plug-in derivative refusals over more than two treatments keep their
    /// message (no reason code, same class) and name what to do instead in the
    /// structured `remedy` field.
    #[test]
    fn plugin_derivatives_over_two_treatments_name_a_remedy() {
        let n: usize = 60;
        let col = |k: f64| (0..n).map(|i| (i as f64 * k).sin()).collect::<Vec<f64>>();
        let (a, b, c, y, x) = (col(0.3), col(0.7), col(1.1), col(0.5), col(0.9));
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("c", c.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let treatments: Arc<[VariableId]> =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)]);
        let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(4)]);
        let refuse = |functional| {
            estimator
                .estimate_identified(
                    &data,
                    &ResponseQuery::new(functional),
                    IdentificationStatus::IdentifiedUnderParametricRestrictions,
                    AssumptionSet::new(),
                )
                .unwrap_err()
        };
        let jacobian = refuse(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments: treatments.clone(),
            at: Arc::from([0.0, 0.0, 0.0]),
            scale: DerivativeScale::Identity,
        });
        assert_eq!(
            jacobian.to_string(),
            "plug-in response Jacobian supports at most two treatments"
        );
        assert!(!matches!(jacobian, EstimationError::Refused { .. }));
        let remedy = jacobian.remedy().expect("the Jacobian refusal names a remedy");
        assert!(remedy.contains("at most two treatments"), "{remedy}");
        assert!(remedy.contains("AverageDerivative"), "{remedy}");

        let directional = refuse(ResponseFunctional::DirectionalDerivative {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments,
            at: Arc::from([0.0, 0.0, 0.0]),
            direction: Arc::from([1.0, 0.0, 0.0]),
        });
        assert_eq!(
            directional.to_string(),
            "plug-in directional derivative supports at most two treatments"
        );
        let remedy = directional.remedy().expect("the directional refusal names a remedy");
        assert!(remedy.contains("ResponseJacobian"), "{remedy}");
    }

    /// Every additive-GAM non-convergence refusal keeps its message and names
    /// the options that let the backfit settle.
    #[test]
    fn unfinished_gam_fit_refusals_name_a_remedy() {
        let n: usize = 80;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v).collect();
        let mut workspace = GamWorkspace::default();
        let mut fit = fit_additive(&x, n, 1, &y, 6, 1.0, &mut workspace).unwrap();
        assert!(fit.converged);
        fit.converged = false;
        for message in [
            GAM_TARGET_NOT_CONVERGED,
            crossfit::GAM_NUISANCE_NOT_CONVERGED,
            crossfit::WEIGHTED_GAM_NUISANCE_NOT_CONVERGED,
        ] {
            let error = require_converged_gam(fit.clone(), message).unwrap_err();
            assert_eq!(error.to_string(), message);
            assert!(!matches!(error, EstimationError::Refused { .. }));
            let remedy = error.remedy().expect("a GAM non-convergence refusal names a remedy");
            assert!(remedy.contains("nuisance_lambda"), "{remedy}");
            assert!(remedy.contains("nuisance_basis"), "{remedy}");
        }
        fit.converged = true;
        assert!(require_converged_gam(fit, GAM_TARGET_NOT_CONVERGED).is_ok());
    }

    /// Two treatments, two outcomes, one adjustment covariate, Gaussian-like noise.
    /// `y1 = scale·(1 + 2a − 0.5b + 0.6a² + x + e₁) + extra·a`.
    fn plugin_gradient_data(scale: f64, extra: f64) -> TabularData {
        let n: usize = 400;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut noise = || {
            // Sum of four uniforms, centred: a deterministic, roughly Gaussian draw.
            (0..4)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .sum::<f64>()
        };
        let (mut a, mut b, mut x, mut y1, mut y2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let z = noise();
            let av = 0.6 * z + noise();
            let bv = -0.3 * z + noise();
            a.push(av);
            b.push(bv);
            x.push(z);
            y1.push(scale * (1.0 + 2.0 * av - 0.5 * bv + 0.6 * av * av + z + noise()) + extra * av);
            y2.push(-1.0 + 0.25 * av + 1.5 * bv - 0.7 * z + noise());
        }
        TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y1", y1.as_slice()),
            ("y2", y2.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap()
    }

    fn plugin_jacobian(scale: DerivativeScale) -> ResponseFunctional {
        ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([0.2, -0.1]),
            scale,
        }
    }

    fn plugin_directional(direction: [f64; 2]) -> ResponseFunctional {
        ResponseFunctional::DirectionalDerivative {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([0.2, -0.1]),
            direction: Arc::from(direction),
        }
    }

    fn published(data: &TabularData, functional: ResponseFunctional) -> CausalResponse {
        ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .estimate_identified(
                data,
                &ResponseQuery::new(functional),
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .unwrap()
    }

    fn internal_band(
        data: &TabularData,
        functional: &ResponseFunctional,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (value, uncertainty) = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
            .plugin_gradient_interval_internal(data, functional)
            .unwrap();
        let values = match value {
            ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => {
                values.to_vec()
            }
            other => panic!("unexpected value {other:?}"),
        };
        let ResponseUncertainty::PointwiseBand { level, lower, upper, interpretation, draws } =
            uncertainty
        else {
            panic!("expected a pointwise band, got {uncertainty:?}");
        };
        assert!((level - ContinuousResponseOptions::default().confidence_level).abs() < 1e-15);
        assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Confidence);
        assert!(draws.is_none());
        (values, lower.to_vec(), upper.to_vec())
    }

    /// The Frequentist Jacobian and directional derivative publish the point only
    /// and say why: the interval route is closed until its coverage is measured.
    #[test]
    fn frequentist_plugin_gradient_withholds_its_interval_and_says_why() {
        let data = plugin_gradient_data(1.0, 0.0);
        for functional in
            [plugin_jacobian(DerivativeScale::Identity), plugin_directional([1.0, 1.0])]
        {
            let response = published(&data, functional);
            assert!(matches!(response.uncertainty, ResponseUncertainty::None));
            let withheld = response
                .support
                .warnings
                .iter()
                .find(|w| w.code.as_ref() == "response.derivative_interval_withheld")
                .expect("the withheld interval is disclosed");
            assert!(withheld.message.contains("not yet measured"), "{}", withheld.message);
            assert!(withheld.message.contains("Bayesian"), "{}", withheld.message);
        }
    }

    /// The closed route's band is the coefficient sandwich of the published point.
    #[test]
    fn plugin_gradient_interval_internal_is_a_sandwich_band_around_the_published_point() {
        let data = plugin_gradient_data(1.0, 0.0);
        let jacobian = plugin_jacobian(DerivativeScale::Identity);
        let (values, lower, upper) = internal_band(&data, &jacobian);
        let ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
            values: public, ..
        }) = published(&data, jacobian.clone()).estimate
        else {
            panic!("expected Jacobian");
        };
        assert_eq!(values, public.to_vec(), "the band is around the published point");
        for j in 0..4 {
            assert!(lower[j] < values[j] && values[j] < upper[j], "coordinate {j}");
            let half = 0.5 * (upper[j] - lower[j]);
            assert!(half > 0.0 && half < 1.0, "coordinate {j}: half-width {half}");
        }
        // The truth (∂y1/∂a = 2 + 1.2·0.2, ∂y1/∂b = −0.5, ∂y2/∂a = 0.25, ∂y2/∂b = 1.5)
        // sits inside a band of a few half-widths on this one dataset.
        for (j, truth) in [2.24, -0.5, 0.25, 1.5].into_iter().enumerate() {
            let half = 0.5 * (upper[j] - lower[j]);
            assert!((values[j] - truth).abs() < 4.0 * half, "coordinate {j}");
        }

        // A unit direction reads one Jacobian column, band included.
        let (dir, dir_lower, dir_upper) = internal_band(&data, &plugin_directional([1.0, 0.0]));
        for (k, j) in [(0, 0), (1, 2)] {
            assert!((dir[k] - values[j]).abs() < 1e-12);
            assert!((dir_lower[k] - lower[j]).abs() < 1e-9);
            assert!((dir_upper[k] - upper[j]).abs() < 1e-9);
        }
        let ResponseIdentification::PointIdentified(ResponseValue::Vector(public_dir)) =
            published(&data, plugin_directional([1.0, 0.0])).estimate
        else {
            panic!("expected a vector");
        };
        assert_eq!(dir, public_dir.to_vec());

        // Equivariance: scaling y1 scales its point and band; adding an exact
        // linear treatment term shifts the point and leaves the band width alone
        // (the cubic basis spans it, so the residuals do not move).
        let half = |lo: &[f64], hi: &[f64], j: usize| 0.5 * (hi[j] - lo[j]);
        let (scaled, s_lower, s_upper) = internal_band(&plugin_gradient_data(2.0, 0.0), &jacobian);
        let (shifted, t_lower, t_upper) = internal_band(&plugin_gradient_data(1.0, 3.0), &jacobian);
        for j in 0..2 {
            let base = half(&lower, &upper, j);
            assert!((scaled[j] - 2.0 * values[j]).abs() < 1e-6, "coordinate {j}");
            assert!((half(&s_lower, &s_upper, j) - 2.0 * base).abs() < 1e-6 * base.max(1.0));
            let expected_shift = if j == 0 { 3.0 } else { 0.0 };
            assert!((shifted[j] - values[j] - expected_shift).abs() < 1e-6, "coordinate {j}");
            assert!((half(&t_lower, &t_upper, j) - base).abs() < 1e-6 * base.max(1.0));
        }
    }

    #[test]
    fn plugin_gradient_interval_internal_refuses_what_it_does_not_construct() {
        let data = plugin_gradient_data(1.0, 0.0);
        let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(4)]);
        let error = estimator
            .plugin_gradient_interval_internal(&data, &plugin_jacobian(DerivativeScale::LogLog))
            .unwrap_err();
        assert!(error.to_string().contains("identity scale"), "{error}");
        let error = estimator
            .plugin_gradient_interval_internal(
                &data,
                &ResponseFunctional::PointDerivative {
                    outcome: VariableId::from_raw(2),
                    treatment: VariableId::from_raw(0),
                    at: 0.2,
                    order: 1,
                    scale: DerivativeScale::Identity,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("Jacobian and directional"), "{error}");
    }
}
