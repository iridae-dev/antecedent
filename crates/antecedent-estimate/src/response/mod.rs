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
    AssumptionStatus, CausalResponse, CausalRng, IdentificationStatus, ObservationSpec,
    ParametricAssumption, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_stats::StatsError;

use crate::EstimationError;

mod band;
mod bayes;
mod crossfit;
mod curve;
mod derivative;
mod nuisance;
mod plugin;
mod policy;
mod sample;
mod support;
#[cfg(test)]
mod tests;
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
    ///   the local-WLS term `w_i · [(XᵀWX)⁻¹]₀ · x_i · (φ_i − x_iᵀβ̂) / sqrt(v_i)` plus the
    ///   covariate-marginalization term `(c_i − c̄)/N` (Kennedy et al. 2017,
    ///   Thm. 3; `c_i = μ̂(A_i, X_i) − ∫μ̂(A_i, x) dP_n(x)`). Sums to zero per
    ///   grid point before finite-fit residual correction; reported pointwise `SE(g) = √(Σ_i ψ²)`, and both bands
    ///   are `m̂ ± c·SE` with these values. Here `v_i = 1 - 2 S_ii + ||S_i||²`
    ///   is the local smoother residual variance factor. Pointwise criticals use
    ///   Student quantiles with Satterthwaite variance df, exported in
    ///   `response.local_variance_degrees_of_freedom`; corrected scores need
    ///   not sum to zero exactly.
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
    pub row_index: Arc<[antecedent_data::GlobalId]>,
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
    antecedent_kernels::bayesian_bootstrap_weights(rng, n, true)
}
