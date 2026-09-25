//! Bayesian mechanisms, g-computation, and posterior functional evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::doc_markdown
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::IdentificationStatus;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, ExecutionContext, PriorAssumption, StreamDomain,
    TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, ConflictSummary, ConjugateGaussianBackend,
    EffectBatch, EffectPrior, GaussianCoefficientPrior, GaussianVarianceModel, HmcGlmBackend,
    HmcOptions, InferenceBackend, InferenceDiagnostics, LaplaceGlmBackend, LaplaceWorkspace,
    PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind, PosteriorSchema,
    PosteriorSummary, PriorSensitivitySummary, PriorSet, PriorSpec, sample_gaussian_mvn,
};
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, DesignColumnRole, FaerBackend, GlmFamily,
    LeastSquaresWorkspace,
};

use crate::adjustment::{PreparedEstimationProblem, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DependenceScope, SerialDependence, long_run_tempering_factor,
};
use crate::util::require_explicit_override;

/// Posterior draws of the linear functional `weights' β`, one per retained draw.
/// `weights` is a design-row average after the intervention overlay.
pub(crate) fn linear_response_draws(
    posterior: &CausalPosterior,
    weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let mut values = vec![0.0; posterior.draws.n_draws];
    for (index, &weight) in weights.iter().enumerate() {
        let column = posterior
            .draws
            .schema
            .quantities
            .iter()
            .position(|q| {
                matches!(q,
            PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index)
            })
            .ok_or_else(|| EstimationError::stats_msg("response posterior missing coefficient"))?;
        for (value, &coefficient) in values.iter_mut().zip(posterior.draws.column(column)?) {
            *value += weight * coefficient;
        }
    }
    Ok(values)
}

/// `(mean, lower, upper, sd)` of linear-functional draws at an equal-tailed `level`.
pub(crate) fn summarize_linear_response_draws(
    mut values: Vec<f64>,
    level: f64,
) -> Result<(f64, f64, f64, f64), EstimationError> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "response posterior needs at least two finite draws",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    values.sort_by(f64::total_cmp);
    // Exchangeable-rank (type-6) ranks so a finite-draw interval covers at its level;
    // the same rule as every other posterior-draw interval in the crate.
    let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
        &values,
        level,
        antecedent_stats::QuantileRule::ExchangeableRank,
    );
    Ok((mean, lower, upper, sd))
}

/// Minimum kept draws for HMC so the MCMC publication gate (Ř≤1.01, ESS≥100)
/// is reachable on typical Gaussian GLMs.
pub const HMC_MIN_DRAWS: usize = 3_000;

/// Stable prefix of the inference-diagnostics note recorded when [`HMC_MIN_DRAWS`]
/// raised the requested draw count (`requested=<n> used=<m>`).
pub const HMC_DRAW_FLOOR_NOTE_PREFIX: &str = "hmc.draw_floor";

/// Diagnostics note when `unit_ids` request random-intercept GLS whitening under a
/// non-Gaussian likelihood. Whitening is a Gaussian linear transform and must not be
/// applied to count or binary outcomes.
pub const RANDOM_INTERCEPT_WHITEN_SKIP_NOTE: &str = "random_intercept.gls_whiten.skipped: compound-symmetry GLS whitening applies only under \
     GaussianIdentity; unit_ids were ignored for this likelihood";

/// `(requested, used)` draw counts when the HMC draw floor raised the request.
#[must_use]
pub fn hmc_draw_floor_from_notes(notes: &[Arc<str>]) -> Option<(usize, usize)> {
    notes.iter().find_map(|note| {
        let rest = note.strip_prefix(HMC_DRAW_FLOOR_NOTE_PREFIX)?;
        let mut requested = None;
        let mut used = None;
        for kv in rest.split_whitespace() {
            if let Some(v) = kv.strip_prefix("requested=") {
                requested = v.parse().ok();
            } else if let Some(v) = kv.strip_prefix("used=") {
                used = v.parse().ok();
            }
        }
        Some((requested?, used?))
    })
}

/// Causal posterior over an identified functional.
#[derive(Clone, Debug)]
pub struct CausalPosterior {
    /// Columnar effect (and optional coefficient) draws.
    pub draws: PosteriorDraws,
    /// Summary of `draws`.
    pub summaries: PosteriorSummary,
    /// Identification status — priors never upgrade this.
    pub identification: IdentificationStatus,
    /// Optional prior-sensitivity grid.
    pub prior_sensitivity: Option<PriorSensitivitySummary>,
    /// Optional external-prior conflict shrink summary.
    pub conflict_summary: Option<ConflictSummary>,
    /// Inference diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Assumptions including prior restrictions.
    pub assumptions: AssumptionSet,
    /// Unidentified graph mass retained when aggregating envelopes (0 if single graph).
    ///
    /// Mass of graph atoms that were structurally not identified. Never
    /// includes atoms a latency tier skipped (see [`Self::subsampled_out_mass`])
    /// or atoms that were identified but whose estimation failed (see
    /// [`Self::unevaluable_mass`]) — a refusal to estimate is not a proof of
    /// non-identification.
    pub unidentified_mass: f64,
    /// Identified graph mass the Interactive latency tier left out of the
    /// envelope subsample (0 outside that tier). Those atoms were not evaluated,
    /// so this is neither unidentified mass nor part of the published mixture;
    /// the identified-atom mixture covers `1 − unidentified − unevaluable − subsampled_out`.
    pub subsampled_out_mass: f64,
    /// Identified graph mass excluded from the mixture because estimation
    /// (design preparation, fitting, composition, or draw extraction) failed
    /// on that atom, not because identification failed. Zero unless the
    /// caller separates the two reasons; a path that has not yet been audited
    /// for this distinction may still fold this mass into
    /// [`Self::unidentified_mass`].
    pub unevaluable_mass: f64,
    /// Adaptive draw early-stop (Laplace / conjugate Gaussian redraw path).
    pub early_stopped: bool,
    /// Source treatment contrast `active − control` used to form the effect.
    ///
    /// Recorded so EffectFunctional hydrate can map ATE → β_T as ATE/Δ.
    /// `None` when the posterior is not an identity-link contrast (or the
    /// contrast was not available at fit time).
    pub treatment_contrast: Option<f64>,
}

impl CausalPosterior {
    /// Primary effect column index (first `Effect` quantity), if any.
    #[must_use]
    pub fn effect_column(&self) -> Option<usize> {
        self.draws
            .schema
            .quantities
            .iter()
            .position(|q| matches!(q, PosteriorQuantityKind::Effect { .. }))
    }

    /// Empirical P(effect < threshold) for the primary effect column.
    ///
    /// # Errors
    ///
    /// Missing effect column.
    pub fn probability_below(&self, threshold: f64) -> Result<f64, EstimationError> {
        let q = self
            .effect_column()
            .ok_or_else(|| EstimationError::stats_msg("CausalPosterior has no effect column"))?;
        self.draws.probability_below(q, threshold).map_err(EstimationError::from)
    }
}

/// Minimum coefficient prior variance when hydrating from a posterior (numerical floor).
const HYDRATE_VAR_FLOOR: f64 = 1e-12;
/// Smallest |Δ| accepted for the identity-link ATE → β_T mapping.
const HYDRATE_CONTRAST_FLOOR: f64 = 1e-12;

/// Bayesian posteriors need at least two draws to form a finite SD / interval.
///
/// This is an API refuse, not a support-matrix cell refuse. Callers must not
/// silently rewrite `0` or `1` to `2`.
///
/// # Errors
///
/// `n_draws < 2`.
pub fn require_bayesian_n_draws(n_draws: usize) -> Result<usize, EstimationError> {
    if n_draws < 2 {
        return Err(EstimationError::stats_msg(format!(
            "Bayesian inference requires n_draws >= 2; got {n_draws} (refusing silent rewrite)"
        )));
    }
    Ok(n_draws)
}

fn identity_ate_to_slope(mean: f64, sd: f64, delta: f64) -> Result<(f64, f64), EstimationError> {
    if !delta.is_finite() || delta.abs() <= HYDRATE_CONTRAST_FLOOR {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: EffectFunctional identity-link ATE→β_T mapping requires a finite nonzero source contrast Δ (active − control)",
        ));
    }
    Ok((mean / delta, sd / delta.abs()))
}

fn effect_functional_bind_col(
    names: &[Arc<str>],
    treatment_col: Option<usize>,
) -> Result<usize, EstimationError> {
    if let Some(idx) = names.iter().position(|n| n.as_ref() == "treatment_at_mean_modifier") {
        return Ok(idx);
    }
    treatment_col.ok_or_else(|| {
        EstimationError::stats_msg(
            "hydrate_prior: EffectFunctional requires treatment_col, or a CATE `treatment_at_mean_modifier` coefficient",
        )
    })
}

/// Build a Gaussian coefficient [`PriorSet`] from posterior quantity summaries.
///
/// Uses coefficient-column posterior means and SDs (index-aligned). Effect
/// columns are ignored. Coefficient SDs are **absolute**; they are converted
/// into conjugate scale `V0` via [`GaussianCoefficientPrior::from_absolute_variance`]
/// using the source residual-variance posterior mean when a
/// [`PosteriorQuantityKind::ResidualVariance`] column is present, otherwise
/// `σ² = 1` (documented in the sequential-prior assumption text).
///
/// Hydration is **diagonal**: off-diagonal posterior covariance is dropped, so
/// a transferred coefficient prior can be tighter than the source marginal on
/// linear combinations (and, for a single coefficient, is the source marginal
/// converted to `V0` — not a claim that every functional stays as wide as the
/// source).
///
/// # Errors
///
/// No coefficient columns, non-finite summaries, non-contiguous indices, or
/// dimension mismatch vs `expected_n_coef`.
pub fn hydrate_prior_from_quantity_summaries(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    expected_n_coef: Option<usize>,
) -> Result<PriorSet, EstimationError> {
    if mean.len() != quantities.len() || sd.len() != quantities.len() {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: mean/sd length must match quantities",
        ));
    }
    let mut coef_cols: Vec<(usize, usize)> = quantities
        .iter()
        .enumerate()
        .filter_map(|(col, q)| match q {
            PosteriorQuantityKind::Coefficient { index, .. } => Some((*index, col)),
            _ => None,
        })
        .collect();
    coef_cols.sort_by_key(|(index, _)| *index);
    let n_coef = coef_cols.len();
    if n_coef == 0 {
        return Err(EstimationError::stats_msg(
            "hydrate_prior_from_posterior: no coefficient columns in posterior",
        ));
    }
    if let Some(expected) = expected_n_coef {
        if n_coef != expected {
            return Err(EstimationError::PriorDimensionMismatch {
                posterior: n_coef,
                design: expected,
            });
        }
    }
    for (i, (index, _)) in coef_cols.iter().enumerate() {
        if *index != i {
            return Err(EstimationError::stats_msg(format!(
                "posterior coefficient indices are not contiguous (expected {i}, got {index})"
            )));
        }
    }
    let sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
    let mut means = Vec::with_capacity(n_coef);
    let mut abs_vars = Vec::with_capacity(n_coef);
    for (_, col) in &coef_cols {
        let m = mean[*col];
        let s = sd[*col];
        if !m.is_finite() || !s.is_finite() {
            return Err(EstimationError::stats_msg(
                "posterior coefficient summary is non-finite; cannot hydrate prior",
            ));
        }
        means.push(m);
        abs_vars.push((s * s).max(HYDRATE_VAR_FLOOR * sigma2));
    }
    let coef = GaussianCoefficientPrior::from_absolute_variance(
        Arc::from(means),
        Arc::from(abs_vars),
        sigma2,
    )
    .map_err(EstimationError::from)?;
    Ok(PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(coef)],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    })
}

/// Source `σ²` for absolute→`V0` hydrate: residual-variance posterior mean, else 1.
fn residual_sigma2_for_hydrate(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
) -> Result<f64, EstimationError> {
    for (i, q) in quantities.iter().enumerate() {
        if matches!(q, PosteriorQuantityKind::ResidualVariance) {
            let s2 = mean[i];
            if s2 <= 0.0 || !s2.is_finite() {
                return Err(EstimationError::stats_msg(
                    "hydrate_prior: residual_variance mean must be finite and > 0",
                ));
            }
            return Ok(s2);
        }
    }
    Ok(1.0)
}

/// Build a Gaussian coefficient [`PriorSet`] from a fitted posterior (sequential Bayes).
///
/// # Errors
///
/// See [`hydrate_prior_from_quantity_summaries`].
pub fn hydrate_prior_from_posterior(
    posterior: &CausalPosterior,
    expected_n_coef: Option<usize>,
) -> Result<PriorSet, EstimationError> {
    hydrate_prior_from_quantity_summaries(
        &posterior.draws.schema.quantities,
        &posterior.summaries.mean,
        &posterior.summaries.sd,
        expected_n_coef,
    )
}

/// Bridge from a banked posterior into a target design's coefficient prior.
///
/// Mirrors `antecedent_io::PriorMapping` without depending on `antecedent-io` (avoids a
/// cycle). Convert at the facade.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HydrateMapping {
    /// Identical coefficient subspace (P1-C sequential Bayes).
    IdenticalCoefficientSubspace,
    /// Effect-functional transfer via a named source quantity (e.g. `"ate"`).
    EffectFunctional {
        /// Source effect / quantity name.
        source_quantity: String,
    },
    /// Explicit source→target quantity name pairs.
    NamedParameters {
        /// `(source_name, target_name)` pairs.
        pairs: Vec<(String, String)>,
    },
}

/// Build a coefficient [`PriorSet`] under a declared [`HydrateMapping`].
///
/// - [`HydrateMapping::IdenticalCoefficientSubspace`]: full coef hydrate; hard-errors
///   when source coef count ≠ baseline length.
/// - [`HydrateMapping::EffectFunctional`]: identity-link ATE → slope bridge
///   `β ~ N(μ_ATE / Δ, (σ_ATE / Δ)²)` onto the treatment coefficient, or onto
///   `treatment_at_mean_modifier` when that CATE coefficient is present.
///   `source_contrast` is the source `active − control`. Missing or ~0 Δ is a
///   typed refuse of the mapping. Other dims keep `baseline`.
/// - [`HydrateMapping::NamedParameters`]: maps named source moments onto named
///   target coefficients; unmapped dims keep `baseline`.
///
/// Records `external_effect_prior` / `external_named_prior` on
/// [`PriorSet::restrictions`].
///
/// # Errors
///
/// Dimension mismatch, missing effect column, unknown names, missing/zero Δ,
/// or invalid baseline.
pub fn hydrate_prior(
    mapping: &HydrateMapping,
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    baseline: &PriorSet,
    target_coef_names: &[Arc<str>],
    treatment_col: Option<usize>,
    source_contrast: Option<f64>,
) -> Result<PriorSet, EstimationError> {
    if mean.len() != quantities.len() || sd.len() != quantities.len() {
        return Err(EstimationError::stats_msg(
            "hydrate_prior: mean/sd length must match quantities",
        ));
    }
    let n_target = target_coef_names.len();
    let base_coef = baseline.gaussian_coefficients().ok_or_else(|| {
        EstimationError::stats_msg("hydrate_prior: baseline missing GaussianCoefficients")
    })?;
    if base_coef.len() != n_target {
        return Err(EstimationError::stats_msg(format!(
            "hydrate_prior: baseline n_coef {} != target_coef_names {}",
            base_coef.len(),
            n_target
        )));
    }

    match mapping {
        HydrateMapping::IdenticalCoefficientSubspace => {
            let mut prior =
                hydrate_prior_from_quantity_summaries(quantities, mean, sd, Some(n_target))?;
            // Preserve residual specs from baseline when present.
            merge_baseline_residuals(&mut prior, baseline);
            Ok(prior)
        }
        HydrateMapping::EffectFunctional { source_quantity } => {
            let t_col = effect_functional_bind_col(target_coef_names, treatment_col)?;
            if t_col >= n_target {
                return Err(EstimationError::stats_msg(format!(
                    "hydrate_prior: treatment_col {t_col} out of range for {n_target} coefs"
                )));
            }
            let delta = source_contrast.ok_or_else(|| {
                EstimationError::stats_msg(
                    "hydrate_prior: EffectFunctional identity-link ATE→β_T mapping requires a finite nonzero source contrast Δ (active − control)",
                )
            })?;
            let (m, s) = quantity_moments(quantities, mean, sd, source_quantity.as_str())?;
            let (slope_mean, slope_sd) = identity_ate_to_slope(m, s, delta)?;
            let effect = EffectPrior::new(slope_mean, slope_sd.max(HYDRATE_VAR_FLOOR.sqrt()))
                .map_err(EstimationError::from)?;
            let sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
            let mut means: Vec<f64> = base_coef.mean.to_vec();
            let mut vars: Vec<f64> = base_coef.variance.to_vec();
            means[t_col] = effect.mean;
            let abs_var = (effect.sd * effect.sd).max(HYDRATE_VAR_FLOOR * sigma2);
            vars[t_col] = abs_var / sigma2;
            let coef =
                GaussianCoefficientPrior { mean: Arc::from(means), variance: Arc::from(vars) };
            coef.validate().map_err(EstimationError::from)?;
            let target_name = target_coef_names[t_col].as_ref();
            let implied = effect.mean * delta;
            let mut prior = PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(coef)],
                contrast: baseline.contrast,
                categorical: baseline.categorical.clone(),
                restrictions: vec![PriorAssumption {
                    id: Arc::from("external_effect_prior"),
                    description: Arc::from(format!(
                        "external effect-functional prior: identity-link ATE→β via N(μ/Δ, (σ/Δ)²) from `{source_quantity}` (Δ={delta}) onto {target_name}; implied NDE/ATE mean {implied}; absolute Var(β) converted to V0 with source σ²={sigma2}"
                    )),
                }],
            };
            merge_baseline_residuals(&mut prior, baseline);
            Ok(prior)
        }
        HydrateMapping::NamedParameters { pairs } => {
            if pairs.is_empty() {
                return Err(EstimationError::stats_msg(
                    "hydrate_prior: NamedParameters requires at least one pair",
                ));
            }
            let mut means: Vec<f64> = base_coef.mean.to_vec();
            let mut vars: Vec<f64> = base_coef.variance.to_vec();
            let name_index: std::collections::HashMap<&str, usize> =
                target_coef_names.iter().enumerate().map(|(i, n)| (n.as_ref(), i)).collect();
            let sigma2 = residual_sigma2_for_hydrate(quantities, mean)?;
            for (src, tgt) in pairs {
                let (m, s) = quantity_moments(quantities, mean, sd, src)?;
                let Some(&idx) = name_index.get(tgt.as_str()) else {
                    return Err(EstimationError::stats_msg(format!(
                        "hydrate_prior: unknown target coefficient name `{tgt}`"
                    )));
                };
                means[idx] = m;
                let abs_var = (s * s).max(HYDRATE_VAR_FLOOR * sigma2);
                vars[idx] = abs_var / sigma2;
            }
            let coef =
                GaussianCoefficientPrior { mean: Arc::from(means), variance: Arc::from(vars) };
            coef.validate().map_err(EstimationError::from)?;
            let pair_desc =
                pairs.iter().map(|(a, b)| format!("{a}->{b}")).collect::<Vec<_>>().join(", ");
            let mut prior = PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(coef)],
                contrast: baseline.contrast,
                categorical: baseline.categorical.clone(),
                restrictions: vec![PriorAssumption {
                    id: Arc::from("external_named_prior"),
                    description: Arc::from(format!(
                        "external named-parameter prior ({pair_desc}); absolute posterior Var converted to V0 with source σ²={sigma2}; diagonal only (off-diagonal covariance dropped)"
                    )),
                }],
            };
            merge_baseline_residuals(&mut prior, baseline);
            Ok(prior)
        }
    }
}

fn merge_baseline_residuals(prior: &mut PriorSet, baseline: &PriorSet) {
    for spec in &baseline.specs {
        match spec {
            PriorSpec::ResidualInvGamma(_) | PriorSpec::KnownResidualVariance(_) => {
                if !prior.specs.iter().any(|s| {
                    matches!(
                        s,
                        PriorSpec::ResidualInvGamma(_) | PriorSpec::KnownResidualVariance(_)
                    )
                }) {
                    prior.specs.push(spec.clone());
                }
            }
            PriorSpec::GaussianCoefficients(_) => {}
        }
    }
}

fn quantity_moments(
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    name: &str,
) -> Result<(f64, f64), EstimationError> {
    for (i, q) in quantities.iter().enumerate() {
        let q_name = match q {
            PosteriorQuantityKind::Effect { name: n }
            | PosteriorQuantityKind::Scalar { name: n } => Some(n.as_ref()),
            PosteriorQuantityKind::Coefficient { name: n, .. } => {
                n.as_ref().map(std::convert::AsRef::as_ref)
            }
            PosteriorQuantityKind::ResidualVariance => Some("residual_variance"),
        };
        if q_name == Some(name) {
            let m = mean[i];
            let s = sd[i];
            if !m.is_finite() || !s.is_finite() {
                return Err(EstimationError::stats_msg(format!(
                    "hydrate_prior: non-finite summary for quantity `{name}`"
                )));
            }
            return Ok((m, s.max(HYDRATE_VAR_FLOOR.sqrt())));
        }
    }
    Err(EstimationError::stats_msg(format!("hydrate_prior: missing quantity `{name}`")))
}

/// Bayesian linear / GLM mechanism fit (coefficient posterior).
#[derive(Clone, Debug)]
pub struct BayesianGlmMechanism {
    /// Fitted coefficient draws (columnar).
    pub coefficient_draws: PosteriorDraws,
    /// Point-estimate coefficients: posterior mode (Laplace) or posterior mean (conjugate, HMC).
    pub map: Vec<f64>,
    /// Likelihood used.
    pub likelihood: BayesLikelihood,
    /// Diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Compiled design retained for g-computation.
    pub design: CompiledDesign,
    /// Treatment column index in the design.
    pub treatment_col: usize,
    /// Active / control levels.
    pub active: f64,
    /// Control level.
    pub control: f64,
}

/// Which inference backend to use for Bayesian g-computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BayesianBackendKind {
    /// Analytic conjugate Gaussian (identity link only).
    ConjugateGaussian,
    /// Native Laplace GLM.
    Laplace,
    /// Native HMC GLM (multi-chain; ESS / R-hat gated).
    Hmc,
}

/// Bayesian g-computation ATE estimator.
#[derive(Clone, Debug)]
pub struct BayesianGComputationAte {
    /// Backend kind.
    pub backend: BayesianBackendKind,
    /// Likelihood (Laplace); conjugate forces GaussianIdentity.
    pub likelihood: BayesLikelihood,
    /// Draw count.
    pub n_draws: usize,
    /// RNG seed.
    pub seed: u64,
    /// Overlap policy (must be ExplicitOverride).
    pub overlap: OverlapPolicy,
    /// Prior scale for isotropic Gaussian coefficients (weakly informative default 10).
    pub prior_scale: f64,
    /// Optional explicit coefficient prior (e.g. hydrated from a previous posterior).
    /// When set, overrides isotropic [`Self::prior_scale`].
    pub prior: Option<PriorSet>,
}

impl Default for BayesianGComputationAte {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianGComputationAte {
    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            seed: 0,
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: 10.0,
            prior: None,
        }
    }

    /// Conjugate Gaussian linear path.
    #[must_use]
    pub fn conjugate() -> Self {
        Self {
            backend: BayesianBackendKind::ConjugateGaussian,
            likelihood: BayesLikelihood::GaussianIdentity,
            ..Self::new()
        }
    }

    /// Set the inference backend kind (conjugate Gaussian, Laplace, or HMC).
    #[must_use]
    pub const fn with_backend(mut self, backend: BayesianBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the likelihood used for the Laplace / HMC backends.
    ///
    /// Ignored by [`BayesianBackendKind::ConjugateGaussian`], which always forces
    /// [`BayesLikelihood::GaussianIdentity`].
    #[must_use]
    pub const fn with_likelihood(mut self, likelihood: BayesLikelihood) -> Self {
        self.likelihood = likelihood;
        self
    }

    /// The likelihood [`Self::fit`] actually uses: the conjugate backend forces
    /// [`BayesLikelihood::GaussianIdentity`]; Laplace and HMC use [`Self::likelihood`].
    #[must_use]
    pub const fn effective_likelihood(&self) -> BayesLikelihood {
        match self.backend {
            BayesianBackendKind::ConjugateGaussian => BayesLikelihood::GaussianIdentity,
            BayesianBackendKind::Laplace | BayesianBackendKind::Hmc => self.likelihood,
        }
    }

    /// GLM mean family of [`Self::effective_likelihood`] (inverse link and observation model).
    #[must_use]
    pub fn glm_family(&self) -> GlmFamily {
        likelihood_to_glm_family(self.effective_likelihood())
    }

    /// Set the target draw count.
    ///
    /// Defaults to 1000. [`BayesianBackendKind::Hmc`] floors this at the schedule needed to
    /// clear the Ř≤1.01 / ESS≥100 publication gate on typical Gaussian GLMs.
    #[must_use]
    pub const fn with_n_draws(mut self, n_draws: usize) -> Self {
        self.n_draws = n_draws;
        self
    }

    /// Set the RNG seed used for sampling / MVN draws.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the overlap policy. `prepare` requires [`OverlapPolicy::ExplicitOverride`].
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the isotropic Gaussian coefficient prior scale (weakly informative default 10).
    ///
    /// Ignored when [`Self::with_prior`] supplies an explicit coefficient prior.
    #[must_use]
    pub const fn with_prior_scale(mut self, prior_scale: f64) -> Self {
        self.prior_scale = prior_scale;
        self
    }

    /// Set an explicit coefficient prior (e.g. hydrated from a previous posterior via
    /// [`hydrate_prior_from_posterior`]), overriding [`Self::with_prior_scale`].
    #[must_use]
    pub fn with_prior(mut self, prior: PriorSet) -> Self {
        self.prior = Some(prior);
        self
    }

    /// The coefficient prior [`Self::fit`] uses on a design with `ncols` columns:
    /// the explicit [`Self::prior`] when set, otherwise the isotropic
    /// `N(0, prior_scale²)` prior. Predictive checks and sensitivity grids must
    /// use this rather than a fresh default so they describe the prior in force.
    #[must_use]
    pub fn prior_in_force(&self, ncols: usize) -> PriorSet {
        self.prior.clone().unwrap_or_else(|| PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(
                antecedent_prob::GaussianCoefficientPrior::isotropic(ncols, self.prior_scale),
            )],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        })
    }

    /// Prepare from data + identified estimand (same IR as frequentist adjustment).
    ///
    /// # Errors
    ///
    /// Overlap / estimand / data failures.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        require_explicit_override(
            self.overlap,
            "BayesianGComputationAte requires ExplicitOverride overlap policy",
        )?;
        if !estimand.is_adjustment_shaped() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "BayesianGComputationAte expects an adjustment-shaped estimand",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported(
                "Bayesian g-comp does not support effect modifiers",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("population_not_estimable"),
                "Bayesian g-comp only supports TargetPopulation::AllObserved",
            ));
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if (active - control).abs() < f64::EPSILON {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        let treatment = query.treatment;
        let outcome = query.outcome;
        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((z, data.float64_masked(z, &row_mask).map_err(EstimationError::from)?));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(EstimationError::from)?;
        let schema = data.schema();
        let treatment_name = schema.get(treatment).map(|v| v.name.as_ref()).unwrap_or("treatment");
        let coef_names = coefficient_names_from_design(&design, treatment_name, |id| {
            schema.get(id).ok().map(|v| Arc::clone(&v.name))
        });
        Ok(PreparedBayesianProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            active,
            control,
            overlap: self.overlap,
            coef_names: Some(coef_names),
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        })
    }

    /// Prepare a Gaussian conditional effect with declared modifiers and one
    /// treatment interaction per modifier. Interactions are centered at their
    /// observed means, so the treatment coefficient is the empirical-population
    /// average contrast for every posterior draw.
    pub fn prepare_conditional(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &antecedent_core::ConditionalEffectQuery,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        query.validate()?;
        let q = &query.inner;
        let mut seen = std::collections::HashSet::new();
        if q.effect_modifiers.iter().any(|modifier| {
            *modifier == q.treatment || *modifier == q.outcome || !seen.insert(*modifier)
        }) {
            return Err(EstimationError::unsupported(
                "Bayesian conditional effects require distinct modifiers separate from treatment and outcome",
            ));
        }
        let mut base_query = q.clone();
        base_query.effect_modifiers = Arc::from([]);
        let mut prep = self.prepare(data, estimand, &base_query)?;
        let mut ids = vec![q.treatment, q.outcome];
        ids.extend_from_slice(&q.effect_modifiers);
        ids.extend_from_slice(&estimand.adjustment_set);
        let mask = data.complete_case_mask(&ids)?;
        let t = data.float64_masked(q.treatment, &mask)?;
        let y = data.float64_masked(q.outcome, &mask)?;
        if t.len() < 8 {
            return Err(EstimationError::data_msg("too few conditional complete rows"));
        }
        let mut modifier_covariates = Vec::with_capacity(q.effect_modifiers.len());
        let mut interaction_covariates = Vec::with_capacity(q.effect_modifiers.len());
        for &modifier in q.effect_modifiers.iter() {
            let w = data.float64_masked(modifier, &mask)?;
            let mean = w.iter().sum::<f64>() / w.len() as f64;
            let interaction: Vec<_> = t.iter().zip(&w).map(|(t, w)| t * (w - mean)).collect();
            modifier_covariates.push((modifier, w));
            interaction_covariates.push((modifier, interaction));
        }
        let mut covs = modifier_covariates;
        covs.extend(interaction_covariates);
        for &id in estimand.adjustment_set.iter().filter(|&&id| !q.effect_modifiers.contains(&id)) {
            covs.push((id, data.float64_masked(id, &mask)?));
        }
        let refs: Vec<_> = covs.iter().map(|(id, values)| (*id, values.as_slice())).collect();
        let rows: Vec<_> =
            mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
        prep.design = CompiledDesign::linear_adjustment(&t, &refs, &y, &rows)?;
        let mut names: Vec<Arc<str>> =
            vec![Arc::from("intercept"), Arc::from("treatment_at_mean_modifier")];
        for &modifier in q.effect_modifiers.iter() {
            names.push(Arc::from(format!("modifier_{}", modifier.raw())));
        }
        for &modifier in q.effect_modifiers.iter() {
            names.push(Arc::from(format!("treatment_centered_modifier_{}", modifier.raw())));
        }
        names.extend(
            covs.iter()
                .skip(q.effect_modifiers.len() * 2)
                .map(|(id, _)| Arc::from(format!("adjustment_{}", id.raw()))),
        );
        // Store the ordered modifier identities in the coefficient names. The draw
        // adjustment helper uses these to preserve shared bootstrap weights.
        prep.coef_names = Some(Arc::from(names));
        Ok(prep)
    }

    /// Adapt a frequentist prepared design (e.g. lag-aligned temporal) for Bayesian fit.
    ///
    /// Used by the temporal pulse/sustained path: prepare via
    /// [`crate::TemporalLinearAdjustment`], then fit with this estimator.
    #[must_use]
    pub fn from_prepared_estimation(prep: &PreparedEstimationProblem) -> PreparedBayesianProblem {
        PreparedBayesianProblem {
            design: prep.design.clone(),
            method: Arc::clone(&prep.method),
            adjustment_set: Arc::clone(&prep.adjustment_set),
            active: prep.active,
            control: prep.control,
            overlap: prep.overlap,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        }
    }

    /// Adapt a lag-aligned temporal Pulse / single-step Sustained design for Bayesian fit.
    ///
    /// Unlike [`Self::from_prepared_estimation`], the problem carries lag-aware durable
    /// coefficient names (see [`crate::temporal_adjustment::temporal_coefficient_names`])
    /// so prior transfer binds by lag, and it declares
    /// [`SerialDependence::LongRunTempering`] on the treatment score.
    ///
    /// # Errors
    ///
    /// `coef_names` does not match the design width.
    pub fn from_prepared_temporal(
        prep: &PreparedEstimationProblem,
        coef_names: Arc<[Arc<str>]>,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        if coef_names.len() != prep.design.ncols {
            return Err(EstimationError::stats_msg(format!(
                "temporal coefficient names ({}) do not match design columns ({})",
                coef_names.len(),
                prep.design.ncols
            )));
        }
        let mut problem = Self::from_prepared_estimation(prep);
        problem.coef_names = Some(coef_names);
        problem.serial_dependence = SerialDependence::LongRunTempering(DependenceScope::Treatment);
        Ok(problem)
    }

    /// Fit mechanism + evaluate ATE g-computation posterior.
    ///
    /// `identification` is recorded as-is; informative priors never change it.
    ///
    /// # Errors
    ///
    /// Backend / evaluation failures.
    pub fn fit(
        &self,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, EstimationError> {
        let sequential = self.prior.is_some();
        if let Some(p) = &self.prior {
            if let Some(coef) = p.gaussian_coefficients() {
                if coef.len() != problem.design.ncols {
                    return Err(EstimationError::stats_msg(format!(
                        "sequential prior coefficient dimension {} != design ncols {}",
                        coef.len(),
                        problem.design.ncols
                    )));
                }
            } else {
                return Err(EstimationError::stats_msg(
                    "sequential prior missing GaussianCoefficients entry",
                ));
            }
        }
        let prior = self.prior_in_force(problem.design.ncols);
        let mut assumptions = AssumptionSet::new();
        let source = if sequential {
            AssumptionSource::Artifact
        } else {
            AssumptionSource::AlgorithmDefault { algorithm: Arc::from("bayesian_gcomp") }
        };
        for spec in &prior.specs {
            let mut pa = spec.as_assumption();
            if sequential {
                pa.description = Arc::from(format!(
                    "{} (sequential prior from posterior artifact; diagonal V0 only — \
                     off-diagonal posterior covariance dropped, so transferred priors can be \
                     tighter than the source marginal on linear combinations; absolute \
                     coefficient SD² converted to conjugate scale V0 via source residual variance)",
                    pa.description
                ));
            }
            assumptions.push(AssumptionRecord {
                assumption: Assumption::PriorRestriction(pa),
                source: source.clone(),
                scope: AssumptionScope::Estimation,
                status: AssumptionStatus::Untestable,
            });
        }
        for pa in &prior.restrictions {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::PriorRestriction(pa.clone()),
                source: AssumptionSource::Artifact,
                scope: AssumptionScope::Estimation,
                status: AssumptionStatus::Untestable,
            });
        }

        let likelihood = self.effective_likelihood();
        // HMC publication (Ř≤1.01, ESS≥100) needs a longer schedule than the
        // Laplace/conjugate default of 1000 draws; floor so under-specified
        // callers still clear the gate rather than refuse with near-miss Ř.
        let requested_draws = require_bayesian_n_draws(self.n_draws)?;
        let max_draws = match self.backend {
            BayesianBackendKind::Hmc => requested_draws.max(HMC_MIN_DRAWS),
            _ => requested_draws,
        };
        let mut extra_notes: Vec<Arc<str>> = Vec::new();
        if max_draws > requested_draws {
            extra_notes.push(Arc::from(format!(
                "{HMC_DRAW_FLOOR_NOTE_PREFIX} requested={requested_draws} used={max_draws}"
            )));
        }
        // Time-ordered rows: temper the likelihood by the long-run-variance ratio
        // of the targeted slope scores (generalized posterior; see serial_dependence).
        let tempering = match &problem.serial_dependence {
            SerialDependence::Iid => None,
            SerialDependence::LongRunTempering(scope) => {
                if problem.unit_ids.is_some() {
                    return Err(EstimationError::unsupported(
                        "long-run tempering applies to one time-ordered series, not stacked \
                         panel units",
                    ));
                }
                Some(long_run_tempering_factor(&problem.design, scope)?)
            }
        };
        let tempering_weights: Option<Vec<f64>> = tempering
            .filter(|factor| factor.kappa > 1.0)
            .map(|factor| vec![factor.kappa.recip(); problem.design.nrows]);
        if let Some(factor) = tempering {
            extra_notes.push(factor.note());
            assumptions.push(AssumptionRecord {
                assumption: Assumption::ParametricRestriction(
                    antecedent_core::ParametricAssumption {
                        id: Arc::from(DEPENDENCE_ASSUMPTION_ID),
                        description: factor.description(),
                    },
                ),
                source: AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from(match factor.scope {
                        "treatment" => "bayesian_temporal_gcomp",
                        "response_levels" => "response.temporal.bayesian",
                        "mediation_paths" => "temporal.mediation.bayesian",
                        _ => "temporal.sequential.gcomp",
                    }),
                },
                scope: AssumptionScope::Estimation,
                status: AssumptionStatus::Declared,
            });
        }
        let adaptive = ctx.adaptive_draws;
        // Adaptive redraws append samples from the fitted Gaussian covariance. Unknown-variance
        // GaussianIdentity is routed by the Laplace backend to the exact conjugate NIG posterior,
        // whose marginal coefficient law is Student-t and has no Gaussian covariance artifact.
        // Materialize the full requested NIG draw count instead of mixing it with MVN redraws.
        let laplace_mvn_redraw_supported = match likelihood {
            BayesLikelihood::GaussianIdentity => matches!(
                GaussianVarianceModel::from_prior_set(&prior).map_err(prob_err)?,
                GaussianVarianceModel::Known { .. }
            ),
            _ => true,
        };
        let laplace_adaptive = adaptive.enabled
            && matches!(self.backend, BayesianBackendKind::Laplace)
            && laplace_mvn_redraw_supported
            && max_draws > adaptive.min_draws.max(2);
        let initial_draws =
            if laplace_adaptive { adaptive.min_draws.max(2).min(max_draws) } else { max_draws };
        let opts = BayesFitOptions {
            n_draws: initial_draws,
            seed: self.seed,
            ..BayesFitOptions::default()
        };
        if let Some(ids) = problem.unit_ids.as_deref() {
            if ids.len() != problem.design.nrows {
                return Err(EstimationError::data_msg(
                    "unit_ids length must match the design row count",
                ));
            }
        }
        // GLS whitening is a Gaussian residual transform. Applying it to Poisson /
        // binomial / other GLM outcomes silently corrupts the likelihood.
        let whitened = match (likelihood, problem.unit_ids.as_deref()) {
            (BayesLikelihood::GaussianIdentity, Some(ids)) => random_intercept_gls_whiten(
                &problem.design.matrix,
                &problem.design.outcome,
                problem.design.nrows,
                problem.design.ncols,
                ids,
            ),
            (_, Some(_)) => {
                extra_notes.push(Arc::from(RANDOM_INTERCEPT_WHITEN_SKIP_NOTE));
                None
            }
            (_, None) => None,
        };
        let (x_fit, y_fit) = match &whitened {
            Some((x, y)) => (x.as_slice(), y.as_slice()),
            None => (problem.design.matrix.as_ref(), problem.design.outcome.as_ref()),
        };
        let design_ref = BayesDesignRef {
            x_colmajor: x_fit,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: y_fit,
            weights: tempering_weights.as_deref(),
            offsets: None,
        };

        let fit = match self.backend {
            BayesianBackendKind::ConjugateGaussian => ConjugateGaussianBackend.fit(
                likelihood,
                design_ref,
                &prior,
                &opts,
                &mut workspace.laplace,
                ctx,
            ),
            BayesianBackendKind::Laplace => LaplaceGlmBackend.fit(
                likelihood,
                design_ref,
                &prior,
                &opts,
                &mut workspace.laplace,
                ctx,
            ),
            BayesianBackendKind::Hmc => {
                // Floor warmup at the oracle schedule that clears Ř≤1.01 /
                // ESS≥100 on Gaussian GLMs; scale with kept draws above that.
                let n_warmup = max_draws.max(1_500);
                HmcGlmBackend::new()
                    .with_options(HmcOptions {
                        n_chains: 4,
                        n_warmup,
                        leapfrog_steps: 16,
                        step_size: 0.04,
                        target_accept: 0.85,
                        ..HmcOptions::default()
                    })
                    .fit(likelihood, design_ref, &prior, &opts, &mut workspace.laplace, ctx)
            }
        }
        .map_err(prob_err)?;

        if !fit.diagnostics.allows_posterior() {
            return Err(EstimationError::stats_msg("Bayesian fit refused without diagnostics"));
        }
        let mut fit = fit;
        fit.diagnostics.notes.extend(extra_notes);

        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;

        let glm_family = likelihood_to_glm_family(likelihood);
        let evaluator = GCompAteEvaluator {
            family: glm_family,
            treatment_col: t_col,
            active: problem.active,
            control: problem.control,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            matrix: Arc::clone(&problem.design.matrix),
        };
        let compiled = evaluator.compile()?;

        let mut early_stopped = false;
        let mut n_draws = fit.draws.n_draws;
        // Capture residual-variance draws before stripping to coefficient-only schema
        // (g-comp evaluates β only; hydrate still needs σ² for absolute→V0 conversion).
        let residual_sigma2_col: Option<Vec<f64>> = fit
            .draws
            .schema
            .quantities
            .iter()
            .enumerate()
            .find(|(_, q)| matches!(q, PosteriorQuantityKind::ResidualVariance))
            .and_then(|(i, _)| fit.draws.column(i).ok().map(<[f64]>::to_vec));
        // Adaptive MVN sampling uses the β-block covariance only; drop residual-variance
        // columns so batch merges match `PosteriorSchema::coefficients`.
        let coef_draws = coefficient_only_draws(&fit.draws)?;

        if laplace_adaptive {
            let cov = fit.cov.as_ref().ok_or_else(|| {
                EstimationError::stats_msg("Laplace adaptive draws require posterior covariance")
            })?;
            let map = fit.map.clone();
            let batch = 32usize;
            let mut effect_acc: Vec<f64> = Vec::with_capacity(max_draws);
            let mut extra_blocks: Vec<PosteriorDraws> = Vec::new();

            // Evaluate initial block.
            {
                workspace.eval.prepare(n_draws, problem.design.ncols);
                let mut effect_out = EffectBatch::default();
                effect_out.prepare(n_draws);
                let batch_view = coef_draws.batch(0, n_draws).map_err(EstimationError::from)?;
                evaluator.evaluate_batch(
                    &compiled,
                    batch_view,
                    &mut effect_out,
                    &mut workspace.eval,
                    ctx,
                )?;
                effect_acc.extend_from_slice(&effect_out.values[..n_draws]);
            }

            loop {
                // Independent MVN draws: ESS is the draw count, a genuine Monte
                // Carlo error measure. The former quantile-width change rule was
                // not one and is no longer consulted.
                let ess = effect_acc.len() as f64;
                if effect_acc.len() >= adaptive.min_draws.max(2) && ess >= adaptive.ess_target {
                    early_stopped = n_draws < max_draws;
                    break;
                }
                if n_draws >= max_draws {
                    break;
                }
                let next = (n_draws + batch).min(max_draws);
                let add = next - n_draws;
                let extra = sample_gaussian_mvn(
                    &map,
                    cov,
                    add,
                    self.seed.wrapping_add(n_draws as u64),
                    &mut workspace.laplace,
                )
                .map_err(EstimationError::from)?;
                let extra_draws = PosteriorDraws::from_column_major(
                    PosteriorSchema::coefficients(problem.design.ncols),
                    add,
                    extra,
                )
                .map_err(EstimationError::from)?;
                workspace.eval.prepare(add, problem.design.ncols);
                let mut effect_out = EffectBatch::default();
                effect_out.prepare(add);
                let batch_view = extra_draws.batch(0, add).map_err(EstimationError::from)?;
                evaluator.evaluate_batch(
                    &compiled,
                    batch_view,
                    &mut effect_out,
                    &mut workspace.eval,
                    ctx,
                )?;
                effect_acc.extend_from_slice(&effect_out.values[..add]);
                // Accumulate blocks; one concatenation after the loop replaces
                // the former per-batch merge + clone (O(D²) copying in D draws).
                extra_blocks.push(extra_draws);
                n_draws = next;
            }

            // Rebuild combined posterior from accumulated effects + final coef draws.
            // Re-attach ResidualVariance (same length as initial block; adaptive MVN
            // path only runs for known-σ² Laplace, which has no residual column).
            let mechanism_draws = concat_coefficient_draws(&coef_draws, &extra_blocks)?;
            let mut quantities = mechanism_draws.schema.quantities.to_vec();
            let residual_idx = if residual_sigma2_col.as_ref().is_some_and(|c| c.len() == n_draws) {
                quantities.push(PosteriorQuantityKind::ResidualVariance);
                Some(quantities.len() - 1)
            } else {
                None
            };
            let effect_idx = quantities.len();
            quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
            let n_q = quantities.len();
            let mut values = vec![0.0; n_draws * n_q];
            for (qi, q) in mechanism_draws.schema.quantities.iter().enumerate() {
                let dest = quantities.iter().position(|qq| qq == q).ok_or_else(|| {
                    EstimationError::stats_msg(format!(
                        "posterior quantity missing from schema: {q:?}"
                    ))
                })?;
                let coef_col = mechanism_draws.column(qi).map_err(EstimationError::from)?;
                values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(coef_col);
            }
            if let (Some(idx), Some(col)) = (residual_idx, residual_sigma2_col.as_ref()) {
                values[idx * n_draws..(idx + 1) * n_draws].copy_from_slice(col);
            }
            values[effect_idx * n_draws..(effect_idx + 1) * n_draws]
                .copy_from_slice(&effect_acc[..n_draws]);
            if let Some(names) = problem.coef_names.as_ref() {
                apply_coefficient_names(&mut quantities, names);
            }
            add_modifier_mean_uncertainty(
                problem,
                glm_family,
                &quantities,
                &mut values,
                n_draws,
                effect_idx,
                self.seed,
            );
            let draws = PosteriorDraws::from_column_major(
                PosteriorSchema { quantities: Arc::from(quantities) },
                n_draws,
                values,
            )
            .map_err(EstimationError::from)?;
            let summaries = draws.summarize();
            return Ok(CausalPosterior {
                subsampled_out_mass: 0.0,
                unevaluable_mass: 0.0,
                draws,
                summaries,
                identification,
                prior_sensitivity: None,
                conflict_summary: None,
                diagnostics: fit.diagnostics,
                assumptions,
                unidentified_mass: 0.0,
                early_stopped,
                treatment_contrast: (self.glm_family() == GlmFamily::GaussianIdentity)
                    .then_some(problem.active - problem.control),
            });
        }

        let mechanism = BayesianGlmMechanism {
            coefficient_draws: coef_draws,
            map: fit.map,
            likelihood,
            diagnostics: fit.diagnostics.clone(),
            design: problem.design.clone(),
            treatment_col: t_col,
            active: problem.active,
            control: problem.control,
        };

        workspace.eval.prepare(n_draws, problem.design.ncols);
        let mut effect_out = EffectBatch::default();
        effect_out.prepare(n_draws);
        let batch = mechanism.coefficient_draws.batch(0, n_draws).map_err(EstimationError::from)?;
        evaluator.evaluate_batch(&compiled, batch, &mut effect_out, &mut workspace.eval, ctx)?;

        let mut quantities = mechanism.coefficient_draws.schema.quantities.to_vec();
        let residual_idx = if residual_sigma2_col.as_ref().is_some_and(|c| c.len() == n_draws) {
            quantities.push(PosteriorQuantityKind::ResidualVariance);
            Some(quantities.len() - 1)
        } else {
            None
        };
        let effect_idx = quantities.len();
        quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        let n_q = quantities.len();
        let mut values = vec![0.0; n_draws * n_q];
        for (qi, q) in mechanism.coefficient_draws.schema.quantities.iter().enumerate() {
            let dest = quantities.iter().position(|qq| qq == q).ok_or_else(|| {
                EstimationError::stats_msg(format!("posterior quantity missing from schema: {q:?}"))
            })?;
            let col = mechanism.coefficient_draws.column(qi).map_err(EstimationError::from)?;
            values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(col);
        }
        if let (Some(idx), Some(col)) = (residual_idx, residual_sigma2_col.as_ref()) {
            values[idx * n_draws..(idx + 1) * n_draws].copy_from_slice(col);
        }
        values[effect_idx * n_draws..(effect_idx + 1) * n_draws]
            .copy_from_slice(&effect_out.values[..n_draws]);

        if let Some(names) = problem.coef_names.as_ref() {
            apply_coefficient_names(&mut quantities, names);
        }
        add_modifier_mean_uncertainty(
            problem,
            glm_family,
            &quantities,
            &mut values,
            n_draws,
            effect_idx,
            self.seed,
        );

        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema { quantities: Arc::from(quantities) },
            n_draws,
            values,
        )
        .map_err(EstimationError::from)?;
        let summaries = draws.summarize();

        let _ = mechanism;
        Ok(CausalPosterior {
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            draws,
            summaries,
            identification,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: fit.diagnostics,
            assumptions,
            unidentified_mass: 0.0,
            early_stopped: false,
            treatment_contrast: (self.glm_family() == GlmFamily::GaussianIdentity)
                .then_some(problem.active - problem.control),
        })
    }
}

/// Bayesian g-computation on a lag-aligned temporal design.
///
/// Prepare with [`crate::TemporalLinearAdjustment::prepare`], convert via
/// [`BayesianGComputationAte::from_prepared_estimation`], then [`BayesianGComputationAte::fit`].
/// This type documents the temporal entry point; fitting delegates to [`BayesianGComputationAte`].
#[derive(Clone, Debug, Default)]
pub struct BayesianTemporalGcomp {
    /// Shared Bayesian estimator configuration.
    pub inner: BayesianGComputationAte,
}

impl BayesianTemporalGcomp {
    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn new() -> Self {
        Self { inner: BayesianGComputationAte::new() }
    }

    /// Conjugate Gaussian linear path.
    #[must_use]
    pub fn conjugate() -> Self {
        Self { inner: BayesianGComputationAte::conjugate() }
    }

    /// Set the shared Bayesian g-computation configuration.
    #[must_use]
    pub fn with_inner(mut self, inner: BayesianGComputationAte) -> Self {
        self.inner = inner;
        self
    }

    /// Convert a temporal prepared design for Bayesian fit.
    #[must_use]
    pub fn from_prepared_estimation(prep: &PreparedEstimationProblem) -> PreparedBayesianProblem {
        BayesianGComputationAte::from_prepared_estimation(prep)
    }

    /// Fit on a prepared Bayesian problem (typically from a temporal design).
    ///
    /// # Errors
    ///
    /// Backend / evaluation failures.
    pub fn fit(
        &self,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, EstimationError> {
        self.inner.fit(problem, identification, workspace, ctx)
    }
}

/// Durable coefficient names from a design + schema name resolver.
///
/// Convention: `intercept`, `coef_{treatment}`, `coef_{covariate}`.
#[must_use]
pub fn coefficient_names_from_design(
    design: &CompiledDesign,
    treatment_name: &str,
    covariate_name: impl Fn(VariableId) -> Option<Arc<str>>,
) -> Arc<[Arc<str>]> {
    let names: Vec<Arc<str>> = design
        .columns
        .iter()
        .map(|col| match col.role {
            DesignColumnRole::Intercept => Arc::from("intercept"),
            DesignColumnRole::Treatment => Arc::from(format!("coef_{treatment_name}")),
            DesignColumnRole::Covariate(id) => covariate_name(id).map_or_else(
                || Arc::from(format!("coef_var_{}", id.raw())),
                |n| Arc::from(format!("coef_{n}")),
            ),
        })
        .collect();
    Arc::from(names)
}

/// Recompute a conditional contrast on the outcome scale for each coefficient draw.
/// One Rubin Bayesian-bootstrap weight vector is shared across all observed
/// modifiers and rows in a draw. Both intervention design rows include their
/// treatment-by-modifier terms, including for nonlinear inverse links.
fn add_modifier_mean_uncertainty(
    problem: &PreparedBayesianProblem,
    family: GlmFamily,
    quantities: &[PosteriorQuantityKind],
    values: &mut [f64],
    n_draws: usize,
    effect_idx: usize,
    seed: u64,
) {
    let Some(names) = problem.coef_names.as_deref() else {
        return;
    };
    let modifier_count = names.iter().filter(|name| name.starts_with("modifier_")).count();
    if modifier_count == 0 || n_draws == 0 || problem.design.ncols < 2 + modifier_count * 2 {
        return;
    }
    let nrows = problem.design.nrows;
    if nrows == 0 {
        return;
    }
    let ncols = problem.design.ncols;
    let mut coefficient_columns = Vec::with_capacity(ncols);
    for coef_index in 0..ncols {
        let Some(column) = quantities.iter().position(|q| {
            matches!(q, PosteriorQuantityKind::Coefficient { index, .. } if *index == coef_index)
        }) else { return; };
        coefficient_columns.push(column);
    }
    let modifier_means: Vec<f64> = (0..modifier_count)
        .map(|modifier| {
            problem.design.matrix[(2 + modifier) * nrows..(3 + modifier) * nrows]
                .iter()
                .sum::<f64>()
                / nrows as f64
        })
        .collect();
    let mut rng = antecedent_core::CausalRng::from_seed(seed ^ 0x4D4F_4449_4649_4552);
    for d in 0..n_draws {
        let weights: Vec<f64> =
            (0..nrows).map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
        let total = weights.iter().sum::<f64>();
        let beta: Vec<f64> =
            coefficient_columns.iter().map(|&column| values[column * n_draws + d]).collect();
        let mut contrast = 0.0;
        for (row, weight) in weights.iter().enumerate() {
            let mut baseline = 0.0;
            for (column, &coefficient) in beta.iter().enumerate() {
                if column != 1 && !(2 + modifier_count..2 + 2 * modifier_count).contains(&column) {
                    baseline += problem.design.matrix[column * nrows + row] * coefficient;
                }
            }
            let mut treatment_slope = beta[1];
            for modifier in 0..modifier_count {
                let centered = problem.design.matrix[(2 + modifier) * nrows + row];
                treatment_slope +=
                    beta[2 + modifier_count + modifier] * (centered - modifier_means[modifier]);
            }
            let active = family.mean_from_eta(baseline + problem.active * treatment_slope);
            let control = family.mean_from_eta(baseline + problem.control * treatment_slope);
            contrast += weight * (active - control);
        }
        values[effect_idx * n_draws + d] = contrast / total;
    }
}

/// Apply durable names onto coefficient quantities (in place).
fn apply_coefficient_names(quantities: &mut [PosteriorQuantityKind], names: &[Arc<str>]) {
    for q in quantities {
        if let PosteriorQuantityKind::Coefficient { index, name } = q {
            if let Some(n) = names.get(*index) {
                *name = Some(Arc::clone(n));
            }
        }
    }
}

/// Compound-symmetry GLS whitening for a random intercept (coefficient fit only).
///
/// Callers must gate on [`BayesLikelihood::GaussianIdentity`]: this transform is a
/// Gaussian linear residual map and is not valid for non-Gaussian GLM outcomes.
///
/// Returns `None` when unit ids are unusable (length mismatch, <2 units, all
/// singleton units, or non-finite MOM variance components). The transform
/// `v*_ij = v_ij - (1 - 1/√μ_i) v̄_i` with `μ_i = 1 + n_i τ²/σ²` yields
/// residual variance σ² I, so the conjugate Gaussian residual model is unchanged.
/// G-computation must keep the original design matrix.
fn random_intercept_gls_whiten(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    unit_ids: &[u32],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if unit_ids.len() != n || y.len() < n || n < 4 || p == 0 || x.len() < n * p {
        return None;
    }
    let mut groups: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &id) in unit_ids.iter().enumerate() {
        groups.entry(id).or_default().push(i);
    }
    let g = groups.len();
    if g < 2 || groups.values().all(|rows| rows.len() < 2) {
        return None;
    }
    let mut ols_ws = LeastSquaresWorkspace::default();
    let ols = FaerBackend.least_squares(x, n, p, y, &mut ols_ws).ok()?;
    let mut unit_mean_resid = Vec::with_capacity(g);
    let mut within_ss = 0.0;
    let mut df_within = 0.0;
    for rows in groups.values() {
        let n_i = rows.len() as f64;
        let mean_e = rows.iter().map(|&i| ols.residuals[i]).sum::<f64>() / n_i;
        unit_mean_resid.push((rows.len(), mean_e));
        for &i in rows {
            let d = ols.residuals[i] - mean_e;
            within_ss += d * d;
        }
        df_within += n_i - 1.0;
    }
    if df_within < 1.0 {
        return None;
    }
    let sigma2 = within_ss / df_within;
    if !sigma2.is_finite() || sigma2 <= 1e-18 {
        return None;
    }
    let grand = unit_mean_resid.iter().map(|&(n_i, m)| n_i as f64 * m).sum::<f64>() / n as f64;
    let ssb =
        unit_mean_resid.iter().map(|&(n_i, m)| n_i as f64 * (m - grand) * (m - grand)).sum::<f64>();
    let msb = ssb / (g as f64 - 1.0);
    let sum_n2: f64 = unit_mean_resid.iter().map(|&(n_i, _)| (n_i * n_i) as f64).sum();
    let n0 = (n as f64 - sum_n2 / n as f64) / (g as f64 - 1.0);
    if !n0.is_finite() || n0 <= 0.0 {
        return None;
    }
    let tau2 = ((msb - sigma2) / n0).max(0.0);
    if tau2 <= 0.0 {
        return None;
    }
    let mut y_star = y.to_vec();
    let mut x_star = x.to_vec();
    for rows in groups.values() {
        let n_i = rows.len() as f64;
        let mu = 1.0 + n_i * tau2 / sigma2;
        let lambda = 1.0 - 1.0 / mu.sqrt();
        let ybar = rows.iter().map(|&i| y[i]).sum::<f64>() / n_i;
        for &i in rows {
            y_star[i] = y[i] - lambda * ybar;
        }
        for c in 0..p {
            let xbar = rows.iter().map(|&i| x[c * n + i]).sum::<f64>() / n_i;
            for &i in rows {
                x_star[c * n + i] = x[c * n + i] - lambda * xbar;
            }
        }
    }
    Some((x_star, y_star))
}

/// Prepared Bayesian g-comp problem.
#[derive(Clone, Debug)]
pub struct PreparedBayesianProblem {
    /// Design.
    pub design: CompiledDesign,
    /// Estimand method.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Active treatment.
    pub active: f64,
    /// Control treatment.
    pub control: f64,
    /// Overlap.
    pub overlap: OverlapPolicy,
    /// Optional durable coefficient names aligned to design columns.
    pub coef_names: Option<Arc<[Arc<str>]>>,
    /// Optional unit / cluster ids aligned to design rows (panel Bayesian GLS).
    pub unit_ids: Option<Vec<u32>>,
    /// Row-dependence model. [`SerialDependence::Iid`] for exchangeable rows; temporal
    /// Pulse / Sustained designs and every horizon of a temporal response use
    /// [`SerialDependence::LongRunTempering`].
    pub serial_dependence: SerialDependence,
}

impl PreparedBayesianProblem {
    /// Set the row-dependence model (see [`crate::serial_dependence`]).
    #[must_use]
    pub fn with_serial_dependence(mut self, dependence: SerialDependence) -> Self {
        self.serial_dependence = dependence;
        self
    }
}

/// Workspace for Bayesian g-comp.
#[derive(Clone, Debug, Default)]
pub struct BayesianGCompWorkspace {
    /// Laplace / conjugate workspace.
    pub laplace: LaplaceWorkspace,
    /// Posterior functional eval scratch.
    pub eval: PosteriorEvalWorkspace,
}

/// Trait for batched posterior functional evaluation.
pub trait PosteriorFunctionalEvaluator {
    /// Compiled plan type.
    type Compiled;

    /// Compile against a posterior schema.
    ///
    /// # Errors
    ///
    /// Incompatible schema.
    fn compile(&self) -> Result<Self::Compiled, EstimationError>;

    /// Evaluate a batch of coefficient draws into effects.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn evaluate_batch(
        &self,
        compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}

/// Compiled g-comp ATE evaluator (finite-difference mean contrast).
#[derive(Clone, Debug)]
pub struct GCompAteEvaluator {
    /// Mean family.
    pub family: GlmFamily,
    /// Treatment column.
    pub treatment_col: usize,
    /// Active level.
    pub active: f64,
    /// Control level.
    pub control: f64,
    /// Rows.
    pub nrows: usize,
    /// Cols.
    pub ncols: usize,
    /// Design matrix (column-major).
    pub matrix: Arc<[f64]>,
}

/// Empty compiled marker (evaluator is self-contained).
#[derive(Clone, Copy, Debug, Default)]
pub struct CompiledGCompAte;

impl PosteriorFunctionalEvaluator for GCompAteEvaluator {
    type Compiled = CompiledGCompAte;

    fn compile(&self) -> Result<Self::Compiled, EstimationError> {
        if self.treatment_col >= self.ncols {
            return Err(EstimationError::stats_msg("treatment column out of range"));
        }
        Ok(CompiledGCompAte)
    }

    fn evaluate_batch(
        &self,
        _compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError> {
        const PARALLEL_WORK_FLOOR: usize = 1 << 16;
        let n_draws = posterior.len;
        workspace.prepare(n_draws, self.ncols);
        output.prepare(n_draws);

        // Coefficient columns 0..ncols from the batch (ignore extra quantities).
        let mut coef_cols: Vec<&[f64]> = Vec::with_capacity(self.ncols);
        for c in 0..self.ncols {
            let col = posterior.column(c).map_err(EstimationError::from)?;
            coef_cols.push(col);
        }

        // Non-identity links average over every row per draw (O(draws × n × p)); the
        // draws are independent and draw-order results are kept, so the pass is spread
        // over the context's thread budget once it is large enough to repay the spawn.
        let contrast = |beta: &[f64]| {
            gcomp_mean_contrast(
                self.family,
                &self.matrix,
                self.nrows,
                self.ncols,
                self.treatment_col,
                beta,
                self.active,
                self.control,
            )
        };
        let nonlinear = !matches!(self.family, GlmFamily::GaussianIdentity);
        if nonlinear && n_draws * self.nrows * self.ncols >= PARALLEL_WORK_FLOOR {
            let values = ctx.map_indexed(n_draws, |d, _| {
                let beta: Vec<f64> = coef_cols.iter().map(|col| col[d]).collect();
                Ok::<f64, EstimationError>(contrast(&beta))
            })?;
            output.values[..n_draws].copy_from_slice(&values);
            return Ok(());
        }
        for d in 0..n_draws {
            for c in 0..self.ncols {
                workspace.row[c] = coef_cols[c][d];
            }
            output.values[d] = contrast(&workspace.row[..self.ncols]);
        }
        Ok(())
    }
}

fn gcomp_mean_contrast(
    family: GlmFamily,
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    beta: &[f64],
    active: f64,
    control: f64,
) -> f64 {
    // Identity link: only the treatment column differs, so the row average is
    // exactly (active − control) β_T. Nonlinear links still need a data pass;
    // they share one residualized η (covariates only) instead of two full Xβ.
    if matches!(family, GlmFamily::GaussianIdentity) {
        return (active - control) * beta[t_col];
    }
    let beta_t = beta[t_col];
    let mut sum = 0.0;
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            if c == t_col {
                continue;
            }
            eta += matrix[c * nrows + r] * beta[c];
        }
        sum += family.mean_from_eta(eta + beta_t * active)
            - family.mean_from_eta(eta + beta_t * control);
    }
    sum / nrows as f64
}

fn likelihood_to_glm_family(l: BayesLikelihood) -> GlmFamily {
    match l {
        BayesLikelihood::GaussianIdentity => GlmFamily::GaussianIdentity,
        BayesLikelihood::BernoulliLogit => GlmFamily::BinomialLogit,
        BayesLikelihood::BernoulliProbit => GlmFamily::BinomialProbit,
        BayesLikelihood::PoissonLog => GlmFamily::PoissonLog,
    }
}

fn prob_err(e: antecedent_prob::ProbError) -> EstimationError {
    EstimationError::from(e)
}

/// Concatenate two coefficient-only posterior draw tables (same schema).
/// Concatenate an initial draw block with follow-on blocks in one pass.
///
/// Column-major layout identical to pairwise-merging the blocks in order,
/// without the quadratic intermediate copies.
fn concat_coefficient_draws(
    first: &PosteriorDraws,
    rest: &[PosteriorDraws],
) -> Result<PosteriorDraws, EstimationError> {
    if rest.is_empty() {
        return Ok(first.clone());
    }
    for block in rest {
        if block.schema != first.schema {
            return Err(EstimationError::stats_msg("concat_coefficient_draws: schema mismatch"));
        }
    }
    let n_q = first.schema.quantities.len();
    let n = first.n_draws + rest.iter().map(|b| b.n_draws).sum::<usize>();
    let mut values = vec![0.0; n * n_q];
    for q in 0..n_q {
        let mut offset = q * n;
        let col = first.column(q).map_err(EstimationError::from)?;
        values[offset..offset + first.n_draws].copy_from_slice(col);
        offset += first.n_draws;
        for block in rest {
            let col = block.column(q).map_err(EstimationError::from)?;
            values[offset..offset + block.n_draws].copy_from_slice(col);
            offset += block.n_draws;
        }
    }
    PosteriorDraws::from_column_major(first.schema.clone(), n, values)
        .map_err(EstimationError::from)
}

/// Keep coefficient columns only (drop residual-variance / other non-β quantities).
fn coefficient_only_draws(draws: &PosteriorDraws) -> Result<PosteriorDraws, EstimationError> {
    let coef_idx: Vec<usize> = draws
        .schema
        .quantities
        .iter()
        .enumerate()
        .filter_map(|(i, q)| matches!(q, PosteriorQuantityKind::Coefficient { .. }).then_some(i))
        .collect();
    if coef_idx.is_empty() {
        return Err(EstimationError::stats_msg(
            "coefficient_only_draws: no coefficient quantities",
        ));
    }
    if coef_idx.len() == draws.schema.quantities.len() {
        return Ok(draws.clone());
    }
    let n = draws.n_draws;
    let n_q = coef_idx.len();
    let mut quantities = Vec::with_capacity(n_q);
    let mut values = vec![0.0; n * n_q];
    for (dest, &src) in coef_idx.iter().enumerate() {
        quantities.push(draws.schema.quantities[src].clone());
        let col = draws.column(src).map_err(EstimationError::from)?;
        values[dest * n..(dest + 1) * n].copy_from_slice(col);
    }
    PosteriorDraws::from_column_major(
        PosteriorSchema { quantities: Arc::from(quantities) },
        n,
        values,
    )
    .map_err(EstimationError::from)
}

/// Build a non-identified posterior artifact that still records priors (exit criterion #2).
///
/// Samples prior-predictive draws of the scalar effect from the explicit effect prior
/// `N(effect_mean, effect_sd²)` so Bayesian envelopes can surface uncertainty without
/// inventing identification; `prior` is recorded as assumptions. The effect prior is
/// passed explicitly because a design-shaped [`PriorSet`] has no single effect
/// coefficient (its first entry is the intercept) and its Gaussian variances are
/// σ²-relative, not an effect SD. Status remains [`IdentificationStatus::NotIdentified`].
///
/// # Errors
///
/// A non-finite `effect_mean` or a non-finite / non-positive `effect_sd`.
pub fn nonidentified_with_prior(
    prior: &PriorSet,
    effect_mean: f64,
    effect_sd: f64,
    diagnostics: InferenceDiagnostics,
    n_draws: usize,
    seed: u64,
) -> Result<CausalPosterior, EstimationError> {
    if !effect_mean.is_finite() || !effect_sd.is_finite() || effect_sd <= 0.0 {
        return Err(EstimationError::unsupported(
            "non-identified prior-predictive effect needs a finite mean and a positive finite sd",
        ));
    }
    let mut assumptions = AssumptionSet::new();
    for spec in &prior.specs {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(spec.as_assumption()),
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Untestable,
        });
    }
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
    };
    let (mean, scale) = (effect_mean, effect_sd);
    let n = n_draws.max(1);
    let mut values = vec![0.0; n];
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Bayesian, 0xBA7E_u64);
    for v in &mut values {
        *v = mean + scale * antecedent_kernels::standard_normal(&mut rng);
    }
    let draws = PosteriorDraws::from_column_major(schema, n, Arc::<[f64]>::from(values))
        .unwrap_or_else(|_| PosteriorDraws {
            schema: PosteriorSchema {
                quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
            },
            n_draws: 0,
            values: Arc::from([]),
        });
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        draws,
        summaries,
        identification: IdentificationStatus::NotIdentified,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 1.0,
        early_stopped: false,
        treatment_contrast: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_prob::InferenceDiagnostics;

    #[test]
    fn gaussian_gcomp_contrast_is_treatment_coef_times_level_gap() {
        let beta = [1.0_f64, 2.5, -0.5];
        let matrix = [
            1.0, 1.0, 1.0, // intercept
            0.0, 1.0, 0.0, // treatment (overwritten by do())
            0.2, 0.4, 0.6, // covariate
        ];
        let ate =
            gcomp_mean_contrast(GlmFamily::GaussianIdentity, &matrix, 3, 3, 1, &beta, 1.0, 0.0);
        assert!((ate - 2.5).abs() < 1e-15);
    }

    fn linear_scm_table(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
        linear_scm_table_with_noise(n, 0.0, 0)
    }

    fn linear_scm_table_shifted(
        n: usize,
        noise: f64,
    ) -> (TabularData, VariableId, VariableId, VariableId) {
        linear_scm_table_with_noise(n, noise, 17)
    }

    fn linear_scm_table_with_noise(
        n: usize,
        noise: f64,
        seed_off: usize,
    ) -> (TabularData, VariableId, VariableId, VariableId) {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let z = VariableId::from_raw(0);
        let t = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let mut zv = vec![0.0; n];
        let mut tv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            zv[i] = (i as f64) * 0.1;
            tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            let e = if noise == 0.0 { 0.0 } else { (((i + seed_off) % 7) as f64 - 3.0) * noise };
            yv[i] = 2.0 * tv[i] + 0.5 * zv[i] + e;
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), t, y, z)
    }

    #[test]
    fn gaussian_conditional_design_supports_multiple_declared_modifiers() {
        let n = 12;
        let mut schema_builder = CausalSchemaBuilder::new();
        for (name, role) in [
            ("W1", RoleHint::Context),
            ("W2", RoleHint::Context),
            ("T", RoleHint::TreatmentCandidate),
            ("Y", RoleHint::OutcomeCandidate),
        ] {
            schema_builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(role),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = schema_builder.build().unwrap();
        let w1 = VariableId::from_raw(0);
        let w2 = VariableId::from_raw(1);
        let t = VariableId::from_raw(2);
        let y = VariableId::from_raw(3);
        let first: Vec<_> = (0..n).map(|i| i as f64).collect();
        let second: Vec<_> = (0..n).map(|i| (i as f64) * 2.0 - 3.0).collect();
        let treatment: Vec<_> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
        let outcome: Vec<_> = (0..n)
            .map(|i| {
                1.0 + treatment[i] * (2.0 + 0.5 * (first[i] - 5.5) - 0.25 * (second[i] - 8.0))
                    + 0.2 * first[i]
                    + 0.1 * second[i]
            })
            .collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(w1, Arc::from(first.clone()), validity.clone()).unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(w2, Arc::from(second.clone()), validity.clone()).unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(t, Arc::from(treatment.clone()), validity.clone()).unwrap(),
            ),
            OwnedColumn::Float64(Float64Column::new(y, Arc::from(outcome), validity).unwrap()),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![w1, w2]),
            ExprId::from_raw(0),
        );
        let mut query = AverageEffectQuery::binary_ate(t, y);
        query.effect_modifiers = Arc::from(vec![w1, w2]);
        let query = antecedent_core::ConditionalEffectQuery::try_new(query).unwrap();
        let prep =
            BayesianGComputationAte::new().prepare_conditional(&data, &estimand, &query).unwrap();

        assert_eq!(prep.design.ncols, 6);
        let nrows = prep.design.nrows;
        let at = |column: usize, row: usize| prep.design.matrix[column * nrows + row];
        assert_eq!(at(1, 0), treatment[0]);
        assert!((at(2, 0) - first[0]).abs() < 1e-12);
        assert!((at(3, 0) - second[0]).abs() < 1e-12);
        assert!((at(4, 0) - treatment[0] * (first[0] - 5.5)).abs() < 1e-12);
        assert!((at(5, 0) - treatment[0] * (second[0] - 8.0)).abs() < 1e-12);
        let names = prep.coef_names.unwrap();
        assert_eq!(names[4].as_ref(), "treatment_centered_modifier_0");
        assert_eq!(names[5].as_ref(), "treatment_centered_modifier_1");

        let mut duplicate = query.inner.clone();
        duplicate.effect_modifiers = Arc::from(vec![w1, w1]);
        let duplicate = antecedent_core::ConditionalEffectQuery::try_new(duplicate).unwrap();
        assert!(
            BayesianGComputationAte::new()
                .prepare_conditional(&data, &estimand, &duplicate)
                .is_err()
        );
    }

    #[test]
    fn conditional_logit_contrast_uses_both_intervention_rows_on_outcome_scale() {
        let treatment = [0.0, 1.0, 0.0, 1.0];
        let modifier = [-2.0, -1.0, 1.0, 2.0];
        let interaction: Vec<f64> = treatment.iter().zip(modifier).map(|(t, w)| t * w).collect();
        let outcome = [0.0; 4];
        let id = VariableId::from_raw(2);
        let design = CompiledDesign::linear_adjustment(
            &treatment,
            &[(id, &modifier), (id, &interaction)],
            &outcome,
            &[],
        )
        .unwrap();
        let problem = PreparedBayesianProblem {
            design,
            method: Arc::from("backdoor.adjustment"),
            adjustment_set: Arc::from([]),
            active: 1.0,
            control: 0.0,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: Some(Arc::from([
                Arc::from("intercept"),
                Arc::from("treatment_at_mean_modifier"),
                Arc::from("modifier_2"),
                Arc::from("treatment_centered_modifier_2"),
            ])),
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        };
        let quantities: Vec<_> = (0..4)
            .map(|index| PosteriorQuantityKind::Coefficient { index, name: None })
            .chain([PosteriorQuantityKind::Effect { name: Arc::from("ate") }])
            .collect();
        let mut values = vec![0.3, 0.7, 0.2, 0.8, 0.0];
        add_modifier_mean_uncertainty(
            &problem,
            GlmFamily::BinomialLogit,
            &quantities,
            &mut values,
            1,
            4,
            17,
        );
        let mut rng = antecedent_core::CausalRng::from_seed(17 ^ 0x4D4F_4449_4649_4552);
        let weights: Vec<f64> =
            (0..4).map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
        let expected = weights
            .iter()
            .zip(modifier)
            .map(|(weight, w)| {
                let baseline = 0.3 + 0.2 * w;
                weight
                    * (GlmFamily::BinomialLogit.mean_from_eta(baseline + 0.7 + 0.8 * w)
                        - GlmFamily::BinomialLogit.mean_from_eta(baseline))
            })
            .sum::<f64>()
            / weights.iter().sum::<f64>();
        assert!((values[4] - expected).abs() < 1e-12);
    }

    fn linear_scm_pooled(
        n_each: usize,
        noise: f64,
    ) -> (TabularData, VariableId, VariableId, VariableId) {
        // Two batches concatenated (batch A seed_off=0, batch B seed_off=17).
        let n = n_each * 2;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let z = VariableId::from_raw(0);
        let t = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let mut zv = vec![0.0; n];
        let mut tv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for batch in 0..2 {
            let seed_off = if batch == 0 { 0 } else { 17 };
            for i in 0..n_each {
                let r = batch * n_each + i;
                zv[r] = (i as f64) * 0.1;
                tv[r] = if i % 2 == 0 { 1.0 } else { 0.0 };
                let e = (((i + seed_off) % 7) as f64 - 3.0) * noise;
                yv[r] = 2.0 * tv[r] + 0.5 * zv[r] + e;
            }
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), t, y, z)
    }

    #[test]
    fn bayesian_and_frequentist_share_ate() {
        let n = 80;
        let (data, t, y, z) = linear_scm_table(n);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);

        let freq = crate::adjustment::LinearAdjustmentAte {
            bootstrap_replicates: 0,
            ..crate::adjustment::LinearAdjustmentAte::new()
        };
        let prep = freq.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let freq_est = freq
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
            .unwrap();

        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 400,
            seed: 5,
            prior_scale: 100.0,
            ..BayesianGComputationAte::new()
        };
        let bprep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut bws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut bws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let eq = post.effect_column().unwrap();
        let mean = post.summaries.mean[eq];
        assert!((freq_est.ate - 2.0).abs() < 1e-6, "frequentist ate={}", freq_est.ate);
        assert!((mean - freq_est.ate).abs() < 0.05, "bayes={mean} freq={}", freq_est.ate);
        assert_eq!(post.identification, IdentificationStatus::NonparametricallyIdentified);
        let coef_names: Vec<_> = post
            .draws
            .schema
            .quantities
            .iter()
            .filter_map(|q| match q {
                PosteriorQuantityKind::Coefficient { name, .. } => name.as_ref().map(AsRef::as_ref),
                _ => None,
            })
            .collect();
        assert!(coef_names.contains(&"intercept"), "{coef_names:?}");
        assert!(coef_names.iter().any(|n| n.starts_with("coef_")), "{coef_names:?}");
    }

    #[test]
    fn adaptive_laplace_unknown_variance_keeps_exact_nig_draws() {
        let (data, t, y, z) = linear_scm_table(80);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);
        let estimator =
            BayesianGComputationAte { n_draws: 96, seed: 17, ..BayesianGComputationAte::new() };
        let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
        let mut workspace = BayesianGCompWorkspace::default();
        let posterior = estimator
            .fit(
                &prepared,
                IdentificationStatus::NonparametricallyIdentified,
                &mut workspace,
                &ExecutionContext::production(17, 1),
            )
            .unwrap();

        assert_eq!(posterior.diagnostics.backend_id.as_ref(), "conjugate_gaussian");
        assert_eq!(posterior.draws.n_draws, 96, "NIG draws must not be replaced by MVN redraws");
        assert!(!posterior.early_stopped, "exact NIG sampling materializes the requested draws");
    }

    #[test]
    fn prior_does_not_create_identification() {
        let prior = PriorSet::weakly_informative(3);
        let post = nonidentified_with_prior(
            &prior,
            0.0,
            10.0,
            InferenceDiagnostics::analytic("none"),
            64,
            1,
        )
        .unwrap();
        assert_eq!(post.identification, IdentificationStatus::NotIdentified);
        assert!(!post.assumptions.is_empty());
        assert!((post.unidentified_mass - 1.0).abs() < 1e-12);
        assert!(post.draws.n_draws > 0, "prior-predictive draws required");
    }

    #[test]
    fn prior_predictive_effect_follows_the_explicit_effect_prior_not_the_intercept() {
        // The design-shaped prior's first coefficient (the intercept) is N(50, 0.1²);
        // the effect prior is N(-3, 0.5²). The draws must follow the effect prior.
        let mut prior = PriorSet::weakly_informative(2);
        if let Some(antecedent_prob::PriorSpec::GaussianCoefficients(g)) = prior.specs.first_mut() {
            *g = antecedent_prob::GaussianCoefficientPrior {
                mean: Arc::from(vec![50.0, 0.0]),
                variance: Arc::from(vec![0.01, 4.0]),
            };
        }
        let post = nonidentified_with_prior(
            &prior,
            -3.0,
            0.5,
            InferenceDiagnostics::analytic("none"),
            4000,
            9,
        )
        .unwrap();
        let draws = post.draws.column(0).unwrap();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let sd = (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64)
            .sqrt();
        // Monte Carlo error of the mean is 0.5/sqrt(4000) = 0.008; of the sd about 0.006.
        assert!((mean + 3.0).abs() < 0.04, "mean={mean}");
        assert!((sd - 0.5).abs() < 0.03, "sd={sd}");
        for (mean, sd) in [(f64::NAN, 1.0), (0.0, 0.0), (0.0, f64::INFINITY), (0.0, -1.0)] {
            assert!(
                nonidentified_with_prior(
                    &prior,
                    mean,
                    sd,
                    InferenceDiagnostics::analytic("none"),
                    8,
                    1
                )
                .is_err()
            );
        }
    }

    #[test]
    fn temporal_prepared_design_conjugate_recovers_pulse() {
        use antecedent_core::{
            CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, TemporalEffectQuery,
            TemporalPolicy, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
            TimeSeriesData, ValidityBitmap,
        };
        use antecedent_graph::{TemporalDag, ensure_lagged};
        use antecedent_identify::TemporalBackdoorIdentifier;

        use crate::temporal_adjustment::TemporalLinearAdjustment;

        let n = 300usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.07).sin();
            y[t] = 0.8 * x[t - 1];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();

        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        let prep = temporal
            .prepare(
                &data,
                estimand,
                &q,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 200,
                seed: 7,
                prior_scale: 100.0,
                ..BayesianGComputationAte::new()
            },
        };
        let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let eq = post.effect_column().unwrap();
        let mean = post.summaries.mean[eq];
        assert!((mean - 0.8).abs() < 0.05, "bayesian temporal pulse mean={mean}");
        assert!(post.probability_below(0.0).unwrap().is_finite());
    }

    #[test]
    fn hierarchical_unit_effects_widen_posterior_vs_stacked() {
        use crate::adjustment::LinearAdjustmentAte;
        use antecedent_core::AverageEffectQuery;

        let n_units = 40usize;
        let t_len = 8usize;
        let n = n_units * t_len;
        let mut t = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut unit_ids = Vec::with_capacity(n);
        for u in 0..n_units {
            let a = if u % 2 == 0 { 1.0 } else { 0.0 };
            let unit_eff = 1.6 * ((u as f64) * 0.31).sin();
            for k in 0..t_len {
                t.push(a);
                z.push(0.0);
                y.push(2.0 * a + unit_eff + 0.15 * ((k + 1) as f64).sin());
                unit_ids.push(u as u32);
            }
        }
        let (data, treatment, outcome, _z) = {
            // Reuse column layout T,Y,Z via a local table.
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "T",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "Y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "Z",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
                ),
            ];
            let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
            (
                TabularData::new(storage),
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                VariableId::from_raw(2),
            )
        };
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([] as [VariableId; 0]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(treatment, outcome);
        let freq = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = freq.prepare(&data, &estimand, &query).unwrap();
        let stacked = BayesianGComputationAte::from_prepared_estimation(&prep);
        let mut hierarchical = stacked.clone();
        hierarchical.unit_ids = Some(unit_ids);
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 400,
            seed: 11,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        };
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let post_s = bayes
            .fit(&stacked, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();
        let post_h = bayes
            .fit(&hierarchical, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();
        let eq = post_s.effect_column().unwrap();
        let sd_s = post_s.summaries.sd[eq];
        let sd_h = post_h.summaries.sd[eq];
        assert!(sd_s.is_finite() && sd_h.is_finite(), "sd stacked={sd_s} hierarchical={sd_h}");
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/panel_hierarchical_vs_stacked/expected.json"
        ))
        .unwrap();
        let min_ratio = pin["min_posterior_sd_ratio_hierarchical_over_stacked"].as_f64().unwrap();
        assert!(
            sd_h > sd_s * min_ratio,
            "hierarchical posterior must be wider than stacked iid (stacked={sd_s}, hierarchical={sd_h}, min_ratio={min_ratio})"
        );
    }

    #[test]
    fn unit_ids_length_mismatch_is_an_error() {
        use crate::adjustment::LinearAdjustmentAte;
        use antecedent_core::AverageEffectQuery;

        let n = 40usize;
        let t = [0.0, 1.0].repeat(n / 2);
        let y: Vec<f64> = t.iter().map(|&a| 2.0 * a).collect();
        let z = vec![0.0; n];
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([] as [VariableId; 0]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let freq = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = freq.prepare(&data, &estimand, &query).unwrap();
        let mut bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
        bprep.unit_ids = Some(vec![0, 1]);
        let bayes = BayesianGComputationAte::conjugate();
        let mut ws = BayesianGCompWorkspace::default();
        let err = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("unit_ids"),
            "length mismatch must fail closed, got {err}"
        );
    }

    /// estimate-bayes-transport-response-3 / R15: GLS whitening must not feed a
    /// Poisson likelihood. With unit_ids present, the fit must consume the raw
    /// count column (same posterior as stacked iid) and disclose the skip.
    #[test]
    fn poisson_random_intercept_does_not_consume_whitened_outcome() {
        use crate::adjustment::LinearAdjustmentAte;
        use antecedent_core::AverageEffectQuery;

        let n_units = 20usize;
        let t_len = 6usize;
        let n = n_units * t_len;
        let mut t = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut unit_ids = Vec::with_capacity(n);
        for u in 0..n_units {
            let a = if u % 2 == 0 { 1.0 } else { 0.0 };
            // Unit intercepts vary within each treatment arm so MOM τ²>0 after OLS.
            let unit_base = 4.0 + (u % 5) as f64;
            for k in 0..t_len {
                t.push(a);
                z.push(0.0);
                // Within-unit count variation so residual σ²>0 (required by the whitener).
                let rate: f64 = unit_base * if a > 0.5 { 2.0 } else { 1.0 } + (k % 3) as f64;
                y.push(rate.round().max(0.0));
                unit_ids.push(u as u32);
            }
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y.clone()), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([] as [VariableId; 0]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let freq = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = freq.prepare(&data, &estimand, &query).unwrap();

        // Whitening is applicable on this panel and would change the outcome column.
        let (x_star, y_star) = random_intercept_gls_whiten(
            &prep.design.matrix,
            &prep.design.outcome,
            prep.design.nrows,
            prep.design.ncols,
            &unit_ids,
        )
        .expect("panel with shared unit intercepts must admit GLS whitening");
        assert!(
            y_star.iter().zip(prep.design.outcome.iter()).any(|(a, b)| (a - b).abs() > 1e-9),
            "sanity: whitened outcome must differ from the raw counts"
        );
        assert_eq!(x_star.len(), prep.design.matrix.len());

        let stacked = BayesianGComputationAte::from_prepared_estimation(&prep);
        let mut hierarchical = stacked.clone();
        hierarchical.unit_ids = Some(unit_ids);

        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::PoissonLog,
            n_draws: 200,
            seed: 19,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        };
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let post_plain = bayes
            .fit(&stacked, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();
        let post_ri = bayes
            .fit(&hierarchical, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();

        let eq = post_plain.effect_column().unwrap();
        let mean_plain = post_plain.summaries.mean[eq];
        let mean_ri = post_ri.summaries.mean[eq];
        assert!(
            (mean_plain - mean_ri).abs() < 1e-12,
            "Poisson + unit_ids must not consume the whitened column \
             (stacked={mean_plain}, with_unit_ids={mean_ri})"
        );
        assert!(
            post_ri
                .diagnostics
                .notes
                .iter()
                .any(|n| n.as_ref() == RANDOM_INTERCEPT_WHITEN_SKIP_NOTE),
            "non-Gaussian random-intercept skip must be disclosed, got {:?}",
            post_ri.diagnostics.notes
        );
        assert!(
            !post_plain
                .diagnostics
                .notes
                .iter()
                .any(|n| n.as_ref() == RANDOM_INTERCEPT_WHITEN_SKIP_NOTE),
            "stacked fit must not emit the skip note"
        );
    }

    /// Seeded standard-normal draws for the calibration DGPs: Box-Muller over an LCG.
    ///
    /// Test-local on purpose — the calibration cases need a stream that is reproducible
    /// from a plain `u64` without pulling in the execution context's RNG plumbing.
    ///
    /// The seed is scrambled (SplitMix64 finalizer) first: seeding the LCG with the
    /// raw replicate index makes the states of seeds `s` and `s + 1` differ by a fixed
    /// `MUL^k` at every step, so consecutive calibration datasets form a lattice
    /// rather than independent draws.
    fn box_muller_lcg(seed: u64) -> impl FnMut() -> f64 {
        const LCG_MUL: u64 = 6_364_136_223_846_793_005;
        const TWO_POW_53: f64 = (1u64 << 53) as f64;
        let mut state = {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut next_unit = move || {
            state = state.wrapping_mul(LCG_MUL).wrapping_add(1);
            ((state >> 11) as f64) / TWO_POW_53
        };
        move || {
            let u = next_unit();
            let v = next_unit();
            (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    fn noisy_lag1_pulse_series(n: usize, seed: u64) -> antecedent_data::TimeSeriesData {
        use antecedent_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut gauss = box_muller_lcg(seed);
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.4 * gauss();
            y[t] = 0.8 * x[t - 1] + 0.35 * gauss();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    fn pulse_graph() -> antecedent_graph::TemporalDag {
        use antecedent_core::Lag;
        use antecedent_graph::{TemporalDag, ensure_lagged};
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        g
    }

    /// Default replicate count for the coverage gate (matches
    /// `crates/antecedent/tests/common/calibration.rs`, which this crate cannot import).
    const CALIBRATION_N_SIM: u32 = 400;

    /// Replicate count, honoring `ANTECEDENT_CALIBRATION_NSIM` for local smoke runs.
    fn calibration_n_sim() -> u32 {
        std::env::var("ANTECEDENT_CALIBRATION_NSIM")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(CALIBRATION_N_SIM)
    }

    /// Two-sided coverage tally: `level ± 3·MCSE`, `MCSE = sqrt(level(1-level)/n)`.
    /// At 400 replicates and 90% that is `[0.855, 0.945]`, so under- and
    /// over-coverage both fail.
    struct CoverageGate {
        name: &'static str,
        level: f64,
        covered: u32,
        scored: u32,
        length_sum: f64,
    }

    impl CoverageGate {
        fn new(name: &'static str, level: f64) -> Self {
            Self { name, level, covered: 0, scored: 0, length_sum: 0.0 }
        }

        /// Score the equal-tailed posterior interval of the effect column.
        fn record(&mut self, post: &CausalPosterior, truth: f64) {
            self.scored += 1;
            let eq = post.effect_column().unwrap();
            let col = post.draws.column(eq).unwrap();
            let mut v: Vec<f64> = col.iter().copied().filter(|x| x.is_finite()).collect();
            if v.len() < 2 {
                return;
            }
            v.sort_by(f64::total_cmp);
            let lo_p = (1.0 - self.level) / 2.0;
            let last = (v.len() - 1) as f64;
            // `q` is a probability and `last >= 0`, so the rounded product is a valid index.
            #[allow(
                clippy::cast_sign_loss,
                reason = "the rounded index is clamped to [0, last] so it is non-negative and in range"
            )]
            let at = |q: f64| v[(last * q).round().clamp(0.0, last) as usize];
            let (lo, hi) = (at(lo_p), at(1.0 - lo_p));
            self.length_sum += hi - lo;
            if truth >= lo && truth <= hi {
                self.covered += 1;
            }
        }

        fn assert(&self) {
            let n = f64::from(self.scored);
            let rate = f64::from(self.covered) / n;
            let mcse = (self.level * (1.0 - self.level) / n).sqrt();
            let lo = (self.level - 3.0 * mcse).max(0.0);
            let hi = (self.level + 3.0 * mcse).min(1.0);
            eprintln!(
                "calibration {}: nominal={:.2} coverage={rate:.3} mcse={mcse:.4} band=[{lo:.3}, {hi:.3}] \
                 mean_length={:.4} ({}/{} covered)",
                self.name,
                self.level,
                self.length_sum / n,
                self.covered,
                self.scored
            );
            assert!(
                rate >= lo && rate <= hi,
                "{} {:.0}% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({}/{})",
                self.name,
                self.level * 100.0,
                self.covered,
                self.scored
            );
        }
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_pulse_conjugate_nominal_90_coverage() {
        use crate::temporal_adjustment::TemporalLinearAdjustment;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_graph::ensure_lagged;
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = calibration_n_sim();
        let n = 160usize;
        let mut gate = CoverageGate::new("bayesian pulse (conjugate, direct estimator)", 0.9);
        let g = pulse_graph();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        for s in 0..n_sim {
            // Vary the sampler seed per replicate: a fixed seed reuses one set of
            // posterior draws, so its Monte-Carlo quantile error is shared by every
            // replicate instead of averaging out.
            let bayes = BayesianTemporalGcomp {
                inner: BayesianGComputationAte {
                    backend: BayesianBackendKind::ConjugateGaussian,
                    n_draws: 240,
                    seed: 21 + u64::from(s),
                    prior_scale: 8.0,
                    ..BayesianGComputationAte::new()
                },
            };
            let data = noisy_lag1_pulse_series(n, 9_000 + u64::from(s));
            let prep = temporal
                .prepare(
                    &data,
                    estimand,
                    &q,
                    &id_res.indexer,
                    None,
                    &ExecutionContext::for_tests(1).kernel_policy,
                )
                .unwrap();
            let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
            let mut ws = BayesianGCompWorkspace::default();
            let post = bayes
                .fit(
                    &bprep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            gate.record(&post, 0.8);
        }
        gate.assert();
        let _ = ensure_lagged;
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_sustained_single_step_conjugate_nominal_90_coverage() {
        use crate::temporal_adjustment::TemporalLinearAdjustment;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = calibration_n_sim();
        let n = 160usize;
        let mut gate = CoverageGate::new("bayesian single-step sustained (conjugate)", 0.9);
        let g = pulse_graph();
        let q = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            -1,
            1.0,
        )
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        for s in 0..n_sim {
            // Vary the sampler seed per replicate: a fixed seed reuses one set of
            // posterior draws, so its Monte-Carlo quantile error is shared by every
            // replicate instead of averaging out.
            let bayes = BayesianTemporalGcomp {
                inner: BayesianGComputationAte {
                    backend: BayesianBackendKind::ConjugateGaussian,
                    n_draws: 240,
                    seed: 22 + u64::from(s),
                    prior_scale: 8.0,
                    ..BayesianGComputationAte::new()
                },
            };
            let data = noisy_lag1_pulse_series(n, 11_000 + u64::from(s));
            let prep = temporal
                .prepare(
                    &data,
                    estimand,
                    &q,
                    &id_res.indexer,
                    None,
                    &ExecutionContext::for_tests(1).kernel_policy,
                )
                .unwrap();
            let bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
            let mut ws = BayesianGCompWorkspace::default();
            let post = bayes
                .fit(
                    &bprep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            gate.record(&post, 0.8);
        }
        gate.assert();
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_sustained_multi_step_conjugate_nominal_90_coverage() {
        use crate::temporal_sequential::estimate_sustained_window;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = calibration_n_sim();
        let n = 160usize;
        let mut gate = CoverageGate::new("bayesian multi-step sustained (conjugate)", 0.9);
        let g = pulse_graph();
        let q = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            -2,
            1.0,
        )
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(2));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        for s in 0..n_sim {
            // Vary the sampler seed per replicate: a fixed seed reuses one set of
            // posterior draws, so its Monte-Carlo quantile error is shared by every
            // replicate instead of averaging out.
            let bayes = BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 240,
                seed: 23 + u64::from(s),
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            };
            let data = noisy_lag1_pulse_series(n, 13_000 + u64::from(s));
            let (_, posterior) = estimate_sustained_window(
                &data,
                &g,
                &id_res.indexer,
                estimand,
                &q,
                IdentificationStatus::NonparametricallyIdentified,
                antecedent_core::AssumptionSet::default(),
                0,
                Some(&bayes),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
            gate.record(posterior.as_ref().unwrap(), 0.8);
        }
        gate.assert();
    }

    fn noisy_lag1_pulse_series_with_unit(
        n: usize,
        seed: u64,
        unit_eff: f64,
        treat: f64,
    ) -> antecedent_data::TimeSeriesData {
        let mut gauss = box_muller_lcg(seed);
        let x = vec![treat; n];
        let mut y = vec![0.0; n];
        // Treatment is constant within unit. Stacked iid treats T repeats as
        // independent looks at the same contrast; random-intercept GLS must widen.
        for t in 1..n {
            y[t] = 0.8 * x[t - 1] + unit_eff + 0.35 * gauss();
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        antecedent_data::TimeSeriesData::try_new(
            storage,
            antecedent_data::TimeIndex {
                regularity: antecedent_data::SamplingRegularity::Regular { interval_ns: 1 },
                length: n,
            },
        )
        .unwrap()
    }

    fn unit_effect_lag1_panel(
        n_units: usize,
        t_len: usize,
        seed: u64,
    ) -> antecedent_data::PanelData {
        use antecedent_data::{PanelData, PanelUnit};
        // Random intercepts are redrawn per replicate: the hierarchical model's
        // interval is for a random-effects population. A fixed intercept pattern
        // (formerly `1.4·sin(0.37u)`) freezes the between-unit imbalance across
        // replicates, so the interval, which correctly prices intercept variance,
        // covered 100% at 400 replicates. SD ≈ 0.99 matches that pattern's spread.
        let mut intercepts = box_muller_lcg(seed ^ 0x0E7F_0E7F);
        let units: Vec<PanelUnit> = (0..n_units)
            .map(|u| {
                let unit_eff = 0.99 * intercepts();
                PanelUnit {
                    unit_id: u as u32,
                    series: noisy_lag1_pulse_series_with_unit(
                        t_len,
                        seed.wrapping_add(u as u64 * 19),
                        unit_eff,
                        if u % 2 == 0 { 1.0 } else { 0.0 },
                    ),
                }
            })
            .collect();
        PanelData::try_new(Arc::from(units)).unwrap()
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_panel_hierarchical_nominal_90_coverage() {
        use crate::temporal_adjustment::TemporalLinearAdjustment;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = calibration_n_sim();
        let n_units = 24usize;
        let t_len = 40usize;
        let mut gate = CoverageGate::new("bayesian panel hierarchical", 0.9);
        let g = pulse_graph();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        for s in 0..n_sim {
            // Vary the sampler seed per replicate: a fixed seed reuses one set of
            // posterior draws, so its Monte-Carlo quantile error is shared by every
            // replicate instead of averaging out.
            let bayes = BayesianTemporalGcomp {
                inner: BayesianGComputationAte {
                    backend: BayesianBackendKind::ConjugateGaussian,
                    n_draws: 200,
                    seed: 23 + u64::from(s),
                    prior_scale: 8.0,
                    ..BayesianGComputationAte::new()
                },
            };
            // Unit `u` draws from `seed + 19·u`; a stride of 1000 keeps every
            // replicate's units disjoint (a unit stride of 1 reused series
            // across replicates, so they were not independent datasets).
            let panel = unit_effect_lag1_panel(n_units, t_len, 13_000 + 1_000 * u64::from(s));
            let (prep, cluster_ids, _) = temporal
                .prepare_panel(
                    &panel,
                    estimand,
                    &q,
                    &id_res.indexer,
                    None,
                    &ExecutionContext::for_tests(1).kernel_policy,
                )
                .unwrap();
            let mut bprep = BayesianTemporalGcomp::from_prepared_estimation(&prep);
            bprep.unit_ids = Some(cluster_ids);
            let mut ws = BayesianGCompWorkspace::default();
            let post = bayes
                .fit(
                    &bprep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            gate.record(&post, 0.8);
        }
        gate.assert();
    }

    #[test]
    fn hydrate_prior_from_posterior_and_refit() {
        let n = 60;
        let (data, t, y, z) = linear_scm_table(n);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 200,
            seed: 3,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let prior = hydrate_prior_from_posterior(&post, Some(prep.design.ncols)).unwrap();
        assert_eq!(prior.gaussian_coefficients().unwrap().len(), prep.design.ncols);
        assert!(hydrate_prior_from_posterior(&post, Some(prep.design.ncols + 1)).is_err());

        let sequential = BayesianGComputationAte { prior: Some(prior), ..bayes };
        let post2 = sequential
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert!(post2.assumptions.entries.iter().any(|a| {
            matches!(a.source, AssumptionSource::Artifact)
                && matches!(&a.assumption, Assumption::PriorRestriction(pa) if pa.description.contains("sequential")
                    && pa.description.contains("diagonal")
                    && pa.description.contains("tighter"))
        }));
        let eq = post2.effect_column().unwrap();
        assert!(post2.summaries.mean[eq].is_finite());
    }

    /// Earns `estimate-bayes-transport-response-1`: absolute SD² must not be stored as V0.
    #[test]
    fn sequential_hydrate_v0_not_orders_narrower_than_pooled() {
        let n = 80;
        let noise = 0.4;
        let (data_a, t, y, z) = linear_scm_table_with_noise(n, noise, 0);
        let (data_b, _, _, _) = linear_scm_table_shifted(n, noise);
        let (pool, _, _, _) = linear_scm_pooled(n, noise);

        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 400,
            seed: 9,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        };
        let mut ws = BayesianGCompWorkspace::default();

        let prep_a = bayes.prepare(&data_a, &estimand, &query).unwrap();
        let post_a = bayes
            .fit(
                &prep_a,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert!(
            post_a
                .draws
                .schema
                .quantities
                .iter()
                .any(|q| matches!(q, PosteriorQuantityKind::ResidualVariance)),
            "source posterior must retain residual_variance for V0 conversion"
        );
        let prior = hydrate_prior_from_posterior(&post_a, Some(prep_a.design.ncols)).unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        let sigma2 = post_a
            .draws
            .schema
            .quantities
            .iter()
            .position(|q| matches!(q, PosteriorQuantityKind::ResidualVariance))
            .map(|i| post_a.summaries.mean[i])
            .unwrap();
        for i in 0..coef.len() {
            let abs = post_a.summaries.sd[i] * post_a.summaries.sd[i];
            let expected_v0 = abs / sigma2;
            assert!(
                (coef.variance[i] - expected_v0).abs() / expected_v0.max(1e-18) < 1e-6,
                "coef {i}: V0 {} vs abs/σ² {expected_v0} (abs={abs}, σ²={sigma2})",
                coef.variance[i]
            );
        }

        let prep_b = bayes.prepare(&data_b, &estimand, &query).unwrap();
        let sequential = BayesianGComputationAte { prior: Some(prior), ..bayes.clone() };
        let post_seq = sequential
            .fit(
                &prep_b,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let prep_pool = bayes.prepare(&pool, &estimand, &query).unwrap();
        let post_pool = bayes
            .fit(
                &prep_pool,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let eq_s = post_seq.effect_column().unwrap();
        let eq_p = post_pool.effect_column().unwrap();
        let sd_seq = post_seq.summaries.sd[eq_s];
        let sd_pool = post_pool.summaries.sd[eq_p];
        assert!(sd_seq.is_finite() && sd_pool.is_finite() && sd_seq > 0.0 && sd_pool > 0.0);
        let ratio = sd_pool / sd_seq;
        assert!(
            ratio < 20.0,
            "sequential SD {sd_seq} is orders narrower than pooled {sd_pool} (ratio {ratio}) — \
             likely absolute SD² stored as V0"
        );
    }

    /// Earns `tests-quality-7`: strong prior vs flat prior vs prior-dominated.
    #[test]
    fn strong_prior_moves_posterior_away_from_flat_and_data_moves_away_from_prior() {
        let n = 100;
        let (data, t, y, z) = linear_scm_table(n);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);
        let mut ws = BayesianGCompWorkspace::default();
        let flat = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 300,
            seed: 5,
            prior_scale: 1e4,
            ..BayesianGComputationAte::new()
        };
        let strong = BayesianGComputationAte { prior_scale: 0.05, ..flat.clone() };
        let prep = flat.prepare(&data, &estimand, &query).unwrap();
        let post_flat = flat
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let post_strong = strong
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let eq = post_flat.effect_column().unwrap();
        let m_flat = post_flat.summaries.mean[eq];
        let m_strong = post_strong.summaries.mean[eq];
        // True ATE ≈ 2; flat recovers it; strong isotropic prior at 0 pulls toward 0.
        assert!((m_flat - m_strong).abs() > 0.15, "prior ignored? flat={m_flat} strong={m_strong}");
        assert!(
            m_strong.abs() < m_flat.abs(),
            "strong prior should shrink toward 0: flat={m_flat} strong={m_strong}"
        );
        // Data not ignored: informative sample must move off the prior mean (0).
        assert!(m_strong.abs() > 0.05, "data ignored? strong posterior {m_strong} ≈ prior mean 0");
    }

    #[test]
    fn hydrate_effect_functional_maps_treatment_coef() {
        let quantities = vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
            PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        ];
        let mean = vec![0.1, 0.5, 2.0];
        let sd = vec![1.0, 1.0, 0.4];
        let names: Vec<Arc<str>> =
            vec![Arc::from("intercept"), Arc::from("coef_t"), Arc::from("coef_z")];
        let baseline = PriorSet::weakly_informative(3);
        let prior = hydrate_prior(
            &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
            &quantities,
            &mean,
            &sd,
            &baseline,
            &names,
            Some(1),
            Some(0.5),
        )
        .unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        // Identity ATE = Δ β_T, so β ~ N(2.0/0.5, (0.4/0.5)²).
        assert!((coef.mean[1] - 4.0).abs() < 1e-12);
        assert!((coef.variance[1] - 0.64).abs() < 1e-12);
        // Unmapped dims keep baseline (isotropic scale 10 → var 100).
        assert!((coef.mean[0] - 0.0).abs() < 1e-12);
        assert!((coef.variance[0] - 100.0).abs() < 1e-12);
        assert!((coef.variance[2] - 100.0).abs() < 1e-12);
        assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == "external_effect_prior"));
    }

    #[test]
    fn hydrate_mapping_hard_errors() {
        let quantities = vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
            PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        ];
        let mean = vec![0.0, 1.0, 2.0];
        let sd = vec![1.0, 1.0, 0.5];
        let names2: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
        let baseline2 = PriorSet::weakly_informative(2);
        // Identical with wrong expected dim via target names of different length than source coefs.
        let names3: Vec<Arc<str>> =
            vec![Arc::from("intercept"), Arc::from("coef_t"), Arc::from("coef_w")];
        let baseline3 = PriorSet::weakly_informative(3);
        assert!(
            hydrate_prior(
                &HydrateMapping::IdenticalCoefficientSubspace,
                &quantities,
                &mean,
                &sd,
                &baseline3,
                &names3,
                None,
                None,
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::EffectFunctional { source_quantity: "missing".into() },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                Some(1),
                Some(1.0),
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                Some(1),
                None,
            )
            .is_err()
        );
        assert!(
            hydrate_prior(
                &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                Some(1),
                Some(0.0),
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::NamedParameters {
                    pairs: vec![("ate".into(), "no_such_coef".into())]
                },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                None,
                None,
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::NamedParameters {
                    pairs: vec![("no_src".into(), "coef_t".into())]
                },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                None,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn hydrate_effect_functional_lands_on_cate_coefficient() {
        let quantities = vec![PosteriorQuantityKind::Effect { name: Arc::from("ate") }];
        let mean = vec![1.2];
        let sd = vec![0.3];
        let names: Vec<Arc<str>> = vec![
            Arc::from("intercept"),
            Arc::from("treatment_at_mean_modifier"),
            Arc::from("modifier"),
            Arc::from("treatment_centered_modifier"),
        ];
        let baseline = PriorSet::weakly_informative(4);
        let prior = hydrate_prior(
            &HydrateMapping::EffectFunctional { source_quantity: "ate".into() },
            &quantities,
            &mean,
            &sd,
            &baseline,
            &names,
            None,
            Some(0.6),
        )
        .unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        assert!((coef.mean[1] - 2.0).abs() < 1e-12);
        assert!((coef.variance[1] - 0.25).abs() < 1e-12);
        assert!((coef.mean[3]).abs() < 1e-12);
    }

    #[test]
    fn hydrate_named_parameters_overwrites_target() {
        let quantities = vec![
            PosteriorQuantityKind::Coefficient { index: 0, name: Some(Arc::from("intercept")) },
            PosteriorQuantityKind::Coefficient { index: 1, name: Some(Arc::from("coef_t")) },
            PosteriorQuantityKind::Effect { name: Arc::from("ate") },
        ];
        let mean = vec![0.0, 0.0, 1.5];
        let sd = vec![1.0, 1.0, 0.2];
        let names: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
        let baseline = PriorSet::weakly_informative(2);
        let prior = hydrate_prior(
            &HydrateMapping::NamedParameters { pairs: vec![("ate".into(), "coef_t".into())] },
            &quantities,
            &mean,
            &sd,
            &baseline,
            &names,
            None,
            None,
        )
        .unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        assert!((coef.mean[1] - 1.5).abs() < 1e-12);
        assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == "external_named_prior"));
    }
}
