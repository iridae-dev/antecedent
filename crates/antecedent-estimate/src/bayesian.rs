//! Bayesian mechanisms, g-computation, and posterior functional evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use antecedent_core::IdentificationStatus;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, ExecutionContext, PriorAssumption, TargetPopulation,
    VariableId,
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
use crate::util::require_explicit_override;

/// Posterior mean and equal-tail interval of a linear response level.
/// `weights` is a design-row average after the intervention overlay.
#[allow(clippy::cast_sign_loss)] // Quantile indices are bounded by [0, n-1].
pub(crate) fn linear_response_summary(
    posterior: &CausalPosterior,
    weights: &[f64],
    level: f64,
) -> Result<(f64, f64, f64, f64), EstimationError> {
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
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "response posterior needs at least two finite draws",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    values.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let x = p * (values.len() - 1) as f64;
        let lo = x.floor() as usize;
        let hi = x.ceil() as usize;
        values[lo] + (values[hi] - values[lo]) * (x - lo as f64)
    };
    Ok((mean, quantile((1.0 - level) / 2.0), quantile((1.0 + level) / 2.0), sd))
}

/// Minimum kept draws for HMC so the MCMC publication gate (Ř≤1.01, ESS≥100)
/// is reachable on typical Gaussian GLMs.
const HMC_MIN_DRAWS: usize = 3_000;

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
    pub unidentified_mass: f64,
    /// Adaptive draw early-stop (Laplace / conjugate Gaussian redraw path).
    pub early_stopped: bool,
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

/// Build a Gaussian coefficient [`PriorSet`] from posterior quantity summaries.
///
/// Uses coefficient-column posterior means and SDs (index-aligned). Effect /
/// residual columns are ignored. When `expected_n_coef` is `Some`, it must match
/// the number of coefficient columns.
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
            return Err(EstimationError::stats_msg(format!(
                "posterior coefficient dimension {n_coef} != expected n_coef {expected}"
            )));
        }
    }
    for (i, (index, _)) in coef_cols.iter().enumerate() {
        if *index != i {
            return Err(EstimationError::stats_msg(format!(
                "posterior coefficient indices are not contiguous (expected {i}, got {index})"
            )));
        }
    }
    let mut means = Vec::with_capacity(n_coef);
    let mut variance = Vec::with_capacity(n_coef);
    for (_, col) in &coef_cols {
        let m = mean[*col];
        let s = sd[*col];
        if !m.is_finite() || !s.is_finite() {
            return Err(EstimationError::stats_msg(
                "posterior coefficient summary is non-finite; cannot hydrate prior",
            ));
        }
        means.push(m);
        variance.push((s * s).max(HYDRATE_VAR_FLOOR));
    }
    let coef = GaussianCoefficientPrior { mean: Arc::from(means), variance: Arc::from(variance) };
    coef.validate().map_err(EstimationError::from)?;
    Ok(PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(coef)],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    })
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
/// - [`HydrateMapping::EffectFunctional`]: maps source effect moments onto the
///   treatment coefficient (identity-link ATE bridge); other dims keep `baseline`.
/// - [`HydrateMapping::NamedParameters`]: maps named source moments onto named
///   target coefficients; unmapped dims keep `baseline`.
///
/// Records `external_effect_prior` / `external_named_prior` on
/// [`PriorSet::restrictions`].
///
/// # Errors
///
/// Dimension mismatch, missing effect column, unknown names, or invalid baseline.
pub fn hydrate_prior(
    mapping: &HydrateMapping,
    quantities: &[PosteriorQuantityKind],
    mean: &[f64],
    sd: &[f64],
    baseline: &PriorSet,
    target_coef_names: &[Arc<str>],
    treatment_col: Option<usize>,
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
            let t_col = treatment_col.ok_or_else(|| {
                EstimationError::stats_msg("hydrate_prior: EffectFunctional requires treatment_col")
            })?;
            if t_col >= n_target {
                return Err(EstimationError::stats_msg(format!(
                    "hydrate_prior: treatment_col {t_col} out of range for {n_target} coefs"
                )));
            }
            let (m, s) = quantity_moments(quantities, mean, sd, source_quantity.as_str())?;
            let effect = EffectPrior::new(m, s.max(HYDRATE_VAR_FLOOR.sqrt()))
                .map_err(EstimationError::from)?;
            let mut means: Vec<f64> = base_coef.mean.to_vec();
            let mut vars: Vec<f64> = base_coef.variance.to_vec();
            means[t_col] = effect.mean;
            vars[t_col] = (effect.sd * effect.sd).max(HYDRATE_VAR_FLOOR);
            let coef =
                GaussianCoefficientPrior { mean: Arc::from(means), variance: Arc::from(vars) };
            coef.validate().map_err(EstimationError::from)?;
            let mut prior = PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(coef)],
                contrast: baseline.contrast,
                categorical: baseline.categorical.clone(),
                restrictions: vec![PriorAssumption {
                    id: Arc::from("external_effect_prior"),
                    description: Arc::from(format!(
                        "external effect-functional prior from quantity `{source_quantity}` onto treatment coefficient"
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
            for (src, tgt) in pairs {
                let (m, s) = quantity_moments(quantities, mean, sd, src)?;
                let Some(&idx) = name_index.get(tgt.as_str()) else {
                    return Err(EstimationError::stats_msg(format!(
                        "hydrate_prior: unknown target coefficient name `{tgt}`"
                    )));
                };
                means[idx] = m;
                vars[idx] = (s * s).max(HYDRATE_VAR_FLOOR);
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
                    description: Arc::from(format!("external named-parameter prior ({pair_desc})")),
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
    /// MAP / posterior mode coefficients.
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
            return Err(EstimationError::unsupported(
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
        })
    }

    /// Prepare a Gaussian conditional effect with one modifier. Centering the
    /// interaction at the observed modifier mean makes the treatment coefficient
    /// exactly the population-average contrast for every posterior draw.
    pub fn prepare_conditional(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &antecedent_core::ConditionalEffectQuery,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        query.validate()?;
        if self.likelihood != BayesLikelihood::GaussianIdentity {
            return Err(EstimationError::unsupported(
                "Bayesian conditional effects require GaussianIdentity",
            ));
        }
        let q = &query.inner;
        if q.effect_modifiers.len() != 1 {
            return Err(EstimationError::unsupported(
                "Bayesian conditional effects require one modifier",
            ));
        }
        let modifier = q.effect_modifiers[0];
        let mut base_query = q.clone();
        base_query.effect_modifiers = Arc::from([]);
        let mut prep = self.prepare(data, estimand, &base_query)?;
        let mut ids = vec![q.treatment, q.outcome, modifier];
        ids.extend_from_slice(&estimand.adjustment_set);
        let mask = data.complete_case_mask(&ids)?;
        let t = data.float64_masked(q.treatment, &mask)?;
        let y = data.float64_masked(q.outcome, &mask)?;
        let w = data.float64_masked(modifier, &mask)?;
        if t.len() < 8 {
            return Err(EstimationError::data_msg("too few conditional complete rows"));
        }
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        let interaction: Vec<_> = t.iter().zip(&w).map(|(t, w)| t * (w - mean)).collect();
        let mut covs = vec![(modifier, w), (modifier, interaction)];
        for &id in estimand.adjustment_set.iter().filter(|&&id| id != modifier) {
            covs.push((id, data.float64_masked(id, &mask)?));
        }
        let refs: Vec<_> = covs.iter().map(|(id, values)| (*id, values.as_slice())).collect();
        let rows: Vec<_> =
            mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
        prep.design = CompiledDesign::linear_adjustment(&t, &refs, &y, &rows)?;
        let mut names: Vec<Arc<str>> = vec![
            Arc::from("intercept"),
            Arc::from("treatment_at_mean_modifier"),
            Arc::from("modifier"),
            Arc::from("treatment_centered_modifier"),
        ];
        names.extend(
            covs.iter().skip(2).map(|(id, _)| Arc::from(format!("adjustment_{}", id.raw()))),
        );
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
        }
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
        let prior = if let Some(p) = &self.prior {
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
            p.clone()
        } else {
            PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(
                    antecedent_prob::GaussianCoefficientPrior::isotropic(
                        problem.design.ncols,
                        self.prior_scale,
                    ),
                )],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            }
        };
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
                    "{} (sequential prior from posterior artifact)",
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

        let likelihood = match self.backend {
            BayesianBackendKind::ConjugateGaussian => BayesLikelihood::GaussianIdentity,
            BayesianBackendKind::Laplace | BayesianBackendKind::Hmc => self.likelihood,
        };
        // HMC publication (Ř≤1.01, ESS≥100) needs a longer schedule than the
        // Laplace/conjugate default of 1000 draws; floor so under-specified
        // callers still clear the gate rather than refuse with near-miss Ř.
        let max_draws = match self.backend {
            BayesianBackendKind::Hmc => self.n_draws.max(HMC_MIN_DRAWS),
            _ => self.n_draws.max(1),
        };
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
        let whitened = problem.unit_ids.as_deref().and_then(|ids| {
            random_intercept_gls_whiten(
                &problem.design.matrix,
                &problem.design.outcome,
                problem.design.nrows,
                problem.design.ncols,
                ids,
            )
        });
        let (x_fit, y_fit) = match &whitened {
            Some((x, y)) => (x.as_slice(), y.as_slice()),
            None => (problem.design.matrix.as_ref(), problem.design.outcome.as_ref()),
        };
        let design_ref = BayesDesignRef {
            x_colmajor: x_fit,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: y_fit,
            weights: None,
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
            let mut width_prev: Option<f64> = None;

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
                let width = quantile_width_95(&effect_acc);
                let ess = effect_acc.len() as f64; // independent MVN draws
                if effect_acc.len() >= adaptive.min_draws.max(2) {
                    let width_ok = width_prev.is_some_and(|prev| {
                        let rel = (width - prev).abs() / prev.abs().max(1e-12);
                        rel < adaptive.quantile_width_rel_epsilon
                    });
                    if width_ok || ess >= adaptive.ess_target {
                        early_stopped = n_draws < max_draws;
                        break;
                    }
                }
                width_prev = Some(width);
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
            let mechanism_draws = concat_coefficient_draws(&coef_draws, &extra_blocks)?;
            let mut quantities = mechanism_draws.schema.quantities.to_vec();
            quantities.retain(|q| !matches!(q, PosteriorQuantityKind::ResidualVariance));
            let effect_idx = quantities.len();
            quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
            let n_q = quantities.len();
            let mut values = vec![0.0; n_draws * n_q];
            for (qi, q) in mechanism_draws.schema.quantities.iter().enumerate() {
                if matches!(q, PosteriorQuantityKind::ResidualVariance) {
                    continue;
                }
                let dest = quantities.iter().position(|qq| qq == q).ok_or_else(|| {
                    EstimationError::stats_msg(format!(
                        "posterior quantity missing from schema: {q:?}"
                    ))
                })?;
                let coef_col = mechanism_draws.column(qi).map_err(EstimationError::from)?;
                values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(coef_col);
            }
            values[effect_idx * n_draws..(effect_idx + 1) * n_draws]
                .copy_from_slice(&effect_acc[..n_draws]);
            if let Some(names) = problem.coef_names.as_ref() {
                apply_coefficient_names(&mut quantities, names);
            }
            let draws = PosteriorDraws::from_column_major(
                PosteriorSchema { quantities: Arc::from(quantities) },
                n_draws,
                values,
            )
            .map_err(EstimationError::from)?;
            let summaries = draws.summarize();
            return Ok(CausalPosterior {
                draws,
                summaries,
                identification,
                prior_sensitivity: None,
                conflict_summary: None,
                diagnostics: fit.diagnostics,
                assumptions,
                unidentified_mass: 0.0,
                early_stopped,
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
        // Drop residual variance column from combined effect artifact if present — keep coefs + effect.
        quantities.retain(|q| !matches!(q, PosteriorQuantityKind::ResidualVariance));
        let effect_idx = quantities.len();
        quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        let n_q = quantities.len();
        let mut values = vec![0.0; n_draws * n_q];
        for (qi, q) in mechanism.coefficient_draws.schema.quantities.iter().enumerate() {
            if matches!(q, PosteriorQuantityKind::ResidualVariance) {
                continue;
            }
            let dest = quantities.iter().position(|qq| qq == q).ok_or_else(|| {
                EstimationError::stats_msg(format!("posterior quantity missing from schema: {q:?}"))
            })?;
            let col = mechanism.coefficient_draws.column(qi).map_err(EstimationError::from)?;
            values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(col);
        }
        values[effect_idx * n_draws..(effect_idx + 1) * n_draws]
            .copy_from_slice(&effect_out.values[..n_draws]);

        if let Some(names) = problem.coef_names.as_ref() {
            apply_coefficient_names(&mut quantities, names);
        }

        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema { quantities: Arc::from(quantities) },
            n_draws,
            values,
        )
        .map_err(EstimationError::from)?;
        let summaries = draws.summarize();

        let _ = mechanism;
        Ok(CausalPosterior {
            draws,
            summaries,
            identification,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: fit.diagnostics,
            assumptions,
            unidentified_mass: 0.0,
            early_stopped: false,
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
        _ctx: &ExecutionContext,
    ) -> Result<(), EstimationError> {
        let n_draws = posterior.len;
        workspace.prepare(n_draws, self.ncols);
        output.prepare(n_draws);

        // Coefficient columns 0..ncols from the batch (ignore extra quantities).
        let mut coef_cols: Vec<&[f64]> = Vec::with_capacity(self.ncols);
        for c in 0..self.ncols {
            let col = posterior.column(c).map_err(EstimationError::from)?;
            coef_cols.push(col);
        }

        for d in 0..n_draws {
            for c in 0..self.ncols {
                workspace.row[c] = coef_cols[c][d];
            }
            let beta = &workspace.row[..self.ncols];
            output.values[d] = gcomp_mean_contrast(
                self.family,
                &self.matrix,
                self.nrows,
                self.ncols,
                self.treatment_col,
                beta,
                self.active,
                self.control,
            );
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

/// 95% quantile width of a scalar draw vector.
fn quantile_width_95(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    // Reuse posterior summarization for consistent quantiles.
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("w") }]),
    };
    let Ok(draws) = PosteriorDraws::from_column_major(schema, values.len(), values.to_vec()) else {
        return f64::NAN;
    };
    let s = draws.summarize();
    s.q975[0] - s.q025[0]
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
/// Samples prior-predictive draws for a scalar effect mean (isotropic Gaussian / weakly
/// informative scale from `prior`) so Bayesian envelopes can surface uncertainty without
/// inventing identification. Status remains [`IdentificationStatus::NotIdentified`].
#[must_use]
pub fn nonidentified_with_prior(
    prior: &PriorSet,
    diagnostics: InferenceDiagnostics,
    n_draws: usize,
    seed: u64,
) -> CausalPosterior {
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
    let (mean, scale) = prior_predictive_effect_params(prior);
    let n = n_draws.max(1);
    let mut values = vec![0.0; n];
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xBA7E_u64);
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
    CausalPosterior {
        draws,
        summaries,
        identification: IdentificationStatus::NotIdentified,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 1.0,
        early_stopped: false,
    }
}

fn prior_predictive_effect_params(prior: &PriorSet) -> (f64, f64) {
    if let Some(g) = prior.gaussian_coefficients() {
        let mean = g.mean.first().copied().unwrap_or(0.0);
        let var = g.variance.first().copied().unwrap_or(100.0).max(1e-12);
        return (mean, var.sqrt());
    }
    (0.0, 10.0)
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
            yv[i] = 2.0 * tv[i] + 0.5 * zv[i];
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
        let post = nonidentified_with_prior(&prior, InferenceDiagnostics::analytic("none"), 64, 1);
        assert_eq!(post.identification, IdentificationStatus::NotIdentified);
        assert!(!post.assumptions.is_empty());
        assert!((post.unidentified_mass - 1.0).abs() < 1e-12);
        assert!(post.draws.n_draws > 0, "prior-predictive draws required");
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

    /// Seeded standard-normal draws for the calibration DGPs: Box-Muller over an LCG.
    ///
    /// Test-local on purpose — the calibration cases need a stream that is reproducible
    /// from a plain `u64` without pulling in the execution context's RNG plumbing.
    fn box_muller_lcg(seed: u64) -> impl FnMut() -> f64 {
        const LCG_MUL: u64 = 6_364_136_223_846_793_005;
        const TWO_POW_53: f64 = (1u64 << 53) as f64;
        let mut state = seed;
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

    fn interval_covers(post: &CausalPosterior, truth: f64, level: f64) -> bool {
        let eq = post.effect_column().unwrap();
        let col = post.draws.column(eq).unwrap();
        let mut v = col.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo_p = (1.0 - level) / 2.0;
        let hi_p = 1.0 - lo_p;
        // `q` is a probability and `last >= 0`, so the rounded product is a valid
        // index; the clamp makes that explicit for the boundary levels.
        #[allow(clippy::cast_sign_loss)]
        let at = |q: f64| {
            let last = (v.len() - 1) as f64;
            v[(last * q).round().clamp(0.0, last) as usize]
        };
        truth >= at(lo_p) && truth <= at(hi_p)
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_pulse_conjugate_nominal_90_coverage() {
        use crate::temporal_adjustment::TemporalLinearAdjustment;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_graph::ensure_lagged;
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = 80u32;
        let n = 160usize;
        let mut covered = 0u32;
        let g = pulse_graph();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 240,
                seed: 21,
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        for s in 0..n_sim {
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
            if interval_covers(&post, 0.8, 0.9) {
                covered += 1;
            }
        }
        let rate = f64::from(covered) / f64::from(n_sim);
        let se = (0.9 * 0.1 / f64::from(n_sim)).sqrt();
        let lo = (0.9 - 4.0 * se).max(0.70);
        let hi = (0.9 + 4.0 * se).min(1.0);
        assert!(
            rate >= lo && rate <= hi,
            "bayesian pulse 90% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({covered}/{n_sim})"
        );
        let _ = ensure_lagged;
    }

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn bayesian_sustained_single_step_conjugate_nominal_90_coverage() {
        use crate::temporal_adjustment::TemporalLinearAdjustment;
        use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
        use antecedent_identify::TemporalBackdoorIdentifier;

        let n_sim = 80u32;
        let n = 160usize;
        let mut covered = 0u32;
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
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 240,
                seed: 22,
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        for s in 0..n_sim {
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
            if interval_covers(&post, 0.8, 0.9) {
                covered += 1;
            }
        }
        let rate = f64::from(covered) / f64::from(n_sim);
        let se = (0.9 * 0.1 / f64::from(n_sim)).sqrt();
        let lo = (0.9 - 4.0 * se).max(0.70);
        let hi = (0.9 + 4.0 * se).min(1.0);
        assert!(
            rate >= lo && rate <= hi,
            "bayesian sustained 90% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({covered}/{n_sim})"
        );
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
        let units: Vec<PanelUnit> = (0..n_units)
            .map(|u| {
                let unit_eff = 1.4 * ((u as f64) * 0.37).sin();
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

        let n_sim = 60u32;
        let n_units = 24usize;
        let t_len = 40usize;
        let mut covered = 0u32;
        let g = pulse_graph();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let temporal = TemporalLinearAdjustment::new();
        let bayes = BayesianTemporalGcomp {
            inner: BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 200,
                seed: 23,
                prior_scale: 8.0,
                ..BayesianGComputationAte::new()
            },
        };
        for s in 0..n_sim {
            let panel = unit_effect_lag1_panel(n_units, t_len, 13_000 + u64::from(s));
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
            if interval_covers(&post, 0.8, 0.9) {
                covered += 1;
            }
        }
        let rate = f64::from(covered) / f64::from(n_sim);
        let se = (0.9 * 0.1 / f64::from(n_sim)).sqrt();
        let lo = (0.9 - 4.0 * se).max(0.70);
        let hi = (0.9 + 4.0 * se).min(1.0);
        assert!(
            rate >= lo && rate <= hi,
            "bayesian panel hierarchical 90% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({covered}/{n_sim})"
        );
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
                && matches!(&a.assumption, Assumption::PriorRestriction(pa) if pa.description.contains("sequential"))
        }));
        let eq = post2.effect_column().unwrap();
        assert!(post2.summaries.mean[eq].is_finite());
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
        )
        .unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        assert!((coef.mean[1] - 2.0).abs() < 1e-12);
        assert!((coef.variance[1] - 0.16).abs() < 1e-12);
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
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::NamedParameters {
                    pairs: vec![("ate".into(), "no_such_coef".into())],
                },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                None,
            )
            .is_err()
        );

        assert!(
            hydrate_prior(
                &HydrateMapping::NamedParameters {
                    pairs: vec![("no_src".into(), "coef_t".into())],
                },
                &quantities,
                &mean,
                &sd,
                &baseline2,
                &names2,
                None,
            )
            .is_err()
        );
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
        )
        .unwrap();
        let coef = prior.gaussian_coefficients().unwrap();
        assert!((coef.mean[1] - 1.5).abs() < 1e-12);
        assert!(prior.restrictions.iter().any(|r| r.id.as_ref() == "external_named_prior"));
    }
}
