//! Bayesian mechanisms, g-computation, and posterior functional evaluation .
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
    EffectBatch, EffectPrior, GaussianCoefficientPrior, HmcGlmBackend, HmcOptions,
    InferenceBackend, InferenceDiagnostics, LaplaceGlmBackend, LaplaceWorkspace, PosteriorBatch,
    PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind, PosteriorSchema,
    PosteriorSummary, PriorSensitivitySummary, PriorSet, PriorSpec, sample_gaussian_mvn,
};
use antecedent_stats::{CompiledDesign, DesignColumnRole, GlmFamily};

use crate::adjustment::{PreparedEstimationProblem, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::require_explicit_override;

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
/// Mirrors [`antecedent_io::PriorMapping`] without depending on `antecedent-io` (avoids a
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
        if !matches!(
            estimand.method_kind().ok(),
            Some(
                antecedent_expr::EstimandMethod::BackdoorAdjustment
                    | antecedent_expr::EstimandMethod::BackdoorEfficient
            )
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "BayesianGComputationAte expects backdoor.adjustment/efficient",
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
        })
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
        let max_draws = self.n_draws.max(1);
        let adaptive = ctx.adaptive_draws;
        let laplace_adaptive = adaptive.enabled
            && matches!(self.backend, BayesianBackendKind::Laplace)
            && max_draws > adaptive.min_draws.max(2);
        let initial_draws =
            if laplace_adaptive { adaptive.min_draws.max(2).min(max_draws) } else { max_draws };
        let opts = BayesFitOptions {
            n_draws: initial_draws,
            seed: self.seed,
            ..BayesFitOptions::default()
        };
        let design_ref = BayesDesignRef {
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: &problem.design.outcome,
            weights: None,
            offsets: None,
        };

        let mut fit = match self.backend {
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
            BayesianBackendKind::Hmc => HmcGlmBackend::new()
                .with_options(HmcOptions {
                    n_chains: 2,
                    n_warmup: (self.n_draws / 2).max(50),
                    ..HmcOptions::default()
                })
                .fit(likelihood, design_ref, &prior, &opts, &mut workspace.laplace, ctx),
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
        let mut coef_draws = fit.draws;

        if laplace_adaptive {
            let cov = fit.cov.as_ref().ok_or_else(|| {
                EstimationError::stats_msg("Laplace adaptive draws require posterior covariance")
            })?;
            let map = fit.map.clone();
            let batch = 32usize;
            let mut effect_acc: Vec<f64> = Vec::with_capacity(max_draws);
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
                coef_draws = merge_coefficient_draws(&coef_draws, &extra_draws)?;
                n_draws = next;
                fit.draws = coef_draws.clone();
            }

            // Rebuild combined posterior from accumulated effects + final coef draws.
            let mechanism_draws = coef_draws;
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
            let mut sum = 0.0;
            for r in 0..self.nrows {
                let mu_a = predict_row(
                    self.family,
                    &self.matrix,
                    self.nrows,
                    self.ncols,
                    self.treatment_col,
                    beta,
                    r,
                    self.active,
                );
                let mu_c = predict_row(
                    self.family,
                    &self.matrix,
                    self.nrows,
                    self.ncols,
                    self.treatment_col,
                    beta,
                    r,
                    self.control,
                );
                sum += mu_a - mu_c;
            }
            output.values[d] = sum / self.nrows as f64;
        }
        Ok(())
    }
}

fn predict_row(
    family: GlmFamily,
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    beta: &[f64],
    row: usize,
    t_value: f64,
) -> f64 {
    let mut eta = 0.0;
    for c in 0..ncols {
        let x = if c == t_col { t_value } else { matrix[c * nrows + row] };
        eta += x * beta[c];
    }
    family.mean_from_eta(eta)
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
fn merge_coefficient_draws(
    a: &PosteriorDraws,
    b: &PosteriorDraws,
) -> Result<PosteriorDraws, EstimationError> {
    if a.schema != b.schema {
        return Err(EstimationError::stats_msg("merge_coefficient_draws: schema mismatch"));
    }
    let n_q = a.schema.quantities.len();
    let n = a.n_draws + b.n_draws;
    let mut values = vec![0.0; n * n_q];
    for q in 0..n_q {
        let col_a = a.column(q).map_err(EstimationError::from)?;
        let col_b = b.column(q).map_err(EstimationError::from)?;
        values[q * n..q * n + a.n_draws].copy_from_slice(col_a);
        values[q * n + a.n_draws..(q + 1) * n].copy_from_slice(col_b);
    }
    PosteriorDraws::from_column_major(a.schema.clone(), n, values).map_err(EstimationError::from)
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
