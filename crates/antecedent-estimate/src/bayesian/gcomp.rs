//! Bayesian g-computation estimator, prepared problem, and fit phases.

use std::sync::Arc;

use antecedent_core::IdentificationStatus;
use antecedent_core::{
    AdaptiveDrawBudget, Assumption, AssumptionRecord, AssumptionScope, AssumptionSet,
    AssumptionSource, AssumptionStatus, AverageEffectQuery, ExecutionContext, TargetPopulation,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, ConjugateGaussianBackend,
    EffectBatch, GaussianVarianceModel, HmcGlmBackend, HmcOptions, InferenceBackend,
    LaplaceGlmBackend, LaplaceWorkspace, PosteriorDraws, PosteriorEvalWorkspace,
    PosteriorQuantityKind, PosteriorSchema, PriorSet, PriorSpec, ProbError, sample_gaussian_mvn,
};
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, DesignColumnRole, FaerBackend, GlmFamily,
    LeastSquaresWorkspace,
};

use crate::adjustment::{PreparedEstimationProblem, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DependenceScope, ResidualFitReuse, SerialDependence,
    long_run_tempering_factor,
};
use crate::util::require_explicit_override;

use super::evaluator::{
    CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator, coefficient_only_draws,
    concat_coefficient_draws, likelihood_to_glm_family, prob_err,
};
use super::{
    CausalPosterior, HMC_DRAW_FLOOR_NOTE_PREFIX, HMC_MIN_DRAWS,
    RANDOM_INTERCEPT_WHITEN_FALLBACK_NOTE, RANDOM_INTERCEPT_WHITEN_SKIP_NOTE,
    require_bayesian_n_draws,
};

/// Draw count below which a Laplace posterior's effect interval is the exchangeable-rank
/// (type-6) interval of its draws rather than the type-7 interval its records were measured with
/// at this count and above. Keyed separately (`rank_quantile_interval`).
pub const LAPLACE_RANK_INTERVAL_BELOW_DRAWS: usize = 1000;

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
    /// Likelihood (Laplace); conjugate forces `GaussianIdentity`.
    pub likelihood: BayesLikelihood,
    /// Draw count.
    pub n_draws: usize,
    /// RNG seed.
    pub seed: u64,
    /// Overlap policy (must be `ExplicitOverride`).
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
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("option_not_applicable"),
                "Bayesian g-comp does not support effect modifiers",
            ));
        }
        let reference_arm = match &query.target_population {
            TargetPopulation::AllObserved => None,
            TargetPopulation::Treated => Some(ReferenceArm::Active),
            TargetPopulation::Untreated => Some(ReferenceArm::Control),
            _ => {
                return Err(EstimationError::refused(
                    antecedent_core::reason_code!("population_not_estimable"),
                    "Bayesian g-comp supports TargetPopulation::AllObserved, and Treated or \
                     Untreated as a model-based effect under the Gaussian identity outcome model",
                ));
            }
        };
        if reference_arm.is_some()
            && self.effective_likelihood() != BayesLikelihood::GaussianIdentity
        {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("population_not_estimable"),
                "the Bayesian model-based effect on the treated or untreated centres treatment \
                 interactions on the arm, which makes the treatment coefficient that arm's \
                 contrast on the identity link only; use the Gaussian identity outcome model",
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
        let crate::prepare::AdmittedColumns { data, row_mask, receipt } =
            crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
        let design = crate::prepare::direct_adjustment_design(
            data,
            treatment,
            outcome,
            &estimand.adjustment_set,
            &[],
            &row_mask,
        )?;
        let schema = data.schema();
        let treatment_name = schema.get(treatment).map(|v| v.name.as_ref()).unwrap_or("treatment");
        let coef_names = coefficient_names_from_design(&design, treatment_name, |id| {
            schema.get(id).ok().map(|v| Arc::clone(&v.name))
        });
        let (design, coef_names) = match reference_arm {
            Some(arm) => {
                let level = match arm {
                    ReferenceArm::Active => active,
                    ReferenceArm::Control => control,
                };
                center_interactions_on_arm(&design, &coef_names, level)?
            }
            None => (design, coef_names),
        };
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
            preparation: Some(crate::prepare::preparation_summary(&receipt)),
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
        let crate::prepare::AdmittedColumns { data, row_mask: mask, receipt } =
            crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
        let t = data.float64_selected(q.treatment, &mask)?;
        let y = data.float64_selected(q.outcome, &mask)?;
        if t.len() < 8 {
            return Err(EstimationError::data_msg("too few conditional complete rows"));
        }
        let mut modifier_covariates: Vec<(VariableId, std::borrow::Cow<'_, [f64]>)> =
            Vec::with_capacity(q.effect_modifiers.len());
        let mut interaction_covariates: Vec<(VariableId, std::borrow::Cow<'_, [f64]>)> =
            Vec::with_capacity(q.effect_modifiers.len());
        for &modifier in q.effect_modifiers.iter() {
            let w = data.float64_selected(modifier, &mask)?;
            let mean = w.iter().sum::<f64>() / w.len() as f64;
            let interaction: Vec<_> = t.iter().zip(w.iter()).map(|(t, w)| t * (w - mean)).collect();
            modifier_covariates.push((modifier, w));
            interaction_covariates.push((modifier, std::borrow::Cow::Owned(interaction)));
        }
        let mut covs = modifier_covariates;
        covs.extend(interaction_covariates);
        for &id in estimand.adjustment_set.iter().filter(|&&id| !q.effect_modifiers.contains(&id)) {
            covs.push((id, data.float64_selected(id, &mask)?));
        }
        let refs: Vec<_> = covs.iter().map(|(id, values)| (*id, &values[..])).collect();
        let rows: Vec<_> =
            mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
        prep.design = CompiledDesign::linear_adjustment(&t, &refs, &y, &rows)?;
        prep.preparation = Some(crate::prepare::preparation_summary(&receipt));
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
            preparation: None,
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

    /// [`Self::from_prepared_temporal`] for a design built from the intact windows of a series
    /// with gaps: the likelihood is tempered by the segment-wise long-run variance, so no
    /// autocovariance crosses a gap (see [`SerialDependence::SegmentedTempering`]).
    ///
    /// # Errors
    ///
    /// As [`Self::from_prepared_temporal`].
    pub fn from_prepared_temporal_gapped(
        prep: &PreparedEstimationProblem,
        coef_names: Arc<[Arc<str>]>,
        segments: Arc<[u32]>,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        let mut problem = Self::from_prepared_temporal(prep, coef_names)?;
        problem.serial_dependence =
            SerialDependence::SegmentedTempering { scope: DependenceScope::Treatment, segments };
        Ok(problem)
    }

    /// Fit mechanism + evaluate ATE g-computation posterior.
    ///
    /// `identification` is recorded as-is; informative priors never change it.
    ///
    /// # Errors
    ///
    /// Backend / evaluation failures.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear fit: draw, optional interval supplement, then summaries"
    )]
    pub fn fit(
        &self,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, EstimationError> {
        self.check_sequential_prior(problem)?;
        let prior = self.prior_in_force(problem.design.ncols);
        let mut assumptions = self.prior_assumptions(&prior);

        let likelihood = self.effective_likelihood();
        let (max_draws, mut extra_notes) = self.draw_budget()?;
        let tempering_weights =
            temper_likelihood(problem, &mut assumptions, &mut extra_notes, ctx)?;
        let (laplace_adaptive, initial_draws) =
            self.adaptive_schedule(likelihood, &prior, ctx, max_draws)?;

        let whitened = whiten_design(problem, likelihood, &mut extra_notes)?;
        let (x_fit, y_fit) = match &whitened {
            Some((x, y)) => (x.as_slice(), y.as_slice()),
            None => (problem.design.matrix.as_ref(), problem.design.outcome.as_ref()),
        };
        let design_ref: BayesDesignRef<'_> = BayesDesignRef {
            x_colmajor: x_fit,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: y_fit,
            weights: tempering_weights.as_deref(),
            offsets: None,
        };
        if ctx.cache_policy.enabled
            && ctx.cache_policy.max_bytes.is_none()
            && ctx.admission.byte_budget() == u64::MAX
            && likelihood == BayesLikelihood::GaussianIdentity
            && whitened.is_none()
            && tempering_weights.is_none()
            && self.backend != BayesianBackendKind::Hmc
        {
            workspace
                .laplace
                .bind_immutable_gaussian_design(
                    Arc::clone(&problem.design.matrix),
                    Arc::clone(&problem.design.outcome),
                    problem.design.nrows,
                    problem.design.ncols,
                )
                .map_err(prob_err)?;
        }
        let opts = BayesFitOptions {
            n_draws: initial_draws,
            seed: self.seed,
            ..BayesFitOptions::default()
        };
        let mut fit =
            BackendInputs { likelihood, design: design_ref, prior: &prior, opts, max_draws }
                .fit(self.backend, workspace, ctx)
                .map_err(prob_err)?;
        if let Some(error) =
            fit.diagnostics.publication_error("Bayesian fit refused without diagnostics")
        {
            return Err(error.into());
        }
        fit.diagnostics.notes.extend(extra_notes);

        let glm_family = likelihood_to_glm_family(likelihood);
        let evaluator = gcomp_evaluator(problem, glm_family)?;
        let compiled = evaluator.compile()?;
        let eval = EffectEvaluation {
            evaluator: &evaluator,
            compiled: &compiled,
            ncols: problem.design.ncols,
        };

        let residual = ResidualVarianceDraws::from_fit(likelihood, &fit.draws, &prior)?;
        // Adaptive MVN sampling uses the β-block covariance only; drop residual-variance
        // columns so batch merges match `PosteriorSchema::coefficients`.
        let coef_draws = coefficient_only_draws(&fit.draws)?;

        let n_draws = fit.draws.n_draws;
        let (mechanism_draws, effects, n_draws, early_stopped) = if laplace_adaptive {
            let cov = fit.cov.as_deref().ok_or_else(|| {
                EstimationError::stats_msg("Laplace adaptive draws require posterior covariance")
            })?;
            let plan = AdaptiveRedraw {
                map: &fit.map,
                cov,
                seed: self.seed,
                max_draws,
                budget: ctx.adaptive_draws,
            };
            let outcome = eval.redraw(&plan, &coef_draws, n_draws, workspace, ctx)?;
            let mechanism_draws = concat_coefficient_draws(&coef_draws, &outcome.extra_blocks)?;
            (mechanism_draws, outcome.effects, outcome.n_draws, outcome.early_stopped)
        } else {
            let effects = eval.effects(&coef_draws, n_draws, workspace, ctx)?;
            (coef_draws, effects, n_draws, false)
        };

        // Adaptive MVN path only runs for known-σ² Laplace, whose residual column is
        // the constant σ², so it is re-attached at the final draw count.
        let draws = assemble_posterior_draws(
            problem,
            glm_family,
            &mechanism_draws,
            residual.at(n_draws).as_deref(),
            &effects,
            self.seed,
        )?;
        let summaries = draws.summarize();
        // A Laplace interval is the type-7 quantile interval of the draws, whose expected posterior
        // mass `0.95(m-1)/(m+1)` is 0.921 at 64 draws and 0.949 at 1,000. Below the measured draw
        // count the effect's interval is the exchangeable-rank (type-6) interval of the same
        // draws instead, whose expected mass is exactly 0.95 at every count. It is a function of
        // the embedded draws alone, so artifacts stay self-verifying, and the declared draws,
        // their count and every other summary are untouched.
        let mut summaries = summaries;
        if matches!(self.backend, BayesianBackendKind::Laplace)
            && n_draws < LAPLACE_RANK_INTERVAL_BELOW_DRAWS
        {
            let column = draws
                .schema
                .quantities
                .iter()
                .position(|quantity| matches!(quantity, PosteriorQuantityKind::Effect { .. }));
            let values = column.and_then(|column| draws.column(column).ok());
            if let (Some(column), Some(values)) = (column, values) {
                if values.len() >= 2 && values.iter().all(|value| value.is_finite()) {
                    let mut sorted = values.to_vec();
                    sorted.sort_by(f64::total_cmp);
                    let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
                        &sorted,
                        0.95,
                        antecedent_stats::QuantileRule::ExchangeableRank,
                    );
                    let mut q025 = summaries.q025.to_vec();
                    let mut q975 = summaries.q975.to_vec();
                    q025[column] = lower;
                    q975[column] = upper;
                    summaries.q025 = Arc::from(q025);
                    summaries.q975 = Arc::from(q975);
                }
            }
        }
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
            early_stopped,
            treatment_contrast: (self.glm_family() == GlmFamily::GaussianIdentity)
                .then_some(problem.active - problem.control),
            preparation: problem.preparation.clone(),
        })
    }

    /// A sequential prior must be a Gaussian coefficient prior of the design's width.
    fn check_sequential_prior(
        &self,
        problem: &PreparedBayesianProblem,
    ) -> Result<(), EstimationError> {
        let Some(p) = &self.prior else {
            return Ok(());
        };
        let Some(coef) = p.gaussian_coefficients() else {
            return Err(EstimationError::stats_msg(
                "sequential prior missing GaussianCoefficients entry",
            ));
        };
        if coef.len() != problem.design.ncols {
            return Err(EstimationError::stats_msg(format!(
                "sequential prior coefficient dimension {} != design ncols {}",
                coef.len(),
                problem.design.ncols
            )));
        }
        Ok(())
    }

    /// Prior restrictions recorded as estimation-scope assumptions.
    fn prior_assumptions(&self, prior: &PriorSet) -> AssumptionSet {
        let sequential = self.prior.is_some();
        let mut assumptions = AssumptionSet::new();
        let source = if sequential {
            AssumptionSource::Artifact
        } else {
            AssumptionSource::AlgorithmDefault { algorithm: Arc::from("bayesian_gcomp") }
        };
        let conversion = if prior.absolute_coefficient_scale().is_some() {
            "absolute coefficient SD² kept on the absolute scale and converted to conjugate \
             scale V0 at fit time via the target residual-variance estimate"
        } else {
            "absolute coefficient SD² converted to conjugate scale V0 via source residual variance"
        };
        let dense = prior.coefficient_correlation().is_some();
        for spec in &prior.specs {
            let mut pa = spec.as_assumption();
            if sequential && dense {
                pa.description = Arc::from(format!(
                    "{} (sequential prior from posterior artifact; dense V0 carrying the source \
                     posterior coefficient covariance; {conversion})",
                    pa.description
                ));
            } else if sequential {
                pa.description = Arc::from(format!(
                    "{} (sequential prior from posterior artifact; diagonal V0 only — \
                     off-diagonal posterior covariance dropped, so transferred priors can be \
                     tighter than the source marginal on linear combinations; {conversion})",
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
        assumptions
    }

    /// Kept-draw count (HMC floored) and the notes recording a floor.
    fn draw_budget(&self) -> Result<(usize, Vec<Arc<str>>), EstimationError> {
        // HMC publication (Ř≤1.01, ESS≥100) needs a longer schedule than the
        // Laplace/conjugate default of 1000 draws; floor so under-specified
        // callers still clear the gate rather than refuse with near-miss Ř.
        let requested_draws = require_bayesian_n_draws(self.n_draws)?;
        let max_draws = match self.backend {
            BayesianBackendKind::Hmc => requested_draws.max(HMC_MIN_DRAWS),
            _ => requested_draws,
        };
        let mut notes: Vec<Arc<str>> = Vec::new();
        if max_draws > requested_draws {
            notes.push(Arc::from(format!(
                "{HMC_DRAW_FLOOR_NOTE_PREFIX} requested={requested_draws} used={max_draws}"
            )));
        }
        Ok((max_draws, notes))
    }

    /// Whether Laplace adaptive redraws apply, and the size of the first block.
    fn adaptive_schedule(
        &self,
        likelihood: BayesLikelihood,
        prior: &PriorSet,
        ctx: &ExecutionContext,
        max_draws: usize,
    ) -> Result<(bool, usize), EstimationError> {
        let adaptive = ctx.adaptive_draws;
        // Adaptive redraws append samples from the fitted Gaussian covariance. Unknown-variance
        // GaussianIdentity is routed by the Laplace backend to the exact conjugate NIG posterior,
        // whose marginal coefficient law is Student-t and has no Gaussian covariance artifact.
        // Materialize the full requested NIG draw count instead of mixing it with MVN redraws.
        let laplace_mvn_redraw_supported = match likelihood {
            BayesLikelihood::GaussianIdentity => matches!(
                GaussianVarianceModel::from_prior_set(prior).map_err(prob_err)?,
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
        Ok((laplace_adaptive, initial_draws))
    }
}

fn gcomp_evaluator(
    problem: &PreparedBayesianProblem,
    family: GlmFamily,
) -> Result<GCompAteEvaluator, EstimationError> {
    let treatment_col = problem
        .design
        .treatment_column()
        .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
    Ok(GCompAteEvaluator {
        family,
        treatment_col,
        active: problem.active,
        control: problem.control,
        nrows: problem.design.nrows,
        ncols: problem.design.ncols,
        matrix: Arc::clone(&problem.design.matrix),
    })
}

/// Everything a backend needs to fit one Bayesian g-computation mechanism.
struct BackendInputs<'a> {
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'a>,
    prior: &'a PriorSet,
    opts: BayesFitOptions,
    max_draws: usize,
}

impl BackendInputs<'_> {
    fn fit(
        self,
        backend: BayesianBackendKind,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BayesFitResult, ProbError> {
        let Self { likelihood, design, prior, opts, max_draws } = self;
        let opts = &opts;
        match backend {
            BayesianBackendKind::ConjugateGaussian => ConjugateGaussianBackend.fit(
                likelihood,
                design,
                prior,
                opts,
                &mut workspace.laplace,
                ctx,
            ),
            BayesianBackendKind::Laplace => {
                LaplaceGlmBackend.fit(likelihood, design, prior, opts, &mut workspace.laplace, ctx)
            }
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
                    .fit(likelihood, design, prior, opts, &mut workspace.laplace, ctx)
            }
        }
    }
}

/// Temper the likelihood of time-ordered rows by the long-run-variance ratio of
/// the targeted slope scores (generalized posterior; see `serial_dependence`).
/// Returns per-row likelihood weights when the ratio exceeds one.
fn temper_likelihood(
    problem: &PreparedBayesianProblem,
    assumptions: &mut AssumptionSet,
    extra_notes: &mut Vec<Arc<str>>,
    ctx: &ExecutionContext,
) -> Result<Option<Vec<f64>>, EstimationError> {
    let tempering = match &problem.serial_dependence {
        SerialDependence::Iid => None,
        SerialDependence::LongRunTempering(scope) => {
            if problem.unit_ids.is_some() {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("option_not_applicable"),
                    "long-run tempering applies to one time-ordered series, not stacked \
                     panel units",
                ));
            }
            Some(antecedent_core::numerical_admission::with_context(ctx, || {
                long_run_tempering_factor(
                    &problem.design,
                    scope,
                    ResidualFitReuse::for_context(ctx),
                )
            })?)
        }
        SerialDependence::SegmentedTempering { scope, segments } => {
            if problem.unit_ids.is_some() {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("option_not_applicable"),
                    "segmented tempering applies to one series with gaps, not stacked panel units",
                ));
            }
            Some(antecedent_core::numerical_admission::with_context(ctx, || {
                crate::serial_dependence::segmented_tempering_factor(
                    &problem.design,
                    scope,
                    segments,
                )
            })?)
        }
    };
    let weights = tempering
        .filter(|factor| factor.kappa > 1.0)
        .map(|factor| vec![factor.kappa.recip(); problem.design.nrows]);
    if let Some(factor) = tempering {
        extra_notes.push(factor.note());
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
                id: Arc::from(DEPENDENCE_ASSUMPTION_ID),
                description: factor.description(),
            }),
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
    Ok(weights)
}

/// Random-intercept GLS whitening of a stacked panel design.
///
/// GLS whitening is a Gaussian residual transform; applying it to Poisson /
/// binomial / other GLM outcomes silently corrupts the likelihood, so those
/// are skipped with a note.
/// Whitened `(design, outcome)` columns.
type WhitenedDesign = (Vec<f64>, Vec<f64>);

fn whiten_design(
    problem: &PreparedBayesianProblem,
    likelihood: BayesLikelihood,
    extra_notes: &mut Vec<Arc<str>>,
) -> Result<Option<WhitenedDesign>, EstimationError> {
    if let Some(ids) = problem.unit_ids.as_deref() {
        if ids.len() != problem.design.nrows {
            return Err(EstimationError::data_msg(
                "unit_ids length must match the design row count",
            ));
        }
    }
    Ok(match (likelihood, problem.unit_ids.as_deref()) {
        (BayesLikelihood::GaussianIdentity, Some(ids)) => {
            let whitened = random_intercept_gls_whiten(
                &problem.design.matrix,
                &problem.design.outcome,
                problem.design.nrows,
                problem.design.ncols,
                ids,
            );
            if whitened.is_none() {
                extra_notes.push(Arc::from(RANDOM_INTERCEPT_WHITEN_FALLBACK_NOTE));
            }
            whitened
        }
        (_, Some(_)) => {
            extra_notes.push(Arc::from(RANDOM_INTERCEPT_WHITEN_SKIP_NOTE));
            None
        }
        (_, None) => None,
    })
}

/// Residual-variance draws carried alongside the coefficient draws so a prior
/// hydrated from this posterior can convert absolute Var(β) to V0.
struct ResidualVarianceDraws {
    /// Residual-variance column of the fitted draws, captured before stripping to the
    /// coefficient-only schema (g-comp evaluates β only).
    column: Option<Vec<f64>>,
    /// A known-σ² Gaussian fit has no residual-variance draws; it records its σ² as a
    /// constant column so hydration uses the source σ² instead of the target's estimate.
    known_sigma2: Option<f64>,
}

impl ResidualVarianceDraws {
    fn from_fit(
        likelihood: BayesLikelihood,
        draws: &PosteriorDraws,
        prior: &PriorSet,
    ) -> Result<Self, EstimationError> {
        let column: Option<Vec<f64>> = draws
            .schema
            .quantities
            .iter()
            .enumerate()
            .find(|(_, q)| matches!(q, PosteriorQuantityKind::ResidualVariance))
            .and_then(|(i, _)| draws.column(i).ok().map(<[f64]>::to_vec));
        let known_sigma2 = match (likelihood, &column) {
            (BayesLikelihood::GaussianIdentity, None) => {
                match GaussianVarianceModel::from_prior_set(prior).map_err(prob_err)? {
                    GaussianVarianceModel::Known { sigma2 } => Some(sigma2),
                    GaussianVarianceModel::InvGamma { .. } => None,
                }
            }
            _ => None,
        };
        Ok(Self { column, known_sigma2 })
    }

    /// The column for a posterior of `n_draws` draws, if one is available.
    fn at(&self, n_draws: usize) -> Option<Vec<f64>> {
        match (&self.column, self.known_sigma2) {
            (Some(col), _) if col.len() == n_draws => Some(col.clone()),
            (None, Some(sigma2)) => Some(vec![sigma2; n_draws]),
            _ => None,
        }
    }
}

/// The compiled g-computation functional, evaluated block by block.
struct EffectEvaluation<'a> {
    evaluator: &'a GCompAteEvaluator,
    compiled: &'a CompiledGCompAte,
    ncols: usize,
}

impl EffectEvaluation<'_> {
    /// Effect of each of the first `n` draws of `coefficients`.
    fn effects(
        &self,
        coefficients: &PosteriorDraws,
        n: usize,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, EstimationError> {
        workspace.eval.prepare(n, self.ncols);
        let mut effect_out = EffectBatch::default();
        effect_out.prepare(n);
        let batch = coefficients.batch(0, n).map_err(EstimationError::from)?;
        self.evaluator.evaluate_batch(
            self.compiled,
            batch,
            &mut effect_out,
            &mut workspace.eval,
            ctx,
        )?;
        Ok(effect_out.values[..n].to_vec())
    }
}

/// Fitted Gaussian posterior and the schedule for appending MVN redraws to it.
struct AdaptiveRedraw<'a> {
    map: &'a [f64],
    cov: &'a [f64],
    seed: u64,
    max_draws: usize,
    budget: AdaptiveDrawBudget,
}

struct AdaptiveOutcome {
    effects: Vec<f64>,
    extra_blocks: Vec<PosteriorDraws>,
    n_draws: usize,
    early_stopped: bool,
}

impl EffectEvaluation<'_> {
    /// Append independent MVN draws in batches until the effect ESS target or the
    /// draw cap is reached.
    fn redraw(
        &self,
        plan: &AdaptiveRedraw<'_>,
        coef_draws: &PosteriorDraws,
        mut n_draws: usize,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<AdaptiveOutcome, EstimationError> {
        let batch = 32usize;
        let mut effects = self.effects(coef_draws, n_draws, workspace, ctx)?;
        let mut extra_blocks: Vec<PosteriorDraws> = Vec::new();
        let mut early_stopped = false;
        loop {
            // Independent MVN draws: ESS is the draw count, a genuine Monte
            // Carlo error measure. The former quantile-width change rule was
            // not one and is no longer consulted.
            let ess = effects.len() as f64;
            if effects.len() >= plan.budget.min_draws.max(2) && ess >= plan.budget.ess_target {
                early_stopped = n_draws < plan.max_draws;
                break;
            }
            if n_draws >= plan.max_draws {
                break;
            }
            let next = (n_draws + batch).min(plan.max_draws);
            let add = next - n_draws;
            let extra = sample_gaussian_mvn(
                plan.map,
                plan.cov,
                add,
                plan.seed.wrapping_add(n_draws as u64),
                &mut workspace.laplace,
            )
            .map_err(EstimationError::from)?;
            let extra_draws = PosteriorDraws::from_column_major(
                PosteriorSchema::coefficients(self.ncols),
                add,
                extra,
            )
            .map_err(EstimationError::from)?;
            effects.extend_from_slice(&self.effects(&extra_draws, add, workspace, ctx)?);
            // Accumulate blocks; one concatenation after the loop replaces
            // the former per-batch merge + clone (O(D²) copying in D draws).
            extra_blocks.push(extra_draws);
            n_draws = next;
        }
        Ok(AdaptiveOutcome { effects, extra_blocks, n_draws, early_stopped })
    }
}

/// Coefficient draws, the optional residual-variance column, and the effect draws as one
/// posterior with durable coefficient names and modifier-mean uncertainty applied.
fn assemble_posterior_draws(
    problem: &PreparedBayesianProblem,
    glm_family: GlmFamily,
    mechanism_draws: &PosteriorDraws,
    residual_col: Option<&[f64]>,
    effects: &[f64],
    seed: u64,
) -> Result<PosteriorDraws, EstimationError> {
    let n_draws = effects.len();
    let mut quantities = mechanism_draws.schema.quantities.to_vec();
    let residual_idx = if residual_col.is_some() {
        quantities.push(PosteriorQuantityKind::ResidualVariance);
        Some(quantities.len() - 1)
    } else {
        None
    };
    let effect_idx = quantities.len();
    quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
    let n_q = quantities.len();
    let _admitted = crate::util::admit_bound_array::<f64>(n_draws, n_q)?;
    let mut values = vec![0.0; n_draws * n_q];
    for (qi, q) in mechanism_draws.schema.quantities.iter().enumerate() {
        let dest = quantities.iter().position(|qq| qq == q).ok_or_else(|| {
            EstimationError::stats_msg(format!("posterior quantity missing from schema: {q:?}"))
        })?;
        let col = mechanism_draws.column(qi).map_err(EstimationError::from)?;
        values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(col);
    }
    if let (Some(idx), Some(col)) = (residual_idx, residual_col) {
        values[idx * n_draws..(idx + 1) * n_draws].copy_from_slice(col);
    }
    values[effect_idx * n_draws..(effect_idx + 1) * n_draws].copy_from_slice(effects);
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
        seed,
    );
    PosteriorDraws::from_column_major(
        PosteriorSchema { quantities: Arc::from(quantities) },
        n_draws,
        values,
    )
    .map_err(EstimationError::from)
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
pub(crate) fn coefficient_names_from_design(
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
pub(super) fn add_modifier_mean_uncertainty(
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
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, nrows, false);
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
pub(super) fn random_intercept_gls_whiten(
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

/// The arm whose rows a model-based Bayesian effect is averaged over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReferenceArm {
    /// Rows observed at the active treatment level (the effect on the treated).
    Active,
    /// Rows observed at the control level (the effect on the untreated).
    Control,
}

/// A design with its coefficient names.
type CenteredDesign = (CompiledDesign, Arc<[Arc<str>]>);

/// Append one treatment-by-covariate interaction per covariate column, each covariate
/// centred at its mean over the rows observed at `level`.
///
/// The outcome model is then `y = b0 + b_T t + g'z + d' t (z - zbar_arm)`. Averaged over the
/// reference arm the centred interactions vanish, so on the identity link `(active - control)
/// b_T` is the model-based contrast on that arm for every posterior draw. It is the effect of
/// the declared outcome model, not a matching estimand.
fn center_interactions_on_arm(
    design: &CompiledDesign,
    names: &Arc<[Arc<str>]>,
    level: f64,
) -> Result<CenteredDesign, EstimationError> {
    let n = design.nrows;
    let treatment = &design.matrix[n..2 * n];
    let in_arm: Vec<bool> = treatment.iter().map(|&t| (t - level).abs() <= 1e-12).collect();
    let arm_rows = in_arm.iter().filter(|&&flag| flag).count();
    if arm_rows < 2 {
        return Err(EstimationError::data_msg(
            "the Bayesian model-based effect on an arm needs at least two rows observed at that \
             arm's treatment level",
        ));
    }
    let covariate_cols: Vec<usize> = design
        .columns
        .iter()
        .enumerate()
        .filter(|(_, column)| matches!(column.role, DesignColumnRole::Covariate(_)))
        .map(|(index, _)| index)
        .collect();
    if design.ncols + covariate_cols.len() >= n {
        return Err(EstimationError::data_msg(
            "the Bayesian model-based effect on an arm needs more rows than coefficients",
        ));
    }
    let mut matrix = design.matrix.to_vec();
    let mut roles: Vec<DesignColumnRole> =
        design.columns.iter().map(|column| column.role).collect();
    let mut all_names: Vec<Arc<str>> = names.to_vec();
    for &col in &covariate_cols {
        let values = &design.matrix[col * n..(col + 1) * n];
        let mean =
            values.iter().zip(&in_arm).filter(|(_, flag)| **flag).map(|(v, _)| *v).sum::<f64>()
                / arm_rows as f64;
        matrix.extend(treatment.iter().zip(values).map(|(t, x)| t * (x - mean)));
        roles.push(roles[col]);
        let base = names.get(col).map_or("covariate", |name| name.as_ref());
        all_names
            .push(Arc::from(format!("treatment_x_centered_{}", base.trim_start_matches("coef_"))));
    }
    let ncols = roles.len();
    let widened = CompiledDesign {
        ncols,
        matrix: Arc::from(matrix),
        columns: antecedent_stats::DesignColumnMap::from_roles(roles),
        ..design.clone()
    };
    Ok((widened, Arc::from(all_names)))
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
    /// How the columns this problem was built from were prepared, when they were admitted
    /// through the preparation path.
    pub preparation: Option<Arc<crate::PreparationSummary>>,
}

impl PreparedBayesianProblem {
    /// Record how the columns were prepared.
    #[must_use]
    pub fn with_preparation(mut self, preparation: Option<Arc<crate::PreparationSummary>>) -> Self {
        self.preparation = preparation;
        self
    }

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
