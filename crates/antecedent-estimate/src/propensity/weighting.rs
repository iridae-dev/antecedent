//! Inverse-probability weighting estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions};

use crate::inference::InferenceSpec;
use crate::arm::TreatmentArm;
use super::balanced::{BalancedLinearization, balance_arm_weights, balanced_influence_values};
use super::levels::OtherLevels;
use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, prepare_propensity_problem_with_levels, trim_of,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
use crate::sampling_weights::SamplingWeights;
use crate::se::AnalyticSeKind;
use crate::util::BootstrapOutcome;

/// A score-inverted interval for the calibrated weighted-population contrast. The common
/// population-mass denominator and its covariance with the full effect influence enter the
/// Fieller quadratic, so the endpoints need not be symmetric about the point estimate.
#[derive(Clone, Debug)]
pub struct BalancedWeightingInterval {
    /// Calibrated point estimate with the interval construction's analytic SE.
    pub estimate: EffectEstimate,
    /// Lower endpoint of the bounded confidence set.
    pub lower: f64,
    /// Upper endpoint of the bounded confidence set.
    pub upper: f64,
    /// Requested probability level.
    pub level: f64,
    /// Variance of the normalized population-mass mean.
    pub mass_variance: f64,
    /// Covariance of the effect influence mean and normalized population-mass mean.
    pub effect_mass_covariance: f64,
    /// Working-Gaussian quadratic-form Satterthwaite reference df, when used.
    /// Normal population-score and cluster-score references record `None`.
    pub reference_df: Option<f64>,
    /// Noise and random-target variance decomposition for the declared linear model.
    pub linear_variance: Option<PopulationLinearVariance>,
}

/// Variance of a calibrated contrast under independent homoskedastic Gaussian arm-linear
/// outcomes. The complete point coefficient carries outcome noise; the fitted contrast
/// carries the additional uncertainty in the sampled target population.
#[derive(Clone, Debug)]
pub struct PopulationLinearVariance {
    /// Declared outcome model required by this variance construction.
    pub outcome_model: &'static str,
    /// Conditional outcome-noise contribution to the point variance.
    pub noise: f64,
    /// Random target-population contribution to the point variance.
    pub target: f64,
    /// Estimated common outcome-noise variance, independent of the point's noise.
    pub noise_scale: f64,
    /// Rank of the independent outcome-residual projector.
    pub noise_degrees_of_freedom: usize,
}

/// Inverse-probability weighting estimator (ATE/ATT/ATC via `TargetPopulation`).
///
/// Point estimate is the Hajek (self-normalized) weighted difference of means. Positivity is
/// mandatory: [`OverlapPolicy::ExplicitOverride`] is refused.
///
/// **Sampling weights.** With [`Self::sampling_weights`] the propensity model is the
/// weighted logistic fit, every row's inverse-probability weight is multiplied by its sampling
/// weight (and by a custom target weight when there is one), and the estimand is the effect in
/// the weighted population. The analytic SE is the linearization of the stacked weighted
/// logistic and weighted-mean estimating equations, so the estimated propensity enters the
/// variance.
///
/// **Clusters.** With cluster labels the analytic SE is the cluster sandwich of that same
/// corrected influence function (the propensity-estimation term is a per-row score, so it sums
/// within clusters), and the bootstrap resamples whole clusters, refitting the propensity on
/// each replicate.
#[derive(Clone, Debug)]
pub struct PropensityWeighting {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Resampling and dependence declarations: bootstrap size and interval request, cluster
    /// labels (either per complete-case row or per source-table row, exclusive) and the
    /// sampling design of the rows. The analytic standard error is fixed by the estimator, so
    /// `se_kind`, multiway labels and panel times are refused, not ignored.
    pub inference: InferenceSpec,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Exponentially calibrate each arm's IPW weights to the sampling-weighted population's
    /// covariate moments. Declares linear arm outcome models for the residual HC3 correction;
    /// the full sandwich includes fitted propensity and estimated moment-target uncertainty.
    pub covariate_balance: bool,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Policy for treatment rows at neither declared level. The default refuses; the
    /// exclusion that restricts the target population to the rows at the two declared levels
    /// must be declared.
    pub other_levels: OtherLevels,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    pub sampling_weights: Option<SamplingWeights>,
}

impl Default for PropensityWeighting {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityWeighting {
    /// Defaults: 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            inference: InferenceSpec::new(200),
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            covariate_balance: false,
            population_registry: None,
            other_levels: OtherLevels::default(),
            sampling_weights: None,
        }
    }

    crate::inference::with_bootstrap_interval!(inference);

    crate::inference::with_bootstrap_replicates!(inference);

    /// Set the overlap policy. Positivity is mandatory here:
    /// [`OverlapPolicy::ExplicitOverride`] is refused by `prepare`.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the GLM fitting options for the propensity model.
    #[must_use]
    pub const fn with_glm_options(mut self, glm_options: GlmOptions) -> Self {
        self.glm_options = glm_options;
        self
    }

    /// Balance each arm to the weighted population's intercept/covariate moments by positive
    /// exponential calibration of its IPW weights. Supported for untrimmed ATEs. This is a
    /// distinct point/variance construction from uncalibrated Hajek weighting.
    #[must_use]
    pub const fn with_covariate_balance(mut self, enabled: bool) -> Self {
        self.covariate_balance = enabled;
        self
    }

    /// Declare what to do with rows whose treatment is at neither declared level (see
    /// [`OtherLevels`]). The default refuses.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    /// Declare sampling weights for every row of the source table.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    crate::inference::with_cluster_ids!(inference);

    crate::inference::with_source_cluster_ids!(inference);

    crate::inference::with_sampling_design!(inference);

    /// Cluster label of each row of `problem`, when clusters are declared.
    fn clusters(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<Option<std::borrow::Cow<'_, [u32]>>, EstimationError> {
        let source_rows = problem.source_rows(self.inference.source_cluster_ids.is_some())?;
        let labels = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            source_rows,
        )?;
        if let Some(labels) = &labels {
            crate::se::require_clusters(Some(labels.as_ref()), problem.nrows)?;
        }
        Ok(labels)
    }

    /// Prepare the covariate design from tabular data, identified estimand, and query.
    ///
    /// # Errors
    ///
    /// Overlap policy is `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        super::require_supported_inference(&self.inference)?;
        if let Some(source) = self.inference.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.inference.cluster_ids.as_deref(),
                data.row_count(),
            )?;
        }
        prepare_propensity_problem_with_levels(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
            self.other_levels,
            self.sampling_weights.as_ref(),
        )
    }

    /// Fit the propensity model and compute the Hajek-weighted effect, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// Unsupported target population or GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        super::require_supported_inference(&self.inference)?;
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Invert the calibrated contrast's population estimating equation using its full joint
    /// covariance with the estimated population mass (Fieller's ratio construction). The
    /// returned endpoints are a separate construction from a symmetric analytic-SE interval.
    /// For iid rows, a working-Gaussian, noncentral quadratic-form Satterthwaite reference
    /// carries variance-estimator uncertainty; its noise model uses unweighted arm OLS.
    /// The cluster construction retains the cluster sandwich's normal score reference.
    /// Requires [`Self::with_covariate_balance`] and an untrimmed ATE.
    ///
    /// # Errors
    /// Invalid level, cancellation, failed calibrated fit, or an unbounded confidence set.
    pub fn population_score_interval(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        level: f64,
    ) -> Result<BalancedWeightingInterval, EstimationError> {
        if !self.covariate_balance || !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(EstimationError::data_msg(
                "population score interval requires calibrated weights and 0 < level < 1",
            ));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::data_msg("population score interval cancelled"));
        }
        let (estimate, psi, linearization) =
            self.fit_point_influence(problem, workspace, assumptions)?;
        let mass: Vec<f64> = problem
            .sampling_weights
            .as_ref()
            .map_or_else(|| vec![1.0; problem.nrows], |s| s.weights.to_vec());
        let groups = self.clusters(problem)?;
        let variance = |values: &[f64]| -> Result<f64, EstimationError> {
            let se = match groups.as_deref() {
                Some(groups) => crate::se::cluster_influence_se(values, groups)?,
                None => hajek_iid_se(values, problem.design_ncols),
            };
            Ok(se * se)
        };
        let v_effect = estimate.se_analytic().powi(2);
        let v_mass = variance(&mass)?;
        let joint: Vec<f64> = psi.iter().zip(&mass).map(|(p, m)| p + m).collect();
        let covariance = 0.5 * (variance(&joint)? - v_effect - v_mass);
        let reference_df = if groups.is_none() {
            Some(
                linearization
                    .ok_or_else(|| {
                        EstimationError::stats_msg("missing calibrated variance operator")
                    })?
                    .reference_df()?,
            )
        } else {
            None
        };
        let z = reference_df.map_or_else(
            || antecedent_stats::normal_ppf((1.0 + level) / 2.0),
            |df| antecedent_stats::student_t_ppf((1.0 + level) / 2.0, df),
        );
        let a = 1.0 - z * z * v_mass;
        if !a.is_finite() || a <= 0.0 {
            return Err(EstimationError::stats_msg("population-mass confidence set is unbounded"));
        }
        let centre = -z * z * covariance / a;
        let radius = (z.powi(4) * covariance * covariance + a * z * z * v_effect).sqrt() / a;
        Ok(BalancedWeightingInterval {
            lower: estimate.ate + centre - radius,
            upper: estimate.ate + centre + radius,
            estimate,
            level,
            mass_variance: v_mass,
            effect_mass_covariance: covariance,
            reference_df,
            linear_variance: None,
        })
    }

    /// Invert a population score under independent homoskedastic Gaussian arm-linear
    /// outcomes, with a separate variance term for the random target population.
    ///
    /// The point is exactly linear in the outcomes conditional on covariates, treatment and
    /// sampling weights. Its noise variance is the squared norm of that complete coefficient
    /// times an unbiased residual noise estimate. Unweighted arm OLS, augmented by the point
    /// coefficient, makes this noise estimate independent of the point's Gaussian noise.
    /// Fitted treatment-effect heterogeneity and its covariance with population mass remain
    /// in the Fieller score. The population reference is asymptotically normal; this does not
    /// assert an exact Student pivot for the combined noise and random-target variance.
    ///
    /// This declared model construction is separate from the robust sandwich published by
    /// [`Self::fit`] and the working quadratic reference in [`Self::population_score_interval`].
    /// It requires [`Self::with_covariate_balance`], an untrimmed ATE, and independent rows.
    ///
    /// # Errors
    /// Invalid level, cancellation, unsupported clustering, failed fit, insufficient residual
    /// degrees of freedom, or an unbounded population-mass confidence set.
    pub fn population_linear_score_interval(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        level: f64,
    ) -> Result<BalancedWeightingInterval, EstimationError> {
        if !self.covariate_balance || !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(EstimationError::data_msg(
                "population linear score requires calibrated weights and 0 < level < 1",
            ));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::data_msg("population linear score cancelled"));
        }
        if self.clusters(problem)?.is_some() {
            return Err(EstimationError::unsupported(
                "population linear score requires independent rows",
            ));
        }
        let (mut estimate, _, linear) =
            self.fit_point_influence(problem, workspace, assumptions)?;
        let linear = linear
            .ok_or_else(|| EstimationError::stats_msg("missing calibrated point linearization"))?;
        let components = linear.population_linear_variance(problem)?;
        let signal = linear.operator.apply(&linear.working_mean);
        let mass = problem
            .sampling_weights
            .as_ref()
            .map_or_else(|| vec![1.0; problem.nrows], |s| s.weights.to_vec());
        let variance = |values: &[f64]| hajek_iid_se(values, problem.design_ncols).powi(2);
        let mass_variance = variance(&mass);
        let joint: Vec<f64> = signal.iter().zip(&mass).map(|(a, b)| a + b).collect();
        let effect_mass_covariance = (variance(&joint) - components.target - mass_variance) / 2.0;
        let effect_variance = components.noise + components.target;
        let z = antecedent_stats::normal_ppf((1.0 + level) / 2.0);
        let a = 1.0 - z * z * mass_variance;
        if !a.is_finite() || a <= 0.0 {
            return Err(EstimationError::stats_msg("population-mass confidence set is unbounded"));
        }
        let centre = -z * z * effect_mass_covariance / a;
        let radius =
            (z.powi(4) * effect_mass_covariance.powi(2) + a * z * z * effect_variance).sqrt() / a;
        estimate.uncertainty.analytic.se = effect_variance.sqrt();
        estimate.uncertainty.analytic.kind = Some(AnalyticSeKind::Homoskedastic);
        Ok(BalancedWeightingInterval {
            lower: estimate.ate + centre - radius,
            upper: estimate.ate + centre + radius,
            estimate,
            level,
            mass_variance,
            effect_mass_covariance,
            reference_df: None,
            linear_variance: Some(components),
        })
    }

    /// Attach the bootstrap SE onto a point estimate from [`Self::fit`] (progressive
    /// uncertainty stage). Equal to what `fit` publishes when `bootstrap_replicates > 0`,
    /// without refitting the point model.
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        if self.inference.bootstrap_replicates == 0 {
            return Ok(point);
        }
        let target = IpwTarget::from_population(&problem.target_population)?;
        let trim = trim_of(problem.overlap);
        let boot = self.bootstrap_se(problem, target, trim, workspace, ctx)?;
        Ok(point.with_bootstrap_outcome(boot))
    }

    fn fit_point(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        self.fit_point_influence(problem, workspace, assumptions).map(|(estimate, _, _)| estimate)
    }

    fn fit_point_influence(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<(EffectEstimate, Vec<f64>, Option<BalancedLinearization>), EstimationError> {
        let target = IpwTarget::from_population(&problem.target_population)?;
        if matches!(target, IpwTarget::Custom) && problem.target_weights.is_none() {
            return Err(EstimationError::unsupported(
                "CustomDistribution requires PopulationRegistry weights on the prepared problem",
            ));
        }
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;

        let sampling = problem.sampling_weights.as_ref().map(|s| &*s.weights);
        let clusters = self.clusters(problem)?;
        let mut weights = compute_ipw_weights(
            &problem.treatment,
            &model.clipped_scores,
            &model.fit.scores,
            target,
            trim,
        );
        apply_target_weights(&mut weights, problem.target_weights.as_deref());
        apply_target_weights(&mut weights, sampling);
        if self.covariate_balance {
            if !matches!(target, IpwTarget::Ate) || trim.is_some() {
                return Err(EstimationError::unsupported(
                    "covariate-balanced weighting currently requires an untrimmed ATE",
                ));
            }
            balance_arm_weights(
                &problem.design_matrix,
                problem.design_ncols,
                &problem.treatment,
                sampling,
                &mut weights,
            )?;
        }
        let ate = hajek_difference(&problem.arms, &problem.outcome, &weights)?;
        let (psi, linearization) = if self.covariate_balance {
            let (psi, linearization) = balanced_influence_values(
                &problem.design_matrix,
                problem.design_ncols,
                &problem.treatment,
                &problem.outcome,
                sampling,
                &model.fit.scores,
                clip_of(problem.overlap),
                &weights,
                ate,
            )?;
            (psi, Some(linearization))
        } else {
            (
                hajek_influence_values(
                    &problem.arms,
                    &problem.outcome,
                    &weights,
                    Some(HajekPropensity {
                        scores: &model.fit.scores,
                        design: &problem.design_matrix,
                        ncols: problem.design_ncols,
                        target,
                        clip: clip_of(problem.overlap),
                        row_mass: sampling,
                    }),
                )?,
                None,
            )
        };
        let se_analytic = match clusters.as_deref() {
            Some(groups) => crate::se::cluster_influence_se(&psi, groups)?,
            None => hajek_iid_se(&psi, problem.design_ncols),
        };

        let overlap_report = Some(crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            Some(&weights),
            Some(target),
        ));

        let n_obs = weights.iter().filter(|w| w.is_finite() && **w > 0.0).count();
        let mut estimate = EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_preparation(Some(problem.preparation_summary()))
            .with_n_obs(u64::try_from(n_obs).unwrap_or(u64::MAX))
            .with_overlap_report(overlap_report)
            .with_treatment_levels(problem.treatment_levels)
            .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
            .with_retained_memory_bytes(Some(workspace.retained_memory_bytes()));
        if clusters.is_some() {
            estimate = estimate.with_se_kind(AnalyticSeKind::Cluster);
        }
        Ok((estimate, psi, linearization))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        target: IpwTarget,
        trim: Option<f64>,
        _workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapOutcome, EstimationError> {
        let n = problem.nrows;
        // Replicates own a cold workspace. A warm logistic start would change the replicate fit.
        let replicate = |scratch: &mut IpwReplicate, idx: &[usize]| {
            Ok(self.replicate_ate(problem, target, trim, scratch, idx))
        };
        let labels = self.clusters(problem)?;
        // Each worker gathers the design and five row vectors.
        let _gather = crate::util::admit_gather_scratch(
            ctx,
            n,
            problem.design_ncols + 5,
            self.inference.bootstrap_replicates,
        )?;
        crate::util::bootstrap_outcome_by_design(
            self.inference.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: labels.as_deref(),
                series_block: None,
                declared: self.inference.sampling_design,
            },
            self.inference.bootstrap_replicates,
            ctx,
            0x9A17_u64,
            n,
            IpwReplicate::default,
            replicate,
        )
    }

    /// The Hajek effect of one resample of the prepared rows, refitting the propensity model
    /// on it; `None` when the replicate cannot be fit. `idx` selects the rows, whatever their
    /// number: a whole-cluster resample is longer or shorter than the data.
    fn replicate_ate(
        &self,
        problem: &PreparedPropensityProblem,
        target: IpwTarget,
        trim: Option<f64>,
        scratch: &mut IpwReplicate,
        idx: &[usize],
    ) -> Option<f64> {
        let clip = clip_of(problem.overlap);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let m = idx.len();
        let tw = problem.target_weights.as_deref();
        let sampling = problem.sampling_weights.as_ref().map(|s| &*s.weights);
        let IpwReplicate { workspace, x, t, arms, y, clipped, w, s } = scratch;
        x.resize(m * ncols, 0.0);
        for buffer in [&mut *t, &mut *y, &mut *clipped, &mut *w, &mut *s] {
            buffer.resize(m, 0.0);
        }
        crate::util::gather_bootstrap_vector(t, &problem.treatment, idx);
        arms.clear();
        arms.extend(idx.iter().map(|&i| problem.arms[i]));
        crate::util::gather_bootstrap_vector(y, &problem.outcome, idx);
        crate::util::gather_cols(x, &problem.design_matrix, n, ncols, idx);
        let fitted = match sampling {
            Some(full) => {
                crate::util::gather_bootstrap_vector(s, full, idx);
                antecedent_stats::fit_propensity_weighted_in_place(
                    x,
                    m,
                    ncols,
                    t,
                    s,
                    &self.backend,
                    &mut workspace.propensity,
                    &self.glm_options,
                )
            }
            None => antecedent_stats::fit_propensity_in_place(
                x,
                m,
                ncols,
                t,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ),
        };
        if fitted.is_err() {
            return None;
        }
        let raw = &workspace.propensity.scores[..m];
        clipped.copy_from_slice(raw);
        if let Some(c) = clip {
            clamp_scores(clipped, c);
        }
        compute_ipw_weights_into(w, t, clipped, raw, target, trim);
        for (r, &src) in idx.iter().enumerate() {
            if let Some(full_tw) = tw {
                w[r] *= full_tw[src];
            }
            if let Some(full) = sampling {
                w[r] *= full[src];
            }
        }
        if self.covariate_balance {
            balance_arm_weights(x, ncols, t, sampling.map(|_| s.as_slice()), w).ok()?;
        }
        hajek_difference(arms, y, w).ok()
    }
}

/// Per-worker buffers of one IPW bootstrap replicate; resized to each resample's length.
#[derive(Default)]
struct IpwReplicate {
    workspace: PropensityEstimationWorkspace,
    x: Vec<f64>,
    t: Vec<f64>,
    arms: Vec<TreatmentArm>,
    y: Vec<f64>,
    clipped: Vec<f64>,
    w: Vec<f64>,
    s: Vec<f64>,
}

// IPW weights + Hajek estimator (shared by `PropensityWeighting`)
// ---------------------------------------------------------------------------------------------

fn apply_target_weights(weights: &mut [f64], target_weights: Option<&[f64]>) {
    let Some(tw) = target_weights else {
        return;
    };
    for (w, &t) in weights.iter_mut().zip(tw) {
        *w *= t;
    }
}

/// `scores_for_weight` feeds the weight formula (typically clipped); `scores_for_trim` feeds
/// the trim decision (typically the raw, pre-clip scores) — they may be the same slice.
pub(crate) fn compute_ipw_weights(
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) -> Vec<f64> {
    let mut out = vec![0.0; treatment.len()];
    compute_ipw_weights_into(&mut out, treatment, scores_for_weight, scores_for_trim, target, trim);
    out
}

/// In-place form of [`compute_ipw_weights`] for replicate loops with a reused buffer.
pub(crate) fn compute_ipw_weights_into(
    out: &mut [f64],
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) {
    for (slot, ((&t, &e), &raw)) in
        out.iter_mut().zip(treatment.iter().zip(scores_for_weight).zip(scores_for_trim))
    {
        *slot = if let Some(tr) = trim {
            if raw < tr || raw > 1.0 - tr { 0.0 } else { target.weight(t, e) }
        } else {
            target.weight(t, e)
        };
    }
}

/// Compensated `(Σ w·y, Σ w)` of the rows of one arm (`want_treated` selects the treated arm,
/// see [`TreatmentArm`]).
fn arm_weighted_sums(
    arms: &[TreatmentArm],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
) -> (f64, f64) {
    use antecedent_kernels::numeric::CompensatedSum;
    let (mut numerator, mut denominator) = (CompensatedSum::default(), CompensatedSum::default());
    for ((&arm, &y), &w) in arms.iter().zip(outcome).zip(weights) {
        if arm.is_treated() == want_treated {
            numerator.add(w * y);
            denominator.add(w);
        }
    }
    (numerator.total(), denominator.total())
}

/// Hájek difference of the two weighted arm means.
///
/// An arm without positive total weight is an error here: the difference of two means has no
/// value without both arms, and a caller reports the refusal. See [`hajek_weighted_mean`] for
/// the single-arm form, whose unavailable arm is a `NaN` that the replicate loops discard.
pub(crate) fn hajek_difference(
    arms: &[TreatmentArm],
    outcome: &[f64],
    weights: &[f64],
) -> Result<f64, EstimationError> {
    let (num1, den1) = arm_weighted_sums(arms, outcome, weights, true);
    let (num0, den0) = arm_weighted_sums(arms, outcome, weights, false);
    if den1 <= 0.0 || den0 <= 0.0 {
        return Err(EstimationError::data_msg(
            "IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)",
        ));
    }
    Ok(num1 / den1 - num0 / den0)
}

/// Hájek weighted mean of one arm; `NaN` when the arm has no positive total weight, so a
/// per-arm quantity inside a bootstrap or cross-fit loop stays a value whose non-finiteness the
/// loop's own finiteness check reports (see [`hajek_difference`] for the refusing form).
pub(crate) fn hajek_weighted_mean(
    arms: &[TreatmentArm],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
) -> f64 {
    let (num, den) = arm_weighted_sums(arms, outcome, weights, want_treated);
    if den > 0.0 { num / den } else { f64::NAN }
}

/// Logistic nuisance fit and the weighting rule differentiated by the sandwich.
pub(crate) struct HajekPropensity<'a> {
    scores: &'a [f64],
    design: &'a [f64],
    ncols: usize,
    target: IpwTarget,
    clip: Option<f64>,
    /// Sampling weight of each row when the logistic model was fit under them: they scale the
    /// model's score and information, which are the estimating equation being linearized.
    row_mass: Option<&'a [f64]>,
}

/// Linearized influence values of the Hajek difference, one per row, with the logistic
/// propensity-estimation term added.
///
/// The IF is the fixed-weight ratio IF plus `B' I⁻¹ x(T-e)`, where `B`
/// differentiates the two weighted means with respect to logistic coefficients.
/// Its derivative depends on the target and is zero beyond clipping thresholds.
/// Trimming membership is locally fixed; all original rows still contribute to
/// the logistic nuisance fit. Bootstrap refits also capture membership changes.
///
/// `weights` are the full per-row weights (inverse-probability times any sampling and target
/// weights), so the values are already the sampling-weighted scores. The correction term
/// is a per-row score, so it sums within clusters like the rest of the influence.
///
/// # Errors
///
/// A mismatch between rows and the propensity inputs, or singular logistic information.
pub(crate) fn hajek_influence_values(
    arms: &[TreatmentArm],
    outcome: &[f64],
    weights: &[f64],
    propensity: Option<HajekPropensity<'_>>,
) -> Result<Vec<f64>, EstimationError> {
    let n = arms.len();
    if n < 2 || weights.len() != n || outcome.len() != n {
        return Ok(Vec::new());
    }
    let ncols = propensity.as_ref().map_or(0, |p| p.ncols);
    if let Some(p) = &propensity {
        if p.scores.len() != n
            || p.design.len() != n * ncols
            || p.row_mass.is_some_and(|mass| mass.len() != n)
        {
            return Err(EstimationError::data_msg("Hajek propensity design/score length mismatch"));
        }
    }
    let mu1 = hajek_weighted_mean(arms, outcome, weights, true);
    let mu0 = hajek_weighted_mean(arms, outcome, weights, false);
    let (mut sum_w1, mut sum_w0) = (0.0, 0.0);
    for i in 0..n {
        if arms[i].is_treated() {
            sum_w1 += weights[i];
        } else {
            sum_w0 += weights[i];
        }
    }
    if sum_w1 <= 0.0 || sum_w0 <= 0.0 {
        return Ok(Vec::new());
    }
    let nf = n as f64;
    let mut psi = vec![0.0; n];
    for i in 0..n {
        let (w1, w0) = if arms[i].is_treated() { (weights[i], 0.0) } else { (0.0, weights[i]) };
        // Linearized Hajek ratio contributions (E[ψ]=0).
        psi[i] = nf * (w1 / sum_w1) * (outcome[i] - mu1) - nf * (w0 / sum_w0) * (outcome[i] - mu0);
    }

    if let Some(p) = propensity {
        let mut score_mult = vec![0.0; n];
        let mut information_weight = vec![0.0; n];
        let mut derivative_weight = vec![0.0; n];
        for i in 0..n {
            let e = p.scores[i];
            let clipped = p.clip.is_some_and(|c| e <= c || e >= 1.0 - c);
            let log_weight_derivative = if clipped {
                0.0
            } else {
                match (p.target, arms[i].is_treated()) {
                    (IpwTarget::Ate | IpwTarget::Custom, true) => -(1.0 - e),
                    (IpwTarget::Ate | IpwTarget::Custom, false) => e,
                    (IpwTarget::Att, true) | (IpwTarget::Atc, false) => 0.0,
                    (IpwTarget::Att, false) => 1.0,
                    (IpwTarget::Atc, true) => -1.0,
                }
            };
            let mass = p.row_mass.map_or(1.0, |mass| mass[i]);
            score_mult[i] = mass * (arms[i].indicator() - e);
            information_weight[i] = mass * e * (1.0 - e);
            derivative_weight[i] = psi[i] * log_weight_derivative;
        }
        crate::se::add_nuisance_correction(
            &mut psi,
            p.design,
            ncols,
            &score_mult,
            &information_weight,
            &derivative_weight,
            "singular logistic information in Hajek SE",
        )?;
    }
    Ok(psi)
}

/// Independent-rows SE of the Hajek difference from its influence values, with the
/// finite-sample inflation for the `ncols` estimated propensity coefficients. `NaN` when the
/// influence could not be formed.
pub(crate) fn hajek_iid_se(psi: &[f64], ncols: usize) -> f64 {
    let n = psi.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = psi.iter().sum::<f64>() / nf;
    let var = psi.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    // Finite-sample inflation for estimated propensity (p design columns).
    let df = (nf - ncols.max(1) as f64).max(1.0);
    (var / nf * (nf / df)).max(0.0).sqrt()
}

/// Hajek analytic SE from stacked logistic and weighted-mean estimating equations
/// ([`hajek_influence_values`] under independent rows).
///
/// # Errors
///
/// As [`hajek_influence_values`].
#[cfg(test)]
pub(crate) fn hajek_influence_se(
    arms: &[TreatmentArm],
    outcome: &[f64],
    weights: &[f64],
    propensity: Option<HajekPropensity<'_>>,
) -> Result<f64, EstimationError> {
    let ncols = propensity.as_ref().map_or(0, |p| p.ncols);
    let psi = hajek_influence_values(arms, outcome, weights, propensity)?;
    Ok(hajek_iid_se(&psi, ncols))
}

#[cfg(test)]
mod tests {
    use crate::arm::TreatmentArm;
    use crate::overlap::IpwTarget;
    use crate::propensity::ipw_oracle::oracle_fit;
    use crate::util::solve_spd;

    #[test]
    fn hajek_se_matches_refitted_case_weight_derivatives() {
        let mut t = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        for group in 0..5 {
            for row in 0..6 {
                let covariate = f64::from(group) - 2.0;
                let treated = f64::from(row <= group);
                x.push(covariate);
                t.push(treated);
                y.push(covariate + treated * (2.0 + covariate) + 0.2 * f64::from(row * row));
            }
        }
        let n = t.len();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&x);
        for target in [IpwTarget::Att, IpwTarget::Atc, IpwTarget::Ate] {
            for (clip, trim) in [(None, None), (Some(0.25), None), (None, Some(0.2))] {
                let mut mass = vec![1.0; n];
                let (_, e) = oracle_fit(&t, &y, &x, &mass, target, clip, trim);
                let clipped: Vec<f64> =
                    e.iter().map(|&p| clip.map_or(p, |c| p.clamp(c, 1.0 - c))).collect();
                let w = super::compute_ipw_weights(&t, &clipped, &e, target, trim);
                let actual = super::hajek_influence_se(
                    &TreatmentArm::classify(&t),
                    &y,
                    &w,
                    Some(super::HajekPropensity {
                        scores: &e,
                        design: &design,
                        ncols: 2,
                        target,
                        clip,
                        row_mass: None,
                    }),
                )
                .unwrap();
                let mut derivatives = Vec::new();
                for row in 0..n {
                    mass[row] = 1.0 + 1e-5;
                    let plus = oracle_fit(&t, &y, &x, &mass, target, clip, trim).0;
                    mass[row] = 1.0 - 1e-5;
                    let minus = oracle_fit(&t, &y, &x, &mass, target, clip, trim).0;
                    mass[row] = 1.0;
                    derivatives.push((plus - minus) / 2e-5);
                }
                let nf = n as f64;
                let expected = (derivatives.iter().map(|d| d * d).sum::<f64>() * nf / (nf - 1.0)
                    * nf
                    / (nf - 2.0))
                    .sqrt();
                assert!(
                    (actual - expected).abs() < 1e-7,
                    "target={target:?} clip={clip:?} trim={trim:?} actual={actual} oracle={expected}"
                );
            }
        }
    }

    #[test]
    fn spd_solve_rejects_badly_scaled_near_singular_matrix() {
        // Rows are nearly parallel at large magnitude — the shape an unnormalized
        // G = S'S / n takes when two propensity-score residual columns are collinear.
        // After one elimination step the remaining pivot is ~1e-6 in absolute terms:
        // comfortably above a fixed 1e-14 absolute threshold (which would wrongly accept
        // this and hand back a garbage adjustment), but far below 1e-12 * scale (~1e-2)
        // once the tolerance is scaled to the matrix's own magnitude (~1e10); the kernel
        // Cholesky refuses it because the second pivot cancels below 64 eps of its diagonal.
        let a = [1e10, 1e10, 1e10, 1e10 + 1e-6];
        let b = [1.0, 1.0];
        assert!(solve_spd(&a, &b, 2).is_none());
    }

    #[test]
    fn spd_solve_still_solves_well_scaled_system() {
        // Sanity check that the relative tolerance doesn't reject ordinary,
        // well-conditioned systems: [4,1;1,3] x = [5,4] ⇒ x = [1,1].
        let a = [4.0, 1.0, 1.0, 3.0];
        let b = [5.0, 4.0];
        let x = solve_spd(&a, &b, 2).expect("well-conditioned");
        assert!((x[0] - 1.0).abs() < 1e-12, "x0={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x1={}", x[1]);
    }

    #[test]
    fn spd_solve_rejects_degenerate_scale() {
        // All-zero diagonal (and a non-finite one) leave no usable scale reference.
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1.0];
        assert!(solve_spd(&a, &b, 2).is_none());

        let a = [f64::INFINITY, 0.0, 0.0, 1.0];
        let b = [1.0, 1.0];
        assert!(solve_spd(&a, &b, 2).is_none());
    }

    #[test]
    fn spd_solve_boundary_is_the_kernel_pivot_floor() {
        // The kernel refuses a pivot that cancels below 64 eps (~1.4e-14) of its diagonal. The
        // former LU used 1e-12 of the largest diagonal, so the band between the two is the
        // only place the verdict moved: a relative pivot of 1e-13 was refused, now accepted.
        let above = [1.0, 1.0, 1.0, 1.0 + 1e-13];
        assert!(solve_spd(&above, &[1.0, 1.0], 2).is_some());
        let below = [1.0, 1.0, 1.0, 1.0 + 1e-15];
        assert!(solve_spd(&below, &[1.0, 1.0], 2).is_none());
    }
}
