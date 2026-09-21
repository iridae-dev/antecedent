//! Linear adjustment ATE estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, Intervention, TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, FirstStageDiagnostics, LassoOptions,
    LeastSquaresWorkspace, MEstimateOptions, fit_huber_m, fit_lasso_with_ones_column, fit_ridge,
    form_xtx, invert_square, predict_lasso, ridge_gram_inverse,
};

use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::prepare::{
    require_adjustment_shaped, treatment_contrast, validate_ate_query_with_targets,
};
use crate::se::{AnalyticSeKind, residual_sandwich_coef_se, score_sandwich_coef_se};

/// Prepared estimation problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedEstimationProblem {
    /// Compiled design.
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
    /// Target population (ATT/ATC use g-computation over the arm’s covariate law).
    pub target_population: TargetPopulation,
    /// Complete-case treatment values (aligned with design rows).
    pub treatment: Arc<[f64]>,
    /// Active treatment level.
    pub active: f64,
    /// Control treatment level.
    pub control: f64,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct EstimationWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Point estimate with uncertainty.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[allow(clippy::struct_excessive_bools)]
pub struct EffectEstimate {
    /// Reported scalar.
    ///
    /// For average-effect claims this is the ATE contrast. For `cell.aipw`
    /// joint cells it is the requested interventional **level**
    /// (`E[Y | do(A=a,…)]` or a single-threshold `F_a(c)`), not a contrast.
    /// Testing this field against zero is only valid when it is a contrast.
    /// Family p-values on joint cells use [`Self::family_contrast`].
    pub ate: f64,
    /// Analytic IID standard error (homoskedastic).
    pub se_analytic: f64,
    /// Bootstrap standard error (if requested and enough survivors).
    pub se_bootstrap: Option<f64>,
    /// Successful bootstrap replicates when bootstrap was requested.
    pub bootstrap_replicates_ok: Option<u32>,
    /// Soft-failed bootstrap replicates when bootstrap was requested.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Bootstrap loop observed cooperative cancellation (partial replicates).
    pub bootstrap_cancelled: bool,
    /// Adaptive bootstrap early-stop (SE relative change).
    pub bootstrap_early_stopped: bool,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Propensity overlap diagnostics when computed.
    pub overlap_report: Option<OverlapReport>,
    /// Weak-instrument diagnostic (IV / 2SLS estimators only); `None` for every other
    /// estimator family. When `f_statistic < 10`, [`Self::se_analytic`] is NaN.
    pub first_stage_diagnostics: Option<FirstStageDiagnostics>,
    /// Estimated retained-memory cost of fitted scratch (bytes), when known.
    pub retained_memory_bytes: Option<u64>,
    /// Cross-fitted AIPW score table when the estimator exported one.
    pub score_table: Option<crate::scores::ScoreTable>,
    /// Simultaneous score-family bands and target-local support.
    pub score_inference: Option<crate::scores::ScoreInference>,
    /// Effects under declared canonical orientation scenarios, in estimand order.
    pub scenario_effects: Option<Arc<[f64]>>,
    /// Simultaneous batch interval (lower, upper, level).
    ///
    /// Joint-cell batches keep this on cell **levels**. Contrast-family bands
    /// live on [`Self::family_contrast_interval`].
    pub simultaneous_interval: Option<(f64, f64, f64)>,
    /// BH and BY adjusted p-values for the fixed batch family.
    pub adjusted_p_values: Option<(f64, f64)>,
    /// Declared family contrast `(value, se)` when batch inference tested a
    /// contrast rather than [`Self::ate`]. `None` for average-effect claims
    /// (`ate` is already the contrast) and when no cell contrast was declared.
    pub family_contrast: Option<(f64, f64)>,
    /// Simultaneous interval `(lower, upper, level)` for [`Self::family_contrast`].
    ///
    /// Formed from contrast-family max-t, not the cell-level covariance used
    /// by [`Self::simultaneous_interval`]. `None` when no contrast was declared
    /// or contrast max-t could not form.
    pub family_contrast_interval: Option<(f64, f64, f64)>,
    /// Simultaneous intervals for canonical scenarios; not bounds on all completions.
    pub scenario_intervals: Option<Arc<[(f64, f64)]>>,
    /// Joint IF covariance across arms / thresholds / cells / claims.
    pub joint_covariance: Option<crate::joint_if::JointCovariance>,
    /// Per-threshold `F_a(c) = 1 - P(Y(a) > c)` after monotone rearrangement.
    pub exceedance_cdf: Option<Arc<[f64]>>,
    /// `true` when `exceedance_cdf` / grid means were isotonically rearranged.
    /// Joint covariance and score-inference bands always describe the raw AIPW scores.
    pub monotone_rearranged: bool,
    /// Whether an additive joint path makes the interaction contrast structurally zero.
    pub interaction_structurally_zero: bool,
    /// Whether the fitted mechanisms on every treatment → outcome path admit no
    /// effect modification, so the reported per-unit effects are equal by
    /// construction of the selected mechanism families, not by measurement.
    pub unit_effects_homogeneous: bool,
    /// Per-row influence for the reported scalar (shared-row joint IF / envelopes).
    pub influence: Option<Arc<[f64]>>,
    /// Held-out outcome R².
    pub outcome_oof_r2: Option<f64>,
    /// Held-out treatment probability log loss.
    pub treatment_oof_logloss: Option<f64>,
    /// Number of nuisance cross-fitting folds.
    pub crossfit_folds: Option<usize>,
    /// Master seed used for nuisance cross-fitting.
    pub crossfit_seed: Option<u64>,
    /// Actual fitted learner identities, in fold order (control, treated, propensity
    /// for AIPW; outcome folds then treatment folds for PLR; final CATE fit last).
    pub learner_provenance: Vec<antecedent_learn::LearnerProvenance>,
    /// Point E-value for the reported effect when a named no-latent premise is in force.
    pub evalue: Option<f64>,
    /// Threshold [`Self::evalue`] was judged against, when it came from an
    /// E-value refuter that reported a pass/fail verdict. `None` when no refuter
    /// ran, or when the E-value is an attached premise diagnostic with no gate.
    pub evalue_threshold: Option<f64>,
    /// Candidate-selection screen recorded on a batch family (artifact payload).
    pub candidate_selection: Option<CandidateSelectionRecord>,
    /// Circular-block geometry of a one-series block-bootstrap SE (the block
    /// length the estimator chose and the lag-aligned rows it resampled).
    pub block_resampling: Option<BlockResampling>,
    /// Observations that actually informed the fit (complete-case / trimmed /
    /// lag-aligned rows). Distinct from the input snapshot row count.
    pub n_obs: Option<u64>,
    /// Analytic SE formula behind [`Self::se_analytic`], recorded by the
    /// estimator that computed it. `None` when the SE is not an
    /// estimator-configured analytic kind (influence-function envelopes,
    /// bootstrap-only paths, posterior summaries).
    pub se_kind: Option<crate::se::AnalyticSeKind>,
    /// Circular-block SE family behind a one-series [`Self::se_bootstrap`],
    /// recorded by the path that ran the circular-block bootstrap.
    pub block_family: Option<crate::temporal_block::CircularBlockFamily>,
    /// Per-row CATE when a heterogeneous-effect estimator produced one.
    pub cate: Option<Arc<[f64]>>,
    /// Retained portable CATE model, separate from the marginal estimate.
    pub fitted_effect: Option<Arc<crate::FittedEffect>>,
    /// Per-row CATE standard errors, when a licensed pointwise formula produced them.
    ///
    /// Linear DR-Learner finals use the HC0 sandwich of the orthogonal scores.
    /// Honest forests use the mean of two-sample leaf variances. Nonlinear or
    /// penalized CATE regressions withhold this field rather than inventing an SE.
    pub cate_se: Option<Arc<[f64]>>,
}

/// Circular-block geometry an estimator used for its one-series bootstrap SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockResampling {
    /// Circular-block length in lag-aligned rows.
    pub block_length: usize,
    /// Lag-aligned rows resampled.
    pub rows: usize,
    /// Bartlett kernel-bias factor of the target influence at `block_length`
    /// (`antecedent_estimate::kernel_bias_scale`), applied to the SE together
    /// with the fixed-b factor.
    pub kernel_bias: f64,
}

/// Screen / estimate split recorded on a batch result artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSelectionRecord {
    /// Caller-stable screen run id, or `"unrecorded"`.
    pub screen_id: Arc<str>,
    /// Selection rule wire name (`max_t`, `bh`, `by`, `unrecorded`).
    pub procedure: Arc<str>,
    /// Winning query index in the supplied family, or `None` when ranking is unavailable.
    pub winner_index: Option<usize>,
    /// Family size.
    pub family_size: usize,
    /// Row indexes used to screen candidates.
    pub screen_rows: Arc<[u32]>,
    /// Row indexes used to estimate the surfaced family.
    pub estimate_rows: Arc<[u32]>,
    /// Whether the two halves are disjoint.
    pub disjoint: bool,
}

impl EffectEstimate {
    /// Construct a point estimate without bootstrap accounting.
    #[must_use]
    pub fn new(
        ate: f64,
        se_analytic: f64,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
    ) -> Self {
        Self {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap,
            overlap_report: None,
            first_stage_diagnostics: None,
            retained_memory_bytes: None,
            score_table: None,
            score_inference: None,
            scenario_effects: None,
            simultaneous_interval: None,
            adjusted_p_values: None,
            family_contrast: None,
            family_contrast_interval: None,
            scenario_intervals: None,
            joint_covariance: None,
            exceedance_cdf: None,
            monotone_rearranged: false,
            interaction_structurally_zero: false,
            unit_effects_homogeneous: false,
            influence: None,
            outcome_oof_r2: None,
            treatment_oof_logloss: None,
            crossfit_folds: None,
            crossfit_seed: None,
            learner_provenance: Vec::new(),
            evalue: None,
            evalue_threshold: None,
            candidate_selection: None,
            block_resampling: None,
            n_obs: None,
            se_kind: None,
            block_family: None,
            cate: None,
            fitted_effect: None,
            cate_se: None,
        }
    }

    /// Record the analysis sample size that produced this interval.
    #[must_use]
    pub fn with_n_obs(mut self, n_obs: u64) -> Self {
        self.n_obs = Some(n_obs);
        self
    }

    /// Full constructor (required outside this crate because the type is `#[non_exhaustive]`).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        ate: f64,
        se_analytic: f64,
        se_bootstrap: Option<f64>,
        bootstrap_replicates_ok: Option<u32>,
        bootstrap_replicates_failed: Option<u32>,
        bootstrap_cancelled: bool,
        bootstrap_early_stopped: bool,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
        overlap_report: Option<OverlapReport>,
        retained_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            ate,
            se_analytic,
            se_bootstrap,
            bootstrap_replicates_ok,
            bootstrap_replicates_failed,
            bootstrap_cancelled,
            bootstrap_early_stopped,
            assumptions,
            overlap,
            overlap_report,
            first_stage_diagnostics: None,
            retained_memory_bytes,
            score_table: None,
            score_inference: None,
            scenario_effects: None,
            simultaneous_interval: None,
            adjusted_p_values: None,
            family_contrast: None,
            family_contrast_interval: None,
            scenario_intervals: None,
            joint_covariance: None,
            exceedance_cdf: None,
            monotone_rearranged: false,
            interaction_structurally_zero: false,
            unit_effects_homogeneous: false,
            influence: None,
            outcome_oof_r2: None,
            treatment_oof_logloss: None,
            crossfit_folds: None,
            crossfit_seed: None,
            learner_provenance: Vec::new(),
            evalue: None,
            evalue_threshold: None,
            candidate_selection: None,
            block_resampling: None,
            n_obs: None,
            se_kind: None,
            block_family: None,
            cate: None,
            fitted_effect: None,
            cate_se: None,
        }
    }

    /// Attach a weak-instrument diagnostic (IV / 2SLS estimators only).
    #[must_use]
    pub fn with_first_stage_diagnostics(
        mut self,
        diagnostics: Option<FirstStageDiagnostics>,
    ) -> Self {
        self.first_stage_diagnostics = diagnostics;
        self
    }

    /// Attach propensity overlap diagnostics (propensity-based estimators only).
    #[must_use]
    pub fn with_overlap_report(mut self, overlap_report: Option<OverlapReport>) -> Self {
        self.overlap_report = overlap_report;
        self
    }

    /// Attach a cross-fitted score table.
    #[must_use]
    pub fn with_score_table(mut self, table: Option<crate::scores::ScoreTable>) -> Self {
        self.score_table = table;
        self
    }

    /// Attach joint IF covariance.
    #[must_use]
    pub fn with_joint_covariance(mut self, cov: Option<crate::joint_if::JointCovariance>) -> Self {
        self.joint_covariance = cov;
        self
    }

    /// Attach per-threshold CDF values `F_a(c)`.
    #[must_use]
    pub fn with_exceedance_cdf(mut self, cdf: Option<Arc<[f64]>>) -> Self {
        self.exceedance_cdf = cdf;
        self
    }

    /// Record whether exceedance means were isotonically rearranged.
    ///
    /// When `true`, [`Self::simultaneous_interval`] is cleared: those intervals
    /// must not mix rearranged means with raw covariance.
    #[must_use]
    pub fn with_monotone_rearranged(mut self, rearranged: bool) -> Self {
        self.monotone_rearranged = rearranged;
        if rearranged {
            self.simultaneous_interval = None;
        }
        self
    }

    /// Mark that a joint interaction contrast is structurally zero on this path.
    #[must_use]
    pub fn with_interaction_structurally_zero(mut self, zero: bool) -> Self {
        self.interaction_structurally_zero = zero;
        self
    }

    /// Mark that the reported per-unit effects are homogeneous by construction of
    /// the selected mechanism families (no effect modification is representable).
    #[must_use]
    pub fn with_unit_effects_homogeneous(mut self, homogeneous: bool) -> Self {
        self.unit_effects_homogeneous = homogeneous;
        self
    }

    /// Attach a per-row influence sequence for the reported scalar.
    #[must_use]
    pub fn with_influence(mut self, influence: Option<Arc<[f64]>>) -> Self {
        self.influence = influence;
        self
    }

    /// Record the analytic SE formula behind [`Self::se_analytic`].
    #[must_use]
    pub fn with_se_kind(mut self, se_kind: crate::se::AnalyticSeKind) -> Self {
        self.se_kind = Some(se_kind);
        self
    }

    /// Attach a per-row CATE (`DrLearner` / `CausalForest`).
    #[must_use]
    pub fn with_cate(mut self, cate: Option<Arc<[f64]>>) -> Self {
        self.cate = cate;
        self
    }

    /// Attach licensed per-row CATE standard errors, or withhold them.
    #[must_use]
    pub fn with_cate_se(mut self, cate_se: Option<Arc<[f64]>>) -> Self {
        self.cate_se = cate_se;
        self
    }

    /// Record the circular-block SE family behind a one-series [`Self::se_bootstrap`].
    #[must_use]
    pub fn with_block_family(mut self, family: crate::temporal_block::CircularBlockFamily) -> Self {
        self.block_family = Some(family);
        self
    }

    /// Attach a point E-value for a named no-latent / unmeasured-confounding premise.
    #[must_use]
    pub fn with_evalue(mut self, evalue: Option<f64>) -> Self {
        self.evalue = evalue;
        self
    }

    /// Attach the pass threshold the E-value refuter judged [`Self::evalue`] against.
    #[must_use]
    pub fn with_evalue_threshold(mut self, threshold: Option<f64>) -> Self {
        self.evalue_threshold = threshold;
        self
    }

    /// Attach the declared family contrast `(value, se)` used for batch FDR.
    #[must_use]
    pub fn with_family_contrast(mut self, contrast: Option<(f64, f64)>) -> Self {
        self.family_contrast = contrast;
        if contrast.is_none() {
            self.family_contrast_interval = None;
        }
        self
    }

    /// Attach the contrast-family simultaneous interval `(lower, upper, level)`.
    #[must_use]
    pub fn with_family_contrast_interval(mut self, interval: Option<(f64, f64, f64)>) -> Self {
        self.family_contrast_interval = interval;
        self
    }

    /// Attach the estimated retained-memory cost of fitted scratch (bytes), when known.
    #[must_use]
    pub fn with_retained_memory_bytes(mut self, retained_memory_bytes: Option<u64>) -> Self {
        self.retained_memory_bytes = retained_memory_bytes;
        self
    }

    /// Attach bootstrap SE accounting (or clear when bootstrap was skipped).
    #[must_use]
    pub fn with_bootstrap(mut self, boot: Option<crate::util::BootstrapSeResult>) -> Self {
        match boot {
            None => {
                self.se_bootstrap = None;
                self.bootstrap_replicates_ok = None;
                self.bootstrap_replicates_failed = None;
                self.bootstrap_cancelled = false;
                self.bootstrap_early_stopped = false;
            }
            Some(b) => {
                self.se_bootstrap = b.se;
                self.bootstrap_replicates_ok = Some(b.replicates_ok);
                self.bootstrap_replicates_failed = Some(b.replicates_failed);
                self.bootstrap_cancelled = b.cancelled;
                self.bootstrap_early_stopped = b.early_stopped;
            }
        }
        self
    }
}

/// Linear fit family for [`LinearAdjustmentAte`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearFitKind {
    /// Ordinary least squares.
    Ols,
    /// Ridge with penalty `lambda` (intercept unpenalized when constant).
    ///
    /// The treatment coefficient is penalized, so the point estimate is shrunk and an
    /// interval around it has no bias term: no analytic SE is published (`se_analytic` is
    /// NaN); use bootstrap. The exported influence function is the ridge one.
    Ridge {
        /// Ridge penalty λ.
        lambda: f64,
    },
    /// Lasso with penalty `lambda`.
    ///
    /// Analytic SE is permanently omitted: classical / active-set sandwich SEs are
    /// invalid after selection, and debiased Lasso changes the point estimator.
    /// Use bootstrap (`bootstrap_replicates > 0`); `se_analytic` is NaN.
    Lasso {
        /// Lasso penalty λ.
        lambda: f64,
    },
    /// Huber M-estimation with tuning constant `c`.
    ///
    /// Analytic SE is the M-estimator's sandwich (Huber 1981), not the OLS formula on Huber
    /// residuals. The MAD scale is treated as known.
    Huber {
        /// Huber tuning constant (default 1.345).
        c: f64,
    },
}

impl Default for LinearFitKind {
    fn default() -> Self {
        Self::Ols
    }
}

/// Linear adjustment estimator for backdoor ATE.
#[derive(Clone, Debug)]
pub struct LinearAdjustmentAte {
    /// Backend.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be explicit in).
    pub overlap: OverlapPolicy,
    /// Analytic SE estimator (default homoskedastic).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids (length = prepared `nrows`) for cluster / panel SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids for [`AnalyticSeKind::Multiway`].
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels (length = prepared `nrows`) for panel HAC.
    pub panel_times: Option<Vec<i64>>,
    /// Linear fit family (default OLS).
    pub fit_kind: LinearFitKind,
    /// Registry for named [`TargetPopulation::Predicate`] selections.
    pub population_registry: Option<antecedent_core::PopulationRegistry>,
}

impl Default for LinearAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override, OLS.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
            fit_kind: LinearFitKind::Ols,
            population_registry: None,
        }
    }

    /// Set the dense linear-algebra backend used for the OLS / ridge / Huber fits.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE
    /// (`NaN` when [`Self::with_fit_kind`] is [`LinearFitKind::Lasso`] or
    /// [`LinearFitKind::Ridge`]).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. `prepare` requires [`OverlapPolicy::ExplicitOverride`] since
    /// this is a regression (not propensity-based) path.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the analytic standard-error kind (default [`AnalyticSeKind::Homoskedastic`]).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids (length must equal the prepared design's row count) for
    /// [`AnalyticSeKind::Cluster`] / panel SE.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension) for
    /// [`AnalyticSeKind::Multiway`].
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels (length must equal the prepared design's row count) for
    /// panel HAC standard errors.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

    /// Set the linear fit family (default [`LinearFitKind::Ols`]).
    ///
    /// **Lasso trap**: with `LinearFitKind::Lasso { .. }`, the analytic SE is permanently
    /// `NaN` — classical / active-set sandwich SEs are invalid after selection, and debiased
    /// Lasso changes the point estimator. Pair this with a nonzero
    /// [`Self::with_bootstrap_replicates`] to get a usable SE; this setter stays dumb and
    /// does not enforce that pairing.
    #[must_use]
    pub const fn with_fit_kind(mut self, fit_kind: LinearFitKind) -> Self {
        self.fit_kind = fit_kind;
        self
    }

    /// Registry used to resolve named [`TargetPopulation::Predicate`] selections.
    #[must_use]
    pub fn with_population_registry(
        mut self,
        registry: antecedent_core::PopulationRegistry,
    ) -> Self {
        self.population_registry = Some(registry);
        self
    }

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// # Errors
    ///
    /// Missing columns, unsupported query options, type errors, or overlap policy not set.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "LinearAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        require_adjustment_shaped(
            estimand,
            "LinearAdjustmentAte expects an adjustment-shaped estimand",
        )?;
        validate_ate_query_with_targets(query)?;
        let treatment = query.treatment;
        let outcome = query.outcome;
        let (active, control, treatment_delta) = treatment_contrast(&query.active, &query.control)?;

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let mut row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        crate::prepare::intersect_predicate_mask(
            &mut row_mask,
            &query.target_population,
            data.row_count(),
            self.population_registry.as_ref(),
        )?;
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
        Ok(PreparedEstimationProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            treatment_delta,
            target_population: query.target_population.clone(),
            treatment: Arc::from(t),
            active,
            control,
        })
    }

    /// Fit ATE with optional IID bootstrap.
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Point estimate + analytic SE only (no bootstrap).
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit_point(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let fit = self.fit_coefficients(problem, workspace)?;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let ate = gcomp_or_coef_ate(problem, &fit.coefficients, t_col)?;
        let nrows = problem.design.nrows;
        let (se_coef, influence) = self.coefficient_se_and_influence(problem, &fit, t_col)?;
        let se_analytic = se_coef * problem.treatment_delta.abs();

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(nrows).unwrap_or(u64::MAX))
            .with_se_kind(self.se_kind)
            .with_influence(influence.map(Arc::from)))
    }

    /// Treatment-coefficient SE and influence function of the estimator that actually ran.
    ///
    /// OLS: `RSS/(n−p)·[(XᵀX)⁻¹]_tt` or the residual sandwich. Huber: the M-estimator's
    /// sandwich, bread `Σ ψ′ x xᵀ` and scores `s·ψ(r/s)`. Ridge and Lasso publish no analytic
    /// SE (shrinkage bias is not in a sampling variance, and post-selection inference is
    /// invalid); their influence function is the ridge one, resp. none, never the OLS one.
    fn coefficient_se_and_influence(
        &self,
        problem: &PreparedEstimationProblem,
        fit: &LinearFit,
        t_col: usize,
    ) -> Result<(f64, Option<Vec<f64>>), EstimationError> {
        let x = &problem.design.matrix;
        let nrows = problem.design.nrows;
        let ncols = problem.design.ncols;
        let n = nrows as f64;
        let p = ncols as f64;
        let delta = problem.treatment_delta;
        match &fit.variance {
            FitVariance::Ols => {
                // `(XᵀX)⁻¹` once: the homoskedastic SE reads its treatment diagonal and
                // the influence function its treatment row.
                let xtx_inv = gram_inverse(x, nrows, ncols);
                let se_coef = if let Some(se) = residual_sandwich_coef_se(
                    self.se_kind,
                    x,
                    nrows,
                    ncols,
                    &fit.residuals,
                    t_col,
                    self.cluster_ids.as_deref(),
                    self.multiway_ids.as_deref(),
                    self.panel_times.as_deref(),
                )? {
                    se
                } else {
                    let sigma2 = fit.rss / (n - p).max(1.0);
                    xtx_inv.as_ref().map_or(f64::NAN, |inv| {
                        (sigma2 * inv[t_col * ncols + t_col].max(0.0)).sqrt()
                    })
                };
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    &fit.residuals,
                    delta,
                    xtx_inv.as_deref(),
                );
                Ok((se_coef, Some(influence)))
            }
            FitVariance::Ridge { lambda } => {
                let bread = ridge_gram_inverse(x, nrows, ncols, *lambda);
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    &fit.residuals,
                    delta,
                    bread.as_deref(),
                );
                Ok((f64::NAN, Some(influence)))
            }
            FitVariance::Lasso => Ok((f64::NAN, None)),
            FitVariance::Huber { scores, curvature } => {
                let bread = curvature_gram_inverse(x, nrows, ncols, curvature);
                let mean_curvature = curvature.iter().sum::<f64>() / n;
                let se_coef = if let Some(se) = score_sandwich_coef_se(
                    self.se_kind,
                    x,
                    nrows,
                    ncols,
                    scores,
                    curvature,
                    t_col,
                    self.cluster_ids.as_deref(),
                    self.multiway_ids.as_deref(),
                    self.panel_times.as_deref(),
                )? {
                    se
                } else if mean_curvature > 0.0 && n > p {
                    // Huber (1981, §7.10): Var(β̂) = κ² · [Σ s²ψ²/(n−p)] / m² · (XᵀX)⁻¹ with
                    // m = mean ψ′ and κ = 1 + (p/n) var(ψ′)/m². For an indicator ψ′,
                    // var(ψ′) = m(1−m).
                    let kappa = 1.0 + (p / n) * (1.0 - mean_curvature) / mean_curvature;
                    let tau2 = scores.iter().map(|e| e * e).sum::<f64>()
                        / (n - p)
                        / (mean_curvature * mean_curvature);
                    gram_inverse(x, nrows, ncols).map_or(f64::NAN, |inv| {
                        (kappa * kappa * tau2 * inv[t_col * ncols + t_col].max(0.0)).sqrt()
                    })
                } else {
                    f64::NAN
                };
                let influence = treatment_coef_influence(
                    x,
                    nrows,
                    ncols,
                    t_col,
                    scores,
                    delta,
                    bread.as_deref(),
                );
                Ok((se_coef, Some(influence)))
            }
        }
    }

    /// Attach bootstrap SE onto a point estimate (progressive uncertainty stage).
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            let t_col = problem
                .design
                .treatment_column()
                .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };
        Ok(point.with_bootstrap(boot))
    }

    fn fit_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
    ) -> Result<LinearFit, EstimationError> {
        let x = &problem.design.matrix;
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let y = &problem.design.outcome;
        match self.fit_kind {
            LinearFitKind::Ols => {
                let fit = problem
                    .design
                    .fit_ols(&self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                Ok(LinearFit {
                    coefficients: fit.coefficients,
                    residuals: fit.residuals,
                    rss: fit.rss,
                    variance: FitVariance::Ols,
                })
            }
            LinearFitKind::Ridge { lambda } => {
                let fit = fit_ridge(x, n, p, y, lambda, &self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                Ok(LinearFit {
                    coefficients: fit.coefficients,
                    residuals: fit.residuals,
                    rss: fit.rss,
                    variance: FitVariance::Ridge { lambda },
                })
            }
            LinearFitKind::Lasso { lambda } => {
                let fit = fit_lasso_with_ones_column(
                    x,
                    n,
                    p,
                    y,
                    &LassoOptions { lambda, fit_intercept: true, ..LassoOptions::default() },
                )
                .map_err(EstimationError::from)?;
                require_converged(fit.converged, LASSO_UNCONVERGED)?;
                // Design keeps an all-ones column; stitch intercept back for g-computation.
                let mut coefficients = Vec::with_capacity(p);
                coefficients.push(fit.intercept);
                coefficients.extend_from_slice(&fit.coefficients);
                let pred = if fit.coefficients.is_empty() {
                    vec![fit.intercept; n]
                } else {
                    predict_lasso(&fit, &x[n..], n, p - 1).map_err(EstimationError::from)?
                };
                let mut residuals = vec![0.0; n];
                let mut rss = 0.0;
                for r in 0..n {
                    let e = y[r] - pred[r];
                    residuals[r] = e;
                    rss += e * e;
                }
                // Permanent policy: no analytic SE for Lasso (bootstrap only).
                Ok(LinearFit { coefficients, residuals, rss, variance: FitVariance::Lasso })
            }
            LinearFitKind::Huber { c } => {
                let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
                let fit = fit_huber_m(x, n, p, y, &opts, &self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                // An unconverged IRLS run is a snapshot of an unfinished reweighting
                // sequence, not the M-estimator's fixed point; publishing it — and an
                // analytic SE derived from it — would attach robust-estimand semantics
                // to a number that has none. Mirrors the GAM backfit gate in
                // `response.rs::fit_additive`.
                require_converged(fit.converged, HUBER_UNCONVERGED)?;
                let mut residuals = vec![0.0; n];
                let mut rss = 0.0;
                for r in 0..n {
                    let mut pred = 0.0;
                    for col in 0..p {
                        pred += x[col * n + r] * fit.coefficients[col];
                    }
                    let e = y[r] - pred;
                    residuals[r] = e;
                    rss += e * e;
                }
                // Estimating equation `Σ x_i · s·ψ_c(r_i/s) = 0`: the score is the residual
                // clipped at `c·s`, and its derivative in the linear predictor is the
                // indicator of the unclipped region.
                let bound = c * fit.scale;
                let scores: Vec<f64> = residuals.iter().map(|r| r.clamp(-bound, bound)).collect();
                let curvature: Vec<f64> =
                    residuals.iter().map(|r| f64::from(u8::from(r.abs() <= bound))).collect();
                Ok(LinearFit {
                    coefficients: fit.coefficients,
                    residuals,
                    rss,
                    variance: FitVariance::Huber { scores, curvature },
                })
            }
        }
    }

    /// ATE from the configured linear fit on a row-resample of a prepared design.
    ///
    /// `row_src[r]` copies design row `row_src[r]` onto output row `r`. Length must
    /// equal `problem.design.nrows`. For `AllObserved` / `Predicate` the ATE is
    /// `β_T · Δ` and does not reread treatment labels.
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
    ) -> Result<f64, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        if row_src.len() != n {
            return Err(EstimationError::data_msg(
                "row resample length must match the prepared design",
            ));
        }
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        self.ate_on_row_indices_into(problem, workspace, row_src, &mut x_boot, &mut y_boot)
    }

    /// [`Self::ate_on_row_indices`] using caller-owned gather buffers.
    ///
    /// Here `row_src` may be shorter than the design (a sub-window resample); the
    /// buffers must hold `row_src.len()` rows (`x_boot`: `row_src.len() · ncols`).
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices_into(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<f64, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let coefs = self.fit_resampled_coefficients(problem, workspace, row_src, x_boot, y_boot)?;
        gcomp_or_coef_ate(problem, &coefs, t_col)
    }

    fn fit_resampled_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let rows = problem.design.nrows;
        let p = problem.design.ncols;
        // Output rows follow the row map, which may be a sub-window of the design
        // (a multi-design shared time window); every source row must exist.
        let n = row_src.len();
        if y_boot.len() != n || x_boot.len() != n * p {
            return Err(EstimationError::data_msg("row resample buffers must match the row map"));
        }
        for (r, &src) in row_src.iter().enumerate() {
            if src >= rows {
                return Err(EstimationError::data_msg("row resample index out of range"));
            }
            y_boot[r] = problem.design.outcome[src];
            for c in 0..p {
                x_boot[c * n + r] = problem.design.matrix[c * rows + src];
            }
        }
        match self.fit_kind {
            LinearFitKind::Ols => {
                let fit = self
                    .backend
                    .least_squares(x_boot, n, p, y_boot, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                Ok(fit.coefficients)
            }
            LinearFitKind::Ridge { lambda } => {
                let fit =
                    fit_ridge(x_boot, n, p, y_boot, lambda, &self.backend, &mut workspace.ols)
                        .map_err(EstimationError::from)?;
                Ok(fit.coefficients)
            }
            LinearFitKind::Lasso { lambda } => {
                let fit = fit_lasso_with_ones_column(
                    x_boot,
                    n,
                    p,
                    y_boot,
                    &LassoOptions { lambda, fit_intercept: true, ..LassoOptions::default() },
                )
                .map_err(EstimationError::from)?;
                // An unconverged replicate is a snapshot, not a draw from the estimator's
                // sampling distribution; the caller counts it as a failed replicate.
                require_converged(fit.converged, LASSO_UNCONVERGED)?;
                let mut coefficients = Vec::with_capacity(p);
                coefficients.push(fit.intercept);
                coefficients.extend_from_slice(&fit.coefficients);
                Ok(coefficients)
            }
            LinearFitKind::Huber { c } => {
                let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
                let fit =
                    fit_huber_m(x_boot, n, p, y_boot, &opts, &self.backend, &mut workspace.ols)
                        .map_err(EstimationError::from)?;
                require_converged(fit.converged, HUBER_UNCONVERGED)?;
                Ok(fit.coefficients)
            }
        }
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<crate::util::BootstrapSeResult, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let _ = workspace;
        crate::util::bootstrap_se_with_scratch(
            self.bootstrap_replicates,
            ctx,
            0xA7E_u64,
            n,
            || (EstimationWorkspace::default(), vec![0.0; n * p], vec![0.0; n]),
            |(ws, x_boot, y_boot), idx| match self
                .fit_resampled_coefficients(problem, ws, idx, x_boot, y_boot)
            {
                Ok(coefs) => Ok(Some(gcomp_or_coef_ate(problem, &coefs, t_col)?)),
                Err(_) => Ok(None),
            },
        )
    }
}

/// G-computation of μ(active,Z)−μ(control,Z) averaged under the target arm’s covariate law.
///
/// Under a linear main-effects model this equals `β_T · Δ` for every target, including ATT/ATC.
fn gcomp_or_coef_ate(
    problem: &PreparedEstimationProblem,
    coefficients: &[f64],
    t_col: usize,
) -> Result<f64, EstimationError> {
    match problem.target_population {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        TargetPopulation::Treated | TargetPopulation::Untreated => {
            let n = problem.design.nrows;
            let ncols = problem.design.ncols;
            let want_treated = matches!(problem.target_population, TargetPopulation::Treated);
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..n {
                let treated = problem.treatment[r] > 0.5;
                if treated != want_treated {
                    continue;
                }
                let mut pred_a = 0.0;
                let mut pred_c = 0.0;
                for c in 0..ncols {
                    let x = if c == t_col {
                        (problem.active, problem.control)
                    } else {
                        let v = problem.design.matrix[c * n + r];
                        (v, v)
                    };
                    pred_a += coefficients[c] * x.0;
                    pred_c += coefficients[c] * x.1;
                }
                sum += pred_a - pred_c;
                count += 1;
            }
            if count == 0 {
                return Err(EstimationError::data_msg(
                    "target population left no rows for g-computation",
                ));
            }
            Ok(sum / count as f64)
        }
        _ => Err(EstimationError::TargetPopulation),
    }
}

pub(crate) fn intervention_f64(intervention: &Intervention) -> Result<f64, EstimationError> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64().ok_or_else(|| {
            EstimationError::unsupported(" linear adjustment requires numeric treatment levels")
        }),
        _ => Err(EstimationError::unsupported(" linear adjustment requires Set interventions")),
    }
}

/// Fitted coefficients plus the variance model of the estimator that produced them.
struct LinearFit {
    coefficients: Vec<f64>,
    /// Raw residuals `y − Xβ̂`.
    residuals: Vec<f64>,
    rss: f64,
    variance: FitVariance,
}

/// Which sampling-variance formula describes a [`LinearFit`].
enum FitVariance {
    Ols,
    /// Bread `(XᵀX + λP)⁻¹`; no analytic SE (shrinkage bias is outside a sampling variance).
    Ridge {
        lambda: f64,
    },
    /// No analytic SE, no influence function (post-selection).
    Lasso,
    /// `scores[i] = s·ψ(r_i/s)`, `curvature[i] = ψ′(r_i/s)`.
    Huber {
        scores: Vec<f64>,
        curvature: Vec<f64>,
    },
}

/// Refuse an iterative fit that stopped at its iteration cap.
fn require_converged(converged: bool, refusal: &'static str) -> Result<(), EstimationError> {
    if converged { Ok(()) } else { Err(EstimationError::unsupported(refusal)) }
}

const LASSO_UNCONVERGED: &str =
    "Lasso coordinate descent did not converge; refuse rather than publish an unfinished fit";
const HUBER_UNCONVERGED: &str =
    "Huber M-estimator did not converge; refuse rather than publish an unfinished fit";

/// `(Σ d_i x_i x_iᵀ)⁻¹` (row-major) for nonnegative row weights `d`; `None` when singular.
fn curvature_gram_inverse(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    d: &[f64],
) -> Option<Vec<f64>> {
    let mut scaled = x_colmajor[..nrows * ncols].to_vec();
    for col in 0..ncols {
        for (v, w) in scaled[col * nrows..(col + 1) * nrows].iter_mut().zip(d) {
            *v *= w.sqrt();
        }
    }
    gram_inverse(&scaled, nrows, ncols)
}

/// `(XᵀX)⁻¹` (row-major) of a column-major design; `None` when singular.
fn gram_inverse(x_colmajor: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    invert_square(&xtx, ncols)
}

/// Frisch–Waugh influence of the treatment coefficient, scaled by `delta`.
///
/// `ψ_i = δ · n · [(XᵀX)⁻¹ x_i]_T · e_i`, which equals
/// `δ · n · t̃_i e_i / Σ t̃²` with `t̃` the treatment residualized on every
/// other design column. Centering the treatment alone (`t − t̄`) is only the
/// unadjusted IF: with covariates correlated with the treatment it understates
/// the variance by `1 − R²(T | Z)`. Non-invertible designs (`xtx_inv` is
/// `None`) return NaN so a downstream mixture SE fails closed instead of
/// reporting zero.
///
/// The leverage `[(XᵀX)⁻¹ x_i]_T` accumulates one design column at a time
/// (contiguous, vectorizable) rather than striding across columns per row.
fn treatment_coef_influence(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    residuals: &[f64],
    delta: f64,
    xtx_inv: Option<&[f64]>,
) -> Vec<f64> {
    if nrows == 0 || residuals.len() != nrows || t_col >= ncols {
        return vec![0.0; residuals.len()];
    }
    let Some(inv) = xtx_inv else {
        return vec![f64::NAN; nrows];
    };
    let row = &inv[t_col * ncols..t_col * ncols + ncols];
    let scale = delta * nrows as f64;
    let mut psi = vec![0.0; nrows];
    for (c, &w) in row.iter().enumerate() {
        let column = &matrix[c * nrows..(c + 1) * nrows];
        for (slot, &x) in psi.iter_mut().zip(column) {
            *slot += w * x;
        }
    }
    for (slot, &e) in psi.iter_mut().zip(residuals) {
        *slot = scale * *slot * e;
    }
    psi
}

impl crate::estimator::Estimator<TabularData> for LinearAdjustmentAte {
    type Fit = EffectEstimate;

    fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        _ctx: &ExecutionContext,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        Self::prepare(self, data, estimand, query)
    }

    fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, EstimationError> {
        Self::fit(self, problem, workspace, ctx, AssumptionSet::new())
    }
}

impl crate::estimator::TabularAteEstimator for LinearAdjustmentAte {}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;

    use super::*;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        toy_with(|_, t, z| 1.0 + 2.0 * t + z)
    }

    fn toy_with(outcome: impl Fn(usize, f64, f64) -> f64) -> (TabularData, IdentifiedEstimand) {
        let n = 100usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
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
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| outcome(i, t[i], z[i])).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
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
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn overlap_report_from_propensities() {
        let ps = [0.1, 0.5, 0.9];
        let ws = [10.0, 2.0, 1.111];
        let report = OverlapReport::from_propensities(
            &ps,
            Some(&ws),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.05) },
            Some(&[0.0, 0.0, 1.0]),
            Some(crate::overlap::IpwTarget::Ate),
            None,
        );
        assert!((report.propensity_min - 0.1).abs() < 1e-12);
        assert!((report.propensity_max - 0.9).abs() < 1e-12);
        assert_eq!(report.extreme_weight_count, 0); // none strictly > 10
        assert_eq!(report.clip, Some(0.05));
        assert!((report.excluded_fraction - 0.0).abs() < 1e-12);
        assert!((report.target_population_support - 1.0).abs() < 1e-12);
        assert_eq!(report.excluded_regions.len(), 2);
        assert!((report.excluded_regions[0].high - 0.05).abs() < 1e-12);
        let sens = report.clip_sensitivity.as_ref().expect("clip sensitivity");
        assert!(sens.thresholds.len() >= 2);
        assert_eq!(sens.ess.len(), sens.thresholds.len());
    }

    #[test]
    fn rejects_require_diagnostics_on_linear_path() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..LinearAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn recovers_ate_two() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 50, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8);
        assert!(effect.se_bootstrap.is_some());
        let identity: Vec<usize> = (0..prep.design.nrows).collect();
        let resampled = est.ate_on_row_indices(&prep, &mut ws, &identity).unwrap();
        assert!((resampled - effect.ate).abs() < 1e-12);
    }

    /// The shared Gram inverse and the column-wise leverage pass reproduce the
    /// row-by-row Frisch–Waugh influence `δ·n·[(XᵀX)⁻¹x_i]_T·e_i` and the
    /// homoskedastic SE `√(σ²·[(XᵀX)⁻¹]_TT)` exactly.
    #[test]
    fn influence_and_homoskedastic_se_match_the_row_wise_gram_formula() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let (n, p) = (prep.design.nrows, prep.design.ncols);
        let t_col = prep.design.treatment_column().unwrap();
        let fit = prep.design.fit_ols(&est.backend, &mut ws.ols).unwrap();
        let mut xtx = vec![0.0; p * p];
        form_xtx(&prep.design.matrix, n, p, &mut xtx);
        let inv = invert_square(&xtx, p).unwrap();
        let sigma2 = fit.rss / (n - p) as f64;
        let se = (sigma2 * inv[t_col * p + t_col]).sqrt() * prep.treatment_delta.abs();
        assert_eq!(effect.se_analytic.to_bits(), se.to_bits());
        let influence = effect.influence.as_ref().unwrap();
        let row = &inv[t_col * p..(t_col + 1) * p];
        let scale = prep.treatment_delta * n as f64;
        for i in 0..n {
            let leverage: f64 = (0..p).map(|c| row[c] * prep.design.matrix[c * n + i]).sum();
            let expected = scale * leverage * fit.residuals[i];
            assert!((influence[i] - expected).abs() == 0.0, "row {i}");
        }
    }

    #[test]
    fn scales_ate_by_level_delta() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        // β_T ≈ 2, delta = 2 → ATE ≈ 4
        assert!((effect.ate - 4.0).abs() < 1e-8);
    }

    #[test]
    fn recovers_att_via_gcomp() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8, "att={}", effect.ate);
    }

    #[test]
    fn predicate_restricts_prepared_rows() {
        use antecedent_core::PredicateExpr;
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let all = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let half = all.clone().with_target_population(TargetPopulation::Predicate(
            PredicateExpr::rows((0..10).collect::<Vec<_>>()),
        ));
        let prep_all = est.prepare(&data, &estimand, &all).unwrap();
        let prep_half = est.prepare(&data, &estimand, &half).unwrap();
        assert_eq!(prep_all.design.nrows, 100);
        assert_eq!(prep_half.design.nrows, 10);
        let named =
            all.with_target_population(TargetPopulation::Predicate(PredicateExpr::named("cohort")));
        let err = est.prepare(&data, &estimand, &named).unwrap_err();
        assert!(
            err.to_string().contains("PopulationRegistry") || err.to_string().contains("named")
        );
    }

    #[test]
    fn hc_sandwich_kinds_yield_finite_se() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            AnalyticSeKind::Hc0,
            AnalyticSeKind::Hc2,
            AnalyticSeKind::Hc3,
            AnalyticSeKind::NeweyWest { lag: 2 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                se_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
                .unwrap();
            assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0, "{kind:?}");
        }
    }

    #[test]
    fn ridge_lasso_huber_fit_kinds_recover_ate() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            LinearFitKind::Ridge { lambda: 1e-3 },
            LinearFitKind::Lasso { lambda: 1e-4 },
            LinearFitKind::Huber { c: 1.345 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                fit_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(2), AssumptionSet::new())
                .unwrap();
            assert!(effect.ate.is_finite(), "{kind:?}");
            assert!((effect.ate - 2.0).abs() < 0.05, "ate={} kind={kind:?}", effect.ate);
        }
    }

    #[test]
    fn lasso_analytic_se_nan_bootstrap_finite() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let no_boot = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let prep = no_boot.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let effect = no_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic.is_nan(), "lasso se_analytic={}", effect.se_analytic);
        assert!(effect.se_bootstrap.is_none());

        let with_boot = LinearAdjustmentAte {
            bootstrap_replicates: 40,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let effect = with_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic.is_nan());
        let boot = effect.se_bootstrap.expect("bootstrap SE");
        assert!(boot.is_finite() && boot >= 0.0, "se_bootstrap={boot}");
    }

    /// Deterministic bounded noise in `[-0.1, 0.1]`.
    fn wiggle(i: usize) -> f64 {
        0.1 * (7.31 * i as f64).sin()
    }

    fn fit_kind_effect(
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        fit_kind: LinearFitKind,
        se_kind: AnalyticSeKind,
    ) -> EffectEstimate {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            fit_kind,
            se_kind,
            ..LinearAdjustmentAte::new()
        };
        let prep = est.prepare(data, estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        est.fit(&prep, &mut ws, &ExecutionContext::for_tests(5), AssumptionSet::new()).unwrap()
    }

    #[test]
    fn huber_with_unclipped_scores_reproduces_ols_variance_and_influence() {
        // With c so large that no residual is clipped, ψ(u)=u and ψ′=1, so the Huber
        // sandwich collapses to the OLS one: τ² = RSS/(n−p), κ = 1, bread (XᵀX)⁻¹.
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
        for se_kind in [AnalyticSeKind::Homoskedastic, AnalyticSeKind::Hc0, AnalyticSeKind::Hc3] {
            let ols = fit_kind_effect(&data, &estimand, LinearFitKind::Ols, se_kind);
            let hub = fit_kind_effect(&data, &estimand, LinearFitKind::Huber { c: 1e9 }, se_kind);
            assert!((hub.ate - ols.ate).abs() < 1e-8, "{se_kind:?}");
            let rel = (hub.se_analytic - ols.se_analytic).abs() / ols.se_analytic;
            assert!(rel < 1e-7, "{se_kind:?}: huber {} ols {}", hub.se_analytic, ols.se_analytic);
            let (hi, oi) = (hub.influence.as_ref().unwrap(), ols.influence.as_ref().unwrap());
            for (a, b) in hi.iter().zip(oi.iter()) {
                assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
            }
        }
    }

    #[test]
    fn huber_se_ignores_gross_outliers_the_estimator_downweights() {
        // 10% of rows (one odd, one even index per 20, so both arms) carry a +50 shock. OLS
        // RSS/(n−p) is dominated by them; Huber's τ² = E[ψ²]/E[ψ′]² only sees them clipped
        // at c·s, with s ≈ MAD-scale of the ±0.1 noise, so its SE must be far below the OLS
        // one and the effect stays near 2.
        let (data, estimand) = toy_with(|i, t, z| {
            let shock = if i % 20 == 3 || i % 20 == 8 { 50.0 } else { 0.0 };
            1.0 + 2.0 * t + z + wiggle(i) + shock
        });
        let ols =
            fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
        let hub = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Huber { c: 1.345 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!((hub.ate - 2.0).abs() < 0.1, "huber ate {}", hub.ate);
        assert!(
            hub.se_analytic < 0.1 * ols.se_analytic,
            "huber se {} vs ols se {}",
            hub.se_analytic,
            ols.se_analytic
        );
    }

    #[test]
    fn ridge_and_lasso_publish_no_analytic_se_and_no_ols_influence() {
        let (data, estimand) = toy_with(|i, t, z| 1.0 + 2.0 * t + z + wiggle(i));
        let ols =
            fit_kind_effect(&data, &estimand, LinearFitKind::Ols, AnalyticSeKind::Homoskedastic);
        let ridge0 = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Ridge { lambda: 0.0 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(ridge0.se_analytic.is_nan());
        // λ = 0 ridge is OLS, so its influence must be the OLS one.
        let (ri, oi) = (ridge0.influence.as_ref().unwrap(), ols.influence.as_ref().unwrap());
        for (a, b) in ri.iter().zip(oi.iter()) {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
        }
        // Heavy ridge shrinks the treatment coefficient, so its bread differs from (XᵀX)⁻¹.
        let ridge = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Ridge { lambda: 20.0 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(ridge.se_analytic.is_nan());
        assert!(ridge.ate < ols.ate - 1e-3);
        let heavy = ridge.influence.as_ref().unwrap();
        assert!(heavy.iter().zip(oi.iter()).any(|(a, b)| (a - b).abs() > 1e-3));
        let lasso = fit_kind_effect(
            &data,
            &estimand,
            LinearFitKind::Lasso { lambda: 1e-4 },
            AnalyticSeKind::Homoskedastic,
        );
        assert!(lasso.se_analytic.is_nan());
        assert!(lasso.influence.is_none());
    }

    #[test]
    fn unconverged_iterative_fits_are_refused() {
        assert!(require_converged(true, LASSO_UNCONVERGED).is_ok());
        for msg in [LASSO_UNCONVERGED, HUBER_UNCONVERGED] {
            let err = require_converged(false, msg).unwrap_err();
            assert!(err.to_string().contains("did not converge"), "{err}");
        }
    }
}
