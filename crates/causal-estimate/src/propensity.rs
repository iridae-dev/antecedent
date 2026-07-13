//! Propensity-based estimators: weighting, stratification, and matching (Phase 4).
//!
//! All estimators here require propensity-based positivity diagnostics
//! ([`OverlapPolicy::RequireDiagnostics`]) — [`OverlapPolicy::ExplicitOverride`] is refused
//! because positivity is mandatory for propensity/matching methods (DESIGN.md §14.3).
//!
//! Bootstrap standard errors **refit the propensity model on every resample** rather than
//! reusing the point-estimate propensity scores. This is more expensive than reusing scores,
//! but it is the honest choice: it propagates first-stage estimation uncertainty into the
//! second-stage effect, which score-reuse would understate. [`causal_stats::PropensityWorkspace`]
//! scratch (IRLS design/Cholesky buffers) is reused across all replicates to keep the
//! per-replicate cost to a single GLM refit with no additional heap churn.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::needless_pass_by_value
)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId};
use causal_data::TabularData;
use causal_identify::IdentifiedEstimand;
use causal_stats::{
    FaerBackend, GlmOptions, MatchingDistance, MatchingIndex, PropensityFit, PropensityWorkspace,
    StatsError, fit_propensity,
};

use crate::adjustment::{EffectEstimate, OverlapPolicy, OverlapReport, intervention_f64};
use crate::error::EstimationError;

/// Prepared covariate design + treatment/outcome shared by every propensity estimator.
///
/// Built once from `(data, estimand, query)`; reused across point estimate and bootstrap.
#[derive(Clone, Debug)]
pub struct PreparedPropensityProblem {
    /// Column-major `[1 | Z…]` design used to fit the propensity model.
    pub design_matrix: Arc<[f64]>,
    /// Number of design columns (`1 + adjustment_set.len()`).
    pub design_ncols: usize,
    /// Number of complete-case rows.
    pub nrows: usize,
    /// Binary treatment indicator (0/1), length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Raw adjustment covariate columns, in `adjustment_set` order (excludes intercept).
    pub covariates: Arc<[Arc<[f64]>]>,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Target population requested by the query.
    pub target_population: TargetPopulation,
}

/// Fitted propensity model shared by weighting, stratification, and matching estimators.
///
/// Retains the raw [`PropensityFit`] (coefficients, scores, GLM diagnostics) plus the
/// clip-adjusted scores actually used for weighting/matching/distance calculations.
#[derive(Clone, Debug)]
pub struct PropensityModel {
    /// Raw logistic fit (pre-clip scores in `fit.scores`).
    pub fit: PropensityFit,
    /// Clip threshold applied to `clipped_scores`, taken from the overlap policy.
    pub clip: Option<f64>,
    /// Propensity scores after clipping into `[clip, 1 - clip]` (identical to `fit.scores`
    /// when `clip` is `None`).
    pub clipped_scores: Vec<f64>,
}

impl PropensityModel {
    /// Fit the logistic propensity model on `problem`'s design, applying the clip threshold
    /// from `problem.overlap` when present.
    ///
    /// # Errors
    ///
    /// Propagates GLM/backend failures.
    pub fn fit(
        problem: &PreparedPropensityProblem,
        backend: &FaerBackend,
        workspace: &mut PropensityWorkspace,
        options: &GlmOptions,
    ) -> Result<Self, EstimationError> {
        let fit = fit_propensity(
            &problem.design_matrix,
            problem.nrows,
            problem.design_ncols,
            &problem.treatment,
            backend,
            workspace,
            options,
        )
        .map_err(stats_err)?;
        let clip = clip_of(problem.overlap);
        let mut clipped_scores = fit.scores.clone();
        if let Some(c) = clip {
            clamp_scores(&mut clipped_scores, c);
        }
        Ok(Self { fit, clip, clipped_scores })
    }
}

/// Reusable scratch for propensity estimators (bootstrap-safe: no per-replicate reallocation
/// beyond what `MatchingIndex` construction requires, since donor rows differ per replicate).
#[derive(Clone, Debug, Default)]
pub struct PropensityEstimationWorkspace {
    /// Logistic IRLS scratch reused across point-estimate and bootstrap propensity refits.
    pub propensity: PropensityWorkspace,
    /// Matching output buffer: matched donor row per query row.
    pub matching_donor_rows: Vec<usize>,
    /// Matching output buffer: distance to the matched donor per query row.
    pub matching_distances: Vec<f64>,
}

/// Default overlap policy for all Phase 4 propensity estimators: diagnostics mandatory,
/// propensities clipped into `[0.01, 0.99]`, no trimming.
#[must_use]
pub const fn default_propensity_overlap() -> OverlapPolicy {
    OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None }
}

// ---------------------------------------------------------------------------------------------
// Shared prepare / small helpers
// ---------------------------------------------------------------------------------------------

pub(crate) fn prepare_propensity_problem(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
) -> Result<PreparedPropensityProblem, EstimationError> {
    if matches!(overlap, OverlapPolicy::ExplicitOverride) {
        return Err(EstimationError::Overlap {
            message: "propensity estimators require RequireDiagnostics overlap policy; positivity is mandatory",
        });
    }
    if &*estimand.method != "backdoor.adjustment" && &*estimand.method != "backdoor.efficient" {
        return Err(EstimationError::IncompatibleEstimand {
            message: "propensity estimators expect backdoor.adjustment or backdoor.efficient",
        });
    }
    query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::UnsupportedQuery(
            "Phase 4 propensity estimators do not support effect modifiers".into(),
        ));
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
        return Err(EstimationError::UnsupportedQuery(
            "propensity estimators require binary treatment coded active=1.0, control=0.0".into(),
        ));
    }

    let treatment = query.treatment;
    let outcome = query.outcome;
    let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.adjustment_set);
    let row_mask =
        data.complete_case_mask(&ids).map_err(|e| EstimationError::Data(e.to_string()))?;
    let t = data
        .float64_masked(treatment, &row_mask)
        .map_err(|e| EstimationError::Data(e.to_string()))?;
    let y = data
        .float64_masked(outcome, &row_mask)
        .map_err(|e| EstimationError::Data(e.to_string()))?;
    let nrows = t.len();
    if nrows == 0 {
        return Err(EstimationError::Data("no complete-case rows for propensity design".into()));
    }

    let ncols = 1 + estimand.adjustment_set.len();
    let mut design = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        design[r] = 1.0;
    }
    let mut covariate_cols: Vec<Arc<[f64]>> = Vec::with_capacity(estimand.adjustment_set.len());
    for (i, &z) in estimand.adjustment_set.iter().enumerate() {
        let col = data
            .float64_masked(z, &row_mask)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let base = (1 + i) * nrows;
        for r in 0..nrows {
            design[base + r] = col[r];
        }
        covariate_cols.push(Arc::from(col));
    }

    Ok(PreparedPropensityProblem {
        design_matrix: Arc::from(design),
        design_ncols: ncols,
        nrows,
        treatment: Arc::from(t),
        outcome: Arc::from(y),
        covariates: Arc::from(covariate_cols),
        method: Arc::clone(&estimand.method),
        adjustment_set: Arc::clone(&estimand.adjustment_set),
        overlap,
        target_population: query.target_population.clone(),
    })
}

pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::Stats(e.to_string())
}

pub(crate) fn clip_of(overlap: OverlapPolicy) -> Option<f64> {
    match overlap {
        OverlapPolicy::RequireDiagnostics { clip, .. } => clip,
        OverlapPolicy::ExplicitOverride => None,
    }
}

fn trim_of(overlap: OverlapPolicy) -> Option<f64> {
    match overlap {
        OverlapPolicy::RequireDiagnostics { trim, .. } => trim,
        OverlapPolicy::ExplicitOverride => None,
    }
}

pub(crate) fn clamp_scores(scores: &mut [f64], clip: f64) {
    for s in scores.iter_mut() {
        *s = s.clamp(clip, 1.0 - clip);
    }
}

pub(crate) fn sample_std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var =
        values.iter().map(|v| { let d = v - mean; d * d }).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn to_row_major(cols: &[Arc<[f64]>], nrows: usize) -> Vec<f64> {
    let dim = cols.len().max(1);
    let mut out = vec![0.0; nrows * dim];
    for (c, col) in cols.iter().enumerate() {
        for r in 0..nrows {
            out[r * dim + c] = col[r];
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// IPW weights + Hajek estimator (shared by `PropensityWeighting`)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IpwTarget {
    Ate,
    Att,
    Atc,
}

impl IpwTarget {
    fn from_population(pop: &TargetPopulation) -> Result<Self, EstimationError> {
        match pop {
            TargetPopulation::AllObserved => Ok(Self::Ate),
            TargetPopulation::Treated => Ok(Self::Att),
            TargetPopulation::Untreated => Ok(Self::Atc),
            _ => Err(EstimationError::UnsupportedQuery(
                "propensity weighting supports AllObserved, Treated, or Untreated target populations".into(),
            )),
        }
    }

    fn weight(self, t: f64, e: f64) -> f64 {
        match self {
            Self::Ate => {
                if t > 0.5 { 1.0 / e } else { 1.0 / (1.0 - e) }
            }
            Self::Att => {
                if t > 0.5 { 1.0 } else { e / (1.0 - e) }
            }
            Self::Atc => {
                if t > 0.5 { (1.0 - e) / e } else { 1.0 }
            }
        }
    }
}

/// `scores_for_weight` feeds the weight formula (typically clipped); `scores_for_trim` feeds
/// the trim decision (typically the raw, pre-clip scores) — they may be the same slice.
fn compute_ipw_weights(
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) -> Vec<f64> {
    treatment
        .iter()
        .zip(scores_for_weight)
        .zip(scores_for_trim)
        .map(|((&t, &e), &raw)| {
            if let Some(tr) = trim {
                if raw < tr || raw > 1.0 - tr {
                    return 0.0;
                }
            }
            target.weight(t, e)
        })
        .collect()
}

fn hajek_difference(treatment: &[f64], outcome: &[f64], weights: &[f64]) -> f64 {
    let (mut num1, mut den1, mut num0, mut den0) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..treatment.len() {
        let w = weights[i];
        if treatment[i] > 0.5 {
            num1 += w * outcome[i];
            den1 += w;
        } else {
            num0 += w * outcome[i];
            den0 += w;
        }
    }
    num1 / den1 - num0 / den0
}

fn hajek_weighted_mean(treatment: &[f64], outcome: &[f64], weights: &[f64], want_treated: bool) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            num += weights[i] * outcome[i];
            den += weights[i];
        }
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

fn hajek_group_variance(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
    mu: f64,
) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            let w = weights[i];
            num += w * w * (outcome[i] - mu).powi(2);
            den += w;
        }
    }
    if den > 0.0 { num / (den * den) } else { f64::NAN }
}

/// Linearized (ratio-estimator) analytic SE of the Hajek ATE/ATT/ATC difference.
fn hajek_analytic_se(treatment: &[f64], outcome: &[f64], weights: &[f64]) -> f64 {
    let mu1 = hajek_weighted_mean(treatment, outcome, weights, true);
    let mu0 = hajek_weighted_mean(treatment, outcome, weights, false);
    let v1 = hajek_group_variance(treatment, outcome, weights, true, mu1);
    let v0 = hajek_group_variance(treatment, outcome, weights, false, mu0);
    (v1 + v0).sqrt()
}

/// Inverse-probability weighting estimator (ATE/ATT/ATC via `TargetPopulation`).
///
/// Point estimate is the Hajek (self-normalized) weighted difference of means. Positivity is
/// mandatory: [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityWeighting {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
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
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
        }
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
        prepare_propensity_problem(data, estimand, query, self.overlap)
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
        let target = IpwTarget::from_population(&problem.target_population)?;
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(problem, &self.backend, &mut workspace.propensity, &self.glm_options)?;

        let weights = compute_ipw_weights(
            &problem.treatment,
            &model.clipped_scores,
            &model.fit.scores,
            target,
            trim,
        );
        let ate = hajek_difference(&problem.treatment, &problem.outcome, &weights);
        let se_analytic = hajek_analytic_se(&problem.treatment, &problem.outcome, &weights);

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, target, trim, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, Some(&weights), problem.overlap));

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        target: IpwTarget,
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<f64, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x9A17_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + idx];
                }
            }
            let fit = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            )
            .map_err(stats_err)?;
            let raw = fit.scores;
            let mut clipped = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut clipped, c);
            }
            let w = compute_ipw_weights(&t_boot, &clipped, &raw, target, trim);
            estimates.push(hajek_difference(&t_boot, &y_boot, &w));
        }
        Ok(sample_std(&estimates))
    }
}

// ---------------------------------------------------------------------------------------------
// Stratification
// ---------------------------------------------------------------------------------------------

fn assign_strata(scores: &[f64], n_strata: usize) -> Vec<usize> {
    let n = scores.len();
    let k = n_strata.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut stratum = vec![0usize; n];
    for (rank, &orig) in order.iter().enumerate() {
        let s = (rank * k) / n.max(1);
        stratum[orig] = s.min(k - 1);
    }
    stratum
}

struct StratifiedResult {
    ate: f64,
    se_analytic: f64,
}

fn stratified_ate(
    treatment: &[f64],
    outcome: &[f64],
    stratum: &[usize],
    n_strata: usize,
) -> Result<StratifiedResult, EstimationError> {
    let mut sum1 = vec![0.0; n_strata];
    let mut sq1 = vec![0.0; n_strata];
    let mut cnt1 = vec![0usize; n_strata];
    let mut sum0 = vec![0.0; n_strata];
    let mut sq0 = vec![0.0; n_strata];
    let mut cnt0 = vec![0usize; n_strata];
    for i in 0..treatment.len() {
        let s = stratum[i];
        if treatment[i] > 0.5 {
            sum1[s] += outcome[i];
            sq1[s] += outcome[i] * outcome[i];
            cnt1[s] += 1;
        } else {
            sum0[s] += outcome[i];
            sq0[s] += outcome[i] * outcome[i];
            cnt0[s] += 1;
        }
    }
    let mut diffs = Vec::new();
    let mut ns = Vec::new();
    let mut vars = Vec::new();
    for s in 0..n_strata {
        if cnt1[s] == 0 || cnt0[s] == 0 {
            continue;
        }
        let n1 = cnt1[s] as f64;
        let n0 = cnt0[s] as f64;
        let mean1 = sum1[s] / n1;
        let mean0 = sum0[s] / n0;
        let var1 = (sq1[s] / n1 - mean1 * mean1).max(0.0);
        let var0 = (sq0[s] / n0 - mean0 * mean0).max(0.0);
        diffs.push(mean1 - mean0);
        ns.push(n1 + n0);
        vars.push(var1 / n1 + var0 / n0);
    }
    let total_n: f64 = ns.iter().sum();
    if total_n <= 0.0 {
        return Err(EstimationError::Data(
            "no strata contain both treated and control units".into(),
        ));
    }
    let ate = diffs.iter().zip(&ns).map(|(d, n)| d * n).sum::<f64>() / total_n;
    let se_var = vars.iter().zip(&ns).map(|(v, n)| v * (n / total_n).powi(2)).sum::<f64>();
    Ok(StratifiedResult { ate, se_analytic: se_var.sqrt() })
}

/// Propensity stratification estimator: within-stratum difference of means pooled by size.
///
/// Supports [`TargetPopulation::AllObserved`] only (stratified ATE). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityStratification {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Number of quantile strata (default 5).
    pub n_strata: u32,
}

impl Default for PropensityStratification {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityStratification {
    /// Defaults: 5 strata, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            n_strata: 5,
        }
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Fit the propensity model, assign quantile strata, and compute the pooled stratified ATE.
    ///
    /// # Errors
    ///
    /// Unsupported target population, no strata with both arms represented, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::UnsupportedQuery(
                "propensity stratification only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let model = PropensityModel::fit(problem, &self.backend, &mut workspace.propensity, &self.glm_options)?;
        let n_strata = (self.n_strata.max(1)) as usize;
        let stratum = assign_strata(&model.clipped_scores, n_strata);
        let result = stratified_ate(&problem.treatment, &problem.outcome, &stratum, n_strata)?;

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, n_strata, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        n_strata: usize,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<f64, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x3D2F_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + idx];
                }
            }
            let fit = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            )
            .map_err(stats_err)?;
            let mut scores = fit.scores;
            if let Some(c) = clip {
                clamp_scores(&mut scores, c);
            }
            let stratum = assign_strata(&scores, n_strata);
            if let Ok(r) = stratified_ate(&t_boot, &y_boot, &stratum, n_strata) {
                estimates.push(r.ate);
            }
        }
        if estimates.len() < 2 {
            return Ok(f64::NAN);
        }
        Ok(sample_std(&estimates))
    }
}

// ---------------------------------------------------------------------------------------------
// Matching (shared by PropensityMatching and DistanceMatching)
// ---------------------------------------------------------------------------------------------

pub(crate) fn split_by_treatment(treatment: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut treated = Vec::new();
    let mut control = Vec::new();
    for (i, &t) in treatment.iter().enumerate() {
        if t > 0.5 { treated.push(i); } else { control.push(i); }
    }
    (treated, control)
}

fn gather(values: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| values[i]).collect()
}

fn gather_rowmajor(matrix: &[f64], dim: usize, idx: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &i in idx {
        out.extend_from_slice(&matrix[i * dim..(i + 1) * dim]);
    }
    out
}

/// Match each `query` row to its nearest `donor` row; returns `query_y[q] - donor_y[matched]`
/// for every query matched within the caliper (caliper-rejected queries are omitted).
fn match_diffs(
    donor_features: &[f64],
    donor_outcome: &[f64],
    dim: usize,
    distance: MatchingDistance,
    query_features: &[f64],
    query_outcome: &[f64],
    caliper: Option<f64>,
    donor_rows_buf: &mut Vec<usize>,
    distances_buf: &mut Vec<f64>,
) -> Result<Vec<f64>, EstimationError> {
    let n_donors = donor_outcome.len();
    if n_donors == 0 {
        return Err(EstimationError::Data("matching requires at least one donor row".into()));
    }
    let donor_ids: Vec<usize> = (0..n_donors).collect();
    let index = MatchingIndex::exact(donor_features, dim, &donor_ids, distance).map_err(stats_err)?;
    let n_queries = query_outcome.len();
    donor_rows_buf.clear();
    donor_rows_buf.resize(n_queries, 0);
    distances_buf.clear();
    distances_buf.resize(n_queries, 0.0);
    index
        .match_all(query_features, n_queries, caliper, donor_rows_buf, distances_buf)
        .map_err(stats_err)?;
    let mut diffs = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let d = donor_rows_buf[q];
        if d != usize::MAX {
            diffs.push(query_outcome[q] - donor_outcome[d]);
        }
    }
    Ok(diffs)
}

struct MatchedEstimate {
    ate: f64,
    se_analytic: f64,
}

/// ATT/ATC/ATE via nearest-neighbor matching on `features` (dim columns, row-major).
///
/// ATT matches treated→nearest control; ATC matches control→nearest treated (sign-flipped);
/// ATE pools both directions' per-unit imputed effects (Abadie–Imbens style).
fn matching_contrast(
    treatment: &[f64],
    outcome: &[f64],
    features: &[f64],
    dim: usize,
    distance: MatchingDistance,
    target: &TargetPopulation,
    caliper: Option<f64>,
    donor_rows_buf: &mut Vec<usize>,
    distances_buf: &mut Vec<f64>,
) -> Result<MatchedEstimate, EstimationError> {
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::Data("matching requires both treated and control rows".into()));
    }
    let treated_feat = gather_rowmajor(features, dim, &treated_idx);
    let control_feat = gather_rowmajor(features, dim, &control_idx);
    let treated_y = gather(outcome, &treated_idx);
    let control_y = gather(outcome, &control_idx);

    let per_unit_effects: Vec<f64> = match target {
        TargetPopulation::Treated => match_diffs(
            &control_feat, &control_y, dim, distance, &treated_feat, &treated_y, caliper,
            donor_rows_buf, distances_buf,
        )?,
        TargetPopulation::Untreated => match_diffs(
            &treated_feat, &treated_y, dim, distance, &control_feat, &control_y, caliper,
            donor_rows_buf, distances_buf,
        )?
        .into_iter()
        .map(|d| -d)
        .collect(),
        TargetPopulation::AllObserved => {
            let mut att_diffs = match_diffs(
                &control_feat, &control_y, dim, distance, &treated_feat, &treated_y, caliper,
                donor_rows_buf, distances_buf,
            )?;
            let atc_diffs: Vec<f64> = match_diffs(
                &treated_feat, &treated_y, dim, distance, &control_feat, &control_y, caliper,
                donor_rows_buf, distances_buf,
            )?
            .into_iter()
            .map(|d| -d)
            .collect();
            att_diffs.extend(atc_diffs);
            att_diffs
        }
        _ => {
            return Err(EstimationError::UnsupportedQuery(
                "matching estimators support AllObserved, Treated, or Untreated target populations".into(),
            ));
        }
    };
    if per_unit_effects.is_empty() {
        return Err(EstimationError::Data("no matched units within caliper".into()));
    }
    let n = per_unit_effects.len() as f64;
    let ate = per_unit_effects.iter().sum::<f64>() / n;
    let se_analytic = sample_std(&per_unit_effects) / n.sqrt();
    Ok(MatchedEstimate { ate, se_analytic })
}

/// Propensity-score nearest-neighbor matching (Absolute distance, optional caliper).
///
/// Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`] is refused. Supports
/// ATT/ATC/ATE via `TargetPopulation`.
#[derive(Clone, Debug)]
pub struct PropensityMatching {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum propensity distance for an accepted match.
    pub caliper: Option<f64>,
}

impl Default for PropensityMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityMatching {
    /// Defaults: no caliper, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
        }
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Fit the propensity model and compute the matched effect.
    ///
    /// # Errors
    ///
    /// Unsupported target population, empty treated/control arm, no matches within the
    /// caliper, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let model = PropensityModel::fit(problem, &self.backend, &mut workspace.propensity, &self.glm_options)?;
        let result = matching_contrast(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            MatchingDistance::Absolute,
            &problem.target_population,
            self.caliper,
            &mut workspace.matching_donor_rows,
            &mut workspace.matching_distances,
        )?;

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<f64, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x51E7_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + idx];
                }
            }
            let fit = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            )
            .map_err(stats_err)?;
            let mut scores = fit.scores;
            if let Some(c) = clip {
                clamp_scores(&mut scores, c);
            }
            if let Ok(m) = matching_contrast(
                &t_boot, &y_boot, &scores, 1, MatchingDistance::Absolute, &problem.target_population,
                self.caliper, &mut workspace.matching_donor_rows, &mut workspace.matching_distances,
            ) {
                estimates.push(m.ate);
            }
        }
        if estimates.len() < 2 {
            return Ok(f64::NAN);
        }
        Ok(sample_std(&estimates))
    }
}

/// Distance matching on raw adjustment covariates (Euclidean), not the propensity score.
///
/// A propensity model is still fit — purely to populate mandatory positivity diagnostics
/// ([`EffectEstimate::overlap_report`]) — but it does not influence the matched contrast.
/// Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct DistanceMatching {
    /// Dense linear-algebra backend used for the diagnostic logistic fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the diagnostic propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum Euclidean distance for an accepted match.
    pub caliper: Option<f64>,
}

impl Default for DistanceMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceMatching {
    /// Defaults: no caliper, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
        }
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Match on raw covariates (Euclidean) and compute the matched effect; fits a diagnostic
    /// propensity model for the mandatory overlap report.
    ///
    /// # Errors
    ///
    /// Empty adjustment set, unsupported target population, empty treated/control arm, no
    /// matches within the caliper, or GLM failure (diagnostic fit).
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if problem.adjustment_set.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "distance matching requires a non-empty adjustment set".into(),
            ));
        }
        let dim = problem.covariates.len();
        let features = to_row_major(&problem.covariates, problem.nrows);
        let result = matching_contrast(
            &problem.treatment,
            &problem.outcome,
            &features,
            dim,
            MatchingDistance::Euclidean,
            &problem.target_population,
            self.caliper,
            &mut workspace.matching_donor_rows,
            &mut workspace.matching_distances,
        )?;

        // Diagnostic-only propensity fit: populates the mandatory overlap report without
        // influencing the covariate-space matched contrast above.
        let diag = PropensityModel::fit(problem, &self.backend, &mut workspace.propensity, &self.glm_options)?;

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, dim, &features, workspace, ctx))
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&diag.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        dim: usize,
        features: &[f64],
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> f64 {
        let mut rng = ctx.rng.stream(0x7C11_u64);
        let n = problem.nrows;
        let mut feat_boot = vec![0.0; n * dim];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for d in 0..dim {
                    feat_boot[r * dim + d] = features[idx * dim + d];
                }
            }
            if let Ok(m) = matching_contrast(
                &t_boot, &y_boot, &feat_boot, dim, MatchingDistance::Euclidean, &problem.target_population,
                self.caliper, &mut workspace.matching_donor_rows, &mut workspace.matching_distances,
            ) {
                estimates.push(m.ate);
            }
        }
        if estimates.len() < 2 {
            return f64::NAN;
        }
        sample_std(&estimates)
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_identify::IdentifiedEstimand;

    use super::*;
    use crate::adjustment::OverlapPolicy;

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2.
    fn standard_normal(rng: &mut causal_core::CausalRng) -> f64 {
        // Box-Muller.
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1234_u64);

        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }

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
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
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

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    #[test]
    fn weighting_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting { bootstrap_replicates: 30, ..PropensityWeighting::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn weighting_att_target_population() {
        let (data, estimand) = confounded_scm(800, 2);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.4, "att={}", effect.ate);
    }

    #[test]
    fn weighting_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting { overlap: OverlapPolicy::ExplicitOverride, ..PropensityWeighting::new() };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn stratification_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 4);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est =
            PropensityStratification { bootstrap_replicates: 30, ..PropensityStratification::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn stratification_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 5);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityStratification {
            overlap: OverlapPolicy::ExplicitOverride,
            ..PropensityStratification::new()
        };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn propensity_matching_recovers_att() {
        let (data, estimand) = confounded_scm(800, 6);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityMatching { bootstrap_replicates: 30, ..PropensityMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "att={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn propensity_matching_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 7);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est =
            PropensityMatching { overlap: OverlapPolicy::ExplicitOverride, ..PropensityMatching::new() };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn distance_matching_recovers_att() {
        let (data, estimand) = confounded_scm(800, 8);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = DistanceMatching { bootstrap_replicates: 30, ..DistanceMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "att={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn distance_matching_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 9);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est =
            DistanceMatching { overlap: OverlapPolicy::ExplicitOverride, ..DistanceMatching::new() };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }
}
