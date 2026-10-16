//! Propensity stratification estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions};

use super::levels::OtherLevels;
use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, gather, gather_optional_row_labels,
    prepare_propensity_problem_with_levels, restrict_to_rows, trim_of, trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
use crate::sampling_weights::SamplingWeights;
use crate::se::AnalyticSeKind;
use crate::util::BootstrapSeResult;

/// Propensity stratification estimator: within-stratum difference of means pooled by size.
///
/// Supports [`TargetPopulation::AllObserved`] only (stratified ATE). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
///
/// **Sampling weights.** The propensity model is the weighted logistic fit; within a stratum
/// each arm's mean is the sampling-weighted mean, and strata are pooled by their weighted
/// population share. Strata are cut at the sample quantiles of the fitted propensity.
///
/// **Clusters.** Declared cluster labels make the analytic SE the cluster sandwich of the
/// stratified estimator's influence values *given the strata* (the stratum boundaries and the
/// propensity fit are held fixed), and make the bootstrap resample whole clusters, refitting
/// the propensity and re-cutting the strata on each replicate. The bootstrap is the
/// construction that carries the nuisance fit and the stratification; the analytic SE does
/// not.
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
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Policy for treatment rows at neither declared level. The default refuses; the
    /// exclusion that restricts the target population to the rows at the two declared levels
    /// must be declared.
    pub other_levels: OtherLevels,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    pub sampling_weights: Option<SamplingWeights>,
    /// Cluster label of each complete-case row. Exclusive with [`Self::source_cluster_ids`].
    pub cluster_ids: Option<Vec<u32>>,
    /// Cluster label of every row of the source table. Exclusive with [`Self::cluster_ids`].
    pub source_cluster_ids: Option<Vec<u32>>,
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
            population_registry: None,
            other_levels: OtherLevels::default(),
            sampling_weights: None,
            cluster_ids: None,
            source_cluster_ids: None,
        }
    }

    /// Declare sampling weights for every row of the source table.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    /// Declare the cluster of each complete-case row.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Declare the cluster of every row of the source table.
    #[must_use]
    pub fn with_source_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.source_cluster_ids = Some(cluster_ids);
        self
    }

    /// Cluster label of each row of `problem`, when clusters are declared.
    fn clusters(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<Option<std::borrow::Cow<'_, [u32]>>, EstimationError> {
        let labels = crate::se::declared_row_labels(
            self.cluster_ids.as_deref(),
            self.source_cluster_ids.as_deref(),
            problem.row_index.iter().map(|&row| row as usize),
        )?;
        if let Some(labels) = &labels {
            crate::se::require_clusters(Some(labels.as_ref()), problem.nrows)?;
        }
        Ok(labels)
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

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

    /// Declare what to do with rows whose treatment is at neither declared level (see
    /// [`OtherLevels`]). The default refuses.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    /// Set the number of quantile strata (default 5).
    #[must_use]
    pub const fn with_n_strata(mut self, n_strata: u32) -> Self {
        self.n_strata = n_strata;
        self
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`](crate::propensity::PropensityWeighting::prepare).
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        if let Some(source) = self.source_cluster_ids.as_deref() {
            crate::se::require_source_labels(
                source,
                self.cluster_ids.as_deref(),
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
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
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
        if self.bootstrap_replicates == 0 {
            return Ok(point);
        }
        let n_strata = (self.n_strata.max(1)) as usize;
        let trim = trim_of(problem.overlap);
        let boot = self.bootstrap_se(problem, n_strata, trim, workspace, ctx)?;
        Ok(point.with_bootstrap(Some(boot)))
    }

    fn fit_point(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(
            problem.target_population,
            TargetPopulation::AllObserved
                | TargetPopulation::Treated
                | TargetPopulation::Untreated
                | TargetPopulation::Predicate(_)
        ) {
            return Err(EstimationError::unsupported(
                "propensity stratification supports AllObserved, Treated, Untreated, or Predicate",
            ));
        }
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        let n_strata = (self.n_strata.max(1)) as usize;
        // Trim on RAW scores (mirrors PropensityWeighting): stratify only common-support rows.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            retained.as_deref(),
        );
        let sampling_used: Option<Vec<f64>> = problem.sampling_weights.as_ref().map(|s| {
            retained.as_deref().map_or_else(|| s.weights.to_vec(), |idx| gather(&s.weights, idx))
        });
        let stratum = assign_strata(&s_used, n_strata);
        let result = stratified_ate(
            &t_used,
            &y_used,
            sampling_used.as_deref(),
            &stratum,
            n_strata,
            &problem.target_population,
        )?;
        let clusters = self.clusters(problem)?;
        let clusters_used = gather_optional_row_labels(
            clusters.as_deref(),
            problem.nrows,
            retained.as_deref(),
            "cluster_ids",
        )?;
        let se_analytic = match clusters_used.as_deref() {
            Some(groups) => crate::se::cluster_influence_se(&result.influence, groups)?,
            None => result.se_analytic,
        };

        let ipw_target = IpwTarget::from_population(&problem.target_population).ok();
        let mut report = crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            None,
            ipw_target,
        );
        // Strata missing a treatment arm are dropped from the pooled contrast; fold the
        // retained fraction into the support figure so the artifact reflects the population
        // the estimate actually targets.
        report.target_population_support *= result.retained_fraction;
        report.retained_fraction *= result.retained_fraction;
        let overlap_report = Some(report);

        let mut estimate =
            EffectEstimate::new(result.ate, se_analytic, assumptions, problem.overlap)
                .with_n_obs(u64::try_from(result.n_obs).unwrap_or(u64::MAX))
                .with_overlap_report(overlap_report)
                .with_treatment_levels(problem.treatment_levels)
                .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
                .with_retained_memory_bytes(Some(workspace.retained_memory_bytes()));
        if clusters_used.is_some() {
            estimate = estimate.with_se_kind(AnalyticSeKind::Cluster);
        }
        Ok(estimate)
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        n_strata: usize,
        trim: Option<f64>,
        _workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        // Replicates own a cold workspace. A warm logistic start would change the replicate fit.
        let replicate = |scratch: &mut StratificationReplicate, idx: &[usize]| {
            Ok(self.replicate_ate(problem, n_strata, trim, scratch, idx))
        };
        match self.clusters(problem)? {
            Some(labels) => crate::util::bootstrap_se_clusters_with_scratch(
                self.bootstrap_replicates,
                ctx,
                0x3D2F_u64,
                &crate::util::ClusterRows::new(&labels)?,
                StratificationReplicate::default,
                replicate,
            ),
            None => crate::util::bootstrap_se_with_scratch(
                self.bootstrap_replicates,
                ctx,
                0x3D2F_u64,
                problem.nrows,
                StratificationReplicate::default,
                replicate,
            ),
        }
    }

    /// The stratified effect of one resample of the prepared rows, refitting the propensity and
    /// re-cutting the strata on it; `None` when the replicate cannot be fit. `idx` selects the
    /// rows, whatever their number.
    fn replicate_ate(
        &self,
        problem: &PreparedPropensityProblem,
        n_strata: usize,
        trim: Option<f64>,
        scratch: &mut StratificationReplicate,
        idx: &[usize],
    ) -> Option<f64> {
        let clip = clip_of(problem.overlap);
        let ncols = problem.design_ncols;
        let m = idx.len();
        let sampling = problem.sampling_weights.as_ref().map(|s| &*s.weights);
        let StratificationReplicate { workspace, x, t, y, s } = scratch;
        x.resize(m * ncols, 0.0);
        for buffer in [&mut *t, &mut *y, &mut *s] {
            buffer.resize(m, 0.0);
        }
        crate::util::gather_bootstrap_vector(t, &problem.treatment, idx);
        crate::util::gather_bootstrap_vector(y, &problem.outcome, idx);
        crate::util::gather_resampled_design(x, &problem.design_matrix, problem.nrows, ncols, idx);
        let fit = match sampling {
            Some(full) => {
                crate::util::gather_bootstrap_vector(s, full, idx);
                antecedent_stats::fit_propensity_weighted(
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
            None => antecedent_stats::fit_propensity(
                x,
                m,
                ncols,
                t,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ),
        };
        let Ok(fit) = fit else {
            return None;
        };
        let raw = fit.scores;
        let mut scores = raw.clone();
        if let Some(c) = clip {
            clamp_scores(&mut scores, c);
        }
        let Ok(retained) = trim_retained_rows(&raw, trim) else {
            return None;
        };
        let (t_used, y_used, s_used) = restrict_to_rows(t, y, &scores, 1, retained.as_deref());
        let weights_used: Option<Vec<f64>> =
            sampling.map(|_| retained.as_deref().map_or_else(|| s.clone(), |kept| gather(s, kept)));
        let stratum = assign_strata(&s_used, n_strata);
        stratified_ate(
            &t_used,
            &y_used,
            weights_used.as_deref(),
            &stratum,
            n_strata,
            &problem.target_population,
        )
        .ok()
        .map(|r| r.ate)
    }
}

/// Per-worker buffers of one stratification bootstrap replicate; resized to each resample.
#[derive(Default)]
struct StratificationReplicate {
    workspace: PropensityEstimationWorkspace,
    x: Vec<f64>,
    t: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
}

// ---------------------------------------------------------------------------------------------
// Matching (shared by PropensityMatching and DistanceMatching)
// ---------------------------------------------------------------------------------------------

pub(crate) fn assign_strata(scores: &[f64], n_strata: usize) -> Vec<usize> {
    let n = scores.len();
    let k = n_strata.max(1);
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    // Cut on score quantiles and assign by `score <= cut`, so identical scores
    // stay in one stratum. Rank cuts split a tie across strata by row position.
    let mut cuts = Vec::with_capacity(k.saturating_sub(1));
    for s in 1..k {
        let idx = (s * n) / k;
        cuts.push(scores[order[idx.min(n - 1)]]);
    }
    scores
        .iter()
        .map(|score| {
            cuts.iter()
                .position(|cut| score.total_cmp(cut).is_le())
                .unwrap_or(cuts.len())
                .min(k - 1)
        })
        .collect()
}

pub(crate) struct StratifiedResult {
    ate: f64,
    /// Independent-rows SE of the pooled contrast given the strata.
    se_analytic: f64,
    /// Influence value of every input row (zero outside a retained stratum), the values whose
    /// sums within clusters give the cluster SE of the same contrast.
    influence: Vec<f64>,
    /// Fraction of input rows that landed in a stratum with both arms represented. Strata
    /// missing an arm are dropped from the pooled contrast, which redefines the target
    /// population; callers surface this via the overlap report's support figure.
    retained_fraction: f64,
    n_obs: usize,
}

/// One arm of one stratum: its (weighted) mean, the sum of its weights and the sum of its
/// squared weighted deviations.
#[derive(Clone, Copy, Default)]
struct ArmCell {
    count: usize,
    mass: f64,
    sum: f64,
    mean: f64,
    squared_deviations: f64,
}

impl ArmCell {
    /// Sample variance of the arm mean: `Σ w²(y − m)² · n/(n−1) / (Σ w)²`. With equal weights
    /// this is the usual `s²/n`.
    fn variance_of_mean(&self, weighted: bool) -> f64 {
        let n = self.count as f64;
        if weighted {
            self.squared_deviations * n / (n - 1.0) / (self.mass * self.mass)
        } else {
            self.squared_deviations / (n - 1.0) / n
        }
    }
}

/// Propensity-stratified contrast: within-stratum differences of arm means, pooled by stratum
/// population (observation counts, or sampling-weight mass when `weights` are given; the
/// treated mass for ATT and the control mass for ATC).
///
/// Weighted arm means are `Σ w y / Σ w`, and their variance is the linearization
/// `Σ w²(y − m)² · n/(n−1) / (Σ w)²`, which is the unweighted `s²/n` for equal weights.
pub(crate) fn stratified_ate(
    treatment: &[f64],
    outcome: &[f64],
    weights: Option<&[f64]>,
    stratum: &[usize],
    n_strata: usize,
    target: &TargetPopulation,
) -> Result<StratifiedResult, EstimationError> {
    let weight_of = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mut cells = vec![[ArmCell::default(); 2]; n_strata];
    for i in 0..treatment.len() {
        let cell = &mut cells[stratum[i]][usize::from(treatment[i] > 0.5)];
        cell.count += 1;
        cell.mass += weight_of(i);
        cell.sum += weight_of(i) * outcome[i];
    }
    for cell in cells.iter_mut().flatten() {
        if cell.count > 0 {
            cell.mean = cell.sum / cell.mass;
        }
    }
    // Second pass: centred sums of squares. `Σy² − n·ȳ²` cancels catastrophically when the
    // outcome's mean dwarfs its spread (relative error ≈ ε·(ȳ/s)²), then the `max(0)` would
    // hide the negative result as a zero variance.
    for i in 0..treatment.len() {
        let cell = &mut cells[stratum[i]][usize::from(treatment[i] > 0.5)];
        let deviation = weight_of(i) * (outcome[i] - cell.mean);
        cell.squared_deviations += if weights.is_some() {
            deviation * deviation
        } else {
            (outcome[i] - cell.mean).powi(2)
        };
    }
    let mut pooled = Vec::new();
    for (s, [control, treated]) in cells.iter().enumerate() {
        if treated.count == 0 || control.count == 0 {
            continue;
        }
        if treated.count < 2 || control.count < 2 {
            return Err(EstimationError::data_msg(
                "analytic stratification uncertainty requires at least two treated and two control observations per retained stratum",
            ));
        }
        let (n1, n0) = (treated.count as f64, control.count as f64);
        let share = match (target, weights.is_some()) {
            (TargetPopulation::Treated, false) => n1,
            (TargetPopulation::Untreated, false) => n0,
            (_, false) => n1 + n0,
            (TargetPopulation::Treated, true) => treated.mass,
            (TargetPopulation::Untreated, true) => control.mass,
            (_, true) => treated.mass + control.mass,
        };
        let variance = treated.variance_of_mean(weights.is_some())
            + control.variance_of_mean(weights.is_some());
        pooled.push((s, treated.mean - control.mean, share, variance));
    }
    let total_w: f64 = pooled.iter().map(|&(_, _, share, _)| share).sum();
    if total_w <= 0.0 {
        return Err(EstimationError::data_msg("no strata contain both treated and control units"));
    }
    let ate = pooled.iter().map(|&(_, d, share, _)| d * share).sum::<f64>() / total_w;
    let se_var = pooled
        .iter()
        .map(|&(_, _, share, variance)| variance * (share / total_w).powi(2))
        .sum::<f64>();
    let retained_n: usize =
        pooled.iter().map(|&(s, ..)| cells[s][0].count + cells[s][1].count).sum();
    let retained_fraction = retained_n as f64 / (treatment.len().max(1) as f64);
    // Influence of each row on the pooled contrast, strata held fixed: the row's weighted
    // deviation from its arm mean, over the arm's mass, times the stratum's share.
    let n = treatment.len() as f64;
    let mut influence = vec![0.0; treatment.len()];
    let mut in_pool = vec![false; n_strata];
    for &(s, ..) in &pooled {
        in_pool[s] = true;
    }
    let share_of: std::collections::HashMap<usize, f64> =
        pooled.iter().map(|&(s, _, share, _)| (s, share)).collect();
    for i in 0..treatment.len() {
        let s = stratum[i];
        if !in_pool[s] {
            continue;
        }
        let arm = usize::from(treatment[i] > 0.5);
        let cell = &cells[s][arm];
        let sign = if arm == 1 { 1.0 } else { -1.0 };
        influence[i] = sign
            * n
            * (share_of[&s] / total_w)
            * (weight_of(i) / cell.mass)
            * (outcome[i] - cell.mean);
    }
    Ok(StratifiedResult {
        ate,
        se_analytic: se_var.sqrt(),
        influence,
        retained_fraction,
        n_obs: retained_n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_propensity_scores_share_a_stratum() {
        let scores = [0.1, 0.9, 0.1, 0.1];
        let stratum = assign_strata(&scores, 2);
        assert_eq!(stratum[0], stratum[2]);
        assert_eq!(stratum[0], stratum[3]);
        assert_ne!(stratum[0], stratum[1]);
    }

    #[test]
    fn stratification_rejects_singleton_arm_uncertainty() {
        let err = match stratified_ate(
            &[0.0, 1.0],
            &[0.0, 1.0],
            None,
            &[0, 0],
            1,
            &TargetPopulation::AllObserved,
        ) {
            Ok(_) => panic!("singleton arms must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, EstimationError::Data(_)));
    }

    #[test]
    fn stratification_variance_survives_a_large_outcome_offset() {
        // Each arm's outcomes are 1e8 ± 0.1: sample variance 0.02, so
        // SE² = 0.02/2 + 0.02/2. Raw second moments (Σy² ≈ 2e16, ulp 4) cannot resolve it.
        let result = stratified_ate(
            &[1.0, 1.0, 0.0, 0.0],
            &[1e8 + 0.1, 1e8 - 0.1, 1e8 + 0.1, 1e8 - 0.1],
            None,
            &[0, 0, 0, 0],
            1,
            &TargetPopulation::AllObserved,
        )
        .unwrap();
        assert!(result.ate.abs() < 1e-7);
        assert!((result.se_analytic - 0.02_f64.sqrt()).abs() < 1e-6, "se {}", result.se_analytic);
    }

    #[test]
    fn stratification_two_per_arm_matches_closed_form_se() {
        let result = stratified_ate(
            &[1.0, 1.0, 0.0, 0.0],
            &[1.0, 3.0, 0.0, 2.0],
            None,
            &[0, 0, 0, 0],
            1,
            &TargetPopulation::AllObserved,
        )
        .unwrap();
        assert!((result.ate - 1.0).abs() <= 1e-12);
        assert!((result.se_analytic - 2.0_f64.sqrt()).abs() <= 1e-12);
        assert!(result.se_analytic.is_finite());
    }
}
