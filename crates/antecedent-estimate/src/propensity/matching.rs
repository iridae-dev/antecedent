//! Propensity-score nearest-neighbor matching.
//!
//! Analytic standard errors follow Abadie–Imbens (2006) with donor-usage counts
//! `Kᵢ` (matching with replacement). A linear within-arm regression bias
//! adjustment (Abadie–Imbens) is applied on the match feature(s).
//!
//! **Bootstrap caution:** the nonparametric bootstrap is invalid for nearest-neighbor
//! matching with a fixed number of matches (Abadie–Imbens 2008). The licensed uncertainty
//! product is the Abadie–Imbens analytic SE ([`EffectEstimate::se_analytic`]). Setting
//! [`PropensityMatching::bootstrap_replicates`] does **not** populate
//! [`EffectEstimate::se_bootstrap`] — the invalid bootstrap is not retained as a CI.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions, MatchingDistance, MatchingIndex, StatsError};

use super::levels::OtherLevels;
use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel,
    default_propensity_overlap, gather, gather_optional_multiway, gather_optional_row_labels,
    gather_rowmajor, prepare_propensity_problem_with_levels, restrict_to_rows, split_by_treatment,
    trim_of, trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
use crate::sampling_weights::SamplingWeights;
use crate::se::{AnalyticSeKind, SeSpec};
use crate::util::{sample_std, stats_err};

/// Queries matched between cancellation checks when several donors are searched per query.
const MATCH_CANCELLATION_CHUNK: usize = 1024;

/// Scale on which the propensity-score matching distance (and [`PropensityMatching::caliper`])
/// is computed.
///
/// The field-standard convention (Rosenbaum & Rubin 1985; Austin 2011, "Optimal caliper
/// widths for propensity-score matching") is that a caliper is applied to the **logit** of
/// the propensity score, `logit(e) = ln(e / (1 - e))` — a caliper of 0.2 means "0.2 standard
/// deviations of the logit propensity," not 0.2 of raw probability. The raw-probability scale
/// compresses badly near 0 and 1 (exactly where matching quality matters most for units with
/// extreme propensities), so a caller who follows the textbook 0.2 rule of thumb but matches
/// on raw probability gets behavior very different from the convention they intended.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaliperScale {
    /// Match on `logit(e) = ln(e / (1 - e))`. This is what the 0.2 rule of thumb means.
    #[default]
    Logit,
    /// Match on the raw (clipped) propensity probability, `e ∈ [clip, 1 - clip]`.
    Raw,
}

/// Propensity-score nearest-neighbor matching (Absolute distance, optional caliper).
///
/// Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`] is refused. Supports
/// ATT/ATC/ATE via `TargetPopulation`.
///
/// Analytic SEs use Abadie–Imbens (2006) donor-reuse variance; see module docs for the
/// bootstrap caveat (Abadie–Imbens 2008).
///
/// **Sampling weights.** With [`Self::sampling_weights`] the propensity model is the weighted
/// logistic fit, donors are still the nearest rows of the sample, and the effect is the
/// weighted mean of the matched unit effects with each matched unit weighted by its sampling
/// weight (times its target weight). Its variance is the Abadie–Imbens form for a weighted
/// matched mean: `σ̂² Σ c_i²`, where `c_i` is a row's own weight plus the weights of the
/// matched units it donates to. The interval it supports is a separate calibration claim.
///
/// **Caliper scale:** [`Self::caliper`] is interpreted on [`Self::caliper_scale`], which
/// defaults to [`CaliperScale::Logit`] — the field-standard convention (Rosenbaum & Rubin
/// 1985; Austin 2011). A caliper of 0.2 under this default means 0.2 on the logit scale, per
/// the literature's rule of thumb; see [`CaliperScale`] for why the raw-probability scale is
/// not equivalent. This is a behavior change for callers who previously relied on the
/// undocumented raw-probability matching distance; set `caliper_scale: CaliperScale::Raw`
/// explicitly to keep the old behavior.
#[derive(Clone, Debug)]
pub struct PropensityMatching {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Accepted for API compatibility; does not populate [`EffectEstimate::se_bootstrap`]
    /// (Abadie–Imbens 2008 — see module docs). Prefer [`EffectEstimate::se_analytic`].
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum matching distance for an accepted match, interpreted on
    /// [`Self::caliper_scale`].
    pub caliper: Option<f64>,
    /// Scale on which matching distance and `caliper` are computed. Defaults to
    /// [`CaliperScale::Logit`] — see the type-level docs.
    pub caliper_scale: CaliperScale,
    /// Analytic SE kind. Only `Homoskedastic` (Abadie–Imbens) is supported: cluster,
    /// multiway and HAC kinds are refused because donor outcomes belong to other
    /// clusters and times than the matched unit.
    pub se_kind: AnalyticSeKind,
    /// Separate outcome residual noise and population treatment-effect heterogeneity in the
    /// matching variance, declaring linear arm outcome models on the matching features.
    pub population_variance: bool,
    /// Optional cluster ids aligned to prepared complete-case rows.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
    /// Policy for treatment rows at neither declared level. The default refuses; the
    /// exclusion that restricts the target population to the rows at the two declared levels
    /// must be declared.
    pub other_levels: OtherLevels,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    pub sampling_weights: Option<SamplingWeights>,
}

impl Default for PropensityMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityMatching {
    /// Defaults: no caliper, `bootstrap_replicates = 200` (ignored for SE), clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
            caliper_scale: CaliperScale::Logit,
            se_kind: AnalyticSeKind::Homoskedastic,
            population_variance: false,
            cluster_ids: None,
            population_registry: None,
            multiway_ids: None,
            panel_times: None,
            other_levels: OtherLevels::default(),
            sampling_weights: None,
        }
    }

    /// Declare sampling weights for every row of the source table.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    /// Declare what to do with rows whose treatment is at neither declared level (see
    /// [`OtherLevels`]). The default refuses.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    /// Record a bootstrap-replicate count without licensing a bootstrap SE.
    ///
    /// The nonparametric bootstrap is invalid for nearest-neighbor matching CIs
    /// (Abadie–Imbens 2008). This setter keeps the field for callers that still set it;
    /// [`PropensityMatching::fit`] never writes [`EffectEstimate::se_bootstrap`]. Prefer
    /// the analytic SE.
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

    /// Set the maximum matching distance for an accepted match, interpreted on
    /// [`Self::caliper_scale`] (default [`CaliperScale::Logit`] — see the type-level docs for
    /// why 0.2 on that scale is the literature's rule of thumb).
    ///
    /// Defaults to `None` (no caliper): every query row is matched to its nearest donor
    /// regardless of distance.
    #[must_use]
    pub const fn with_caliper(mut self, caliper: f64) -> Self {
        self.caliper = Some(caliper);
        self
    }

    /// Set the scale on which matching distance and [`Self::caliper`] are computed.
    #[must_use]
    pub const fn with_caliper_scale(mut self, caliper_scale: CaliperScale) -> Self {
        self.caliper_scale = caliper_scale;
        self
    }

    /// Set the analytic SE kind (only `Homoskedastic`, Abadie–Imbens, is supported).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Declare linear arm outcome models and include both residual donor-reuse noise and
    /// population treatment-effect heterogeneity in the analytic variance. The raw
    /// constant-effect homoskedastic construction remains the default.
    #[must_use]
    pub const fn with_population_variance(mut self, enabled: bool) -> Self {
        self.population_variance = enabled;
        self
    }

    /// Set cluster ids aligned to prepared complete-case rows.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension).
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels for panel HAC.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
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
        // `bootstrap_replicates` is intentionally unread: the nonparametric bootstrap is
        // invalid for fixed-M NN matching (Abadie–Imbens 2008) and must not land in
        // `se_bootstrap` even when a caller still requests replicates.
        let _ = self.bootstrap_replicates;
        validate_caliper(self.caliper)?;
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        // Trim on RAW scores (mirrors PropensityWeighting): both query and donor sets are
        // restricted to common-support rows before matching.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let feature = match self.caliper_scale {
            CaliperScale::Raw => model.clipped_scores.clone(),
            CaliperScale::Logit => model.logit_scores(problem),
        };
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &feature,
            1,
            retained.as_deref(),
        );
        let weights_used = |weights: &[f64]| -> Vec<f64> {
            match &retained {
                Some(idx) => idx.iter().map(|&i| weights[i]).collect(),
                None => weights.to_vec(),
            }
        };
        // Target and sampling weights compose by product: the matched units are averaged over
        // the target law within the weighted population.
        let tw_used: Option<Vec<f64>> =
            match (problem.target_weights.as_deref(), problem.sampling_weights.as_ref()) {
                (None, None) => None,
                (Some(target), None) => Some(weights_used(target)),
                (None, Some(sampling)) => Some(weights_used(&sampling.weights)),
                (Some(target), Some(sampling)) => Some(
                    weights_used(target)
                        .into_iter()
                        .zip(weights_used(&sampling.weights))
                        .map(|(t, s)| t * s)
                        .collect(),
                ),
            };
        let clusters_used = gather_optional_row_labels(
            self.cluster_ids.as_deref(),
            problem.nrows,
            retained.as_deref(),
            "cluster_ids",
        )?;
        let times_used = gather_optional_row_labels(
            self.panel_times.as_deref(),
            problem.nrows,
            retained.as_deref(),
            "panel_times",
        )?;
        let multiway_used = gather_optional_multiway(
            self.multiway_ids.as_deref(),
            problem.nrows,
            retained.as_deref(),
        )?;
        let result = matching_contrast(
            MatchingSample {
                treatment: &t_used,
                outcome: &y_used,
                features: &s_used,
                dim: 1,
                distance: MatchingDistance::Absolute,
                matches: 1,
                population_variance: self.population_variance,
                ctx,
            },
            &problem.target_population,
            self.caliper,
            workspace,
            SeSpec {
                kind: self.se_kind,
                cluster_ids: clusters_used.as_deref(),
                multiway_ids: multiway_used.as_deref(),
                panel_times: times_used.as_deref(),
            },
            tw_used.as_deref(),
        )?;

        let ipw_target = IpwTarget::from_population(&problem.target_population).ok();
        let mut overlap_report = crate::propensity::propensity_overlap_report(
            problem,
            &model.fit.scores,
            None,
            ipw_target,
        );
        overlap_report.retained_fraction *= result.retained_fraction;
        let overlap_report = Some(overlap_report);

        Ok(EffectEstimate::new(result.ate, result.se_analytic, assumptions, problem.overlap)
            .with_se_kind(self.se_kind)
            .with_n_obs(u64::try_from(result.n_obs).unwrap_or(u64::MAX))
            .with_overlap_report(overlap_report)
            .with_treatment_levels(problem.treatment_levels)
            .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
            .with_retained_memory_bytes(Some(workspace.retained_memory_bytes())))
    }
}

/// One side of a match: row-major match features and the aligned outcomes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchSide<'a> {
    pub(crate) features: &'a [f64],
    pub(crate) outcome: &'a [f64],
}

/// The sample a matched contrast is formed over.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchingSample<'a> {
    pub(crate) treatment: &'a [f64],
    pub(crate) outcome: &'a [f64],
    /// Row-major match features (`dim` columns).
    pub(crate) features: &'a [f64],
    pub(crate) dim: usize,
    pub(crate) distance: MatchingDistance,
    /// Nearest donors averaged for each query (`1` is single-nearest-neighbour matching).
    pub(crate) matches: usize,
    pub(crate) population_variance: bool,
    /// Memory limit and cancellation for the index build and the match passes.
    pub(crate) ctx: &'a ExecutionContext,
}

/// The differences and the donor and query rows of one matching pass.
pub(crate) struct MatchedDiffs {
    /// Bias-corrected `query_y[q] − mean(donor_y[matched])`, one per matched query.
    pub(crate) diffs: Vec<f64>,
    /// Local donor indices used by the matched queries, `matches` consecutive entries per
    /// matched query (for `Kᵢ`).
    pub(crate) donors: Vec<usize>,
    /// Local index of each matched query.
    pub(crate) queries: Vec<usize>,
}

/// Match each `query` row to its `matches` nearest `donor` rows; returns bias-corrected
/// `query_y[q] − mean(donor_y[matched])` and the local donor indices used (for `Kᵢ`).
///
/// With `matches == 1` this is single-nearest-neighbour matching. With more, a query is
/// matched only when its `matches`-th nearest donor is within the caliper, so every matched
/// query averages exactly `matches` donors. Euclidean features only.
///
/// Reuses [`PropensityEstimationWorkspace`]'s cached [`MatchingIndex`] when donor geometry
/// is unchanged. One-dimensional absolute distance is answered by [`SortedScalarIndex`]:
/// the same donor a forward scan keeps, including ties, calipers, and replacement.
/// Multivariate queries stay on the cached index and its memory guard.
#[expect(
    clippy::too_many_arguments,
    reason = "the two sides of the match, the metric, the neighbour count, the caliper and the execution state are independent inputs"
)]
pub(crate) fn match_diffs(
    donors: MatchSide<'_>,
    queries: MatchSide<'_>,
    dim: usize,
    distance: MatchingDistance,
    matches: usize,
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<MatchedDiffs, EstimationError> {
    let MatchSide { features: donor_features, outcome: donor_outcome } = donors;
    let MatchSide { features: query_features, outcome: query_outcome } = queries;
    let n_donors = donor_outcome.len();
    if n_donors == 0 {
        return Err(EstimationError::data_msg("matching requires at least one donor row"));
    }
    if matches == 0 {
        return Err(EstimationError::data_msg("matching requires at least one match per query"));
    }
    if n_donors < matches {
        return Err(EstimationError::data_msg(format!(
            "matching with {matches} matches per query needs at least {matches} donor rows; the \
             donor arm has {n_donors}"
        )));
    }
    // Building the cached index is what enforces the memory guard and the reuse
    // counter. Scalar queries are answered afterwards by the sorted index.
    workspace.ensure_matching_index(donor_features, dim, distance, ctx)?;
    let n_queries = query_outcome.len();
    let slots = n_queries.checked_mul(matches).ok_or_else(|| {
        EstimationError::data_msg("matching queries times matches per query overflow usize")
    })?;
    let mut donor_rows = std::mem::take(&mut workspace.matching_donor_rows);
    let mut distances = std::mem::take(&mut workspace.matching_distances);
    donor_rows.clear();
    donor_rows.resize(slots, usize::MAX);
    distances.clear();
    distances.resize(n_queries, 0.0);
    {
        let scalar = matches == 1 && dim == 1 && matches!(distance, MatchingDistance::Absolute);
        if scalar {
            match_scalar_sorted(
                donor_features,
                query_features,
                n_queries,
                caliper,
                &mut donor_rows,
                &mut distances,
                ctx,
            )?;
        } else {
            // Multivariate Euclidean search keeps `MatchingIndex` and its memory guard.
            // An exact spatial index is a separate attempt and does not lift that guard.
            let index = workspace.matching_index.as_ref().expect("ensured");
            if matches == 1 {
                index
                    .match_all_within(
                        query_features,
                        n_queries,
                        caliper,
                        &mut donor_rows,
                        &mut distances,
                        ctx,
                    )
                    .map_err(stats_err)?;
            } else {
                match_nearest_k(
                    index,
                    query_features,
                    matches,
                    caliper,
                    &mut donor_rows,
                    &mut distances,
                    ctx,
                )?;
            }
        }
    }
    let mut diffs = Vec::with_capacity(n_queries);
    let mut used_donors = Vec::with_capacity(slots);
    let mut used_queries = Vec::with_capacity(n_queries);
    let mu_donor = fit_linear_mean(donor_features, donor_outcome, dim);
    // One match is read off directly so single-neighbour results do not pass through a sum.
    let average = |set: &[usize], value: &dyn Fn(usize) -> f64| -> f64 {
        match set {
            [only] => value(*only),
            _ => set.iter().map(|&d| value(d)).sum::<f64>() / set.len() as f64,
        }
    };
    for q in 0..n_queries {
        let set = &donor_rows[q * matches..(q + 1) * matches];
        if set[0] != usize::MAX {
            let raw = query_outcome[q] - average(set, &|d| donor_outcome[d]);
            let bias = match &mu_donor {
                Some(beta) => {
                    let mq = predict_linear(beta, query_features, dim, q);
                    let md = average(set, &|d| predict_linear(beta, donor_features, dim, d));
                    mq - md
                }
                None => 0.0,
            };
            diffs.push(raw - bias);
            used_donors.extend_from_slice(set);
            used_queries.push(q);
        }
    }
    workspace.matching_donor_rows = donor_rows;
    workspace.matching_distances = distances;
    Ok(MatchedDiffs { diffs, donors: used_donors, queries: used_queries })
}

/// A caliper is a maximum distance: finite and non-negative.
fn validate_caliper(caliper: Option<f64>) -> Result<(), EstimationError> {
    match caliper {
        Some(limit) if !(limit.is_finite() && limit >= 0.0) => {
            Err(EstimationError::data_msg("matching caliper must be finite and non-negative"))
        }
        _ => Ok(()),
    }
}

/// Exact nearest-donor index for one scalar score.
///
/// Donors stay in the index after a query matches them, so a later query can use the same
/// donor (matching with replacement). Equal absolute differences resolve to the lowest donor
/// index, which is the donor a forward scan keeps under a strict `<` update.
struct SortedScalarIndex {
    /// `(score, lowest donor index with that score)` in ascending score order.
    by_score: Vec<(f64, usize)>,
}

impl SortedScalarIndex {
    /// Index `donors` in their given order. Scores are finite; the matching index build
    /// refuses a non-finite donor before this is called.
    #[must_use]
    fn from_donors(donors: &[f64]) -> Self {
        let mut by_score: Vec<(f64, usize)> =
            donors.iter().copied().enumerate().map(|(index, score)| (score, index)).collect();
        by_score.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        // The first entry of an equal-score run is its lowest donor index, which is the
        // only one a query can resolve to; the rest never change a result.
        by_score.dedup_by(|later, kept| later.0.total_cmp(&kept.0).is_eq());
        Self { by_score }
    }

    /// Nearest donor to `query`, or `None` when that distance is non-finite or past `caliper`.
    ///
    /// A distance equal to the caliper is a match. `caliper == None` accepts every finite
    /// nearest donor.
    #[must_use = "the donor and distance are the match"]
    fn nearest(&self, query: f64, caliper: Option<f64>) -> Option<(usize, f64)> {
        let split = self.by_score.partition_point(|&(score, _)| score < query);
        let left = split.checked_sub(1).map(|i| (query - self.by_score[i].0).abs());
        let right = self.by_score.get(split).map(|&(score, _)| (query - score).abs());
        let best = match (left, right) {
            (Some(l), Some(r)) => l.min(r),
            (Some(distance), None) | (None, Some(distance)) => distance,
            (None, None) => return None,
        };
        if !best.is_finite() {
            return None;
        }
        // `|query - score|` is non-decreasing as `score` moves away from `query`, so every
        // donor tied with `best` is contiguous around the two neighbours.
        let mut position = usize::MAX;
        let mut i = split;
        while i > 0 {
            i -= 1;
            let (score, donor) = self.by_score[i];
            if (query - score).abs().total_cmp(&best).is_ne() {
                break;
            }
            position = position.min(donor);
        }
        for &(score, donor) in &self.by_score[split..] {
            if (query - score).abs().total_cmp(&best).is_ne() {
                break;
            }
            position = position.min(donor);
        }
        if position == usize::MAX {
            return None;
        }
        if let Some(limit) = caliper {
            // Negated so that a NaN limit refuses instead of accepting every distance.
            if !matches!(
                best.partial_cmp(&limit),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) {
                return None;
            }
        }
        Some((position, best))
    }
}

/// Answer one-dimensional absolute-distance queries from [`SortedScalarIndex`].
///
/// `donor_rows[q] == usize::MAX` and `distances[q] == ∞` when the nearest donor is outside
/// the caliper or at a non-finite distance.
fn match_scalar_sorted(
    donors: &[f64],
    queries: &[f64],
    n_queries: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    if queries.len() != n_queries {
        return Err(stats_err(StatsError::Shape { message: "queries length != n_queries * dim" }));
    }
    let index = SortedScalarIndex::from_donors(donors);
    for (q, &query) in queries.iter().enumerate() {
        if q % MATCH_CANCELLATION_CHUNK == 0 && ctx.cancellation.is_cancelled() {
            return Err(stats_err(StatsError::Cancelled));
        }
        if !query.is_finite() {
            return Err(stats_err(StatsError::Shape { message: "matching query must be finite" }));
        }
        if let Some((donor, distance)) = index.nearest(query, caliper) {
            donor_rows[q] = donor;
            distances[q] = distance;
        } else {
            donor_rows[q] = usize::MAX;
            distances[q] = f64::INFINITY;
        }
    }
    Ok(())
}

/// Fill `donor_rows` (`matches` slots per query) with each query's `matches` nearest donors.
/// A query whose farthest match is beyond the caliper, or at a non-finite distance, keeps
/// `usize::MAX` in its first slot and `distances[q] = ∞`; otherwise `distances[q]` is the
/// farthest of its matches.
fn match_nearest_k(
    index: &MatchingIndex,
    query_features: &[f64],
    matches: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    let dim = index.dim;
    let mut found = Vec::with_capacity(matches);
    for (q, query) in query_features.chunks_exact(dim).enumerate() {
        if q % MATCH_CANCELLATION_CHUNK == 0 && ctx.cancellation.is_cancelled() {
            return Err(stats_err(StatsError::Cancelled));
        }
        index.nearest_k(query, matches, &mut found).map_err(stats_err)?;
        let farthest = found.last().map_or(f64::INFINITY, |&(_, d)| d);
        let within = found.len() == matches
            && farthest.is_finite()
            && caliper.is_none_or(|limit| farthest <= limit);
        if within {
            for (slot, &(row, _)) in
                donor_rows[q * matches..(q + 1) * matches].iter_mut().zip(&found)
            {
                *slot = row;
            }
            distances[q] = farthest;
        } else {
            donor_rows[q * matches] = usize::MAX;
            distances[q] = f64::INFINITY;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct MatchedEstimate {
    pub(crate) ate: f64,
    pub(crate) se_analytic: f64,
    pub(crate) retained_fraction: f64,
    pub(crate) n_obs: usize,
}

/// ATT/ATC/ATE via nearest-neighbor matching on `features` (dim columns, row-major).
///
/// ATT matches treated→nearest control; ATC matches control→nearest treated (sign-flipped);
/// ATE pools both directions' per-unit imputed effects (Abadie–Imbens style).
#[allow(
    clippy::too_many_lines,
    reason = "split by arm, match, then form the point, Abadie-Imbens variance and standard error from the same donor usage"
)]
pub(crate) fn matching_contrast(
    sample: MatchingSample<'_>,
    target: &TargetPopulation,
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
    se: SeSpec<'_>,
    target_weights: Option<&[f64]>,
) -> Result<MatchedEstimate, EstimationError> {
    let MatchingSample {
        treatment,
        outcome,
        features,
        dim,
        distance,
        matches,
        population_variance,
        ctx,
    } = sample;
    let SeSpec { kind: se_kind, cluster_ids, multiway_ids, panel_times } = se;
    if let Some(ids) = cluster_ids {
        if ids.len() != treatment.len() {
            return Err(EstimationError::data_msg("matching cluster_ids length != treatment rows"));
        }
    }
    if let Some(times) = panel_times {
        if times.len() != treatment.len() {
            return Err(EstimationError::data_msg("matching panel_times length != treatment rows"));
        }
    }
    if let Some(dims) = multiway_ids {
        for (i, d) in dims.iter().enumerate() {
            if d.len() != treatment.len() {
                return Err(EstimationError::data_msg(format!(
                    "matching multiway_ids[{i}] length {} != treatment rows",
                    d.len()
                )));
            }
        }
    }
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg("matching requires both treated and control rows"));
    }
    let treated_feat = gather_rowmajor(features, dim, &treated_idx);
    let control_feat = gather_rowmajor(features, dim, &control_idx);
    let treated_y = gather(outcome, &treated_idx);
    let control_y = gather(outcome, &control_idx);

    let (per_unit_effects, donor_usage, n_donors, effect_rows): (
        Vec<f64>,
        Vec<usize>,
        usize,
        Vec<usize>,
    ) = match target {
        TargetPopulation::Treated => {
            let MatchedDiffs { diffs, donors, queries: q_local } = match_diffs(
                MatchSide { features: &control_feat, outcome: &control_y },
                MatchSide { features: &treated_feat, outcome: &treated_y },
                dim,
                distance,
                matches,
                caliper,
                workspace,
                ctx,
            )?;
            let rows: Vec<usize> = q_local.iter().map(|&q| treated_idx[q]).collect();
            (diffs, donors, control_y.len(), rows)
        }
        TargetPopulation::Untreated => {
            let MatchedDiffs { diffs, donors, queries: q_local } = match_diffs(
                MatchSide { features: &treated_feat, outcome: &treated_y },
                MatchSide { features: &control_feat, outcome: &control_y },
                dim,
                distance,
                matches,
                caliper,
                workspace,
                ctx,
            )?;
            let flipped: Vec<f64> = diffs.into_iter().map(|d| -d).collect();
            let rows: Vec<usize> = q_local.iter().map(|&q| control_idx[q]).collect();
            (flipped, donors, treated_y.len(), rows)
        }
        TargetPopulation::AllObserved
        | TargetPopulation::Predicate(_)
        | TargetPopulation::CustomDistribution(_) => {
            let MatchedDiffs { diffs: att_diffs, donors: att_donors, queries: att_q } =
                match_diffs(
                    MatchSide { features: &control_feat, outcome: &control_y },
                    MatchSide { features: &treated_feat, outcome: &treated_y },
                    dim,
                    distance,
                    matches,
                    caliper,
                    workspace,
                    ctx,
                )?;
            let MatchedDiffs { diffs: atc_raw, donors: atc_donors, queries: atc_q } = match_diffs(
                MatchSide { features: &treated_feat, outcome: &treated_y },
                MatchSide { features: &control_feat, outcome: &control_y },
                dim,
                distance,
                matches,
                caliper,
                workspace,
                ctx,
            )?;
            let atc_diffs: Vec<f64> = atc_raw.into_iter().map(|d| -d).collect();
            let n_control = control_y.len();
            let mut effects = att_diffs;
            effects.extend(atc_diffs);
            let mut donors = att_donors;
            donors.extend(atc_donors.into_iter().map(|d| d + n_control));
            let mut rows: Vec<usize> = att_q.iter().map(|&q| treated_idx[q]).collect();
            rows.extend(atc_q.iter().map(|&q| control_idx[q]));
            (effects, donors, n_control + treated_y.len(), rows)
        }
        _ => {
            return Err(EstimationError::unsupported(
                "matching estimators support AllObserved, Treated, Untreated, Predicate, or CustomDistribution",
            ));
        }
    };
    if per_unit_effects.is_empty() {
        return Err(EstimationError::data_msg("no matched units within caliper"));
    }
    let n_eligible = match target {
        TargetPopulation::Treated => treated_idx.len(),
        TargetPopulation::Untreated => control_idx.len(),
        _ => treated_idx.len() + control_idx.len(),
    };
    let retained_fraction = per_unit_effects.len() as f64 / n_eligible.max(1) as f64;
    if matches!(target, TargetPopulation::CustomDistribution(_)) && target_weights.is_none() {
        return Err(EstimationError::unsupported(
            "CustomDistribution requires PopulationRegistry weights on the prepared problem",
        ));
    }
    let ate = if let Some(tw) = target_weights {
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &eff) in per_unit_effects.iter().enumerate() {
            let w = tw.get(effect_rows[i]).copied().unwrap_or(0.0);
            num += w * eff;
            den += w;
        }
        if den.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(EstimationError::data_msg(
                "CustomDistribution weights left no mass on matched units",
            ));
        }
        num / den
    } else {
        per_unit_effects.iter().sum::<f64>() / per_unit_effects.len() as f64
    };
    // One-arm (ATT/ATC) vs two-arm ATE Abadie–Imbens variance.
    let ate_form = !matches!(target, TargetPopulation::Treated | TargetPopulation::Untreated);
    // Under a custom target or sampling weights the estimator is the weighted matched mean, so
    // each unit's outcome enters with its query weight plus the weights of the queries it
    // donates to; the equal-weight `(1 + K)/n` coefficients would describe a different
    // estimator.
    let weighted_coefficients = target_weights.map(|tw| {
        custom_unit_coefficients(
            tw,
            &effect_rows,
            &donor_usage,
            &treated_idx,
            &control_idx,
            treatment.len(),
            ate_form,
            matches!(target, TargetPopulation::Untreated),
            matches,
        )
    });
    let population_se = if population_variance {
        let unit_weights = vec![1.0; treatment.len()];
        let weights = target_weights.unwrap_or(&unit_weights);
        let mut coefficients = custom_unit_coefficients(
            weights,
            &effect_rows,
            &donor_usage,
            &treated_idx,
            &control_idx,
            treatment.len(),
            ate_form,
            matches!(target, TargetPopulation::Untreated),
            matches,
        );
        if ate_form {
            let mass = effect_rows.iter().map(|&row| weights[row]).sum::<f64>();
            let queried: std::collections::HashSet<usize> = effect_rows.iter().copied().collect();
            for (row, coefficient) in coefficients.iter_mut().enumerate() {
                if !queried.contains(&row) {
                    *coefficient -= weights[row] / mass;
                }
            }
        }
        Some(population_matching_se(
            &treated_feat,
            &treated_y,
            &control_feat,
            &control_y,
            &treated_idx,
            &control_idx,
            features,
            dim,
            &effect_rows,
            &coefficients,
            weights,
        )?)
    } else {
        None
    };
    let se_analytic = match se_kind {
        AnalyticSeKind::Homoskedastic if population_se.is_some() => {
            population_se.unwrap_or(f64::NAN)
        }
        AnalyticSeKind::Homoskedastic => match &weighted_coefficients {
            Some(coefficients) => weighted_abadie_imbens_se(
                &per_unit_effects,
                ate,
                &effect_rows,
                coefficients,
                target_weights.unwrap_or(&[]),
                matches,
            ),
            None => abadie_imbens_se(&per_unit_effects, &donor_usage, n_donors, ate_form, matches),
        },
        AnalyticSeKind::Hc0 | AnalyticSeKind::Hc1 | AnalyticSeKind::Hc2 | AnalyticSeKind::Hc3 => {
            return Err(EstimationError::unsupported(
                "matching does not implement HC0–HC3 sandwich SEs; use Homoskedastic (Abadie–Imbens)",
            ));
        }
        // A matched effect mixes the query unit's outcome with its donors' outcomes, and
        // donors come from other clusters and time points. Charging the whole per-query
        // residual to the query's cluster (or time) mis-assigns the donor noise, so a
        // cluster, multiway or HAC sandwich over those contributions is not a valid
        // variance for the matching estimator.
        //
        // Kept as a refusal rather than derived, because no clustered influence
        // function is available for this exact estimator:
        // - Fixed-M nearest-neighbour matching is not asymptotically linear in an influence
        //   function of one observation; its only linear representation is the
        //   martingale one, √N(τ̂ − τ) ≈ N^{-1/2} Σ_i [τ(X_i) − τ + (2W_i − 1)(1 + K_i/M) ε_i]
        //   (Abadie–Imbens 2006, Econometrica 74(1); 2012, JASA 107(498)), proved for
        //   random samples. Its K_i distribution and the negligibility of the
        //   bias-corrected conditional bias are not established under within-cluster or
        //   serial dependence of the covariates and of the matches themselves.
        // - Even granting that representation, a cluster sum needs per-unit ε_i
        //   (within-cluster cross products ε_i ε_j) and per-unit τ(X_i). Abadie–Imbens
        //   recover only the diagonal σ²(X_i) from same-arm matches; here the bias
        //   correction is a linear fit on the match feature (the one-dimensional score
        //   for propensity matching), which is not a consistent μ_w(X), and the
        //   published SE is a pooled ½·Var(τ̂_i) proxy, not the conditional estimator.
        // - Propensity matching on an estimated score needs the Abadie–Imbens (2016,
        //   Econometrica 84(2)) estimated-score adjustment, which is not implemented even
        //   for iid data; a clustered version would also need it.
        // - Resampling does not fill the gap: the (cluster) bootstrap is inconsistent for
        //   fixed-M matching (Abadie–Imbens 2008, Econometrica 76(6)), and the wild
        //   bootstrap of the martingale representation (Otsu–Rai 2017, JASA 112(520)) is
        //   proved for iid samples only.
        AnalyticSeKind::Cluster
        | AnalyticSeKind::Multiway
        | AnalyticSeKind::NeweyWest { .. }
        | AnalyticSeKind::PanelClusterHac { .. } => {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("estimator_inference_mismatch"),
                "matching has no valid cluster, multiway or HAC standard error (donor outcomes \
                 belong to other clusters and times than the matched unit); use Homoskedastic \
                 (Abadie–Imbens), or AIPW, which implements cluster, multiway and HAC standard \
                 errors, for clustered or serially dependent data",
            ));
        }
    };
    Ok(MatchedEstimate { ate, se_analytic, retained_fraction, n_obs: per_unit_effects.len() })
}

/// Coefficient `c_i` of each sample row's outcome in the weighted pooled matching estimator
/// `τ̂ = Σ_q w̃_q τ̂_q` (`w̃_q = w_q / M`, `M = Σ_q w_q` over the matched queries): the row's own
/// query weight `w_i / M` plus the weighted donor reuse `K^w_i / M`, where
/// `K^w_i = Σ_{q: donor(q) = i} w_q` is the weight of the queries row `i` donates to.
///
/// With the ATE form every unit is a query once, so a unit dropped by the caliper still counts
/// its own `w_i / M` (the unweighted Abadie–Imbens ATE sum `Σ_j (1 + K_j)²` runs over every
/// unit, `K_j = 0` included); the one-arm (ATT/ATC) form counts matched queries only.
///
/// At constant weights `w_i = w` this is `c_i = (1 + K_i) / n` (ATE) and `c_q = 1/n`,
/// `c_j = K_j / n` (ATT), the coefficients behind the unweighted `(1 + K_j)²` and `n + ΣK²`
/// sums, whatever the scale of `w`.
#[expect(
    clippy::too_many_arguments,
    reason = "the coefficients need the weights, both arms' rows and the donor mapping"
)]
fn custom_unit_coefficients(
    target_weights: &[f64],
    effect_rows: &[usize],
    donor_local: &[usize],
    treated_idx: &[usize],
    control_idx: &[usize],
    nrows: usize,
    ate_form: bool,
    donors_are_treated: bool,
    matches: usize,
) -> Vec<f64> {
    let n_control = control_idx.len();
    let weight = |row: usize| target_weights.get(row).copied().unwrap_or(0.0);
    let mass: f64 = effect_rows.iter().map(|&r| weight(r)).sum();
    let mut coefficient = vec![0.0; nrows];
    if ate_form {
        for (row, slot) in coefficient.iter_mut().enumerate() {
            *slot = weight(row) / mass;
        }
        for &row in effect_rows {
            coefficient[row] -= weight(row) / mass;
        }
    }
    for (&row, donors) in effect_rows.iter().zip(donor_local.chunks(matches)) {
        let w = weight(row) / mass;
        coefficient[row] += w;
        // Each of a query's `matches` donors carries an equal share of its weight.
        let share = w / matches as f64;
        for &donor in donors {
            // The ATC match draws its donors from the treated arm alone, indexed without the
            // control offset the pooled (ATE) donor list carries for its ATC half.
            let donor_row = if donors_are_treated {
                treated_idx[donor]
            } else if donor < n_control {
                control_idx[donor]
            } else {
                treated_idx[donor - n_control]
            };
            coefficient[donor_row] += share;
        }
    }
    coefficient
}

/// Abadie–Imbens homoskedastic SE of the weighted matched mean:
/// `Var = σ̂² Σ_i c_i²`, `σ̂² = ½ · Σ_q w̃_q (τ̂_q − τ̂)² / (1 − Σ_q w̃_q²)`.
///
/// The denominator is the reliability-weights degrees-of-freedom correction of the weighted
/// variance, so it generalizes the unweighted `Var(τ̂ᵢ)/2` with its `n − 1` divisor. At
/// constant weights `w̃_q = 1/n`, so `1 − Σ w̃² = (n − 1)/n`,
/// `σ̂² = ½ Σ(τ̂_q − τ̂)² / (n − 1)` and, with the coefficients of
/// [`custom_unit_coefficients`], `Var = σ̂² Σ_j (1 + K_j)² / n²` (ATE) or
/// `σ̂² (n + ΣK²) / n²` (ATT/ATC): exactly [`abadie_imbens_se`], independent of the scale of
/// the weights. A single matched query has no variance estimate and returns `NaN`.
fn weighted_abadie_imbens_se(
    effects: &[f64],
    ate: f64,
    effect_rows: &[usize],
    coefficients: &[f64],
    target_weights: &[f64],
    matches: usize,
) -> f64 {
    let weight = |row: usize| target_weights.get(row).copied().unwrap_or(0.0);
    let mass: f64 = effect_rows.iter().map(|&r| weight(r)).sum();
    let concentration: f64 = effect_rows.iter().map(|&r| (weight(r) / mass).powi(2)).sum();
    let denominator = 1.0 - concentration;
    if denominator <= 0.0 {
        return f64::NAN;
    }
    let sigma2: f64 = noise_share(matches)
        * effects
            .iter()
            .zip(effect_rows)
            .map(|(e, &r)| weight(r) / mass * (e - ate).powi(2))
            .sum::<f64>()
        / denominator;
    (sigma2 * coefficients.iter().map(|c| c * c).sum::<f64>()).max(0.0).sqrt()
}

/// Abadie–Imbens population variance: outcome-noise donor reuse plus the variance of the
/// conditional treatment effect over the target population. The fitted arm means are the
/// linear models already declared by matching's bias correction. Estimating noise from the
/// entire matched-contrast spread would charge effect heterogeneity to donor noise as well.
#[allow(clippy::too_many_arguments, reason = "arm regressions and aligned matched population")]
fn population_matching_se(
    treated_features: &[f64],
    treated_y: &[f64],
    control_features: &[f64],
    control_y: &[f64],
    treated_rows: &[usize],
    control_rows: &[usize],
    features: &[f64],
    dim: usize,
    effect_rows: &[usize],
    coefficients: &[f64],
    weights: &[f64],
) -> Result<f64, EstimationError> {
    let beta1 = fit_linear_mean(treated_features, treated_y, dim)
        .ok_or_else(|| EstimationError::stats_msg("matching arm outcome model failed"))?;
    let beta0 = fit_linear_mean(control_features, control_y, dim)
        .ok_or_else(|| EstimationError::stats_msg("matching arm outcome model failed"))?;
    let n = treated_y.len() + control_y.len();
    let parameters = beta1.len() + beta0.len();
    if n <= parameters || effect_rows.len() < 2 {
        return Err(EstimationError::data_msg(
            "population matching variance requires residual degrees of freedom",
        ));
    }
    let mut residual_sum = 0.0;
    for (x, y, beta) in
        [(treated_features, treated_y, &beta1), (control_features, control_y, &beta0)]
    {
        for (row, outcome) in y.iter().enumerate() {
            residual_sum += (outcome - predict_linear(beta, x, dim, row)).powi(2);
        }
    }
    let noise = residual_sum / (n - parameters) as f64;
    let mass = effect_rows.iter().map(|&row| weights[row]).sum::<f64>();
    let contrasts: Vec<f64> = effect_rows
        .iter()
        .map(|&row| {
            predict_linear(&beta1, features, dim, row) - predict_linear(&beta0, features, dim, row)
        })
        .collect();
    let mean = effect_rows
        .iter()
        .zip(&contrasts)
        .map(|(&row, contrast)| weights[row] * contrast)
        .sum::<f64>()
        / mass;
    let count = effect_rows.len() as f64;
    let heterogeneity = effect_rows
        .iter()
        .zip(&contrasts)
        .map(|(&row, contrast)| (weights[row] / mass * (contrast - mean)).powi(2))
        .sum::<f64>()
        * count
        / (count - 1.0);
    let target_design: Vec<f64> = std::iter::once(1.0)
        .chain((0..dim).map(|c| {
            effect_rows.iter().map(|&row| weights[row] * features[row * dim + c]).sum::<f64>()
                / mass
        }))
        .collect();
    let norm = bias_adjusted_coefficient_norm(
        treated_features,
        dim,
        treated_rows,
        coefficients,
        &target_design,
    )? + bias_adjusted_coefficient_norm(
        control_features,
        dim,
        control_rows,
        coefficients,
        &target_design,
    )?;
    Ok((noise * norm + heterogeneity).sqrt())
}

/// Exact coefficients of outcome noise after the shared OLS bias correction. If the raw
/// donor coefficient vector is c, the corrected vector is c + X(X'X)^-1(a-X'c), where a is
/// the target query's covariate mean. Squaring raw donor coefficients alone loses the fitted
/// regression's variance and its covariance with the matched outcomes.
fn bias_adjusted_coefficient_norm(
    features: &[f64],
    dim: usize,
    rows: &[usize],
    coefficients: &[f64],
    target: &[f64],
) -> Result<f64, EstimationError> {
    let p = dim + 1;
    let mut gram = vec![0.0; p * p];
    let mut mismatch = target.to_vec();
    let row = |i: usize| {
        std::iter::once(1.0)
            .chain(features[i * dim..(i + 1) * dim].iter().copied())
            .collect::<Vec<_>>()
    };
    for (i, &source) in rows.iter().enumerate() {
        let x = row(i);
        for c in 0..p {
            mismatch[c] -= x[c] * coefficients[source];
            for d in 0..p {
                gram[c * p + d] += x[c] * x[d];
            }
        }
    }
    let correction = crate::util::solve_spd(&gram, &mismatch, p)
        .ok_or_else(|| EstimationError::stats_msg("singular matching bias-adjustment variance"))?;
    Ok(rows
        .iter()
        .enumerate()
        .map(|(i, &source)| {
            (coefficients[source] + row(i).iter().zip(&correction).map(|(x, b)| x * b).sum::<f64>())
                .powi(2)
        })
        .sum())
}

/// Donor-reuse counts `Kⱼ` from query→donor local indices.
fn donor_counts(donor_local: &[usize], n_donors: usize) -> Vec<usize> {
    let mut k = vec![0usize; n_donors.max(1)];
    for &d in donor_local {
        if d < k.len() {
            k[d] += 1;
        }
    }
    k
}

/// Share of the variance of a matched difference `Yᵢ − mean(Y_donors)` that one outcome's
/// noise accounts for: the difference carries the query's noise plus the average of `matches`
/// donors', so under a constant effect `Var(τ̂ᵢ) = σ² (1 + 1/M)` and `σ̂² = Var(τ̂ᵢ) / (1 + 1/M)`.
/// One match gives the equal-arm one half.
fn noise_share(matches: usize) -> f64 {
    1.0 / (1.0 + 1.0 / matches as f64)
}

/// Abadie–Imbens (2006) SE for `M`-nearest-neighbour matching with replacement
/// (homoskedastic); `M = 1` is single-nearest-neighbour matching.
///
/// `σ̂² = Var(τ̂ᵢ) / (1 + 1/M)` (equal-arm residual variance; one half at `M = 1`). With donor
/// reuse counts `Kⱼ` (each of a query's `M` donors is counted once, so `Σⱼ Kⱼ = M·n`):
/// - **ATT/ATC** (`ate_form = false`): `Var = σ̂² (n + Σⱼ (Kⱼ/M)²) / n²`
/// - **ATE** (`ate_form = true`): `Var = σ̂² Σⱼ (1 + Kⱼ/M)² / n²`
///   `= σ̂² (n_donors + 2 Σ K/M + Σ (K/M)²) / n²` (the ATT form misses the cross term).
fn abadie_imbens_se(
    effects: &[f64],
    donor_local: &[usize],
    n_donors: usize,
    ate_form: bool,
    matches: usize,
) -> f64 {
    let n = effects.len();
    if n < 2 || n.checked_mul(matches) != Some(donor_local.len()) {
        return sample_std(effects) / (n as f64).sqrt();
    }
    let k = donor_counts(donor_local, n_donors);
    let share = matches as f64;
    let mean = effects.iter().sum::<f64>() / n as f64;
    let var_tau = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let sigma2 = (var_tau * noise_share(matches)).max(0.0);
    let weight = if ate_form {
        // Σⱼ (1 + Kⱼ/M)² over the donor-count vector (includes K=0 units).
        k.iter().map(|&kj| (1.0 + kj as f64 / share).powi(2)).sum::<f64>()
    } else {
        let sum_k2: f64 = k.iter().map(|&kj| (kj as f64 / share).powi(2)).sum();
        n as f64 + sum_k2
    };
    let var = sigma2 * weight / (n as f64).powi(2);
    var.sqrt()
}

/// OLS of `y` on `[1, x]` (row-major `x` with `dim` columns). Returns `[intercept, β…]`.
fn fit_linear_mean(features: &[f64], y: &[f64], dim: usize) -> Option<Vec<f64>> {
    let n = y.len();
    if n < dim + 1 || dim == 0 {
        // Fall back to intercept-only mean when underdetermined.
        if n == 0 {
            return None;
        }
        return Some(vec![y.iter().sum::<f64>() / n as f64]);
    }
    let p = dim + 1;
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let mut row = vec![1.0; p];
        for d in 0..dim {
            row[d + 1] = features[i * dim + d];
        }
        for a in 0..p {
            xty[a] += row[a] * y[i];
            for b in 0..p {
                xtx[a * p + b] += row[a] * row[b];
            }
        }
    }
    crate::util::solve_spd(&xtx, &xty, p)
}

fn predict_linear(beta: &[f64], features: &[f64], dim: usize, row: usize) -> f64 {
    if beta.len() == 1 {
        return beta[0];
    }
    let mut y = beta[0];
    for d in 0..dim.min(beta.len().saturating_sub(1)) {
        y += beta[d + 1] * features[row * dim + d];
    }
    y
}

#[cfg(test)]
mod scale_tests;

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;

    use super::*;

    #[test]
    fn fitted_bias_adjustment_noise_variance_matches_outcome_basis_oracle() {
        let n = 12;
        let treatment: Vec<f64> = (0..n).map(|i| f64::from(i % 2 == 1)).collect();
        let features: Vec<f64> =
            (0..n).flat_map(|i| [1.3 * i as f64, (i * i % 7) as f64 + 0.17 * i as f64]).collect();
        let (treated, control) = split_by_treatment(&treatment);
        let mut coefficients = vec![1.0 / n as f64; n];
        for i in 0..n {
            let donor = (0..n)
                .filter(|&j| treatment[i].to_bits() != treatment[j].to_bits())
                .min_by(|&a, &b| {
                    let distance = |j: usize| {
                        (0..2)
                            .map(|d| (features[2 * i + d] - features[2 * j + d]).powi(2))
                            .sum::<f64>()
                    };
                    distance(a).total_cmp(&distance(b)).then(a.cmp(&b))
                })
                .unwrap();
            coefficients[donor] += 1.0 / n as f64;
        }
        let target = [
            1.0,
            (0..n).map(|i| features[2 * i]).sum::<f64>() / n as f64,
            (0..n).map(|i| features[2 * i + 1]).sum::<f64>() / n as f64,
        ];
        let norm = bias_adjusted_coefficient_norm(
            &gather_rowmajor(&features, 2, &treated),
            2,
            &treated,
            &coefficients,
            &target,
        )
        .unwrap()
            + bias_adjusted_coefficient_norm(
                &gather_rowmajor(&features, 2, &control),
                2,
                &control,
                &coefficients,
                &target,
            )
            .unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut workspace = PropensityEstimationWorkspace::default();
        let mut oracle = 0.0;
        // Conditional on X,T, matching and its fitted linear bias correction are exactly
        // linear in y. Evaluating every unit outcome basis gives its complete noise vector.
        for source in 0..n {
            let mut outcome = vec![0.0; n];
            outcome[source] = 1.0;
            let fit = matching_contrast(
                MatchingSample {
                    treatment: &treatment,
                    outcome: &outcome,
                    features: &features,
                    dim: 2,
                    distance: MatchingDistance::Euclidean,
                    matches: 1,
                    population_variance: false,
                    ctx: &ctx,
                },
                &TargetPopulation::AllObserved,
                None,
                &mut workspace,
                SeSpec {
                    kind: AnalyticSeKind::Homoskedastic,
                    cluster_ids: None,
                    multiway_ids: None,
                    panel_times: None,
                },
                None,
            )
            .unwrap();
            oracle += fit.ate.powi(2);
        }
        assert!(
            (norm - oracle).abs() < 1e-12,
            "coefficient norm {norm} vs noise-basis oracle {oracle}"
        );
    }

    #[test]
    fn pivot_floor_follows_the_matrix_scale() {
        // A design in units of 1e-8 is a scaled identity, not a singular system.
        let tiny = [1e-16, 0.0, 0.0, 1e-16];
        let tiny_b = [1e-16, 2e-16];
        let x = crate::util::solve_spd(&tiny, &tiny_b, 2).expect("tiny identity");
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 2.0).abs() < 1e-6);
        // A pair collinear to 1 part in 1e16 of a 1e16 Gram must be refused.
        // An absolute 1e-14 floor accepts the residual pivot of 1.
        let huge = [1e16, 1e16, 1e16, 1e16 + 1.0];
        let huge_b = [1.0, 1.0];
        assert!(crate::util::solve_spd(&huge, &huge_b, 2).is_none());
    }

    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = antecedent_kernels::sigmoid(logit);
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
    fn abadie_imbens_se_grows_with_donor_reuse() {
        let effects = [1.0, 1.2, 0.8, 1.1];
        // Four queries, two unique donors reused twice each (full K vector length n).
        let donors_reuse = vec![0usize, 0, 1, 1];
        let donors_unique = vec![0usize, 1, 2, 3];
        let se_reuse = abadie_imbens_se(&effects, &donors_reuse, 4, true, 1);
        let se_unique = abadie_imbens_se(&effects, &donors_unique, 4, true, 1);
        assert!(se_reuse > se_unique, "reuse={se_reuse} unique={se_unique}");
    }

    /// Algebraic identity: ATE SE = √(σ² Σ(1+K)² / n²) and exceeds the old ATT-form
    /// `(n + Σ K²)` whenever some K > 0.
    #[test]
    fn abadie_imbens_ate_se_one_plus_k_squared_identity() {
        let effects = [1.0, 1.2, 0.8, 1.1];
        // K_M = [2, 2, 0, 0] via n_donors = 4 and donors {0,1} each reused twice.
        let donors = vec![0usize, 0, 1, 1];
        let n = effects.len() as f64;
        let mean = effects.iter().sum::<f64>() / n;
        let var_tau = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma2 = var_tau * 0.5;
        let k = [2.0_f64, 2.0, 0.0, 0.0];
        let sum_1pk2: f64 = k.iter().map(|kj| (1.0 + kj).powi(2)).sum();
        let expected = (sigma2 * sum_1pk2 / n.powi(2)).sqrt();
        let se = abadie_imbens_se(&effects, &donors, 4, true, 1);
        assert!((se - expected).abs() < 1e-14, "se={se} expected={expected} (Σ(1+K)²={sum_1pk2})");
        let sum_k2: f64 = k.iter().map(|kj| kj.powi(2)).sum();
        let old_att_form = (sigma2 * (n + sum_k2) / n.powi(2)).sqrt();
        assert!(
            se > old_att_form,
            "ATE (1+K)² form {se} should exceed ATT-form {old_att_form} when some K>0"
        );
        // Expansion check: Σ(1+K)² = n + 2ΣK + ΣK² when |K|=n.
        let sum_k: f64 = k.iter().sum();
        assert!((sum_1pk2 - (n + 2.0 * sum_k + sum_k2)).abs() < 1e-14);
    }

    #[test]
    fn abadie_imbens_att_keeps_one_arm_formula() {
        let effects = [1.0, 1.2, 0.8, 1.1];
        let donors = vec![0usize, 0, 1, 1];
        let n = effects.len() as f64;
        let mean = effects.iter().sum::<f64>() / n;
        let var_tau = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma2 = var_tau * 0.5;
        // ATT donor pool length 2: K=[2,2], weight = n + ΣK² = 4+8 = 12.
        let expected = (sigma2 * (n + 8.0) / n.powi(2)).sqrt();
        let se = abadie_imbens_se(&effects, &donors, 2, false, 1);
        assert!((se - expected).abs() < 1e-14, "se={se} expected={expected}");
        let se_ate = abadie_imbens_se(&effects, &donors, 4, true, 1);
        assert!(se_ate > se, "ATE form should exceed ATT form under reuse");
    }

    fn homoskedastic() -> SeSpec<'static> {
        SeSpec {
            kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    /// Donor outcomes belong to other clusters and times than the matched unit, so no
    /// cluster, multiway or HAC variance is published for matching.
    #[test]
    fn clustered_and_hac_matching_variance_is_refused() {
        let t = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let y = [2.0, 2.5, 3.0, 1.0, 1.4, 2.1];
        let x = [0.1, 0.5, 0.9, 0.2, 0.6, 1.0];
        let clusters = [0u32, 1, 2, 0, 1, 2];
        let times = [0i64, 1, 2, 0, 1, 2];
        let multiway = vec![clusters.to_vec(), clusters.to_vec()];
        for kind in [
            AnalyticSeKind::Cluster,
            AnalyticSeKind::Multiway,
            AnalyticSeKind::NeweyWest { lag: 1 },
            AnalyticSeKind::PanelClusterHac { lag: 1 },
        ] {
            let mut ws = PropensityEstimationWorkspace::default();
            let err = matching_contrast(
                MatchingSample {
                    treatment: &t,
                    outcome: &y,
                    features: &x,
                    dim: 1,
                    distance: MatchingDistance::Absolute,
                    matches: 1,
                    population_variance: false,
                    ctx: &ExecutionContext::for_tests(1),
                },
                &TargetPopulation::AllObserved,
                None,
                &mut ws,
                SeSpec {
                    kind,
                    cluster_ids: Some(&clusters),
                    multiway_ids: Some(&multiway),
                    panel_times: Some(&times),
                },
                None,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("reason=estimator_inference_mismatch"),
                "{kind:?}: {err}"
            );
        }
    }

    /// Short Monte Carlo: ATE analytic 95% Wald intervals should not sit near 89% coverage.
    #[test]
    fn abadie_imbens_ate_se_coverage_direction() {
        let mut covered = 0usize;
        let reps = 80usize;
        let n = 120usize;
        for seed in 0..reps {
            let mut rng = ExecutionContext::for_tests(seed as u64)
                .rng
                .stream_for(StreamDomain::Estimate, 0xA11Eu64);
            let mut t = vec![0.0; n];
            let mut y = vec![0.0; n];
            let mut x = vec![0.0; n];
            for i in 0..n {
                let zi = standard_normal(&mut rng);
                x[i] = zi;
                t[i] = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
                y[i] = 1.0 * t[i] + 0.3 * zi + standard_normal(&mut rng) * 0.5;
            }
            let mut ws = PropensityEstimationWorkspace::default();
            let est = matching_contrast(
                MatchingSample {
                    treatment: &t,
                    outcome: &y,
                    features: &x,
                    dim: 1,
                    distance: MatchingDistance::Absolute,
                    matches: 1,
                    population_variance: false,
                    ctx: &ExecutionContext::for_tests(1),
                },
                &TargetPopulation::AllObserved,
                None,
                &mut ws,
                homoskedastic(),
                None,
            )
            .unwrap();
            let z = antecedent_stats::NORMAL_Q975;
            if (est.ate - 1.0).abs() <= z * est.se_analytic {
                covered += 1;
            }
        }
        let rate = covered as f64 / reps as f64;
        // Directional check only: must not collapse near the old ~0.89 undercoverage.
        assert!(rate >= 0.85, "nominal-95% coverage {rate} too low (covered {covered}/{reps})");
    }

    /// Regression: requesting bootstrap replicates must not license an SE in
    /// `se_bootstrap` (Abadie–Imbens 2008). Analytic SE remains the uncertainty product.
    #[test]
    fn bootstrap_replicates_do_not_populate_se_bootstrap() {
        let (data, estimand) = confounded_scm(120, 42);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityMatching { bootstrap_replicates: 30, ..PropensityMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect =
            est.fit(&prep, &mut ws, &ExecutionContext::for_tests(7), AssumptionSet::new()).unwrap();
        assert!(
            effect.se_bootstrap().is_none(),
            "NN matching must not store the invalid bootstrap SE"
        );
        assert!(
            effect.se_analytic().is_finite() && effect.se_analytic() > 0.0,
            "se_analytic={}",
            effect.se_analytic()
        );
    }

    /// A NaN, infinite or negative caliper is refused instead of matching at any distance.
    #[test]
    fn invalid_calipers_are_refused() {
        assert!(validate_caliper(None).is_ok());
        assert!(validate_caliper(Some(0.0)).is_ok());
        assert!(validate_caliper(Some(0.2)).is_ok());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
            assert!(validate_caliper(Some(bad)).is_err(), "{bad}");
        }
        let index = SortedScalarIndex::from_donors(&[10.0]);
        assert_eq!(index.nearest(0.0, Some(f64::NAN)), None);
        assert_eq!(index.nearest(0.0, Some(10.0)), Some((0, 10.0)));
    }

    /// A large run of equal scores resolves to the lowest donor index, matching a forward
    /// strict-`<` scan, and collapses to one entry per distinct score.
    #[test]
    fn huge_tie_block_matches_the_forward_scan() {
        let mut donors = Vec::new();
        for i in 0..30_000_usize {
            // Interleave the tied value with others so its indices are far apart.
            donors.push(if i % 3 == 0 { 0.5 } else { (i % 7) as f64 * 0.25 - 0.25 });
        }
        donors[17_000] = 0.5;
        let index = SortedScalarIndex::from_donors(&donors);
        assert_eq!(index.by_score.len(), 7, "one entry per distinct score");
        let queries = [0.5, 0.4, 0.375, 0.6, 0.25, -5.0, 9.0, 0.125];
        for caliper in [None, Some(0.0), Some(0.05), Some(0.125)] {
            for &query in &queries {
                let mut best: Option<(usize, f64)> = None;
                for (i, &score) in donors.iter().enumerate() {
                    let distance = (query - score).abs();
                    if best.is_none_or(|(_, d)| distance < d) {
                        best = Some((i, distance));
                    }
                }
                let expected = best.filter(|&(_, d)| caliper.is_none_or(|limit| d <= limit));
                assert_eq!(
                    index.nearest(query, caliper),
                    expected,
                    "q={query} caliper={caliper:?}"
                );
            }
        }
    }

    /// Regression test for the caliper-scale defect: with a fixed numeric caliper of 0.2 and
    /// propensity scores near the extremes of `[0, 1]`, `CaliperScale::Raw` and
    /// `CaliperScale::Logit` must select materially different match sets. This is the
    /// discriminating case the literature's 0.2 rule of thumb is about — raw-probability
    /// caliper 0.2 is far too permissive near 0/1, while the logit-scale caliper (what 0.2
    /// actually means per Rosenbaum & Rubin 1985 / Austin 2011) correctly excludes those
    /// pairs.
    #[test]
    fn caliper_scale_logit_vs_raw_diverge_near_extremes() {
        // Donor at each extreme plus one in the middle; queries just inside each extreme.
        let donor_probs = vec![0.02, 0.5, 0.98];
        let donor_y = vec![0.0, 0.0, 0.0];
        let query_probs = vec![0.08, 0.5, 0.92];
        let query_y = vec![0.0, 0.0, 0.0];
        let caliper = Some(0.2);

        let logit = |ps: &[f64]| -> Vec<f64> { ps.iter().map(|p| (p / (1.0 - p)).ln()).collect() };
        let raw_donors = donor_probs.clone();
        let raw_queries = query_probs.clone();
        let mut ws_raw = PropensityEstimationWorkspace::default();
        let MatchedDiffs { diffs: raw_diffs, .. } = match_diffs(
            MatchSide { features: &raw_donors, outcome: &donor_y },
            MatchSide { features: &raw_queries, outcome: &query_y },
            1,
            MatchingDistance::Absolute,
            1,
            caliper,
            &mut ws_raw,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();

        let logit_donors = logit(&donor_probs);
        let logit_queries = logit(&query_probs);
        let mut ws_logit = PropensityEstimationWorkspace::default();
        let MatchedDiffs { diffs: logit_diffs, .. } = match_diffs(
            MatchSide { features: &logit_donors, outcome: &donor_y },
            MatchSide { features: &logit_queries, outcome: &query_y },
            1,
            MatchingDistance::Absolute,
            1,
            caliper,
            &mut ws_logit,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();

        // Raw-probability caliper 0.2 is loose everywhere: all three queries land within 0.2
        // of a donor (|0.08-0.02|=0.06, |0.5-0.5|=0, |0.92-0.98|=0.06).
        assert_eq!(
            raw_diffs.len(),
            3,
            "raw-scale caliper=0.2 should admit all 3 near-extreme queries, got {}",
            raw_diffs.len()
        );
        // Logit caliper 0.2 correctly rejects the extreme pairs: logit(0.08) to logit(0.02) is
        // ~1.45 apart (>> 0.2), and symmetrically at the top; only the mid-range query (whose
        // donor and query coincide, logit distance 0) survives.
        assert_eq!(
            logit_diffs.len(),
            1,
            "logit-scale caliper=0.2 should admit only the mid-range query, got {}",
            logit_diffs.len()
        );
    }

    /// Correction: the weighted variance now carries the reliability-weights factor
    /// `1 / (1 − Σ w̃²)` in `σ̂²` (it reduces to the unweighted `n − 1` divisor at equal weights),
    /// so custom-target and sampling-weighted matching SEs are larger than before by that factor.
    /// Four units, 1-D feature, target weights `[4, 1, 1, 2]`. The bias-corrected effects are
    /// hand-derived from the two per-arm donor OLS lines; the weighted variance is
    /// `σ̂²_w · Σ c_i²` with `c` the row coefficients of the weighted pooled mean.
    #[test]
    fn custom_target_se_describes_the_weighted_estimator() {
        let treatment = [1.0, 1.0, 0.0, 0.0];
        let outcome = [3.0, 5.0, 1.0, 2.0];
        let features = [0.0, 1.0, 0.1, 0.9];
        let weights = [4.0, 1.0, 1.0, 2.0];
        // Treated queries (donors: controls, line 1 + 1.25 (x − 0.1)); control queries
        // (donors: treated, line 3 + 2x), sign-flipped to `Y(1) − Y(0)`.
        let effects = [2.125, 2.875, 2.2, 2.8];
        let rows = [0usize, 1, 2, 3];
        let w_sum: f64 = weights.iter().sum();
        let tau: f64 = effects.iter().zip(&weights).map(|(e, w)| e * w).sum::<f64>() / w_sum;
        let concentration: f64 = weights.iter().map(|w| (w / w_sum).powi(2)).sum();
        let sigma2: f64 = 0.5
            * effects.iter().zip(&weights).map(|(e, w)| w / w_sum * (e - tau).powi(2)).sum::<f64>()
            / (1.0 - concentration);
        // Donors: treated q0→control 0, q1→control 1, control q2→treated 0, q3→treated 1.
        let donor_of_query = [2usize, 3, 0, 1];
        let mut c = [0.0; 4];
        for q in 0..4 {
            c[rows[q]] += weights[q] / w_sum;
            c[donor_of_query[q]] += weights[q] / w_sum;
        }
        let expected = (sigma2 * c.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let equal_weight_se = {
            let mean = effects.iter().sum::<f64>() / 4.0;
            let v = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 3.0;
            (0.5 * v * 4.0 * 4.0 / 16.0).sqrt()
        };

        let mut ws = PropensityEstimationWorkspace::default();
        let out = matching_contrast(
            MatchingSample {
                treatment: &treatment,
                outcome: &outcome,
                features: &features,
                dim: 1,
                distance: MatchingDistance::Absolute,
                matches: 1,
                population_variance: false,
                ctx: &ExecutionContext::for_tests(1),
            },
            &TargetPopulation::CustomDistribution(antecedent_core::DistributionRef::from_raw(1)),
            None,
            &mut ws,
            homoskedastic(),
            Some(&weights),
        )
        .unwrap();
        assert!((out.ate - tau).abs() < 1e-12, "ate {} vs {tau}", out.ate);
        assert!((out.se_analytic - expected).abs() < 1e-12, "se {} vs {expected}", out.se_analytic);
        assert!((out.se_analytic - equal_weight_se).abs() > 1e-3, "weights must matter");
    }

    fn contrast(
        treatment: &[f64],
        outcome: &[f64],
        features: &[f64],
        target: &TargetPopulation,
        caliper: Option<f64>,
        weights: Option<&[f64]>,
    ) -> MatchedEstimate {
        matching_contrast(
            MatchingSample {
                treatment,
                outcome,
                features,
                dim: 1,
                distance: MatchingDistance::Absolute,
                matches: 1,
                population_variance: false,
                ctx: &ExecutionContext::for_tests(1),
            },
            target,
            caliper,
            &mut PropensityEstimationWorkspace::default(),
            homoskedastic(),
            weights,
        )
        .unwrap()
    }

    /// Constant weights of any scale are the unweighted construction: the weighted matched
    /// mean, its donor-reuse coefficients and its `(1 − Σ w̃²)`-corrected variance reduce to
    /// the Abadie–Imbens estimator, for the ATE and the one-arm forms and with a unit dropped by
    /// the caliper.
    #[test]
    fn constant_weights_reproduce_the_abadie_imbens_standard_error() {
        let treatment = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let outcome = [3.0, 5.0, 9.0, 1.0, 2.0, 2.5];
        let features = [0.0, 1.0, 5.0, 0.1, 0.9, 1.2];
        for target in
            [TargetPopulation::AllObserved, TargetPopulation::Treated, TargetPopulation::Untreated]
        {
            for caliper in [None, Some(0.5)] {
                let plain = contrast(&treatment, &outcome, &features, &target, caliper, None);
                for scale in [1.0, 3.7, 1.0e-3] {
                    let weights = [scale; 6];
                    let weighted =
                        contrast(&treatment, &outcome, &features, &target, caliper, Some(&weights));
                    assert!((weighted.ate - plain.ate).abs() < 1e-12, "{target:?} ate");
                    assert!(
                        (weighted.se_analytic - plain.se_analytic).abs()
                            < 1e-12 * (1.0 + plain.se_analytic),
                        "{target:?} {caliper:?} scale {scale}: {} vs {}",
                        weighted.se_analytic,
                        plain.se_analytic
                    );
                }
            }
        }
    }

    /// Integer weights on the treated queries of an ATT match are repeated treated rows: a
    /// duplicate query matches the same donor and the donor pool (hence the bias-correction
    /// fit) is unchanged, so the weighted point estimate equals the unweighted estimate of the
    /// replicated rows. The standard error is deliberately not that of the replicated rows:
    /// a sampling weight is one unit standing for `w` units, not `w` independent draws, so
    /// its variance is `σ̂² (Σ w_q² + Σ_j K^w_j²) / W²` with the `(1 − Σ w̃²)` correction.
    #[test]
    fn integer_query_weights_equal_replicated_queries_in_the_point_and_follow_the_definition() {
        let treated_x = [0.0, 1.0, 1.6];
        let treated_y = [3.0, 5.0, 6.0];
        let counts = [3usize, 1, 2];
        let controls_x = [0.1, 0.9, 1.5];
        let controls_y = [1.0, 2.0, 2.5];
        let assemble = |copies: &[usize]| {
            let (mut t, mut y, mut x) = (vec![], vec![], vec![]);
            for i in 0..3 {
                for _ in 0..copies[i] {
                    t.push(1.0);
                    y.push(treated_y[i]);
                    x.push(treated_x[i]);
                }
            }
            for i in 0..3 {
                t.push(0.0);
                y.push(controls_y[i]);
                x.push(controls_x[i]);
            }
            (t, y, x)
        };
        let (t, y, x) = assemble(&[1, 1, 1]);
        let weights = [3.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        let weighted = contrast(&t, &y, &x, &TargetPopulation::Treated, None, Some(&weights));
        let (rt, ry, rx) = assemble(&counts);
        let replicated = contrast(&rt, &ry, &rx, &TargetPopulation::Treated, None, None);
        assert!((weighted.ate - replicated.ate).abs() < 1e-12);

        let unweighted = contrast(&t, &y, &x, &TargetPopulation::Treated, None, None);
        let sparse = contrast(&t, &y, &x, &TargetPopulation::Treated, None, Some(&[1.0; 6]));
        assert!((sparse.se_analytic - unweighted.se_analytic).abs() < 1e-12);
        // Closed form, independent of the library. Bias-corrected unit effects: the nearest
        // control of treated x = 0, 1, 1.6 is x = 0.1, 0.9, 1.5, and the correction slope is
        // the OLS slope of control outcome on feature. Every donor is used by one query, so
        // K^w = (3, 1, 2) = w and Σ w² = Σ (K^w)² = 14, W = 6.
        let mean_x = controls_x.iter().sum::<f64>() / 3.0;
        let mean_y = controls_y.iter().sum::<f64>() / 3.0;
        let slope = controls_x
            .iter()
            .zip(&controls_y)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / controls_x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
        let effects: Vec<f64> = (0..3)
            .map(|q| (treated_y[q] - controls_y[q]) - slope * (treated_x[q] - controls_x[q]))
            .collect();
        let w = [3.0_f64, 1.0, 2.0];
        let total: f64 = w.iter().sum();
        let tau: f64 = effects.iter().zip(&w).map(|(e, w)| e * w).sum::<f64>() / total;
        let concentration: f64 = w.iter().map(|w| (w / total).powi(2)).sum();
        let sigma2 = 0.5
            * effects.iter().zip(&w).map(|(e, w)| w / total * (e - tau).powi(2)).sum::<f64>()
            / (1.0 - concentration);
        let sum_w2: f64 = w.iter().map(|w| w * w).sum();
        let expected = (sigma2 * (sum_w2 + sum_w2) / (total * total)).sqrt();
        assert!((weighted.ate - tau).abs() < 1e-12, "ate {} vs {tau}", weighted.ate);
        assert!(
            (weighted.se_analytic - expected).abs() < 1e-12,
            "se {} vs {expected}",
            weighted.se_analytic
        );
        assert!(
            (weighted.se_analytic - replicated.se_analytic).abs() > 1e-6,
            "a weight is not a frequency: the variance must differ from the replicated rows"
        );
    }

    /// Forward scan the sorted index replaces: strict `<`, so the lowest donor index keeps a tie.
    fn scalar_scan(donors: &[f64], query: f64) -> (usize, f64) {
        let mut best = (0usize, f64::INFINITY);
        for (index, &donor) in donors.iter().enumerate() {
            let distance = (query - donor).abs();
            if distance < best.1 {
                best = (index, distance);
            }
        }
        best
    }

    /// What the scan accepts under `caliper` (`None` accepts every finite nearest donor).
    fn scalar_scan_within(
        donors: &[f64],
        query: f64,
        caliper: Option<f64>,
    ) -> Option<(usize, f64)> {
        let found = scalar_scan(donors, query);
        if !found.1.is_finite() || caliper.is_some_and(|limit| found.1 > limit) {
            None
        } else {
            Some(found)
        }
    }

    /// Ties keep the lowest donor index, a caliper drops the pair it excludes, replacement
    /// reuses a donor, and sampling weights average those same scan matches.
    #[test]
    fn sorted_scalar_index_agrees_with_the_scalar_scan() {
        // Query 1 is equidistant from donor 0 (score 2) and donor 1 (score 0). Donor 0 is
        // the right-hand neighbour in sorted order and still wins, because its index is lower.
        // Donor 3 repeats donor 1's score; the first copy wins. Query 10 sits outside a 0.5
        // caliper of every donor.
        let donors = [2.0, 0.0, 3.0, 0.0];
        let queries = [1.0, 1.0, 0.0, 10.0];
        let index = SortedScalarIndex::from_donors(&donors);
        for &query in &queries {
            for caliper in [None, Some(1.0), Some(0.5)] {
                let scanned = scalar_scan_within(&donors, query, caliper);
                let indexed = index.nearest(query, caliper);
                assert_eq!(indexed.map(|m| m.0), scanned.map(|m| m.0), "query {query} {caliper:?}");
                assert_eq!(
                    indexed.map(|m| m.1.to_bits()),
                    scanned.map(|m| m.1.to_bits()),
                    "query {query} {caliper:?}"
                );
            }
        }
        let (tied, tied_distance) = scalar_scan(&donors, 1.0);
        assert_eq!(tied, 0, "the tie keeps the lower index, not the left neighbour");
        assert!(tied_distance > 0.5);
        assert!(index.nearest(1.0, Some(0.5)).is_none(), "the caliper excludes the tie");
        assert_eq!(index.nearest(1.0, Some(tied_distance)).unwrap().0, tied);
        // Replacement: a second query at the same score matches the same donor.
        assert_eq!(index.nearest(1.0, None).unwrap().0, index.nearest(queries[1], None).unwrap().0);

        let treatment = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let outcome = [10.0, 8.0, 4.0, 1.0, 5.0, 9.0];
        let features = [1.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let weights = [3.0, 1.0, 9.0, 1.0, 4.0, 1.0];
        let mut workspace = PropensityEstimationWorkspace::default();
        let MatchedDiffs { diffs, donors: matched_donors, queries: matched_queries } = match_diffs(
            MatchSide { features: &features[3..], outcome: &outcome[3..] },
            MatchSide { features: &features[..3], outcome: &outcome[..3] },
            1,
            MatchingDistance::Absolute,
            1,
            None,
            &mut workspace,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(matched_queries, vec![0, 1, 2]);
        for (slot, &query) in matched_queries.iter().enumerate() {
            assert_eq!(matched_donors[slot], scalar_scan(&features[3..], features[query]).0);
        }
        // The two treated units at 1.0 tie between controls at 2.0 and 0.0 and both reuse
        // local donor 0. The treated unit at 0.0 matches the control at 0.0.
        assert_eq!(matched_donors, vec![0, 0, 1]);
        let mass: f64 = weights[..3].iter().sum();
        let weighted_mean =
            diffs.iter().zip(&weights[..3]).map(|(diff, weight)| diff * weight).sum::<f64>() / mass;
        let weighted = contrast(
            &treatment,
            &outcome,
            &features,
            &TargetPopulation::Treated,
            None,
            Some(&weights),
        );
        let plain =
            contrast(&treatment, &outcome, &features, &TargetPopulation::Treated, None, None);
        assert_eq!(weighted.ate.to_bits(), weighted_mean.to_bits());
        assert_eq!(plain.ate.to_bits(), (diffs.iter().sum::<f64>() / diffs.len() as f64).to_bits());
        assert!((weighted.ate - plain.ate).abs() > 1e-9);

        let inside = match_diffs(
            MatchSide { features: &features[3..], outcome: &outcome[3..] },
            MatchSide { features: &features[..3], outcome: &outcome[..3] },
            1,
            MatchingDistance::Absolute,
            1,
            Some(0.5),
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(inside.queries, vec![2]);
        assert_eq!(inside.donors, vec![scalar_scan(&features[3..], features[2]).0]);
    }

    /// Multivariate matching still builds `MatchingIndex`, whose memory guard refuses a
    /// donor set that does not fit. A spatial index is not part of this path.
    #[test]
    fn multivariate_size_guard_refuses_an_input_past_its_limit() {
        let n = 8;
        let treatment: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let outcome = vec![1.0; n];
        let features: Vec<f64> = (0..n * 2).map(|i| i as f64 * 0.1).collect();
        let sample = |ctx: &ExecutionContext| {
            matching_contrast(
                MatchingSample {
                    treatment: &treatment,
                    outcome: &outcome,
                    features: &features,
                    dim: 2,
                    distance: MatchingDistance::Euclidean,
                    matches: 1,
                    population_variance: false,
                    ctx,
                },
                &TargetPopulation::AllObserved,
                None,
                &mut PropensityEstimationWorkspace::default(),
                homoskedastic(),
                None,
            )
        };
        let mut allowed = ExecutionContext::for_tests(1);
        allowed.memory.hard_limit_bytes = Some(1024 * 1024);
        assert!(sample(&allowed).is_ok());
        let mut refused = ExecutionContext::for_tests(1);
        refused.memory.hard_limit_bytes = Some(1);
        let err = sample(&refused).unwrap_err();
        assert!(matches!(
            err,
            crate::error::EstimationError::Stats(antecedent_stats::StatsError::ResourceLimit {
                limit_bytes: 1,
                ..
            })
        ));
    }
}
