//! Sequentially randomized unit-period designs: excursion effects of a decision on a later
//! outcome, under a known (possibly history-dependent) assignment law.
//!
//! # Design and target
//!
//! Independent units are observed over periods. At each decision period `t` a unit's
//! treatment is assigned by a law the experimenter controls and records: the probability (or,
//! for a continuous dose, the density) of the realized assignment given the unit's history is
//! known for every decision row. The outcome of interest is the unit's outcome `lag` periods
//! later, `Y_{t+lag}`. The target is the *excursion* contrast of the micro-randomized-trial
//! literature (Boruvka, Almirall, Witkiewitz and Murphy 2018): the mean, over decision rows, of
//! the contrast between assigning one arm and another at `t`, with every earlier and later
//! assignment following the design. Carryover is allowed and is part of the target: an
//! assignment's effect on `Y_{t+lag}` includes whatever reaches it through later periods under
//! the design. That is what separates this target from the switchback direct effect, which
//! assumes no carryover.
//!
//! Identification is by design: sequential randomization with the recorded probabilities,
//! positivity of every arm at every decision row, consistency, and no interference between
//! units (the interference design relaxes the last within declared clusters). None of these is
//! derived from the data.
//!
//! # Estimators
//!
//! Every coordinate is a ratio of per-pair Horvitz-Thompson scores summed within independent
//! clusters (the unit, or the declared interference cluster): `theta = sum_g S_g / sum_g n_g`.
//! The frequentist uncertainty is the cluster-robust (linearized ratio) covariance; serial
//! dependence inside a unit is unrestricted. A simultaneous band over several coordinates uses
//! a Gaussian multiplier sup-t critical value over clusters. The Bayesian counterpart is the
//! Bayesian bootstrap over clusters (a Dirichlet-process posterior in the limit of vanishing
//! concentration) of the same ratio, so every draw reweights whole units. Neither is
//! calibrated: no coverage of these intervals has been measured.
//!
//! The quantile functional, the local-quadratic dose grid, the regime survival procedure and the
//! observational excursion and the whole-regime contrast (whose assignment laws are fitted, not
//! recorded) are not linear in the
//! scores; their modules state their own uncertainty paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod contrasts;
pub mod dose;
mod linalg;
pub mod misclassification;
pub mod observational;
pub mod quantile;
pub mod regime;
pub mod serial;
pub mod survival;

pub(crate) use linalg::{cholesky as cholesky_factor, cholesky_solve as cholesky_apply};

use std::collections::HashMap;

use antecedent_core::{ExecutionContext, StreamDomain};

use crate::error::EstimationError;
use crate::util::{count, refuse};

/// Nominal level of every pointwise interval and band reported here.
pub const NOMINAL_LEVEL: f64 = 0.95;
/// Fewest independent clusters an interval is computed from.
pub const MIN_CLUSTERS: usize = 2;
/// Default Gaussian multiplier replicates of a simultaneous band.
pub const DEFAULT_BAND_REPLICATES: usize = 2000;

/// Stream key of the simultaneous-band multipliers.
const BAND_STREAM: u64 = 0x7E5D_BA4D_0000_0000;
/// Stream key of the Bayesian-bootstrap cluster weights.
const POSTERIOR_STREAM: u64 = 0x7E5D_B0B0_0000_0000;

/// Row layout of a unit-period panel.
#[derive(Clone, Copy, Debug)]
pub struct PanelRows<'a> {
    /// Dense unit index of each row, in `0..units`.
    pub unit: &'a [usize],
    /// Period of each row; `(unit, period)` is unique.
    pub period: &'a [i64],
}

/// Decision rows paired with the outcome row `lag` periods later on the same unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaggedPairs {
    /// Decision row of each pair.
    pub decision: Vec<usize>,
    /// Outcome row of each pair.
    pub outcome: Vec<usize>,
    /// Decision rows whose outcome period is not observed on the unit.
    pub unpaired: usize,
}

/// Pair every decision row with the row of the same unit `lag` periods later.
///
/// A decision whose outcome period is not observed is left unpaired and counted; the target
/// is then the excursion over the paired decisions, which equals the design target when the
/// missing outcome periods are administrative (past the end of follow-up).
///
/// # Errors
/// Misaligned or empty rows, or a repeated `(unit, period)`.
pub fn lagged_pairs(rows: PanelRows<'_>, lag: u32) -> Result<LaggedPairs, EstimationError> {
    let n = rows.unit.len();
    if n == 0 || rows.period.len() != n {
        return Err(invalid("unit and period labels must be aligned and non-empty"));
    }
    let mut index = HashMap::with_capacity(n);
    for (row, (&unit, &period)) in rows.unit.iter().zip(rows.period).enumerate() {
        if index.insert((unit, period), row).is_some() {
            return Err(invalid("each (unit, period) must appear on exactly one row"));
        }
    }
    let mut decision = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut unpaired = 0_usize;
    for (row, (&unit, &period)) in rows.unit.iter().zip(rows.period).enumerate() {
        match period.checked_add(i64::from(lag)).and_then(|target| index.get(&(unit, target))) {
            Some(&target) => {
                decision.push(row);
                outcome.push(target);
            }
            None => unpaired += 1,
        }
    }
    if decision.is_empty() {
        return Err(invalid("no decision row has its outcome period observed at this lag"));
    }
    Ok(LaggedPairs { decision, outcome, unpaired })
}

/// Per-cluster sums of per-pair coordinate scores.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterScores {
    /// Coordinates per pair.
    pub coordinates: usize,
    /// Cluster-major sums: `sums[g * coordinates + k]`.
    pub sums: Vec<f64>,
    /// Pairs in each cluster.
    pub counts: Vec<f64>,
}

impl ClusterScores {
    /// Accumulate pair-major scores (`scores[p * coordinates + k]`) by the cluster of each pair.
    ///
    /// Clusters with no pair are dropped; the order of the remaining clusters is ascending.
    ///
    /// # Errors
    /// Misaligned inputs, a non-finite score, or fewer than [`MIN_CLUSTERS`] clusters.
    pub fn accumulate(
        scores: &[f64],
        coordinates: usize,
        cluster_of_pair: &[usize],
    ) -> Result<Self, EstimationError> {
        Self::accumulate_weighted(scores, coordinates, cluster_of_pair, None)
    }

    /// [`Self::accumulate`] under analysis weights: a pair of weight `s` contributes `s` times
    /// its score and counts `s` toward its cluster, so the ratio of the sums is the weighted
    /// (Hajek) ratio `sum s z / sum s` and the cluster residuals `S_g - n_g theta` are the
    /// sampling-weighted linearization of the weighted estimating equation.
    ///
    /// The weights act only through their ratios: scaling every weight changes neither the
    /// estimate nor the covariance.
    ///
    /// # Errors
    /// As [`Self::accumulate`], and a weight that is not finite and positive, or a weight vector
    /// that does not have one entry per pair.
    pub fn accumulate_weighted(
        scores: &[f64],
        coordinates: usize,
        cluster_of_pair: &[usize],
        pair_weights: Option<&[f64]>,
    ) -> Result<Self, EstimationError> {
        let pairs = cluster_of_pair.len();
        if coordinates == 0 || scores.len() != pairs * coordinates {
            return Err(invalid("pair scores must be aligned with the pair clusters"));
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(invalid("a pair score is not finite"));
        }
        if let Some(weights) = pair_weights {
            if weights.len() != pairs || weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
                return Err(invalid(
                    "pair weights must be finite and positive, one per pair (a zero-weight pair \
                     is removed before scoring)",
                ));
            }
        }
        let mut dense: Vec<usize> = cluster_of_pair.to_vec();
        dense.sort_unstable();
        dense.dedup();
        if dense.len() < MIN_CLUSTERS {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "temporal_design.too_few_clusters",
                "the design needs at least two independent units or clusters",
            ));
        }
        let position: HashMap<usize, usize> =
            dense.iter().enumerate().map(|(index, cluster)| (*cluster, index)).collect();
        let mut sums = vec![0.0; dense.len() * coordinates];
        let mut counts = vec![0.0; dense.len()];
        for (pair, cluster) in cluster_of_pair.iter().enumerate() {
            let g = position[cluster];
            let weight = pair_weights.map_or(1.0, |weights| weights[pair]);
            counts[g] += weight;
            for k in 0..coordinates {
                sums[g * coordinates + k] += weight * scores[pair * coordinates + k];
            }
        }
        Ok(Self { coordinates, sums, counts })
    }

    /// Number of clusters.
    #[must_use]
    pub fn clusters(&self) -> usize {
        self.counts.len()
    }

    /// The ratio estimate of every coordinate.
    #[must_use]
    pub fn estimate(&self) -> Vec<f64> {
        self.weighted_estimate(None)
    }

    /// The ratio estimate under cluster weights (`None` weighs every cluster one).
    #[must_use]
    pub fn weighted_estimate(&self, weights: Option<&[f64]>) -> Vec<f64> {
        let k = self.coordinates;
        let mut numerator = vec![0.0; k];
        let mut denominator = 0.0;
        for (g, count_g) in self.counts.iter().enumerate() {
            let w = weights.map_or(1.0, |weights| weights[g]);
            denominator += w * count_g;
            for (j, value) in numerator.iter_mut().enumerate() {
                *value += w * self.sums[g * k + j];
            }
        }
        numerator.into_iter().map(|value| value / denominator).collect()
    }

    /// Linearized residual of every cluster at `estimate`: `S_g - n_g * theta`.
    pub(crate) fn residuals(&self, estimate: &[f64]) -> Vec<f64> {
        let k = self.coordinates;
        let mut residuals = vec![0.0; self.sums.len()];
        for (g, count_g) in self.counts.iter().enumerate() {
            for j in 0..k {
                residuals[g * k + j] = self.sums[g * k + j] - count_g * estimate[j];
            }
        }
        residuals
    }

    /// Cluster-robust covariance (row-major `k x k`) of the ratio estimate, with the
    /// small-sample factor `G / (G - 1)`.
    #[must_use]
    pub fn covariance(&self, estimate: &[f64]) -> Vec<f64> {
        let k = self.coordinates;
        let g_count = self.clusters();
        let total: f64 = self.counts.iter().sum();
        let residuals = self.residuals(estimate);
        let scale = count(g_count) / (count(g_count) - 1.0) / (total * total);
        let mut covariance = vec![0.0; k * k];
        for g in 0..g_count {
            let row = &residuals[g * k..(g + 1) * k];
            for a in 0..k {
                for b in 0..k {
                    covariance[a * k + b] += row[a] * row[b];
                }
            }
        }
        for value in &mut covariance {
            *value *= scale;
        }
        covariance
    }
}

/// Frequentist summary of a set of coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingSummary {
    /// Coordinate estimates.
    pub estimate: Vec<f64>,
    /// Standard errors.
    pub standard_error: Vec<f64>,
    /// Pointwise lower endpoints at [`NOMINAL_LEVEL`].
    pub lower: Vec<f64>,
    /// Pointwise upper endpoints at [`NOMINAL_LEVEL`].
    pub upper: Vec<f64>,
    /// Row-major covariance across coordinates.
    pub covariance: Vec<f64>,
    /// Simultaneous band over the band coordinates, when requested.
    pub band: Option<SimultaneousBand>,
    /// Holm-adjusted two-sided p-values of the family coordinates, when requested.
    pub holm_adjusted_p: Option<Vec<f64>>,
    /// Worst-case smoothing bias of each coordinate under a declared curvature bound, when
    /// the coordinate is a smoother whose bias is not part of its target.
    pub smoothing_bias_bound: Option<Vec<f64>>,
    /// How the uncertainty was computed.
    pub method: &'static str,
}

/// A simultaneous band over a subset of coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SimultaneousBand {
    /// Coordinates the band covers.
    pub coordinates: Vec<usize>,
    /// Critical value of the centered sup-t statistic.
    pub critical_value: f64,
    /// Lower envelope, one per covered coordinate.
    pub lower: Vec<f64>,
    /// Upper envelope, one per covered coordinate.
    pub upper: Vec<f64>,
    /// Multiplier replicates behind the critical value.
    pub replicates: usize,
}

/// Bayesian summary of a set of coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorSummary {
    /// Coordinate-major draws: `draws[k][s]`.
    pub draws: Vec<Vec<f64>>,
    /// Posterior means.
    pub mean: Vec<f64>,
    /// Posterior standard deviations.
    pub standard_deviation: Vec<f64>,
    /// Equal-tailed lower endpoints at [`NOMINAL_LEVEL`].
    pub lower: Vec<f64>,
    /// Equal-tailed upper endpoints at [`NOMINAL_LEVEL`].
    pub upper: Vec<f64>,
    /// How the posterior was computed.
    pub method: &'static str,
}

/// The pointwise normal summary of a ratio of cluster scores.
///
/// # Errors
/// A non-finite estimate or covariance.
pub fn sampling_summary(scores: &ClusterScores) -> Result<SamplingSummary, EstimationError> {
    let estimate = scores.estimate();
    let covariance = scores.covariance(&estimate);
    let k = scores.coordinates;
    let standard_error: Vec<f64> = (0..k).map(|j| covariance[j * k + j].max(0.0).sqrt()).collect();
    if estimate.iter().chain(&standard_error).any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg("the design estimate is not finite"));
    }
    let z = antecedent_stats::NORMAL_Q975;
    Ok(SamplingSummary {
        lower: estimate.iter().zip(&standard_error).map(|(e, s)| e - z * s).collect(),
        upper: estimate.iter().zip(&standard_error).map(|(e, s)| e + z * s).collect(),
        estimate,
        standard_error,
        covariance,
        band: None,
        holm_adjusted_p: None,
        smoothing_bias_bound: None,
        method: "cluster_robust_ratio_normal",
    })
}

/// Attach a Gaussian multiplier sup-t band over clusters for the listed coordinates.
///
/// The band half-width of coordinate `j` is `c * se_j` plus its smoothing-bias bound when the
/// summary carries one, so a bias that is not part of the target widens the band instead of
/// being left outside it.
///
/// # Errors
/// Fewer than two replicates, a coordinate out of range, a zero standard error, or cancellation.
pub fn sup_t_band(
    scores: &ClusterScores,
    summary: &mut SamplingSummary,
    band_coordinates: &[usize],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    let residuals = scores.residuals(&summary.estimate);
    let g_count = scores.clusters();
    let total: f64 = scores.counts.iter().sum();
    let scale = (count(g_count) / (count(g_count) - 1.0)).sqrt() / total;
    let critical = sup_t_critical(
        &residuals,
        scores.coordinates,
        scale,
        &summary.standard_error,
        band_coordinates,
        replicates,
        ctx,
    )?;
    attach_band(summary, band_coordinates, critical, replicates);
    Ok(())
}

/// Record a band of critical value `critical` on `summary`.
pub(crate) fn attach_band(
    summary: &mut SamplingSummary,
    band_coordinates: &[usize],
    critical: f64,
    replicates: usize,
) {
    let bias = summary.smoothing_bias_bound.clone();
    let mut lower = Vec::with_capacity(band_coordinates.len());
    let mut upper = Vec::with_capacity(band_coordinates.len());
    for &j in band_coordinates {
        let half = critical * summary.standard_error[j] + bias.as_ref().map_or(0.0, |b| b[j]);
        lower.push(summary.estimate[j] - half);
        upper.push(summary.estimate[j] + half);
    }
    summary.band = Some(SimultaneousBand {
        coordinates: band_coordinates.to_vec(),
        critical_value: critical,
        lower,
        upper,
        replicates,
    });
}

/// Gaussian multiplier sup-t critical value from cluster-major residual contributions
/// (`residuals[g * coordinates + k]`).
///
/// Each replicate draws one standard normal per cluster; its statistic is the largest
/// `|sum_g xi_g r_gk| * scale / se_k` over the band coordinates. The critical value is the
/// Monte Carlo rank of [`NOMINAL_LEVEL`] among the replicate statistics.
///
/// # Errors
/// Fewer than two replicates, a coordinate out of range, a zero standard error, or cancellation.
pub fn sup_t_critical(
    residuals: &[f64],
    coordinates: usize,
    scale: f64,
    standard_error: &[f64],
    band_coordinates: &[usize],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<f64, EstimationError> {
    let k = coordinates;
    if replicates < 2
        || k == 0
        || residuals.len() % k != 0
        || band_coordinates.is_empty()
        || band_coordinates.iter().any(|j| *j >= k)
    {
        return Err(invalid("a band needs at least two replicates and coordinates in range"));
    }
    if band_coordinates.iter().any(|j| {
        let se = standard_error[*j];
        !se.is_finite() || se <= 0.0
    }) {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "temporal_design.degenerate_band",
            "a band coordinate has zero standard error",
        ));
    }
    let g_count = residuals.len() / k;
    let _lease = crate::util::admit_array::<f64>(ctx, replicates + g_count, 1)?;
    let mut maxima = Vec::with_capacity(replicates);
    let mut multipliers = vec![0.0; g_count];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, BAND_STREAM | replicate as u64);
        antecedent_kernels::fill_standard_normal(&mut rng, &mut multipliers);
        let mut largest = 0.0_f64;
        for &j in band_coordinates {
            let sum: f64 =
                multipliers.iter().enumerate().map(|(g, xi)| xi * residuals[g * k + j]).sum();
            largest = largest.max((sum * scale / standard_error[j]).abs());
        }
        maxima.push(largest);
    }
    maxima.sort_by(f64::total_cmp);
    Ok(crate::util::monte_carlo_critical(&maxima, NOMINAL_LEVEL))
}

/// Critical value `c` of a bias-aware interval: `P(|Z + t| <= c) = level` for `Z ~ N(0, 1)`
/// and a worst-case standardized bias `t >= 0` (Armstrong and Kolesar 2020).
#[must_use]
pub fn folded_normal_critical(t: f64, level: f64) -> f64 {
    let t = t.abs();
    let coverage =
        |c: f64| antecedent_kernels::norm_cdf(c - t) - antecedent_kernels::norm_cdf(-c - t);
    let mut low = 0.0;
    let mut high = t + 10.0;
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if coverage(middle) < level {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

/// Holm step-down adjusted two-sided normal p-values of `family` coordinates against zero.
#[must_use]
pub fn holm_adjusted(summary: &SamplingSummary, family: &[usize]) -> Vec<f64> {
    let raw: Vec<f64> = family
        .iter()
        .map(|&j| {
            let se = summary.standard_error[j];
            if se > 0.0 { crate::util::normal_two_sided_p(summary.estimate[j] / se) } else { 1.0 }
        })
        .collect();
    let m = raw.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|a, b| raw[*a].total_cmp(&raw[*b]));
    let mut adjusted = vec![0.0; m];
    let mut running = 0.0_f64;
    for (rank, &index) in order.iter().enumerate() {
        let value = (count(m - rank) * raw[index]).min(1.0);
        running = running.max(value);
        adjusted[index] = running;
    }
    adjusted
}

/// Bayesian-bootstrap posterior over clusters of the ratio of cluster scores.
///
/// # Errors
/// Fewer than two draws, a non-finite draw, admission or cancellation.
pub fn bayesian_bootstrap(
    scores: &ClusterScores,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<PosteriorSummary, EstimationError> {
    posterior_by_cluster_weights(scores.clusters(), scores.coordinates, draws, ctx, |weights| {
        Ok(scores.weighted_estimate(Some(weights)))
    })
    .map(|mut summary| {
        summary.method = "bayesian_bootstrap_over_clusters";
        summary
    })
}

/// Posterior draws of any functional of Dirichlet cluster weights.
///
/// Each draw takes one Dirichlet(1, ..., 1) weight vector over the clusters (rescaled to mean
/// one) from its own stream, so the draws do not depend on the thread count.
///
/// # Errors
/// Fewer than two draws, a functional that fails or returns a non-finite value, admission or
/// cancellation.
pub fn posterior_by_cluster_weights(
    clusters: usize,
    coordinates: usize,
    draws: usize,
    ctx: &ExecutionContext,
    mut functional: impl FnMut(&[f64]) -> Result<Vec<f64>, EstimationError>,
) -> Result<PosteriorSummary, EstimationError> {
    if draws < 2 || clusters < MIN_CLUSTERS || coordinates == 0 {
        return Err(invalid("a posterior needs two draws, two clusters and a coordinate"));
    }
    let _lease = crate::util::admit_array::<f64>(ctx, draws, coordinates + clusters)?;
    let mut columns: Vec<Vec<f64>> = (0..coordinates).map(|_| Vec::with_capacity(draws)).collect();
    for draw in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, POSTERIOR_STREAM | draw as u64);
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, clusters, true);
        let values = functional(&weights)?;
        if values.len() != coordinates || values.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::stats_msg("a posterior draw is not finite"));
        }
        for (column, value) in columns.iter_mut().zip(values) {
            column.push(value);
        }
    }
    Ok(summarize_draws(columns, "dirichlet_cluster_weights"))
}

/// Summaries of coordinate-major draws.
#[must_use]
pub fn summarize_draws(draws: Vec<Vec<f64>>, method: &'static str) -> PosteriorSummary {
    let mut mean = Vec::with_capacity(draws.len());
    let mut standard_deviation = Vec::with_capacity(draws.len());
    let mut lower = Vec::with_capacity(draws.len());
    let mut upper = Vec::with_capacity(draws.len());
    for column in &draws {
        mean.push(crate::util::mean(column));
        standard_deviation.push(crate::util::sample_std(column));
        let mut sorted = column.clone();
        sorted.sort_by(f64::total_cmp);
        let (lo, hi) = antecedent_stats::equal_tail_interval_sorted(
            &sorted,
            NOMINAL_LEVEL,
            antecedent_stats::QuantileRule::ExchangeableRank,
        );
        lower.push(lo);
        upper.push(hi);
    }
    PosteriorSummary { draws, mean, standard_deviation, lower, upper, method }
}

pub(crate) fn invalid(message: &str) -> EstimationError {
    refuse(antecedent_core::reason_code!("invalid_argument"), "temporal_design.invalid", message)
}

/// Dense index `0..` of string labels in first-appearance order, and the label count.
#[must_use]
pub fn dense_labels<T: Eq + std::hash::Hash + Clone>(labels: &[T]) -> (Vec<usize>, usize) {
    let mut index: HashMap<T, usize> = HashMap::new();
    let mut dense = Vec::with_capacity(labels.len());
    for label in labels {
        let next = index.len();
        dense.push(*index.entry(label.clone()).or_insert(next));
    }
    let n = index.len();
    (dense, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_follow_the_lag_within_each_unit() {
        let unit = [0, 0, 0, 1, 1];
        let period = [0, 1, 2, 0, 2];
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 1).unwrap();
        assert_eq!(pairs.decision, vec![0, 1]);
        assert_eq!(pairs.outcome, vec![1, 2]);
        assert_eq!(pairs.unpaired, 3);
        let same = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        assert_eq!(same.decision, vec![0, 1, 2, 3, 4]);
        assert!(lagged_pairs(PanelRows { unit: &[0, 0], period: &[1, 1] }, 0).is_err());
    }

    #[test]
    fn the_ratio_and_its_cluster_covariance_match_the_hand_formula() {
        // Two coordinates, three clusters of sizes 1, 2, 1.
        let scores = [1.0, 0.0, 2.0, 1.0, 4.0, 1.0, 3.0, 2.0];
        let clusters = [0, 1, 1, 2];
        let accumulated = ClusterScores::accumulate(&scores, 2, &clusters).unwrap();
        let estimate = accumulated.estimate();
        assert!((estimate[0] - 10.0 / 4.0).abs() < 1e-12);
        assert!((estimate[1] - 4.0 / 4.0).abs() < 1e-12);
        let cov = accumulated.covariance(&estimate);
        // Residuals of coordinate 0: 1 - 2.5, 6 - 5, 3 - 2.5.
        let r = [-1.5_f64, 1.0, 0.5];
        let expected = 1.5 * r.iter().map(|v| v * v).sum::<f64>() / 16.0;
        assert!((cov[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn holm_is_monotone_in_the_raw_order() {
        let summary = SamplingSummary {
            estimate: vec![3.0, 0.1, 2.0],
            standard_error: vec![1.0, 1.0, 1.0],
            lower: vec![0.0; 3],
            upper: vec![0.0; 3],
            covariance: vec![0.0; 9],
            band: None,
            holm_adjusted_p: None,
            smoothing_bias_bound: None,
            method: "test",
        };
        let adjusted = holm_adjusted(&summary, &[0, 1, 2]);
        assert!(adjusted[0] <= adjusted[2] && adjusted[2] <= adjusted[1]);
        assert!(adjusted[1] <= 1.0);
    }

    #[test]
    fn posterior_draws_repeat_under_the_same_seed() {
        let scores = [1.0, 2.0, 0.5, 3.0, 1.5];
        let clusters = [0, 1, 2, 3, 4];
        let accumulated = ClusterScores::accumulate(&scores, 1, &clusters).unwrap();
        let ctx = ExecutionContext::for_tests(5);
        let first = bayesian_bootstrap(&accumulated, 64, &ctx).unwrap();
        let second = bayesian_bootstrap(&accumulated, 64, &ctx).unwrap();
        assert_eq!(first, second);
        assert!((first.mean[0] - 1.6).abs() < 0.3);
    }
}
