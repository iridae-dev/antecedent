//! Quantile excursion of a two-arm sequentially randomized design.
//!
//! # Target
//!
//! For a level `tau` in `(0, 1)` and arm `a`, `q_a(tau)` is the `tau`-quantile of the law of
//! `Y_{t+lag}(a)` pooled over decision rows: the outcome `lag` periods after assigning arm `a`
//! at a decision, with every other assignment following the design. The pooled law is
//! `F_a(y) = E[sum_t 1{Y_{t+lag}(a) <= y}] / E[#decisions]`; it is the distribution a randomly
//! chosen decision of the trial would see under the excursion. The target is
//! `q_1(tau) - q_0(tau)`, with both arm quantiles. This is a quantile of the outcome under the
//! dynamic regime "follow the design, except assign `a` now"; it is not a quantile of a
//! unit-level effect, which no randomized design identifies.
//!
//! # Estimator
//!
//! `F_a` is the Hajek inverse-probability weighted distribution function of the paired
//! outcomes, `sum_p w_pa 1{Y_p <= y} / sum_p w_pa` with `w_p1 = A_t / P(A_t = 1 | history)` and
//! `w_p0 = (1 - A_t) / P(A_t = 0 | history)`; the quantile is its left-continuous inverse at
//! the observed outcomes. Cluster weights multiply every pair of a cluster.
//!
//! # Uncertainty
//!
//! The frequentist interval is the percentile interval of a whole-procedure cluster bootstrap:
//! each replicate resamples independent units with replacement and recomputes both weighted
//! distribution functions and their quantiles. The Bayesian posterior reweights units with
//! Dirichlet weights and recomputes the same functional. Neither is calibrated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use super::{LaggedPairs, PosteriorSummary, invalid, posterior_by_cluster_weights};
use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the cluster bootstrap.
const BOOTSTRAP_STREAM: u64 = 0x7E5D_9A47_0000_0000;
/// Largest share of bootstrap replicates allowed to fail before the interval is refused.
const MAX_FAILED_SHARE: f64 = 0.01;

/// Pairs prepared for the quantile functional.
#[derive(Clone, Debug)]
struct QuantilePairs {
    outcome: Vec<f64>,
    treated_weight: Vec<f64>,
    control_weight: Vec<f64>,
    cluster: Vec<usize>,
    clusters: usize,
    /// Pair order by ascending outcome.
    order: Vec<usize>,
}

/// A fitted quantile excursion.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileFit {
    /// `[quantile effect, treated quantile, control quantile]`.
    pub estimate: [f64; 3],
    /// Bootstrap standard deviation of each coordinate, under frequentist inference.
    pub standard_error: Option<[f64; 3]>,
    /// Percentile interval of each coordinate, under frequentist inference.
    pub interval: Option<[(f64, f64); 3]>,
    /// Replicates that succeeded, and those that failed (an arm with no resampled support).
    pub replicates: (usize, usize),
    /// Posterior, under Bayesian inference.
    pub posterior: Option<PosteriorSummary>,
    /// Pairs and independent clusters.
    pub pairs: usize,
    /// Independent clusters.
    pub clusters: usize,
}

/// Coordinate names of a quantile fit.
pub const QUANTILE_COORDINATES: [&str; 3] =
    ["quantile_effect", "treated_quantile", "control_quantile"];

fn prepare(
    pairs: &LaggedPairs,
    outcome: &[f64],
    assignment: &[bool],
    probability: &[f64],
    cluster: &[usize],
    floor: f64,
) -> Result<QuantilePairs, EstimationError> {
    let rows = outcome.len();
    if assignment.len() != rows || probability.len() != rows || cluster.len() != rows {
        return Err(invalid("design arrays must be aligned with the table rows"));
    }
    let n = pairs.decision.len();
    let mut values = Vec::with_capacity(n);
    let mut treated_weight = Vec::with_capacity(n);
    let mut control_weight = Vec::with_capacity(n);
    let mut raw_cluster = Vec::with_capacity(n);
    for (&t, &o) in pairs.decision.iter().zip(&pairs.outcome) {
        let (y, p) = (outcome[o], probability[t]);
        if !y.is_finite() || !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(invalid("paired outcomes must be finite and probabilities in (0, 1)"));
        }
        let realized = p.min(1.0 - p);
        if realized < floor {
            return Err(refuse(
                antecedent_core::reason_code!("positivity_violated"),
                "temporal_design.positivity",
                &format!("a realized assignment probability {realized} is below the floor {floor}"),
            ));
        }
        values.push(y);
        treated_weight.push(if assignment[t] { 1.0 / p } else { 0.0 });
        control_weight.push(if assignment[t] { 0.0 } else { 1.0 / (1.0 - p) });
        raw_cluster.push(cluster[t]);
    }
    let (dense, clusters) = super::dense_labels(&raw_cluster);
    if clusters < super::MIN_CLUSTERS {
        return Err(invalid("the quantile design needs at least two independent units"));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    Ok(QuantilePairs {
        outcome: values,
        treated_weight,
        control_weight,
        cluster: dense,
        clusters,
        order,
    })
}

/// Left-continuous inverse of a weighted distribution function at `tau`.
fn weighted_quantile(
    prepared: &QuantilePairs,
    arm_weight: &[f64],
    cluster_weights: Option<&[f64]>,
    tau: f64,
) -> Option<f64> {
    let weight = |p: usize| {
        arm_weight[p] * cluster_weights.map_or(1.0, |weights| weights[prepared.cluster[p]])
    };
    let total: f64 = (0..prepared.outcome.len()).map(weight).sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let mut cumulative = 0.0;
    for &p in &prepared.order {
        cumulative += weight(p);
        if cumulative >= tau * total - 1e-12 * total {
            return Some(prepared.outcome[p]);
        }
    }
    prepared.order.last().map(|p| prepared.outcome[*p])
}

fn evaluate(
    prepared: &QuantilePairs,
    cluster_weights: Option<&[f64]>,
    tau: f64,
) -> Option<[f64; 3]> {
    let treated = weighted_quantile(prepared, &prepared.treated_weight, cluster_weights, tau)?;
    let control = weighted_quantile(prepared, &prepared.control_weight, cluster_weights, tau)?;
    Some([treated - control, treated, control])
}

/// Fit the quantile excursion at level `tau`.
///
/// `posterior_draws` selects the Bayesian bootstrap; otherwise `replicates` whole-procedure
/// cluster-bootstrap replicates give the percentile interval.
///
/// # Errors
/// A level outside `(0, 1)`, misaligned inputs, a probability below the floor, an arm with no
/// paired decision, fewer than 99 bootstrap replicates, more than one percent failed
/// replicates, admission or cancellation.
#[allow(
    clippy::too_many_arguments,
    reason = "one design: pairs, outcome, assignment, probability, clusters, floor, level and budget"
)]
pub fn fit_quantile(
    pairs: &LaggedPairs,
    outcome: &[f64],
    assignment: &[bool],
    probability: &[f64],
    cluster: &[usize],
    floor: f64,
    tau: f64,
    replicates: usize,
    posterior_draws: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<QuantileFit, EstimationError> {
    if !(tau.is_finite() && tau > 0.0 && tau < 1.0) {
        return Err(invalid("the quantile level must be in (0, 1)"));
    }
    let prepared = prepare(pairs, outcome, assignment, probability, cluster, floor)?;
    let estimate = evaluate(&prepared, None, tau).ok_or_else(|| {
        refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "temporal_design.arm_not_populated",
            "an arm has no paired decision, so its distribution function is not estimable",
        )
    })?;
    let n = prepared.outcome.len();
    if let Some(draws) = posterior_draws {
        let posterior = posterior_by_cluster_weights(prepared.clusters, 3, draws, ctx, |weights| {
            evaluate(&prepared, Some(weights), tau).map(|value| value.to_vec()).ok_or_else(|| {
                EstimationError::stats_msg("a posterior draw left an arm without weight")
            })
        })
        .map(|mut summary| {
            summary.method = "bayesian_bootstrap_weighted_quantile";
            summary
        })?;
        return Ok(QuantileFit {
            estimate,
            standard_error: None,
            interval: None,
            replicates: (0, 0),
            posterior: Some(posterior),
            pairs: n,
            clusters: prepared.clusters,
        });
    }
    if replicates < 99 {
        return Err(refuse(
            antecedent_core::reason_code!("required_option_missing"),
            "temporal_design.bootstrap_replicates",
            "the quantile interval is a whole-procedure cluster bootstrap and needs at least 99 \
             replicates. Next attempt: set bootstrap_replicates to 99 or more",
        ));
    }
    let g_count = prepared.clusters;
    let _lease = crate::util::admit_array::<f64>(ctx, replicates, 3 + g_count)?;
    let mut columns: [Vec<f64>; 3] = [
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
    ];
    let mut counts = vec![0.0; g_count];
    let mut failed = 0_usize;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, BOOTSTRAP_STREAM | replicate as u64);
        counts.iter_mut().for_each(|value| *value = 0.0);
        for _ in 0..g_count {
            counts[antecedent_kernels::unbiased_index(&mut rng, g_count)] += 1.0;
        }
        match evaluate(&prepared, Some(&counts), tau) {
            Some(values) => {
                for (column, value) in columns.iter_mut().zip(values) {
                    column.push(value);
                }
            }
            None => failed += 1,
        }
    }
    if failed as f64 > MAX_FAILED_SHARE * replicates as f64 {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "temporal_design.bootstrap_arm_support",
            &format!(
                "{failed} of {replicates} cluster-bootstrap replicates left an arm without a \
                 paired decision. Next attempt: more units, or arm probabilities further from \
                 the boundary"
            ),
        ));
    }
    let mut standard_error = [0.0; 3];
    let mut interval = [(0.0, 0.0); 3];
    for (j, column) in columns.iter_mut().enumerate() {
        standard_error[j] = crate::util::sample_std(column);
        column.sort_by(f64::total_cmp);
        interval[j] = antecedent_stats::equal_tail_interval_sorted(
            column,
            super::NOMINAL_LEVEL,
            antecedent_stats::QuantileRule::ExchangeableRank,
        );
    }
    Ok(QuantileFit {
        estimate,
        standard_error: Some(standard_error),
        interval: Some(interval),
        replicates: (replicates - failed, failed),
        posterior: None,
        pairs: n,
        clusters: g_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{PanelRows, lagged_pairs};
    use super::*;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x6A09_E667_F3BC_C908;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn the_median_shift_of_a_location_effect_is_recovered() {
        // Uniform(0, 1) noise; treatment shifts the outcome by 2, so every quantile moves by 2.
        let (mut unit, mut period, mut assignment, mut probability, mut outcome) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..250_u64 {
            for t in 0..6_u64 {
                let p = if t % 2 == 0 { 0.4 } else { 0.6 };
                let a = uniform(i * 31 + t) < p;
                unit.push(usize::try_from(i).unwrap());
                period.push(i64::try_from(t).unwrap());
                assignment.push(a);
                probability.push(p);
                outcome.push(uniform(i * 31 + t + 777_777) + if a { 2.0 } else { 0.0 });
            }
        }
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        let ctx = ExecutionContext::for_tests(11);
        let fit = fit_quantile(
            &pairs,
            &outcome,
            &assignment,
            &probability,
            &unit,
            0.05,
            0.5,
            199,
            None,
            &ctx,
        )
        .unwrap();
        assert!((fit.estimate[0] - 2.0).abs() < 0.1, "{:?}", fit.estimate);
        let (low, high) = fit.interval.unwrap()[0];
        assert!(low < fit.estimate[0] && fit.estimate[0] < high);
        assert!(high - low < 0.3);
        let bayes = fit_quantile(
            &pairs,
            &outcome,
            &assignment,
            &probability,
            &unit,
            0.05,
            0.5,
            0,
            Some(200),
            &ctx,
        )
        .unwrap();
        assert!((bayes.posterior.unwrap().mean[0] - 2.0).abs() < 0.1);
        assert!(
            fit_quantile(
                &pairs,
                &outcome,
                &assignment,
                &probability,
                &unit,
                0.05,
                0.5,
                10,
                None,
                &ctx
            )
            .is_err()
        );
    }
}
